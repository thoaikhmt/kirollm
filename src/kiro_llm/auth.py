"""
Authentication: load the Kiro CLI's stored API token and automatically obtain /
refresh a valid access token -- the same mechanism the Kiro CLI uses.

Where Kiro keeps its credentials
--------------------------------
The Kiro CLI stores auth material in a SQLite "secret store" at
``~/.local/share/kiro-cli/data.sqlite3`` in a table ``auth_kv(key TEXT, value TEXT)``.
The relevant keys (observed in the CLI binary) are:

* ``kirocli:odic:token``               - the Builder ID / IdC token record
* ``kirocli:odic:device-registration`` - the OIDC client registration (used to refresh)
* ``kirocli:social:token`` / ``kirocli:external-idp:token`` - other login methods

A token record is JSON with fields such as ``accessToken``, ``refreshToken``,
``expiresAt`` (RFC3339), ``region``, ``startUrl``, ``profileArn``. The device
registration holds ``clientId`` / ``clientSecret``.

Refresh
-------
When the access token is expired (or within a safety window), we refresh it via
AWS SSO-OIDC ``CreateToken`` with ``grant_type=refresh_token`` against
``https://oidc.<region>.amazonaws.com/token`` -- exactly what the CLI does. The
refreshed token is written back to the store so subsequent runs reuse it.

This module has no dependency beyond ``requests`` (already required).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

import requests

DEFAULT_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".local", "share", "kiro-cli", "data.sqlite3"
)

TOKEN_KEY_IDC = "kirocli:odic:token"
TOKEN_KEY_SOCIAL = "kirocli:social:token"
TOKEN_KEY_EXTERNAL = "kirocli:external-idp:token"
DEVICE_REGISTRATION_KEY = "kirocli:odic:device-registration"

# Refresh a little before actual expiry to avoid mid-request expiration.
_EXPIRY_SKEW_SECONDS = 120


class AuthError(Exception):
    """Raised when a token cannot be loaded or refreshed."""


class CredentialProvider(Protocol):
    """Minimal interface consumed by :class:`KiroLLMClient`."""

    @property
    def region(self) -> str: ...

    @property
    def profile_arn(self) -> Optional[str]: ...

    def get_access_token(self, force_refresh: bool = False) -> str: ...


def _parse_rfc3339(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    v = str(value).strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        try:
            return _parse_rfc3339(float(v))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class Token:
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    region: str = "us-east-1"
    start_url: Optional[str] = None
    profile_arn: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)  # original JSON record

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "Token":
        return cls(
            access_token=record.get("accessToken") or record.get("access_token") or "",
            refresh_token=record.get("refreshToken") or record.get("refresh_token"),
            expires_at=_parse_rfc3339(record.get("expiresAt") or record.get("expires_at") or ""),
            region=record.get("region") or "us-east-1",
            start_url=record.get("startUrl") or record.get("start_url"),
            profile_arn=record.get("profileArn") or record.get("profile_arn"),
            raw=record,
        )

    def is_expired(self, skew_seconds: int = _EXPIRY_SKEW_SECONDS) -> bool:
        if self.expires_at is None:
            return False  # no expiry info -> assume usable
        now = datetime.now(timezone.utc)
        return (self.expires_at - now).total_seconds() <= skew_seconds


class SecretStore:
    """Read/write the Kiro CLI ``auth_kv`` SQLite secret store."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        if not os.path.exists(self.db_path):
            raise AuthError(
                f"Kiro credential store not found at {self.db_path}. "
                "Run `kiro-cli login` first, or pass an access_token explicitly."
            )
        return sqlite3.connect(self.db_path, timeout=10.0)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT value FROM auth_kv WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
        if not row or row[0] is None:
            return None
        try:
            return json.loads(row[0])
        except (ValueError, TypeError):
            return None

    def set(self, key: str, record: Dict[str, Any]) -> None:
        conn = self._connect()
        serialized = json.dumps(record)
        try:
            cursor = conn.execute(
                "UPDATE auth_kv SET value = ? WHERE key = ?", (serialized, key)
            )
            if cursor.rowcount == 0:
                conn.execute(
                    "INSERT INTO auth_kv (key, value) VALUES (?, ?)",
                    (key, serialized),
                )
            conn.commit()
        finally:
            conn.close()


class KiroCredentialProvider:
    """
    Loads a token from the Kiro secret store and hands out a valid access token,
    refreshing automatically when expired. Thread-safe.

    Usage::

        provider = KiroCredentialProvider.load()
        token = provider.get_access_token()   # always valid
        region = provider.region
        profile_arn = provider.profile_arn
    """

    def __init__(
        self,
        token: Token,
        store: Optional[SecretStore] = None,
        token_key: str = TOKEN_KEY_IDC,
        registration: Optional[Dict[str, Any]] = None,
    ):
        self._token = token
        self._store = store
        self._token_key = token_key
        self._registration = registration or {}
        self._lock = threading.Lock()

    # ---- construction -------------------------------------------------- #
    @classmethod
    def load(
        cls,
        db_path: str = DEFAULT_DB_PATH,
        token_key: Optional[str] = None,
    ) -> "KiroCredentialProvider":
        """
        Load a credential provider from the Kiro secret store. If ``token_key``
        is not given, tries the IdC, social, then external-IdP keys in order.
        """
        store = SecretStore(db_path)
        keys = [token_key] if token_key else [
            TOKEN_KEY_IDC, TOKEN_KEY_SOCIAL, TOKEN_KEY_EXTERNAL,
        ]
        last_key = None
        record = None
        for key in keys:
            record = store.get(key)
            if record and (record.get("accessToken") or record.get("access_token")):
                last_key = key
                break
        if not record:
            raise AuthError(
                "No Kiro token found in the credential store "
                f"({db_path}). Log in with `kiro-cli login` first."
            )

        token = Token.from_record(record)
        registration = store.get(DEVICE_REGISTRATION_KEY) or {}
        if registration:
            token.client_id = registration.get("clientId") or registration.get("client_id")
            token.client_secret = (
                registration.get("clientSecret") or registration.get("client_secret")
            )
        assert last_key is not None
        return cls(token, store=store, token_key=last_key, registration=registration)

    @classmethod
    def from_token(
        cls,
        access_token: str,
        *,
        refresh_token: Optional[str] = None,
        expires_at: Optional[str] = None,
        region: str = "us-east-1",
        profile_arn: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> "KiroCredentialProvider":
        """Build a provider from explicit values (no secret store on disk)."""
        token = Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=_parse_rfc3339(expires_at) if expires_at else None,
            region=region,
            profile_arn=profile_arn,
            client_id=client_id,
            client_secret=client_secret,
        )
        return cls(token, store=None)

    # ---- accessors ----------------------------------------------------- #
    @property
    def region(self) -> str:
        return self._token.region

    @property
    def profile_arn(self) -> Optional[str]:
        return self._token.profile_arn

    def get_access_token(self, force_refresh: bool = False) -> str:
        """Return a valid access token, refreshing if necessary."""
        with self._lock:
            if force_refresh or self._token.is_expired():
                self._refresh_locked()
            return self._token.access_token

    # ---- refresh ------------------------------------------------------- #
    def _oidc_endpoint(self) -> str:
        return f"https://oidc.{self._token.region}.amazonaws.com/token"

    def _refresh_locked(self) -> None:
        if self._token_key != TOKEN_KEY_IDC:
            raise AuthError(
                "Direct refresh is only supported for the Kiro IdC token. "
                "Use KiroCliCredentialProvider so the Kiro CLI can refresh this "
                f"{self._token_key!r} credential with its provider-specific flow."
            )
        if not self._token.refresh_token:
            raise AuthError(
                "Access token is expired and no refresh token is available. "
                "Re-run `kiro-cli login`."
            )
        if not self._token.client_id or not self._token.client_secret:
            raise AuthError(
                "Cannot refresh: missing OIDC client registration "
                "(clientId/clientSecret). Re-run `kiro-cli login`."
            )

        payload = {
            "grantType": "refresh_token",
            "clientId": self._token.client_id,
            "clientSecret": self._token.client_secret,
            "refreshToken": self._token.refresh_token,
        }
        try:
            resp = requests.post(
                self._oidc_endpoint(),
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise AuthError("Token refresh request failed") from exc
        try:
            if resp.status_code >= 400:
                request_id = resp.headers.get("x-amzn-requestid", "unknown")
                raise AuthError(
                    f"Token refresh failed (HTTP {resp.status_code}, "
                    f"request id {request_id})"
                )
            try:
                data = resp.json()
            except (ValueError, requests.JSONDecodeError) as exc:
                raise AuthError("Token refresh returned invalid JSON") from exc
        finally:
            resp.close()

        new_access = data.get("accessToken") or data.get("access_token")
        if not new_access:
            raise AuthError("Token refresh response did not contain an access token.")

        self._token.access_token = new_access
        # SSO-OIDC may or may not rotate the refresh token.
        self._token.refresh_token = (
            data.get("refreshToken") or data.get("refresh_token")
            or self._token.refresh_token
        )
        expires_in = data.get("expiresIn") or data.get("expires_in")
        if expires_in:
            from datetime import timedelta
            self._token.expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=int(expires_in)
            )

        self._persist()

    def _persist(self) -> None:
        """Write the refreshed token back to the secret store, if we have one."""
        if not self._store or not self._token_key:
            return
        record = dict(self._token.raw or {})
        record["accessToken"] = self._token.access_token
        if self._token.refresh_token:
            record["refreshToken"] = self._token.refresh_token
        if self._token.expires_at:
            record["expiresAt"] = self._token.expires_at.astimezone(
                timezone.utc
            ).isoformat().replace("+00:00", "Z")
        try:
            self._store.set(self._token_key, record)
            self._token.raw = record
        except (AuthError, sqlite3.Error, OSError) as exc:
            warnings.warn(
                f"Refreshed token is valid in memory but could not be persisted: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


class KiroCliCredentialProvider:
    """Acquire and refresh credentials through Kiro's own token helper.

    This is the preferred provider because the CLI knows how to refresh Builder
    ID, social, external-IdP, and API-key sessions. Credential values are parsed
    from stdout in memory and are never logged or persisted by this library.
    """

    def __init__(
        self,
        executable: Optional[str] = None,
        *,
        timeout: float = 30.0,
        region: str = "us-east-1",
    ):
        self.executable = executable or shutil.which("kiro-cli") or shutil.which("kiro")
        if not self.executable:
            raise AuthError(
                "Kiro CLI executable not found. Install/login to Kiro, set "
                "KIRO_ACCESS_TOKEN, or use KiroCredentialProvider.load()."
            )
        self.timeout = timeout
        self._region = region
        self._profile_arn: Optional[str] = None
        self._token: Optional[Token] = None
        self._lock = threading.Lock()

    @property
    def region(self) -> str:
        return self._region

    @property
    def profile_arn(self) -> Optional[str]:
        return self._profile_arn

    def get_access_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            if force_refresh or self._token is None or self._token.is_expired():
                self._load_from_cli(force_refresh=force_refresh)
            assert self._token is not None
            return self._token.access_token

    def _load_from_cli(self, *, force_refresh: bool) -> None:
        command = [self.executable, "chat", "_", "get-kas-token"]
        if force_refresh:
            command.append("--force-refresh")
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AuthError("Failed to invoke the Kiro credential helper") from exc

        try:
            envelope = json.loads(result.stdout)
        except (TypeError, ValueError) as exc:
            raise AuthError(
                "Kiro credential helper returned an invalid response"
            ) from exc
        if not isinstance(envelope, dict) or envelope.get("kind") != "getKasToken":
            data = envelope.get("data") if isinstance(envelope, dict) else None
            message = data.get("message") if isinstance(data, dict) else None
            raise AuthError(message or "Kiro CLI could not provide an access token")
        data = envelope.get("data")
        if not isinstance(data, dict):
            raise AuthError("Kiro credential helper response has no data object")

        token = Token.from_record(data)
        if not token.access_token:
            raise AuthError("Kiro credential helper response has no access token")
        self._token = token
        self._region = token.region
        self._profile_arn = token.profile_arn


def load_default_credentials() -> CredentialProvider:
    """Resolve credentials without exposing secrets.

    Explicit environment credentials win. Otherwise delegate to the installed
    Kiro CLI, which owns all provider-specific refresh behavior.
    """

    explicit = os.environ.get("KIRO_ACCESS_TOKEN") or os.environ.get("KIRO_API_KEY")
    if explicit:
        return KiroCredentialProvider.from_token(explicit)
    return KiroCliCredentialProvider()
