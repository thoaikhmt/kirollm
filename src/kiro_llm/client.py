"""
The client that performs the ``GenerateAssistantResponse`` streaming call the
same way the Kiro CLI does:

* Bearer-token auth via the ``Authorization: Bearer <token>`` header (Kiro sends
  an SSO/IdC bearer token; the underlying Smithy client uses ``httpBearerAuth``).
* ``application/x-amz-json-1.0`` request with an ``X-Amz-Target`` header naming
  the operation ``AmazonCodeWhispererStreamingService.GenerateAssistantResponse``.
* A ``conversationState`` body built from :mod:`kiro_llm.models`.
* Response decoded as an ``application/vnd.amazon.eventstream`` byte stream and
  surfaced as typed ``ChatResponseStream`` events.

Region endpoints mirror the Kiro CLI's endpoint table (``q.<region>.amazonaws.com``
and the ``*.kiro.dev`` runtime hosts).
"""

from __future__ import annotations

import json
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterator, Optional

import requests

from .auth import (
    CredentialProvider,
    KiroCliCredentialProvider,
    KiroCredentialProvider,
    load_default_credentials,
)
from .eventstream import EventStreamMessage, iter_messages
from .models import (
    AssistantResponseEvent,
    ChatResponseStreamEvent,
    CodeReferenceEvent,
    ConversationState,
    FollowupPromptEvent,
    InvalidStateEvent,
    MetadataEvent,
    ReasoningEvent,
    ToolUseEvent,
    UnknownEvent,
)

_TARGET_SERVICE = "AmazonCodeWhispererStreamingService"
_GENERATE_ASSISTANT_RESPONSE = f"{_TARGET_SERVICE}.GenerateAssistantResponse"

# Region -> service endpoint, mirroring the Kiro CLI endpoint table.
CODEWHISPERER_ENDPOINTS: Dict[str, str] = {
    "us-east-1": "https://q.us-east-1.amazonaws.com",
    "eu-central-1": "https://q.eu-central-1.amazonaws.com",
    "us-gov-east-1": "https://q.us-gov-east-1.amazonaws.com",
    "us-gov-west-1": "https://q.us-gov-west-1.amazonaws.com",
}

# The Kiro-native runtime hosts (the "*.kiro.dev" family).
KIRO_RUNTIME_ENDPOINTS: Dict[str, str] = {
    "us-east-1": "https://runtime.us-east-1.kiro.dev",
    "eu-central-1": "https://runtime.eu-central-1.kiro.dev",
    "us-gov-east-1": "https://runtime.us-gov-east-1.kiro.dev",
    "us-gov-west-1": "https://runtime.us-gov-west-1.kiro.dev",
}


class KiroApiError(Exception):
    """Raised on a non-2xx HTTP response or an ``exception`` event-stream frame."""

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 error_type: Optional[str] = None, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.payload = payload


def resolve_endpoint(region: str, *, runtime: bool = False,
                     endpoint_override: Optional[str] = None) -> str:
    if endpoint_override:
        return endpoint_override.rstrip("/")
    table = KIRO_RUNTIME_ENDPOINTS if runtime else CODEWHISPERER_ENDPOINTS
    try:
        return table[region]
    except KeyError:
        raise ValueError(
            f"No endpoint known for region {region!r}. "
            f"Known: {sorted(table)}. Pass endpoint_override= to use a custom host."
        )


def _retry_delay(response: Any, attempt: int) -> float:
    """Return a bounded Retry-After or exponential delay in seconds."""
    value = response.headers.get("Retry-After") if hasattr(response, "headers") else None
    if value:
        try:
            return min(max(float(value), 0.0), 30.0)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                delay = retry_at.timestamp() - time.time()
                return min(max(delay, 0.0), 30.0)
            except (TypeError, ValueError, OverflowError):
                pass
    return min(0.5 * (2 ** (attempt - 1)), 8.0)


def _http_error_detail(response: Any) -> str:
    """Extract modeled error fields without echoing arbitrary response bodies."""
    try:
        payload = response.json()
    except (TypeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    reason = payload.get("reason") or payload.get("__type")
    message = payload.get("message")
    parts = [str(part).replace("\n", " ")[:200] for part in (reason, message) if part]
    return ": " + " - ".join(parts) if parts else ""


class KiroLLMClient:
    """
    Low-level client. Construct with a bearer token and region, then call
    :meth:`generate_assistant_response` (or the higher-level helpers in
    :mod:`kiro_llm.chat`).
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        *,
        region: Optional[str] = None,
        profile_arn: Optional[str] = None,
        agent_mode: Optional[str] = None,
        runtime: bool = False,
        endpoint_override: Optional[str] = None,
        user_agent: str = "kiro-llm-py/0.1.0",
        session: Optional[requests.Session] = None,
        timeout: float = 120.0,
        credential_provider: Optional[CredentialProvider] = None,
        max_attempts: int = 3,
    ):
        """
        Provide auth in one of three ways:

        * ``access_token=`` - use a bearer token you already have.
        * ``credential_provider=`` - a :class:`KiroCredentialProvider` that loads
          and auto-refreshes a token (see :meth:`from_kiro_login`).
        * neither - the client reads the local Kiro SQLite credential store or
          environment credentials; installing kiro-cli is not required.
        """
        if credential_provider is None and access_token is None:
            credential_provider = load_default_credentials()
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self._provider = credential_provider
        self._static_token = access_token

        # Region / profile_arn: explicit args win, else inherit from the provider.
        if region is None:
            region = credential_provider.region if credential_provider else "us-east-1"
        if profile_arn is None and credential_provider is not None:
            profile_arn = credential_provider.profile_arn

        self.region = region
        self.profile_arn = profile_arn
        self.agent_mode = agent_mode
        self.endpoint = resolve_endpoint(region, runtime=runtime,
                                         endpoint_override=endpoint_override)
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._session = session or requests.Session()

    @classmethod
    def from_kiro_cli(
        cls,
        *,
        executable: Optional[str] = None,
        credential_timeout: float = 30.0,
        **kwargs: Any,
    ) -> "KiroLLMClient":
        """Delegate token acquisition and refresh to the installed Kiro CLI."""
        provider = KiroCliCredentialProvider(
            executable=executable, timeout=credential_timeout
        )
        return cls(credential_provider=provider, **kwargs)

    @classmethod
    def from_kiro_login(
        cls,
        *,
        db_path: Optional[str] = None,
        token_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "KiroLLMClient":
        """Load the legacy local SQLite credential store directly.

        This path does not require the Kiro CLI. Direct automatic refresh there
        supports IdC credentials; use :meth:`from_kiro_cli` only when you
        explicitly want the CLI's provider-specific refresh behavior.
        """
        from .auth import DEFAULT_DB_PATH

        provider = KiroCredentialProvider.load(
            db_path=db_path or DEFAULT_DB_PATH, token_key=token_key
        )
        return cls(credential_provider=provider, **kwargs)

    def _current_access_token(self) -> str:
        if self._provider is not None:
            return self._provider.get_access_token()
        if self._static_token:
            return self._static_token
        raise KiroApiError("No access token or credential provider configured.")

    # ------------------------------------------------------------------ #
    # Header + body construction
    # ------------------------------------------------------------------ #
    def _headers(self, attempt: int = 1) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._current_access_token()}",
            "Content-Type": "application/x-amz-json-1.0",
            "X-Amz-Target": _GENERATE_ASSISTANT_RESPONSE,
            "User-Agent": self.user_agent,
            "Accept": "application/vnd.amazon.eventstream",
            "x-kiro-attempt": str(attempt),
        }
        # Header names observed in the Kiro CLI binary. A CLI-backed provider
        # learns the profile ARN only after its first token-helper invocation.
        profile_arn = self.profile_arn
        if profile_arn is None and self._provider is not None:
            profile_arn = self._provider.profile_arn
        if profile_arn:
            headers["x-amzn-kiro-profile-arn"] = profile_arn
        if self.agent_mode:
            headers["x-amzn-kiro-agent-mode"] = self.agent_mode
        return headers

    @staticmethod
    def _build_body(conversation_state: ConversationState) -> Dict[str, Any]:
        return {"conversationState": conversation_state.to_dict()}

    # ------------------------------------------------------------------ #
    # The streaming call
    # ------------------------------------------------------------------ #
    def generate_assistant_response(
        self, conversation_state: ConversationState
    ) -> Iterator[ChatResponseStreamEvent]:
        """
        POST a ``ConversationState`` and yield parsed ``ChatResponseStream``
        events as they stream back.
        """
        body = json.dumps(self._build_body(conversation_state))
        auth_refreshed = False

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._session.post(
                    self.endpoint + "/",
                    headers=self._headers(attempt),
                    data=body,
                    stream=True,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_attempts:
                    raise KiroApiError(
                        f"Request to {self.endpoint} failed after {attempt} attempts"
                    ) from exc
                time.sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))
                continue

            if (
                response.status_code in (401, 403)
                and self._provider is not None
                and not auth_refreshed
            ):
                response.close()
                self._provider.get_access_token(force_refresh=True)
                auth_refreshed = True
                continue

            if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_attempts:
                delay = _retry_delay(response, attempt)
                response.close()
                time.sleep(delay)
                continue

            if response.status_code >= 400:
                try:
                    detail = _http_error_detail(response)
                finally:
                    response.close()
                raise KiroApiError(
                    f"HTTP {response.status_code} from {self.endpoint}{detail}",
                    status_code=response.status_code,
                )

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/vnd.amazon.eventstream":
                response.close()
                raise KiroApiError(
                    "Expected application/vnd.amazon.eventstream response, "
                    f"received {content_type or 'no Content-Type'}",
                    status_code=response.status_code,
                )

            try:
                for message in iter_messages(response.iter_content(chunk_size=64 * 1024)):
                    yield _parse_event(message)
            finally:
                response.close()
            return

        raise KiroApiError(f"Request to {self.endpoint} exhausted all attempts")

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "KiroLLMClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Map a decoded event-stream message to a typed ChatResponseStream event.
# --------------------------------------------------------------------------- #

def _payload_json(message: EventStreamMessage) -> Dict[str, Any]:
    if not message.payload:
        return {}
    try:
        payload = json.loads(message.payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise KiroApiError(
            f"Invalid JSON payload in {message.event_type or message.message_type or 'stream'} event"
        ) from exc
    if not isinstance(payload, dict):
        raise KiroApiError(
            f"Expected object payload in {message.event_type or 'stream'} event"
        )
    return payload


def _parse_event(message: EventStreamMessage) -> ChatResponseStreamEvent:
    if message.is_exception:
        payload = _payload_json(message)
        etype = message.exception_type or message.event_type or "UnknownException"
        raise KiroApiError(
            payload.get("message", f"Stream returned exception {etype}"),
            error_type=etype,
            payload=payload,
        )

    event_type = message.event_type
    payload = _payload_json(message)

    if event_type == "assistantResponseEvent":
        return AssistantResponseEvent(content=payload.get("content", ""))

    if event_type == "reasoningEvent":
        return ReasoningEvent(content=payload.get("content", payload.get("text", "")))

    if event_type == "toolUseEvent":
        return ToolUseEvent(
            tool_use_id=payload.get("toolUseId"),
            name=payload.get("name"),
            input=payload.get("input"),
            stop=bool(payload.get("stop", False)),
        )

    if event_type == "codeReferenceEvent":
        return CodeReferenceEvent(references=payload.get("references", []))

    if event_type == "followupPromptEvent":
        followup = payload.get("followupPrompt", {})
        return FollowupPromptEvent(content=followup.get("content", payload.get("content", "")))

    if event_type in ("metadataEvent", "messageMetadataEvent", "meteringEvent",
                      "contextUsageEvent"):
        return MetadataEvent(raw=payload)

    if event_type == "invalidStateEvent":
        return InvalidStateEvent(
            reason=payload.get("reason", ""),
            message=payload.get("message", ""),
        )

    return UnknownEvent(event_type=event_type, raw=payload)
