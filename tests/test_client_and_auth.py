import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from kiro_llm import (
    AssistantResponseEvent,
    ChatSession,
    ConversationState,
    KiroApiError,
    KiroLLMClient,
    ToolSpecification,
    UserInputMessage,
    UserInputMessageContext,
)
from kiro_llm.auth import (
    KiroCliCredentialProvider,
    KiroCredentialProvider,
    SecretStore,
    load_default_credentials,
)
from kiro_llm.eventstream import encode_message


# --------------------------------------------------------------------------- #
# Payload building
# --------------------------------------------------------------------------- #

def test_conversation_state_payload_shape():
    tool = ToolSpecification(
        name="get_weather",
        description="Get weather",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    msg = UserInputMessage(
        content="hi",
        model_id="claude-sonnet-4",
        context=UserInputMessageContext(tools=[tool]),
    )
    state = ConversationState(current_message=msg)
    body = state.to_dict()

    assert "conversationId" in body
    assert body["chatTriggerType"] == "MANUAL"
    uim = body["currentMessage"]["userInputMessage"]
    assert uim["content"] == "hi"
    assert uim["modelId"] == "claude-sonnet-4"
    spec = uim["userInputMessageContext"]["tools"][0]["toolSpecification"]
    assert spec["name"] == "get_weather"
    assert spec["inputSchema"]["json"]["type"] == "object"


# --------------------------------------------------------------------------- #
# Client streaming against a mocked event-stream response
# --------------------------------------------------------------------------- #

def _event(event_type, payload):
    return encode_message(
        {":message-type": "event", ":event-type": event_type},
        json.dumps(payload).encode("utf-8"),
    )


class _FakeResponse:
    def __init__(self, body: bytes, status_code: int = 200, headers=None):
        self._body = body
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/vnd.amazon.eventstream"}
        self.text = body.decode("utf-8", "replace")
        self.closed = False

    def iter_content(self, chunk_size=None):
        # Stream in small chunks to exercise reassembly.
        for i in range(0, len(self._body), 7):
            yield self._body[i:i + 7]

    def json(self):
        return json.loads(self._body)

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, response):
        self._responses = list(response) if isinstance(response, list) else [response]
        self.last_headers = None
        self.last_body = None
        self.calls = 0

    def post(self, url, headers=None, data=None, json=None, stream=None, timeout=None):
        self.last_headers = headers
        self.last_body = data if data is not None else json
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response

    def close(self):
        pass


def test_client_streams_and_sets_bearer_header():
    body = (
        _event("assistantResponseEvent", {"content": "Hello "})
        + _event("assistantResponseEvent", {"content": "world"})
        + _event("metadataEvent", {"requestId": "abc"})
    )
    session = _FakeSession(_FakeResponse(body))
    client = KiroLLMClient(access_token="tok-123", region="us-east-1", session=session)

    events = list(
        client.generate_assistant_response(
            ConversationState(current_message=UserInputMessage(content="hi"))
        )
    )
    texts = [e.content for e in events if isinstance(e, AssistantResponseEvent)]
    assert texts == ["Hello ", "world"]
    assert session.last_headers["Authorization"] == "Bearer tok-123"
    assert (
        session.last_headers["X-Amz-Target"]
        == "AmazonCodeWhispererStreamingService.GenerateAssistantResponse"
    )
    assert session.last_headers["Content-Type"] == "application/x-amz-json-1.0"


def test_chat_session_send_and_history():
    body = (
        _event("assistantResponseEvent", {"content": "Hi!"})
        + _event("metadataEvent", {"requestId": "x"})
    )
    session = _FakeSession(_FakeResponse(body))
    client = KiroLLMClient(access_token="t", region="us-east-1", session=session)
    chat = ChatSession(client, model_id="m")

    reply = chat.send("hello")
    assert reply == "Hi!"
    # history: user turn + assistant turn
    assert len(chat.history) == 2
    assert chat.history[0]["userInputMessage"]["content"] == "hello"
    assert chat.history[1]["assistantResponseMessage"]["content"] == "Hi!"


def test_tool_use_events_collected():
    body = (
        _event("toolUseEvent", {"toolUseId": "t1", "name": "get_weather", "input": '{"city":'})
        + _event("toolUseEvent", {"toolUseId": "t1", "input": '"Paris"}', "stop": True})
    )
    session = _FakeSession(_FakeResponse(body))
    client = KiroLLMClient(access_token="t", region="us-east-1", session=session)
    chat = ChatSession(client, model_id="m")
    chat.send("weather?")
    assert len(chat.pending_tool_uses) == 1
    assert chat.pending_tool_uses[0].name == "get_weather"
    assert chat.pending_tool_uses[0].input == {"city": "Paris"}
    assert chat.history[-1]["assistantResponseMessage"]["toolUses"][0]["toolUseId"] == "t1"


# --------------------------------------------------------------------------- #
# Auth: load + refresh from a temp SQLite secret store
# --------------------------------------------------------------------------- #

def _make_store(tmp_path, token_record, registration=None):
    db = tmp_path / "data.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE auth_kv (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO auth_kv VALUES (?, ?)",
        ("kirocli:odic:token", json.dumps(token_record)),
    )
    if registration:
        conn.execute(
            "INSERT INTO auth_kv VALUES (?, ?)",
            ("kirocli:odic:device-registration", json.dumps(registration)),
        )
    conn.commit()
    conn.close()
    return str(db)


def test_load_valid_token(tmp_path):
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    db = _make_store(
        tmp_path,
        {"accessToken": "AT", "refreshToken": "RT", "expiresAt": future,
         "region": "us-east-1", "profileArn": "arn:aws:kiro:profile/x"},
    )
    provider = KiroCredentialProvider.load(db_path=db)
    assert provider.get_access_token() == "AT"
    assert provider.region == "us-east-1"
    assert provider.profile_arn == "arn:aws:kiro:profile/x"


def test_expired_token_triggers_refresh(tmp_path, monkeypatch):
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    db = _make_store(
        tmp_path,
        {"accessToken": "OLD", "refreshToken": "RT", "expiresAt": past, "region": "us-east-1"},
        registration={"clientId": "cid", "clientSecret": "secret"},
    )
    provider = KiroCredentialProvider.load(db_path=db)

    captured = {}

    class _Resp:
        status_code = 200
        headers = {}
        closed = False

        def json(self):
            return {"accessToken": "NEW", "refreshToken": "RT2", "expiresIn": 3600}

        def close(self):
            self.closed = True

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr("kiro_llm.auth.requests.post", fake_post)

    token = provider.get_access_token()
    assert token == "NEW"
    assert captured["url"] == "https://oidc.us-east-1.amazonaws.com/token"
    assert captured["payload"]["grantType"] == "refresh_token"
    assert captured["payload"]["refreshToken"] == "RT"

    # Refreshed token persisted back to the store.
    reloaded = SecretStore(db).get("kirocli:odic:token")
    assert reloaded["accessToken"] == "NEW"
    assert reloaded["refreshToken"] == "RT2"


def test_client_uses_credential_provider(tmp_path):
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    db = _make_store(
        tmp_path,
        {"accessToken": "PROV-TOK", "refreshToken": "RT", "expiresAt": future,
         "region": "us-east-1"},
    )
    body = _event("assistantResponseEvent", {"content": "ok"})
    session = _FakeSession(_FakeResponse(body))

    client = KiroLLMClient.from_kiro_login(db_path=db, session=session)
    list(client.generate_assistant_response(
        ConversationState(current_message=UserInputMessage(content="hi"))
    ))
    assert session.last_headers["Authorization"] == "Bearer PROV-TOK"


def test_response_is_closed_after_stream():
    response = _FakeResponse(_event("assistantResponseEvent", {"content": "ok"}))
    client = KiroLLMClient(access_token="t", session=_FakeSession(response))
    list(client.generate_assistant_response(
        ConversationState(current_message=UserInputMessage(content="hi"))
    ))
    assert response.closed is True


def test_non_eventstream_response_is_rejected_and_closed():
    response = _FakeResponse(
        b'{"message":"proxy response"}',
        headers={"Content-Type": "application/json"},
    )
    client = KiroLLMClient(access_token="t", session=_FakeSession(response))
    with pytest.raises(KiroApiError, match="Expected application/vnd.amazon.eventstream"):
        list(client.generate_assistant_response(
            ConversationState(current_message=UserInputMessage(content="hi"))
        ))
    assert response.closed is True


def test_unauthorized_response_forces_one_provider_refresh():
    class Provider:
        region = "us-east-1"
        profile_arn = None

        def __init__(self):
            self.token = "old"
            self.force_calls = 0

        def get_access_token(self, force_refresh=False):
            if force_refresh:
                self.force_calls += 1
                self.token = "new"
            return self.token

    provider = Provider()
    unauthorized = _FakeResponse(
        b'{"message":"expired"}',
        status_code=401,
        headers={"Content-Type": "application/json"},
    )
    success = _FakeResponse(_event("assistantResponseEvent", {"content": "ok"}))
    session = _FakeSession([unauthorized, success])
    client = KiroLLMClient(credential_provider=provider, session=session)

    events = list(client.generate_assistant_response(
        ConversationState(current_message=UserInputMessage(content="hi"))
    ))
    assert [event.content for event in events] == ["ok"]
    assert provider.force_calls == 1
    assert session.last_headers["Authorization"] == "Bearer new"
    assert unauthorized.closed is True


def test_secret_store_set_works_without_unique_constraint(tmp_path):
    db = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(str(db))
    connection.execute("CREATE TABLE auth_kv (key TEXT, value TEXT)")
    connection.execute("INSERT INTO auth_kv VALUES ('k', '{}')")
    connection.commit()
    connection.close()

    store = SecretStore(str(db))
    store.set("k", {"updated": True})
    assert store.get("k") == {"updated": True}


def test_cli_provider_parses_token_without_exposing_it(monkeypatch):
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace(
        "+00:00", "Z"
    )

    class Result:
        returncode = 0
        stderr = ""
        stdout = json.dumps({
            "kind": "getKasToken",
            "data": {
                "accessToken": "secret-token",
                "expiresAt": future,
                "region": "eu-central-1",
                "profileArn": "arn:example",
            },
        })

    monkeypatch.setattr("kiro_llm.auth.subprocess.run", lambda *args, **kwargs: Result())
    provider = KiroCliCredentialProvider(executable="kiro-cli")
    assert provider.get_access_token() == "secret-token"
    assert provider.region == "eu-central-1"
    assert provider.profile_arn == "arn:example"


def test_default_credentials_use_environment_without_cli(monkeypatch):
    monkeypatch.delenv("KIRO_API_KEY", raising=False)
    monkeypatch.setenv("KIRO_ACCESS_TOKEN", "environment-token")
    monkeypatch.setattr(
        "kiro_llm.auth.KiroCliCredentialProvider",
        lambda: pytest.fail("default auth must not instantiate kiro-cli provider"),
    )
    provider = load_default_credentials()
    assert provider.get_access_token() == "environment-token"


def test_api_key_mode_sets_kiro_headers_and_origin():
    body = _event("assistantResponseEvent", {"content": "ok"})
    session = _FakeSession(_FakeResponse(body))
    client = KiroLLMClient(
        access_token="ksk_test_key",
        region="us-east-1",
        session=session,
    )
    list(client.generate_assistant_response(
        ConversationState(current_message=UserInputMessage(content="hi"))
    ))
    assert session.last_headers["Authorization"] == "Bearer ksk_test_key"
    assert session.last_headers["tokentype"] == "API_KEY"
    assert session.last_headers["x-amzn-kiro-agent-mode"] == "vibe"
    assert session.last_headers["x-amzn-codewhisperer-optout"] == "true"
    payload = json.loads(session.last_body)
    assert (
        payload["conversationState"]["currentMessage"]["userInputMessage"]["origin"]
        == "AI_EDITOR"
    )


def test_list_models_parses_server_catalog():
    payload = {
        "defaultModel": {"modelId": "claude-sonnet-4"},
        "models": [
            {
                "modelId": "claude-sonnet-4",
                "modelName": "Claude Sonnet 4",
                "description": "fast",
                "supportedInputTypes": ["TEXT", "IMAGE"],
                "rateMultiplier": 1.0,
                "tokenLimits": {"maxInputTokens": 200000, "maxOutputTokens": 8192},
            },
            {"modelId": "custom-model", "name": "Custom"},
        ],
    }
    response = _FakeResponse(json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    session = _FakeSession(response)
    client = KiroLLMClient(access_token="ksk_test_key", session=session)

    catalog = client.list_models()
    assert catalog.default_model == "claude-sonnet-4"
    assert catalog.model_ids == ("claude-sonnet-4", "custom-model")
    assert catalog.get("claude-sonnet-4").supported_input_types == ("TEXT", "IMAGE")
    assert session.last_headers["X-Amz-Target"] == "AmazonCodeWhispererService.ListAvailableModels"
    assert session.last_body == {"origin": "AI_EDITOR"}
