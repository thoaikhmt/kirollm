import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from kiro_llm import KiroLLMClient
from kiro_llm.eventstream import encode_message
from kiro_llm.langchain import ChatKiro, _messages_to_kiro


def _event(event_type, payload):
    return encode_message(
        {":message-type": "event", ":event-type": event_type},
        json.dumps(payload).encode("utf-8"),
    )


class _FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.text = body.decode("utf-8", "replace")
        self.headers = {"Content-Type": "application/vnd.amazon.eventstream"}

    def iter_content(self, chunk_size=None):
        yield self._body

    def close(self):
        pass


class _FakeSession:
    def __init__(self, response):
        self._response = response
        self.last_body = None

    def post(self, url, headers=None, data=None, stream=None, timeout=None):
        self.last_body = json.loads(data)
        return self._response

    def close(self):
        pass


def _client(body):
    return KiroLLMClient(access_token="t", region="us-east-1",
                         session=_FakeSession(_FakeResponse(body)))


def test_is_langchain_chat_model():
    from langchain_core.language_models.chat_models import BaseChatModel

    llm = ChatKiro(client=_client(b""), model="m")
    assert isinstance(llm, BaseChatModel)
    assert llm._llm_type == "kiro"


def test_invoke_returns_ai_message():
    body = _event("assistantResponseEvent", {"content": "Hello from Kiro"})
    llm = ChatKiro(client=_client(body), model="m")
    result = llm.invoke("hi")
    assert isinstance(result, AIMessage)
    assert result.content == "Hello from Kiro"


def test_stream_yields_chunks():
    body = (
        _event("assistantResponseEvent", {"content": "a"})
        + _event("assistantResponseEvent", {"content": "b"})
    )
    llm = ChatKiro(client=_client(body), model="m")
    chunks = [c.content for c in llm.stream("hi")]
    assert "".join(chunks) == "ab"


def test_bind_tools_and_tool_calls():
    body = (
        _event("toolUseEvent", {"toolUseId": "t1", "name": "get_weather", "input": '{"city":"Paris"}', "stop": True})
    )
    client = _client(body)

    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    from langchain_core.tools import tool

    llm = ChatKiro(client=client, model="m").bind_tools([tool(get_weather)])
    result = llm.invoke("weather in Paris?")
    assert result.tool_calls
    assert result.tool_calls[0]["name"] == "get_weather"
    assert result.tool_calls[0]["args"] == {"city": "Paris"}


def test_message_translation_system_and_history():
    msgs = [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="first"),
        AIMessage(content="reply"),
        HumanMessage(content="second"),
    ]
    history, current, images, tool_results, system_prompt = _messages_to_kiro(msgs)
    assert len(history) == 2  # first user + assistant reply
    assert current == "second"
    assert system_prompt == "You are helpful."
    assert tool_results == []


def test_tool_message_becomes_result():
    msgs = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"id": "t1", "name": "f", "args": {}, "type": "tool_call"}]),
        ToolMessage(content="42", tool_call_id="t1"),
    ]
    history, current, images, tool_results, system_prompt = _messages_to_kiro(msgs)
    assert len(tool_results) == 1
    assert tool_results[0].tool_use_id == "t1"


def test_stream_preserves_tool_call_chunks():
    body = (
        _event("toolUseEvent", {
            "toolUseId": "t1", "name": "get_weather", "input": '{"city":'
        })
        + _event("toolUseEvent", {
            "toolUseId": "t1", "input": '"Paris"}', "stop": True
        })
    )
    llm = ChatKiro(client=_client(body), model="m")
    chunks = list(llm.stream("weather?"))
    merged = sum(chunks[1:], chunks[0])
    assert merged.tool_calls[0]["name"] == "get_weather"
    assert merged.tool_calls[0]["args"] == {"city": "Paris"}


def test_parallel_tool_messages_become_one_current_result_turn():
    msgs = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[
            {"id": "t1", "name": "a", "args": {}, "type": "tool_call"},
            {"id": "t2", "name": "b", "args": {}, "type": "tool_call"},
        ]),
        ToolMessage(content="one", tool_call_id="t1"),
        ToolMessage(content="two", tool_call_id="t2"),
    ]
    history, current, images, tool_results, system_prompt = _messages_to_kiro(msgs)
    assert current == ""
    assert [result.tool_use_id for result in tool_results] == ["t1", "t2"]
    assert len(history) == 2


def test_access_token_is_redacted_in_model_repr():
    llm = ChatKiro(access_token="super-secret", region="us-east-1", model="m")
    assert "super-secret" not in repr(llm)


def test_parallel_tool_messages_in_history_are_coalesced():
    msgs = [
        HumanMessage(content="first"),
        AIMessage(content="", tool_calls=[
            {"id": "t1", "name": "a", "args": {}, "type": "tool_call"},
            {"id": "t2", "name": "b", "args": {}, "type": "tool_call"},
        ]),
        ToolMessage(content="one", tool_call_id="t1"),
        ToolMessage(content="two", tool_call_id="t2"),
        AIMessage(content="done"),
        HumanMessage(content="next"),
    ]
    history, current, images, tool_results, system_prompt = _messages_to_kiro(msgs)
    assert current == "next"
    assert len(history) == 4
    assert len(history[2]["userInputMessage"]["userInputMessageContext"]["toolResults"]) == 2


def test_invoke_preserves_reasoning_and_metadata():
    body = (
        _event("reasoningEvent", {"content": "thinking"})
        + _event("assistantResponseEvent", {"content": "answer"})
        + _event("metadataEvent", {
            "requestId": "req-1",
            "usage": {"inputTokens": 3, "outputTokens": 2},
        })
    )
    result = ChatKiro(client=_client(body), model="m").invoke("hi")
    assert result.content == "answer"
    assert result.additional_kwargs["reasoning_content"] == "thinking"
    assert result.response_metadata["requestId"] == "req-1"
    assert result.usage_metadata["total_tokens"] == 5


def test_invalid_state_event_fails_langchain_generation():
    body = _event("invalidStateEvent", {"reason": "bad", "message": "invalid"})
    with pytest.raises(Exception, match="invalid"):
        ChatKiro(client=_client(body), model="m").invoke("hi")


def test_async_stream_is_incremental_api():
    import asyncio

    body = (
        _event("assistantResponseEvent", {"content": "a"})
        + _event("assistantResponseEvent", {"content": "b"})
    )
    llm = ChatKiro(client=_client(body), model="m")

    async def collect():
        return [chunk async for chunk in llm.astream("hi")]

    chunks = asyncio.run(collect())
    assert "".join(chunk.content for chunk in chunks) == "ab"
