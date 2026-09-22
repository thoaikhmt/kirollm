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


def test_with_structured_output_pydantic_returns_model():
    from pydantic import BaseModel

    class Answer(BaseModel):
        """A concise answer with confidence."""

        answer: str
        confidence: float

    body = _event(
        "toolUseEvent",
        {
            "toolUseId": "structured-1",
            "name": "Answer",
            "input": '{"answer":"yes","confidence":0.9}',
            "stop": True,
        },
    )
    runnable = ChatKiro(client=_client(body)).with_structured_output(Answer)
    result = runnable.invoke("Return an answer")
    assert isinstance(result, Answer)
    assert result.answer == "yes"
    assert result.confidence == 0.9


def test_with_structured_output_include_raw_matches_chatopenai_shape():
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    body = _event(
        "toolUseEvent",
        {
            "toolUseId": "structured-2",
            "name": "Answer",
            "input": '{"answer":"ok"}',
            "stop": True,
        },
    )
    result = ChatKiro(client=_client(body)).with_structured_output(
        Answer, include_raw=True, strict=True
    ).invoke("Return structured output")
    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert isinstance(result["raw"], AIMessage)
    assert result["parsed"] == Answer(answer="ok")
    assert result["parsing_error"] is None


def test_with_structured_output_typeddict_returns_dict():
    from typing import TypedDict

    class AnswerDict(TypedDict):
        answer: str
        count: int

    body = _event(
        "toolUseEvent",
        {
            "toolUseId": "structured-3",
            "name": "AnswerDict",
            "input": '{"answer":"ok","count":2}',
            "stop": True,
        },
    )
    result = ChatKiro(client=_client(body)).with_structured_output(
        AnswerDict, method="function_calling"
    ).invoke("Return a dict")
    assert result == {"answer": "ok", "count": 2}


def test_with_structured_output_json_schema_returns_dict():
    schema = {
        "title": "WeatherAnswer",
        "description": "Weather response",
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    body = _event(
        "toolUseEvent",
        {
            "toolUseId": "structured-4",
            "name": "WeatherAnswer",
            "input": '{"city":"Paris"}',
            "stop": True,
        },
    )
    result = ChatKiro(client=_client(body)).with_structured_output(schema).invoke(
        "Return weather"
    )
    assert result == {"city": "Paris"}


@pytest.mark.parametrize(
    "tool_choice",
    [
        None,
        False,
        "auto",
        "any",
        "required",
        True,
        "get_weather",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
)
def test_bind_tools_accepts_chatopenai_tool_choice_forms(tool_choice):
    def get_weather(city: str) -> str:
        """Get weather."""
        return city

    bound = ChatKiro(client=_client(b"")).bind_tools(
        [get_weather],
        tool_choice=tool_choice,
        strict=True,
        parallel_tool_calls=False,
        ls_structured_output_format={"schema": "internal"},
    )
    assert bound is not None


def test_bind_tools_rejects_unknown_named_choice():
    def known(value: str) -> str:
        """Known tool."""
        return value

    with pytest.raises(ValueError, match="unknown tool"):
        ChatKiro(client=_client(b"")).bind_tools([known], tool_choice="missing")


def test_parallel_tool_calls_false_is_enforced_locally():
    def first(value: str) -> str:
        """First tool."""
        return value

    def second(value: str) -> str:
        """Second tool."""
        return value

    body = (
        _event(
            "toolUseEvent",
            {"toolUseId": "one", "name": "first", "input": '{"value":"a"}'},
        )
        + _event(
            "toolUseEvent",
            {"toolUseId": "two", "name": "second", "input": '{"value":"b"}'},
        )
    )
    llm = ChatKiro(client=_client(body)).bind_tools(
        [first, second], parallel_tool_calls=False
    )
    with pytest.raises(Exception, match="parallel tool calls"):
        llm.invoke("Call tools")


def test_with_structured_output_rejects_openai_only_methods():
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    llm = ChatKiro(client=_client(b""))
    for method in ("json_mode", "json_schema"):
        with pytest.raises(ValueError, match="function_calling"):
            llm.with_structured_output(Answer, method=method)


def test_bind_tools_response_format_rejected_with_guidance():
    def tool(value: str) -> str:
        """A tool."""
        return value

    with pytest.raises(ValueError, match="with_structured_output"):
        ChatKiro(client=_client(b"")).bind_tools(
            [tool], response_format={"type": "json_object"}
        )


def test_with_structured_output_include_raw_captures_validation_error():
    from pydantic import BaseModel

    class Answer(BaseModel):
        count: int

    body = _event(
        "toolUseEvent",
        {
            "toolUseId": "structured-invalid",
            "name": "Answer",
            "input": '{"count":"not-an-integer"}',
            "stop": True,
        },
    )
    result = ChatKiro(client=_client(body)).with_structured_output(
        Answer, include_raw=True
    ).invoke("Return invalid output")
    assert isinstance(result["raw"], AIMessage)
    assert result["parsed"] is None
    assert result["parsing_error"] is not None


def test_langgraph_tool_loop_uses_chatkiro():
    from typing import Annotated, TypedDict

    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    first = _FakeResponse(
        _event(
            "toolUseEvent",
            {
                "toolUseId": "add-1",
                "name": "add",
                "input": '{"a":2,"b":3}',
                "stop": True,
            },
        )
    )
    second = _FakeResponse(
        _event("assistantResponseEvent", {"content": "5"})
    )

    class SequentialSession:
        def __init__(self):
            self.responses = iter([first, second])

        def post(self, *args, **kwargs):
            return next(self.responses)

        def close(self):
            pass

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    client = KiroLLMClient(
        access_token="test-token",
        session=SequentialSession(),
    )
    model = ChatKiro(client=client).bind_tools([add])

    def call_model(state: State):
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode([add]))
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", tools_condition, {"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "model")
    result = builder.compile().invoke(
        {"messages": [HumanMessage(content="What is 2 + 3?")]}
    )

    assert result["messages"][-1].content == "5"
    assert any(isinstance(message, ToolMessage) for message in result["messages"])


def test_chatopenai_compatible_public_parameter_names():
    import inspect

    bind_parameters = inspect.signature(ChatKiro.bind_tools).parameters
    assert {
        "tools",
        "tool_choice",
        "strict",
        "parallel_tool_calls",
        "response_format",
    }.issubset(bind_parameters)

    structured_parameters = inspect.signature(
        ChatKiro.with_structured_output
    ).parameters
    assert {
        "schema",
        "method",
        "include_raw",
        "strict",
        "tools",
    }.issubset(structured_parameters)


def test_required_tool_choice_raises_when_model_does_not_call_tool():
    def lookup(value: str) -> str:
        """Look up a value."""
        return value

    body = _event("assistantResponseEvent", {"content": "direct answer"})
    llm = ChatKiro(client=_client(body)).bind_tools(
        [lookup], tool_choice="required"
    )
    with pytest.raises(Exception, match="required tool call"):
        llm.invoke("Use the tool")


def test_with_structured_output_validates_arguments():
    llm = ChatKiro(client=_client(b""))
    with pytest.raises(ValueError, match="schema is required"):
        llm.with_structured_output(None)
    with pytest.raises(ValueError, match="Additional tools"):
        llm.with_structured_output(
            {"title": "Answer", "type": "object", "properties": {}},
            tools=[{"name": "other"}],
        )
    with pytest.raises(ValueError, match="unsupported arguments"):
        llm.with_structured_output(
            {"title": "Answer", "type": "object", "properties": {}},
            unsupported=True,
        )
