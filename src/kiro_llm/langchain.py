"""
LangChain / LangGraph integration.

``ChatKiro`` is a LangChain chat model (a subclass of
``langchain_core.language_models.chat_models.BaseChatModel`` -- the same base
class ``ChatOpenAI`` extends). That means it plugs directly into LangChain chains
and LangGraph graphs, supports ``.invoke`` / ``.stream`` / ``.bind_tools`` and the
standard message types, while under the hood it calls the model exactly the way
the Kiro CLI does (``GenerateAssistantResponse`` streaming + event-stream decode).

LangChain is an optional dependency; importing this module without it installed
raises a clear error.

Example
-------
    from kiro_llm.langchain import ChatKiro

    # Auto-loads the Kiro CLI's stored login and auto-refreshes the token.
    llm = ChatKiro(model="claude-sonnet-4")
    print(llm.invoke("Write a haiku about event streams").content)

    # Streaming
    for chunk in llm.stream("Count to five"):
        print(chunk.content, end="")

    # Tool calling / LangGraph
    llm_with_tools = llm.bind_tools([my_tool])
"""

from __future__ import annotations

# pyright: reportPossiblyUnboundVariable=false, reportRedeclaration=false

import json
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union, cast

from .client import KiroLLMClient
from .models import (
    AssistantResponseEvent,
    AssistantResponseMessage,
    ChatTriggerType,
    ConversationState,
    ToolResult,
    ToolSpecification,
    ToolUse,
    ToolUseEvent,
    UserInputMessage,
    UserInputMessageContext,
)

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import Field, PrivateAttr, SecretStr

    _LANGCHAIN_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - exercised only without langchain
    _LANGCHAIN_AVAILABLE = False
    _IMPORT_ERROR = _exc


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "ChatKiro requires langchain-core. Install it with "
            "`pip install langchain-core` (or `pip install kiro-llm[langchain]`)."
        ) from _IMPORT_ERROR


if _LANGCHAIN_AVAILABLE:

    class ChatKiro(BaseChatModel):
        """A LangChain chat model backed by the Kiro ``GenerateAssistantResponse`` API."""

        model: Optional[str] = Field(default=None, description="Model id to send.")
        region: Optional[str] = Field(default=None)
        profile_arn: Optional[str] = Field(default=None)
        access_token: Optional[SecretStr] = Field(default=None, repr=False)
        agent_mode: Optional[str] = Field(default=None)
        endpoint_override: Optional[str] = Field(default=None)
        runtime: bool = Field(default=False)
        timeout: float = Field(default=120.0)

        _client: KiroLLMClient = PrivateAttr()
        _bound_tools: List[ToolSpecification] = PrivateAttr(default_factory=list)

        def __init__(self, client: Optional[KiroLLMClient] = None, **kwargs: Any):
            _require_langchain()
            super().__init__(**kwargs)
            if client is None:
                client = KiroLLMClient(
                    access_token=(
                        self.access_token.get_secret_value()
                        if self.access_token is not None
                        else None
                    ),
                    region=self.region,
                    profile_arn=self.profile_arn,
                    agent_mode=self.agent_mode,
                    runtime=self.runtime,
                    endpoint_override=self.endpoint_override,
                    timeout=self.timeout,
                )
            self._client = client
            self._bound_tools = []

        # -- LangChain plumbing ----------------------------------------- #
        @property
        def _llm_type(self) -> str:
            return "kiro"

        @property
        def _identifying_params(self) -> Dict[str, Any]:
            return {"model": self.model, "region": self._client.region}

        # -- tool binding ----------------------------------------------- #
        def bind_tools(
            self,
            tools: Sequence[
                Union[Dict[str, Any], type, Callable[..., Any], "BaseTool"]
            ],
            *,
            tool_choice: Optional[str] = None,
            **kwargs: Any,
        ) -> "Runnable[LanguageModelInput, AIMessage]":
            if tool_choice not in (None, "auto"):
                raise NotImplementedError(
                    "Kiro does not expose a portable tool_choice control"
                )
            if kwargs:
                raise TypeError(
                    f"Unsupported bind_tools options: {', '.join(sorted(kwargs))}"
                )
            specs = [_to_tool_specification(t) for t in tools]
            new = ChatKiro(
                client=self._client,
                model=self.model,
                region=self.region,
                profile_arn=self.profile_arn,
                agent_mode=self.agent_mode,
                runtime=self.runtime,
                endpoint_override=self.endpoint_override,
                timeout=self.timeout,
            )
            new._bound_tools = list(specs)  # pyright: ignore[reportAttributeAccessIssue]
            return new  # pyright: ignore[reportReturnType]

        # -- core generation -------------------------------------------- #
        def _build_state(self, messages: List[BaseMessage]) -> ConversationState:
            history, current, tool_results = _messages_to_kiro(messages)
            ctx = UserInputMessageContext(
                tools=self._bound_tools or None,
                tool_results=tool_results or None,
            )
            current_message = UserInputMessage(
                content=current,
                model_id=self.model,
                context=ctx if (self._bound_tools or tool_results) else None,
            )
            return ConversationState(
                current_message=current_message,
                chat_trigger_type=ChatTriggerType.MANUAL,
                history=history,
            )

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional["CallbackManagerForLLMRun"] = None,
            **kwargs: Any,
        ) -> "ChatResult":
            if stop:
                raise NotImplementedError("Kiro does not support stop sequences")
            state = self._build_state(messages)
            text_parts: List[str] = []
            tool_calls = _ToolCallAccumulator()

            for event in self._client.generate_assistant_response(state):
                if isinstance(event, AssistantResponseEvent):
                    text_parts.append(event.content)
                    if run_manager:
                        run_manager.on_llm_new_token(event.content)
                elif isinstance(event, ToolUseEvent):
                    tool_calls.add(event)

            message = AIMessage(
                content="".join(text_parts),
                tool_calls=tool_calls.finalize(),
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

        def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional["CallbackManagerForLLMRun"] = None,
            **kwargs: Any,
        ) -> Iterator["ChatGenerationChunk"]:
            if stop:
                raise NotImplementedError("Kiro does not support stop sequences")
            state = self._build_state(messages)
            tool_indexes: Dict[str, int] = {}
            for event in self._client.generate_assistant_response(state):
                if isinstance(event, AssistantResponseEvent):
                    if run_manager:
                        run_manager.on_llm_new_token(event.content)
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=event.content)
                    )
                elif isinstance(event, ToolUseEvent):
                    key = event.tool_use_id or (
                        next(reversed(tool_indexes)) if tool_indexes else "tool_0"
                    )
                    if key not in tool_indexes:
                        tool_indexes[key] = len(tool_indexes)
                    if isinstance(event.input, dict):
                        arguments = json.dumps(event.input, separators=(",", ":"))
                    else:
                        arguments = event.input or ""
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            tool_call_chunks=[{
                                "name": event.name,
                                "args": arguments,
                                "id": event.tool_use_id,
                                "index": tool_indexes[key],
                                "type": "tool_call_chunk",
                            }],
                        )
                    )

else:  # pragma: no cover

    class ChatKiro:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            _require_langchain()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_tool_specification(tool: Any) -> ToolSpecification:
    """Convert a LangChain tool / pydantic / dict into a Kiro ToolSpecification."""
    openai_tool = convert_to_openai_tool(tool)
    fn = openai_tool["function"]
    return ToolSpecification(
        name=fn["name"],
        description=fn.get("description", ""),
        input_schema=fn.get("parameters", {"type": "object", "properties": {}}),
    )


def _messages_to_kiro(messages: List["BaseMessage"]):
    """
    Translate LangChain messages into (history, current_content, tool_results).

    * SystemMessage is prepended to the first user turn's content.
    * The trailing message determines the "current" turn.
    * ToolMessages become Kiro ToolResults on the current turn.
    """
    history: List[Dict[str, Any]] = []
    system_prefix = ""
    tool_results: List[ToolResult] = []

    non_system: List["BaseMessage"] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system_prefix += _text(message.content) + "\n"
        else:
            non_system.append(message)

    if not non_system:
        return history, system_prefix.strip(), tool_results

    trailing_tool_start = len(non_system)
    while trailing_tool_start > 0 and isinstance(
        non_system[trailing_tool_start - 1], ToolMessage
    ):
        trailing_tool_start -= 1

    if trailing_tool_start < len(non_system):
        prior = non_system[:trailing_tool_start]
        trailing_tools = non_system[trailing_tool_start:]
        tool_results = [
            _tool_message_to_result(cast(ToolMessage, message))
            for message in trailing_tools
        ]
        current_content = ""
    else:
        prior = non_system[:-1]
        last = non_system[-1]
        if not isinstance(last, HumanMessage):
            raise ValueError(
                "The final LangChain message must be HumanMessage or ToolMessage"
            )
        current_content = _text(last.content)

    for message in prior:
        if isinstance(message, HumanMessage):
            history.append(UserInputMessage(content=_text(message.content)).to_dict())
        elif isinstance(message, AIMessage):
            tool_uses = _ai_tool_uses(cast(AIMessage, message))
            history.append(
                AssistantResponseMessage(
                    content=_text(message.content), tool_uses=tool_uses or None
                ).to_dict()
            )
        elif isinstance(message, ToolMessage):
            history.append(
                UserInputMessage(
                    content="",
                    context=UserInputMessageContext(
                        tool_results=[
                            _tool_message_to_result(cast(ToolMessage, message))
                        ]
                    ),
                ).to_dict()
            )
        else:
            raise TypeError(f"Unsupported LangChain message type: {type(message).__name__}")

    if system_prefix:
        prefix = system_prefix.strip()
        current_content = f"{prefix}\n\n{current_content}" if current_content else prefix

    return history, current_content, tool_results


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif isinstance(c, str):
                parts.append(c)
        return "".join(parts)
    return str(content)


def _ai_tool_uses(message: "AIMessage") -> List[ToolUse]:
    uses: List[ToolUse] = []
    for tc in getattr(message, "tool_calls", None) or []:
        uses.append(
            ToolUse(
                tool_use_id=tc.get("id") or str(uuid.uuid4()),
                name=tc.get("name", ""),
                input=tc.get("args", {}) or {},
            )
        )
    return uses


def _tool_message_to_result(message: "ToolMessage") -> ToolResult:
    content = message.content
    text = content if isinstance(content, str) else json.dumps(content)
    status = "success"
    if getattr(message, "status", None) == "error":
        status = "error"
    return ToolResult.text(message.tool_call_id, text, status=status)


class _ToolCallAccumulator:
    """Reassemble streamed toolUseEvent fragments into LangChain tool_calls."""

    def __init__(self) -> None:
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []

    def add(self, event: "ToolUseEvent") -> None:
        key = event.tool_use_id or (self._order[-1] if self._order else "tool_0")
        if key not in self._by_id:
            self._by_id[key] = {"id": key, "name": event.name or "", "input": ""}
            self._order.append(key)
        if event.name:
            self._by_id[key]["name"] = event.name
        if isinstance(event.input, dict):
            self._by_id[key]["input"] = event.input
        elif isinstance(event.input, str):
            current = self._by_id[key]["input"]
            if isinstance(current, dict):
                current = json.dumps(current, separators=(",", ":"))
            self._by_id[key]["input"] = current + event.input

    def finalize(self) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for key in self._order:
            entry = self._by_id[key]
            raw = entry["input"]
            if isinstance(raw, dict):
                args = raw
            else:
                try:
                    args = json.loads(raw) if raw else {}
                except (ValueError, TypeError):
                    args = {"__raw__": raw}
            calls.append({"id": entry["id"], "name": entry["name"], "args": args, "type": "tool_call"})
        return calls
