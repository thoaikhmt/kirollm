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

import asyncio
import json
import threading
import uuid
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Sequence, Union, cast

from .client import KiroApiError, KiroLLMClient
from .media import ImageInput, image_from_data_url
from .models import (
    AssistantResponseEvent,
    AssistantResponseMessage,
    ChatTriggerType,
    ContextUsageEvent,
    ConversationState,
    InvalidStateEvent,
    KiroModelCatalog,
    MetadataEvent,
    ModelUnavailableEvent,
    ReasoningEvent,
    ToolResult,
    ToolSpecification,
    ToolUse,
    ToolUseEvent,
    SupplementaryWebLinksEvent,
    UserInputMessage,
    UserInputMessageContext,
)

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
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

        def list_models(self) -> "KiroModelCatalog":
            """Discover all models available to this Kiro credential."""
            return self._client.list_models()

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
            history, current, current_images, tool_results, system_prompt = _messages_to_kiro(messages)
            ctx = UserInputMessageContext(
                tools=self._bound_tools or None,
                tool_results=tool_results or None,
            )
            current_message = UserInputMessage(
                content=current,
                model_id=self.model,
                images=current_images or None,
                context=ctx if (self._bound_tools or tool_results) else None,
            )
            return ConversationState(
                current_message=current_message,
                chat_trigger_type=ChatTriggerType.MANUAL,
                history=history,
                system_prompt=system_prompt or None,
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
            reasoning_parts: List[str] = []
            response_metadata: Dict[str, Any] = {}
            tool_calls = _ToolCallAccumulator()

            for event in self._client.generate_assistant_response(state):
                if isinstance(event, AssistantResponseEvent):
                    text_parts.append(event.content)
                    if run_manager:
                        run_manager.on_llm_new_token(event.content)
                elif isinstance(event, ReasoningEvent):
                    reasoning_parts.append(event.content)
                elif isinstance(event, MetadataEvent):
                    response_metadata.update(event.raw)
                elif isinstance(event, ContextUsageEvent):
                    response_metadata["context_usage"] = event.raw
                elif isinstance(event, SupplementaryWebLinksEvent):
                    response_metadata["supplementary_web_links"] = event.links
                elif isinstance(event, ModelUnavailableEvent):
                    raise KiroApiError(event.message, error_type="ModelUnavailable")
                elif isinstance(event, ToolUseEvent):
                    tool_calls.add(event)
                elif isinstance(event, InvalidStateEvent):
                    raise KiroApiError(
                        event.message or "Kiro returned an invalid conversation state",
                        error_type=event.reason or "InvalidState",
                    )

            additional_kwargs = {}
            if reasoning_parts:
                additional_kwargs["reasoning_content"] = "".join(reasoning_parts)
            usage_metadata = _usage_metadata(response_metadata)
            message = AIMessage(
                content="".join(text_parts),
                tool_calls=tool_calls.finalize(),
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
                usage_metadata=cast(Any, usage_metadata or None),
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
            response_metadata: Dict[str, Any] = {}
            reasoning_parts: List[str] = []
            for event in self._client.generate_assistant_response(state):
                if isinstance(event, AssistantResponseEvent):
                    if run_manager:
                        run_manager.on_llm_new_token(event.content)
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=event.content)
                    )
                elif isinstance(event, ReasoningEvent):
                    reasoning_parts.append(event.content)
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            additional_kwargs={"reasoning_content": event.content},
                        )
                    )
                elif isinstance(event, MetadataEvent):
                    response_metadata.update(event.raw)
                elif isinstance(event, ContextUsageEvent):
                    response_metadata["context_usage"] = event.raw
                elif isinstance(event, SupplementaryWebLinksEvent):
                    response_metadata["supplementary_web_links"] = event.links
                elif isinstance(event, ModelUnavailableEvent):
                    raise KiroApiError(event.message, error_type="ModelUnavailable")
                elif isinstance(event, InvalidStateEvent):
                    raise KiroApiError(
                        event.message or "Kiro returned an invalid conversation state",
                        error_type=event.reason or "InvalidState",
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
            if response_metadata or reasoning_parts:
                additional_kwargs = (
                    {"reasoning_content": "".join(reasoning_parts)}
                    if reasoning_parts
                    else {}
                )
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        additional_kwargs=additional_kwargs,
                        response_metadata=response_metadata,
                        usage_metadata=cast(
                            Any, _usage_metadata(response_metadata) or None
                        ),
                    )
                )

        async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional["AsyncCallbackManagerForLLMRun"] = None,
            **kwargs: Any,
        ) -> "ChatResult":
            return await asyncio.to_thread(
                self._generate, messages, stop, None, **kwargs
            )

        async def _astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional["AsyncCallbackManagerForLLMRun"] = None,
            **kwargs: Any,
        ) -> AsyncIterator["ChatGenerationChunk"]:
            queue: asyncio.Queue[Any] = asyncio.Queue()
            sentinel = object()
            errors: List[BaseException] = []
            loop = asyncio.get_running_loop()

            def produce() -> None:
                try:
                    for chunk in self._stream(messages, stop, None, **kwargs):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)

            threading.Thread(target=produce, daemon=True).start()
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield cast("ChatGenerationChunk", item)
            if errors:
                raise errors[0]

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
    """Translate LangChain messages into Kiro history and current-turn fields."""
    history: List[Dict[str, Any]] = []
    system_parts: List[str] = []
    tool_results: List[ToolResult] = []

    non_system: List["BaseMessage"] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(_text(message.content))
        else:
            non_system.append(message)
    system_prompt = "\n\n".join(part for part in system_parts if part)

    if not non_system:
        return history, "", [], tool_results, system_prompt

    trailing_tool_start = len(non_system)
    while trailing_tool_start > 0 and isinstance(
        non_system[trailing_tool_start - 1], ToolMessage
    ):
        trailing_tool_start -= 1

    current_images: List[ImageInput] = []
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
        current_content, current_images = _text_and_images(last.content)

    index = 0
    while index < len(prior):
        message = prior[index]
        if isinstance(message, HumanMessage):
            content, images = _text_and_images(message.content)
            history.append(
                UserInputMessage(content=content, images=images or None).to_dict()
            )
            index += 1
        elif isinstance(message, AIMessage):
            tool_uses = _ai_tool_uses(cast(AIMessage, message))
            history.append(
                AssistantResponseMessage(
                    content=_text(message.content), tool_uses=tool_uses or None
                ).to_dict()
            )
            index += 1
        elif isinstance(message, ToolMessage):
            results: List[ToolResult] = []
            while index < len(prior) and isinstance(prior[index], ToolMessage):
                results.append(
                    _tool_message_to_result(cast(ToolMessage, prior[index]))
                )
                index += 1
            history.append(
                UserInputMessage(
                    content="",
                    context=UserInputMessageContext(tool_results=results),
                ).to_dict()
            )
        else:
            raise TypeError(f"Unsupported LangChain message type: {type(message).__name__}")

    return history, current_content, current_images, tool_results, system_prompt


def _text_and_images(content: Any) -> tuple[str, List[ImageInput]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    text_parts: List[str] = []
    images: List[ImageInput] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
            continue
        if block_type == "image_url":
            image_url = block.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str) or not url.startswith("data:"):
                raise ValueError(
                    "Kiro image input requires a base64 data URL; remote image URLs "
                    "must be downloaded by the caller first"
                )
            images.append(image_from_data_url(url))
            continue
        if block_type == "image":
            source = block.get("source", block)
            if isinstance(source, dict):
                data = source.get("data") or source.get("bytes")
                mime_type = source.get("media_type") or source.get("mime_type")
                if isinstance(data, str) and isinstance(mime_type, str):
                    images.append(
                        ImageInput.from_data_url(
                            f"data:{mime_type};base64,{data}"
                        )
                    )
                    continue
            data = block.get("data")
            mime_type = block.get("mimeType") or block.get("mime_type")
            if isinstance(data, str) and isinstance(mime_type, str):
                images.append(
                    ImageInput.from_data_url(
                        f"data:{mime_type};base64,{data}"
                    )
                )
                continue
        # Preserve unknown structured blocks as text rather than silently
        # dropping user content.
        if block_type:
            text_parts.append(str(block))

    return "".join(text_parts), images


def _text(content: Any) -> str:
    return _text_and_images(content)[0]


def _usage_metadata(metadata: Dict[str, Any]) -> Dict[str, int]:
    usage = metadata.get("usage") or metadata.get("tokenUsage") or metadata
    if not isinstance(usage, dict):
        return {}
    aliases = {
        "input_tokens": ("input_tokens", "inputTokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "outputTokens", "completion_tokens"),
        "total_tokens": ("total_tokens", "totalTokens"),
    }
    result: Dict[str, int] = {}
    for target, keys in aliases.items():
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                result[target] = int(value)
                break
    if "total_tokens" not in result and {
        "input_tokens", "output_tokens"
    }.issubset(result):
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


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
