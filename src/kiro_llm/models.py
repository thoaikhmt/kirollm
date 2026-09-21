"""
Data models mirroring the request/response shapes the Kiro CLI uses when it
calls ``AmazonCodeWhispererStreamingService.GenerateAssistantResponse``.

The field names here follow the wire (JSON) names observed in the Kiro CLI
binary's ``ConversationState`` / ``UserInputMessage`` / ``ToolSpecification``
serializers, e.g. ``conversationId``, ``currentMessage``, ``userInputMessage``,
``userInputMessageContext``, ``chatTriggerType``, ``toolSpecification``.

Everything serializes via :meth:`to_dict` so the client can drop the result
straight into the JSON request body.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ChatTriggerType(str, Enum):
    MANUAL = "MANUAL"
    DIAGNOSTIC = "DIAGNOSTIC"
    INLINE_CHAT = "INLINE_CHAT"


class UserIntent(str, Enum):
    APPLY_COMMON_BEST_PRACTICES = "APPLY_COMMON_BEST_PRACTICES"
    CITE_SOURCES = "CITE_SOURCES"
    CODE_GENERATION = "CODE_GENERATION"
    EXPLAIN_CODE_SELECTION = "EXPLAIN_CODE_SELECTION"
    EXPLAIN_LINE_BY_LINE = "EXPLAIN_LINE_BY_LINE"
    GENERATE_UNIT_TESTS = "GENERATE_UNIT_TESTS"
    IMPROVE_CODE = "IMPROVE_CODE"
    SHOW_EXAMPLES = "SHOW_EXAMPLES"
    SUGGEST_ALTERNATE_IMPLEMENTATION = "SUGGEST_ALTERNATE_IMPLEMENTATION"


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


@dataclass
class ToolSpecification:
    """A tool the model may call. ``input_schema`` is a JSON Schema object."""

    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "toolSpecification": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {"json": self.input_schema},
            }
        }


@dataclass
class ToolUse:
    """A tool call previously emitted by the assistant (echoed back in history)."""

    tool_use_id: str
    name: str
    input: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"toolUseId": self.tool_use_id, "name": self.name, "input": self.input}


@dataclass
class ToolResult:
    """The result of running a tool, sent back to the model on the next turn."""

    tool_use_id: str
    content: List[Dict[str, Any]]
    status: str = "success"  # "success" | "error"

    @classmethod
    def text(cls, tool_use_id: str, text: str, status: str = "success") -> "ToolResult":
        return cls(tool_use_id=tool_use_id, content=[{"text": text}], status=status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "toolUseId": self.tool_use_id,
            "content": self.content,
            "status": self.status,
        }


@dataclass
class UserInputMessageContext:
    """Optional context bag attached to a user message."""

    tools: Optional[List[ToolSpecification]] = None
    tool_results: Optional[List[ToolResult]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.tools:
            out["tools"] = [t.to_dict() for t in self.tools]
        if self.tool_results:
            out["toolResults"] = [r.to_dict() for r in self.tool_results]
        return out


@dataclass
class UserInputMessage:
    content: str
    model_id: Optional[str] = None
    origin: str = "CLI"
    user_intent: Optional[UserIntent] = None
    context: Optional[UserInputMessageContext] = None
    images: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"content": self.content, "origin": self.origin}
        if self.model_id:
            msg["modelId"] = self.model_id
        if self.user_intent:
            msg["userIntent"] = self.user_intent.value
        if self.images:
            msg["images"] = self.images
        ctx = self.context.to_dict() if self.context else {}
        if ctx:
            msg["userInputMessageContext"] = ctx
        return {"userInputMessage": msg}


@dataclass
class AssistantResponseMessage:
    content: str
    tool_uses: Optional[List[ToolUse]] = None

    def to_dict(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"content": self.content}
        if self.tool_uses:
            msg["toolUses"] = [t.to_dict() for t in self.tool_uses]
        return {"assistantResponseMessage": msg}


@dataclass
class ConversationState:
    """
    The top-level payload of a ``GenerateAssistantResponse`` request.

    ``history`` is a list of alternating user / assistant message dicts (each
    produced by ``UserInputMessage.to_dict()`` / ``AssistantResponseMessage``),
    and ``current_message`` is the user turn being sent now.
    """

    current_message: UserInputMessage
    chat_trigger_type: ChatTriggerType = ChatTriggerType.MANUAL
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: List[Dict[str, Any]] = field(default_factory=list)
    agent_continuation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        state = {
            "conversationId": self.conversation_id,
            "currentMessage": self.current_message.to_dict(),
            "chatTriggerType": self.chat_trigger_type.value,
            "history": self.history or None,
            "agentContinuationId": self.agent_continuation_id,
        }
        return _drop_none(state)


# --------------------------------------------------------------------------- #
# Response side: parsed ChatResponseStream events.
# --------------------------------------------------------------------------- #

@dataclass
class AssistantResponseEvent:
    """A chunk of assistant text."""
    content: str


@dataclass
class ToolUseEvent:
    """A (possibly partial) tool-use request from the assistant."""
    tool_use_id: Optional[str]
    name: Optional[str]
    input: Optional[Union[str, Dict[str, Any]]]
    stop: bool = False


@dataclass
class ReasoningEvent:
    """A chunk of the model's reasoning / thinking text."""
    content: str


@dataclass
class CodeReferenceEvent:
    references: List[Dict[str, Any]]


@dataclass
class FollowupPromptEvent:
    content: str


@dataclass
class MetadataEvent:
    """Terminal metadata: request id, token usage, etc."""
    raw: Dict[str, Any]


@dataclass
class InvalidStateEvent:
    reason: str
    message: str


@dataclass
class UnknownEvent:
    """Any event variant we don't model explicitly."""
    event_type: Optional[str]
    raw: Dict[str, Any]


# Union alias for typing/documentation purposes.
ChatResponseStreamEvent = Union[
    AssistantResponseEvent,
    ToolUseEvent,
    ReasoningEvent,
    CodeReferenceEvent,
    FollowupPromptEvent,
    MetadataEvent,
    InvalidStateEvent,
    UnknownEvent,
]
