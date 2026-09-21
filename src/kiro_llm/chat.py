"""
High-level convenience API on top of :class:`kiro_llm.client.KiroLLMClient`.

:class:`ChatSession` keeps conversation history and exposes:

* :meth:`stream` - yield assistant text chunks as they arrive.
* :meth:`send`   - block and return the full assistant reply as a string.

Both accept optional ``tools`` (a list of :class:`ToolSpecification`) and collect
any tool-use requests the model makes so the caller can run them and reply with
:meth:`send_tool_results`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from .client import KiroLLMClient
from .media import AudioInput, DocumentInput, ImageInput, image_from_file
from .voice import Transcriber, transcribe_audio
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


class ChatSession:
    def __init__(
        self,
        client: KiroLLMClient,
        *,
        model_id: Optional[str] = None,
        tools: Optional[List[ToolSpecification]] = None,
    ):
        self.client = client
        self.model_id = model_id
        self.tools = tools
        self._history: List[Dict[str, Any]] = []
        self.conversation_id: Optional[str] = None
        # Complete tool-use requests collected from the most recent turn.
        self.pending_tool_uses: List[ToolUse] = []

    def _make_state(self, message: UserInputMessage) -> ConversationState:
        state = ConversationState(
            current_message=message,
            chat_trigger_type=ChatTriggerType.MANUAL,
            history=list(self._history),
        )
        if self.conversation_id:
            state.conversation_id = self.conversation_id
        else:
            self.conversation_id = state.conversation_id
        return state

    def _user_message(
        self,
        content: str,
        tool_results: Optional[List[ToolResult]] = None,
        images: Optional[Sequence[Union[ImageInput, str, Path]]] = None,
        documents: Optional[Sequence[Union[DocumentInput, str, Path]]] = None,
    ) -> UserInputMessage:
        normalized_images = [
            image_from_file(image) if isinstance(image, (str, Path)) else image
            for image in (images or [])
        ]
        normalized_documents = [
            DocumentInput.from_file(document)
            if isinstance(document, (str, Path))
            else document
            for document in (documents or [])
        ]
        ctx = UserInputMessageContext(tools=self.tools, tool_results=tool_results)
        return UserInputMessage(
            content=content,
            model_id=self.model_id,
            images=normalized_images or None,
            documents=normalized_documents or None,
            context=ctx if (self.tools or tool_results) else None,
        )

    def _run_turn(self, message: UserInputMessage) -> Iterator[str]:
        state = self._make_state(message)
        self.pending_tool_uses = []
        assistant_text_parts: List[str] = []
        tool_parts: Dict[str, Dict[str, Any]] = {}
        tool_order: List[str] = []

        for event in self.client.generate_assistant_response(state):
            if isinstance(event, AssistantResponseEvent):
                assistant_text_parts.append(event.content)
                yield event.content
            elif isinstance(event, ToolUseEvent):
                key = event.tool_use_id or (tool_order[-1] if tool_order else "tool_0")
                if key not in tool_parts:
                    tool_parts[key] = {"name": event.name or "", "input": ""}
                    tool_order.append(key)
                if event.name:
                    tool_parts[key]["name"] = event.name
                if isinstance(event.input, str):
                    tool_parts[key]["input"] += event.input
                elif isinstance(event.input, dict):
                    tool_parts[key]["input"] = event.input

        for key in tool_order:
            part = tool_parts[key]
            raw_input = part["input"]
            if isinstance(raw_input, dict):
                tool_input = raw_input
            else:
                try:
                    tool_input = json.loads(raw_input) if raw_input else {}
                except ValueError:
                    tool_input = {"__raw__": raw_input}
            self.pending_tool_uses.append(
                ToolUse(tool_use_id=key, name=part["name"], input=tool_input)
            )

        self._history.append(message.to_dict())
        self._history.append(
            AssistantResponseMessage(
                content="".join(assistant_text_parts),
                tool_uses=self.pending_tool_uses or None,
            ).to_dict()
        )

    def stream(
        self,
        content: str,
        *,
        images: Optional[Sequence[Union[ImageInput, str, Path]]] = None,
        documents: Optional[Sequence[Union[DocumentInput, str, Path]]] = None,
    ) -> Iterator[str]:
        """Send a text/image/document message; yield assistant text chunks."""
        yield from self._run_turn(
            self._user_message(content, images=images, documents=documents)
        )

    def send(
        self,
        content: str,
        *,
        images: Optional[Sequence[Union[ImageInput, str, Path]]] = None,
        documents: Optional[Sequence[Union[DocumentInput, str, Path]]] = None,
    ) -> str:
        """Send a text/image/document message; return the full reply."""
        return "".join(self.stream(content, images=images, documents=documents))

    def send_voice(
        self,
        audio: AudioInput,
        transcriber: Transcriber,
        *,
        images: Optional[Sequence[Union[ImageInput, str, Path]]] = None,
    ) -> str:
        """Transcribe audio with an adapter, then send the text to Kiro.

        This deliberately does not pretend to call a private Kiro voice
        endpoint: the verified Kiro API path accepts the resulting text turn.
        """
        return self.send(
            transcribe_audio(audio, transcriber), images=images
        )

    def send_tool_results(self, results: List[ToolResult]) -> str:
        """
        Reply to the model with the results of tool calls it requested. Content
        is empty because the payload is the tool results themselves.
        """
        return "".join(self._run_turn(self._user_message("", tool_results=results)))

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)
