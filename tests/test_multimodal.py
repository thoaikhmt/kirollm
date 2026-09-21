import base64
import json

import pytest

from kiro_llm import (
    AudioInput,
    ChatSession,
    ImageInput,
    ImageInputError,
    KiroLLMClient,
    VoiceInputError,
    image_from_data_url,
)
from kiro_llm.eventstream import encode_message
from kiro_llm.langchain import _messages_to_kiro
from langchain_core.messages import HumanMessage


PNG = b"\x89PNG\r\n\x1a\nminimal"


def _event(event_type, payload):
    return encode_message(
        {":message-type": "event", ":event-type": event_type},
        json.dumps(payload).encode("utf-8"),
    )


class _Response:
    status_code = 200
    headers = {"Content-Type": "application/vnd.amazon.eventstream"}

    def __init__(self, body):
        self.body = body

    def iter_content(self, chunk_size=None):
        yield self.body

    def close(self):
        pass


class _Session:
    def __init__(self, body):
        self.body = body
        self.last_body = None

    def post(self, url, headers=None, data=None, stream=None, timeout=None):
        self.last_body = json.loads(data)
        return _Response(self.body)

    def close(self):
        pass


def test_image_helpers_serialize_kipro_image_block():
    image = ImageInput.from_bytes(PNG, "image/png")
    assert image.format == "png"
    assert image.to_dict() == {
        "format": "png",
        "source": {"bytes": base64.b64encode(PNG).decode("ascii")},
    }

    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")
    assert image_from_data_url(data_url) == image


def test_image_helper_rejects_unsupported_and_invalid_data():
    with pytest.raises(ImageInputError):
        ImageInput.from_bytes(PNG, "image/tiff")
    with pytest.raises(ImageInputError):
        image_from_data_url("data:image/png;base64,not-valid!!!")


def test_chat_session_sends_image_in_current_message(tmp_path):
    image_path = tmp_path / "picture.png"
    image_path.write_bytes(PNG)
    body = _event("assistantResponseEvent", {"content": "seen"})
    session = _Session(body)
    chat = ChatSession(KiroLLMClient(access_token="t", session=session))

    assert chat.send("Describe", images=[image_path]) == "seen"
    current = session.last_body["conversationState"]["currentMessage"]["userInputMessage"]
    assert current["images"][0]["format"] == "png"
    assert current["images"][0]["source"]["bytes"] == base64.b64encode(PNG).decode("ascii")


def test_langchain_image_url_becomes_kiro_image():
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")
    messages = [HumanMessage(content=[
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ])]
    history, current, images, tool_results, system_prompt = _messages_to_kiro(messages)
    assert history == []
    assert current == "What is this?"
    assert len(images) == 1
    assert images[0].to_dict()["format"] == "png"
    assert tool_results == []


def test_langchain_remote_image_url_is_explicitly_rejected():
    with pytest.raises(ValueError, match="base64 data URL"):
        _messages_to_kiro([HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        ])])


def test_voice_adapter_transcribes_then_sends_text():
    body = _event("assistantResponseEvent", {"content": "reply"})
    session = _Session(body)
    chat = ChatSession(KiroLLMClient(access_token="t", session=session))

    result = chat.send_voice(
        AudioInput(b"wav", "audio/wav"),
        lambda audio: "  transcribed question  ",
    )
    assert result == "reply"
    current = session.last_body["conversationState"]["currentMessage"]["userInputMessage"]
    assert current["content"] == "transcribed question"


def test_voice_adapter_rejects_empty_transcription():
    session = _Session(_event("assistantResponseEvent", {"content": "unused"}))
    chat = ChatSession(KiroLLMClient(access_token="t", session=session))
    with pytest.raises(VoiceInputError):
        chat.send_voice(AudioInput(b"wav"), lambda audio: "  ")


def test_chat_session_sends_document_in_current_message(tmp_path):
    document_path = tmp_path / "notes.txt"
    document_path.write_text("hello document")
    body = _event("assistantResponseEvent", {"content": "seen"})
    session = _Session(body)
    chat = ChatSession(KiroLLMClient(access_token="t", session=session))

    assert chat.send("Read this", documents=[document_path]) == "seen"
    current = session.last_body["conversationState"]["currentMessage"]["userInputMessage"]
    assert current["documents"][0]["format"] == "txt"
    assert current["documents"][0]["name"] == "notes.txt"
