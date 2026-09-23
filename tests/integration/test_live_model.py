"""Opt-in live test for the Kiro GenerateAssistantResponse model endpoint."""

import os
import struct
import zlib

import pytest

from kiro_llm import (
    AssistantResponseEvent,
    ChatSession,
    ConversationState,
    KiroLLMClient,
    ImageInput,
)
from kiro_llm.models import UserInputMessage

pytestmark = pytest.mark.integration

IMAGE_MODEL_ID = "claude-haiku-4.5"


def _solid_png(red: int, green: int, blue: int, size: int = 16) -> bytes:
    """Build a tiny solid-colour PNG without third-party image libraries."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes([red, green, blue]) * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _live_client(api_key: str) -> KiroLLMClient:
    return KiroLLMClient(
        access_token=api_key,
        region=os.getenv("KIRO_REGION", "us-east-1"),
        endpoint_override=os.getenv("KIRO_ENDPOINT"),
        timeout=90,
        max_attempts=1,
    )


@pytest.mark.integration
def test_live_kiro_model_returns_assistant_text():
    """Make one real model request when KIRO_API_KEY is explicitly provided."""
    api_key = os.getenv("KIRO_API_KEY")
    if not api_key:
        pytest.skip("KIRO_API_KEY is not set; live model test is opt-in")

    client = _live_client(api_key)
    try:
        catalog = client.list_models()
        assert catalog.models, "Kiro advertised no models"
        model_id = os.getenv("KIRO_MODEL_ID", "qwen3-coder-next")
        if model_id not in catalog.model_ids:
            raise AssertionError(f"KIRO_MODEL_ID is not advertised: {model_id}")
        events = client.generate_assistant_response(
            ConversationState(
                current_message=UserInputMessage(
                    content="Reply with exactly the word OK.",
                    model_id=model_id,
                    origin="AI_EDITOR",
                )
            )
        )
        text = "".join(
            event.content
            for event in events
            if isinstance(event, AssistantResponseEvent)
        )
    finally:
        client.close()

    assert text.strip(), "Kiro returned no assistant text"


@pytest.mark.integration
def test_live_image_task_uses_image_capable_model():
    """Image tasks use claude-haiku-4.5, the lowest-cost model that accepts images."""
    api_key = os.getenv("KIRO_API_KEY")
    if not api_key:
        pytest.skip("KIRO_API_KEY is not set; live model test is opt-in")

    client = _live_client(api_key)
    try:
        catalog = client.list_models()
        assert IMAGE_MODEL_ID in catalog.model_ids, (
            f"{IMAGE_MODEL_ID} is not advertised"
        )
        image = ImageInput.from_bytes(_solid_png(255, 0, 0), "image/png")
        reply = ChatSession(client, model_id=IMAGE_MODEL_ID).send(
            "What is the dominant colour? Reply with one word.",
            images=[image],
        )
    finally:
        client.close()

    assert reply.strip(), "Kiro returned no assistant text"
