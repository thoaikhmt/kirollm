"""Opt-in live test for the Kiro GenerateAssistantResponse model endpoint."""

import os

import pytest

from kiro_llm import AssistantResponseEvent, ConversationState, KiroLLMClient
from kiro_llm.models import UserInputMessage

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_live_kiro_model_returns_assistant_text():
    """Make one real model request when KIRO_API_KEY is explicitly provided."""
    api_key = os.getenv("KIRO_API_KEY")
    if not api_key:
        pytest.skip("KIRO_API_KEY is not set; live model test is opt-in")

    client = KiroLLMClient(
        access_token=api_key,
        region=os.getenv("KIRO_REGION", "us-east-1"),
        endpoint_override=os.getenv("KIRO_ENDPOINT"),
        timeout=90,
        max_attempts=1,
    )
    try:
        events = client.generate_assistant_response(
            ConversationState(
                current_message=UserInputMessage(
                    content="Reply with exactly the word OK."
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
