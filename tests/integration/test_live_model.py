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
        catalog = client.list_models()
        assert catalog.models, "Kiro advertised no models"
        model_id = os.getenv("KIRO_MODEL_ID") or catalog.default_model or catalog.models[0].model_id
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
