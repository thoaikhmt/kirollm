"""
Example: stream an assistant reply, exactly the way the Kiro CLI's agent
backend calls GenerateAssistantResponse.

Usage:
    export KIRO_ACCESS_TOKEN="<your bearer token>"
    export KIRO_MODEL_ID="claude-sonnet-4"        # optional
    python examples/basic_stream.py "Write a haiku about event streams"

The bearer token is the same SSO/IdC access token the Kiro CLI stores after
`kiro-cli login`. This library does NOT perform the OAuth login flow; supply a
token you already have.
"""

import os
import sys

from kiro_llm import ChatSession, KiroLLMClient, KiroApiError


def main() -> int:
    prompt = " ".join(sys.argv[1:]) or "Say hello in one sentence."

    token = os.environ.get("KIRO_ACCESS_TOKEN")
    # If KIRO_ACCESS_TOKEN is set, use it; otherwise auto-load the Kiro CLI's
    # stored login token and refresh it automatically.
    if token:
        client = KiroLLMClient(
            access_token=token,
            region=os.environ.get("KIRO_REGION", "us-east-1"),
            profile_arn=os.environ.get("KIRO_PROFILE_ARN"),
        )
    else:
        client = KiroLLMClient()  # auto-auth from the Kiro CLI login store

    chat = ChatSession(client, model_id=os.environ.get("KIRO_MODEL_ID"))

    try:
        for chunk in chat.stream(prompt):
            print(chunk, end="", flush=True)
        print()
    except KiroApiError as e:
        print(f"\nAPI error ({e.status_code or e.error_type}): {e}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
