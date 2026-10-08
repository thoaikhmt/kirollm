"""
kirollm: call an LLM the same way the Kiro CLI does.

Quick start::

    from kirollm import KiroLLMClient, ChatSession

    client = KiroLLMClient(access_token="<bearer-token>", region="us-east-1")
    chat = ChatSession(client, model_id="claude-sonnet-4")
    for chunk in chat.stream("Write a haiku about event streams"):
        print(chunk, end="", flush=True)
"""

from .chat import ChatSession
from .auth import (
    AuthError,
    CredentialProvider,
    KiroCliCredentialProvider,
    KiroCredentialProvider,
    SecretStore,
    Token,
    load_default_credentials,
)
from .client import (
    CODEWHISPERER_ENDPOINTS,
    KIRO_RUNTIME_ENDPOINTS,
    KiroApiError,
    KiroLLMClient,
    resolve_endpoint,
)
from .eventstream import (
    EventStreamError,
    EventStreamMessage,
    decode_message,
    encode_message,
    iter_messages,
)
from .media import (
    AudioInput,
    DocumentInput,
    ImageInput,
    ImageInputError,
    image_from_bytes,
    image_from_data_url,
    image_from_file,
)
from .voice import VoiceInputError, VoiceTranscriber, transcribe_audio
from .models import (
    AssistantResponseEvent,
    AssistantResponseMessage,
    ChatTriggerType,
    CodeReferenceEvent,
    ContextUsageEvent,
    ConversationState,
    FollowupPromptEvent,
    InvalidStateEvent,
    KiroModel,
    KiroModelCatalog,
    MetadataEvent,
    ModelUnavailableEvent,
    ReasoningEvent,
    ToolResult,
    ToolSpecification,
    ToolUse,
    ToolUseEvent,
    SupplementaryWebLinksEvent,
    UnknownEvent,
    UserInputMessage,
    UserInputMessageContext,
    UserIntent,
)

__version__ = "0.1.0"

__all__ = [
    "KiroLLMClient",
    "KiroApiError",
    "ChatSession",
    "KiroCredentialProvider",
    "KiroCliCredentialProvider",
    "CredentialProvider",
    "load_default_credentials",
    "SecretStore",
    "Token",
    "AuthError",
    "resolve_endpoint",
    "CODEWHISPERER_ENDPOINTS",
    "KIRO_RUNTIME_ENDPOINTS",
    "ConversationState",
    "UserInputMessage",
    "UserInputMessageContext",
    "AssistantResponseMessage",
    "ToolSpecification",
    "ToolUse",
    "ToolResult",
    "ChatTriggerType",
    "UserIntent",
    "AssistantResponseEvent",
    "ReasoningEvent",
    "ToolUseEvent",
    "CodeReferenceEvent",
    "FollowupPromptEvent",
    "MetadataEvent",
    "ContextUsageEvent",
    "ModelUnavailableEvent",
    "SupplementaryWebLinksEvent",
    "InvalidStateEvent",
    "UnknownEvent",
    "KiroModel",
    "KiroModelCatalog",
    "EventStreamMessage",
    "EventStreamError",
    "decode_message",
    "iter_messages",
    "encode_message",
    "ImageInput",
    "ImageInputError",
    "image_from_bytes",
    "image_from_file",
    "image_from_data_url",
    "AudioInput",
    "DocumentInput",
    "VoiceTranscriber",
    "VoiceInputError",
    "transcribe_audio",
]
