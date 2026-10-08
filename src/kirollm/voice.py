"""Voice input adapters.

The installed Kiro CLI exposes voice-related commands, but this release does
not expose a stable public voice HTTP schema. This module therefore keeps voice
transport provider-neutral: an application supplies an ASR/transcription
callable, and the resulting text is sent through the verified Kiro text API.
"""

from __future__ import annotations

from typing import Callable, Protocol, Union

from .media import AudioInput


class VoiceInputError(RuntimeError):
    """Raised when an audio input cannot be transcribed."""


class VoiceTranscriber(Protocol):
    """Protocol implemented by Whisper, cloud ASR, or application adapters."""

    def transcribe(self, audio: AudioInput) -> str: ...


Transcriber = Union[VoiceTranscriber, Callable[[AudioInput], str]]


def transcribe_audio(audio: AudioInput, transcriber: Transcriber) -> str:
    """Transcribe audio with an adapter and validate its text result."""
    try:
        method = getattr(transcriber, "transcribe", None)
        if callable(method):
            text = method(audio)
        elif callable(transcriber):
            text = transcriber(audio)
        else:
            raise TypeError("transcriber must be callable or expose transcribe()")
    except Exception as exc:
        raise VoiceInputError("Audio transcription failed") from exc
    if not isinstance(text, str) or not text.strip():
        raise VoiceInputError("Audio transcriber returned no text")
    return text.strip()
