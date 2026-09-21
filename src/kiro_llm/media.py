"""Image and audio input helpers for multimodal Kiro conversations."""

from __future__ import annotations

import base64
import binascii
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


_SUPPORTED_IMAGE_MIME_TYPES = {
    "image/gif": "gif",
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
}
_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>image/[a-z0-9.+-]+);base64,(?P<data>[a-z0-9+/=\s]+)$",
    re.IGNORECASE,
)


class ImageInputError(ValueError):
    """Raised when image input cannot be represented by Kiro's image block."""


def _image_format(mime_type: str) -> str:
    normalized = mime_type.strip().lower()
    try:
        return _SUPPORTED_IMAGE_MIME_TYPES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_SUPPORTED_IMAGE_MIME_TYPES))
        raise ImageInputError(
            f"Unsupported image MIME type {mime_type!r}; supported: {supported}"
        ) from exc


@dataclass(frozen=True)
class ImageInput:
    """An image encoded as a Kiro ``ImageBlock`` source."""

    data: bytes
    mime_type: str

    def __post_init__(self) -> None:
        if not self.data:
            raise ImageInputError("Image data cannot be empty")
        _image_format(self.mime_type)

    @property
    def format(self) -> str:
        return _image_format(self.mime_type)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> "ImageInput":
        if not isinstance(data, bytes):
            raise TypeError("ImageInput.from_bytes expects bytes")
        return cls(data=data, mime_type=mime_type)

    @classmethod
    def from_file(
        cls, path: Union[str, Path], mime_type: Optional[str] = None
    ) -> "ImageInput":
        file_path = Path(path)
        if not file_path.is_file():
            raise ImageInputError(f"Image file does not exist: {file_path}")
        detected = mime_type or mimetypes.guess_type(file_path.name)[0]
        if not detected:
            raise ImageInputError(
                f"Could not determine image MIME type for {file_path}; pass mime_type="
            )
        return cls(data=file_path.read_bytes(), mime_type=detected)

    @classmethod
    def from_data_url(cls, data_url: str) -> "ImageInput":
        match = _DATA_URL_RE.match(data_url.strip())
        if not match:
            raise ImageInputError(
                "Expected a base64 image data URL such as "
                "data:image/png;base64,..."
            )
        try:
            data = base64.b64decode(
                re.sub(r"\s+", "", match.group("data")), validate=True
            )
        except (binascii.Error, ValueError) as exc:
            raise ImageInputError("Image data URL contains invalid base64") from exc
        return cls(data=data, mime_type=match.group("mime"))

    def to_dict(self) -> dict:
        """Serialize to the observed Kiro ``ImageBlock`` wire shape."""
        return {
            "format": self.format,
            "source": {"bytes": base64.b64encode(self.data).decode("ascii")},
        }


def image_from_bytes(data: bytes, mime_type: str) -> ImageInput:
    """Functional form of :meth:`ImageInput.from_bytes`."""
    return ImageInput.from_bytes(data, mime_type)


def image_from_file(
    path: Union[str, Path], mime_type: Optional[str] = None
) -> ImageInput:
    """Functional form of :meth:`ImageInput.from_file`."""
    return ImageInput.from_file(path, mime_type)


def image_from_data_url(data_url: str) -> ImageInput:
    """Functional form of :meth:`ImageInput.from_data_url`."""
    return ImageInput.from_data_url(data_url)


@dataclass(frozen=True)
class AudioInput:
    """Audio bytes for an external transcription adapter."""

    data: bytes
    mime_type: str = "audio/wav"

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("Audio data cannot be empty")
        if not self.mime_type.startswith("audio/"):
            raise ValueError("AudioInput mime_type must start with audio/")
