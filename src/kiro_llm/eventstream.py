"""
AWS event-stream (``application/vnd.amazon.eventstream``) decoder.

Kiro's backend returns the ``GenerateAssistantResponse`` body as a stream of
binary "event stream" messages. This is the same framing used by
``bedrock-runtime`` ``ConverseStream``, Transcribe, etc. Each message is:

    [ prelude ][ prelude_crc ][ headers ][ payload ][ message_crc ]

    prelude       = total_length (u32 BE) + headers_length (u32 BE)   (8 bytes)
    prelude_crc   = CRC32 of the prelude                              (4 bytes)
    headers       = headers_length bytes of encoded headers
    payload       = total_length - headers_length - 16 bytes
    message_crc   = CRC32 of everything before it                     (4 bytes)

Each header is:

    name_len (u8) | name (utf-8) | value_type (u8) | value...

We only need enough header-value types to read the ``:message-type``,
``:event-type``, ``:content-type`` and ``:exception-type`` string headers that
identify each ``ChatResponseStream`` variant, but the full set is implemented
for correctness.

This module has no third-party dependencies.
"""

from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Union

# Header value type ids as defined by the event-stream spec.
_HEADER_BOOL_TRUE = 0
_HEADER_BOOL_FALSE = 1
_HEADER_BYTE = 2
_HEADER_SHORT = 3
_HEADER_INT = 4
_HEADER_LONG = 5
_HEADER_BYTES = 6
_HEADER_STRING = 7
_HEADER_TIMESTAMP = 8
_HEADER_UUID = 9

_PRELUDE_LEN = 8
_PRELUDE_CRC_LEN = 4
_MESSAGE_CRC_LEN = 4
_MIN_MESSAGE_LEN = _PRELUDE_LEN + _PRELUDE_CRC_LEN + _MESSAGE_CRC_LEN
# Limits defined by the AWS event-stream specification.
MAX_HEADERS_LENGTH = 128 * 1024
MAX_MESSAGE_LENGTH = 16 * 1024 * 1024


class EventStreamError(Exception):
    """Raised when the event-stream bytes are malformed or fail a checksum."""


HeaderValue = Union[bool, int, bytes, str]


@dataclass
class EventStreamMessage:
    """A single decoded event-stream message."""

    headers: Dict[str, HeaderValue] = field(default_factory=dict)
    payload: bytes = b""

    @property
    def message_type(self) -> Optional[str]:
        """``:message-type`` header, e.g. ``event`` or ``exception``."""
        return _as_str(self.headers.get(":message-type"))

    @property
    def event_type(self) -> Optional[str]:
        """``:event-type`` header, e.g. ``assistantResponseEvent``."""
        return _as_str(self.headers.get(":event-type"))

    @property
    def exception_type(self) -> Optional[str]:
        """``:exception-type`` header, present when message_type == exception."""
        return _as_str(self.headers.get(":exception-type"))

    @property
    def is_exception(self) -> bool:
        return self.message_type == "exception" or self.exception_type is not None


def _as_str(value: Optional[HeaderValue]) -> Optional[str]:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xFFFFFFFF


def _decode_headers(raw: bytes) -> Dict[str, HeaderValue]:
    headers: Dict[str, HeaderValue] = {}
    offset = 0
    length = len(raw)

    def take(size: int, label: str) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > length:
            raise EventStreamError(f"Truncated event-stream header {label}")
        value = raw[offset:offset + size]
        offset += size
        return value

    def unpack(fmt: str, size: int, label: str) -> int:
        return struct.unpack(fmt, take(size, label))[0]

    while offset < length:
        name_len = unpack("!B", 1, "name length")
        if name_len == 0:
            raise EventStreamError("Event-stream header name cannot be empty")
        try:
            name = take(name_len, "name").decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EventStreamError("Event-stream header name is not UTF-8") from exc
        value_type = unpack("!B", 1, "value type")

        if value_type == _HEADER_BOOL_TRUE:
            value: HeaderValue = True
        elif value_type == _HEADER_BOOL_FALSE:
            value = False
        elif value_type == _HEADER_BYTE:
            value = unpack("!b", 1, "byte value")
        elif value_type == _HEADER_SHORT:
            value = unpack("!h", 2, "short value")
        elif value_type == _HEADER_INT:
            value = unpack("!i", 4, "integer value")
        elif value_type in (_HEADER_LONG, _HEADER_TIMESTAMP):
            value = unpack("!q", 8, "long value")
        elif value_type in (_HEADER_BYTES, _HEADER_STRING):
            value_len = unpack("!H", 2, "value length")
            encoded = take(value_len, "value")
            if value_type == _HEADER_STRING:
                try:
                    value = encoded.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise EventStreamError(
                        f"Event-stream header {name!r} is not UTF-8"
                    ) from exc
            else:
                value = encoded
        elif value_type == _HEADER_UUID:
            value = take(16, "UUID value")
        else:
            raise EventStreamError(f"Unknown header value type: {value_type}")
        headers[name] = value

    return headers


def decode_message(buffer: bytes) -> EventStreamMessage:
    """Decode a single complete event-stream message from ``buffer``."""
    if len(buffer) < _MIN_MESSAGE_LEN:
        raise EventStreamError("Buffer too small to be a valid message")

    total_length, headers_length = struct.unpack_from("!II", buffer, 0)
    if total_length < _MIN_MESSAGE_LEN or total_length > MAX_MESSAGE_LENGTH:
        raise EventStreamError(f"Invalid message length: {total_length}")
    if headers_length > MAX_HEADERS_LENGTH:
        raise EventStreamError(f"Headers exceed {MAX_HEADERS_LENGTH} byte limit")
    if headers_length > total_length - _MIN_MESSAGE_LEN:
        raise EventStreamError("Headers length exceeds message body")
    if total_length != len(buffer):
        raise EventStreamError(
            f"Declared message length {total_length} != buffer length {len(buffer)}"
        )

    prelude = buffer[:_PRELUDE_LEN]
    (expected_prelude_crc,) = struct.unpack_from("!I", buffer, _PRELUDE_LEN)
    if _crc32(prelude) != expected_prelude_crc:
        raise EventStreamError("Prelude CRC mismatch")

    (expected_msg_crc,) = struct.unpack_from("!I", buffer, total_length - _MESSAGE_CRC_LEN)
    if _crc32(buffer[: total_length - _MESSAGE_CRC_LEN]) != expected_msg_crc:
        raise EventStreamError("Message CRC mismatch")

    headers_start = _PRELUDE_LEN + _PRELUDE_CRC_LEN
    headers_end = headers_start + headers_length
    headers = _decode_headers(buffer[headers_start:headers_end])
    payload = buffer[headers_end: total_length - _MESSAGE_CRC_LEN]
    return EventStreamMessage(headers=headers, payload=payload)


def iter_messages(chunks: Iterator[bytes]) -> Iterator[EventStreamMessage]:
    """
    Consume an iterator of arbitrary byte chunks (e.g. ``response.iter_content``)
    and yield fully-assembled :class:`EventStreamMessage` objects.

    The decoder buffers partial data until a complete message (as declared by the
    prelude's ``total_length``) is available, so callers can stream bytes in any
    chunk size.
    """
    buffer = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        buffer.extend(chunk)
        while True:
            if len(buffer) < _PRELUDE_LEN + _PRELUDE_CRC_LEN:
                break
            total_length, headers_length = struct.unpack_from("!II", buffer, 0)
            if total_length < _MIN_MESSAGE_LEN or total_length > MAX_MESSAGE_LENGTH:
                raise EventStreamError(f"Invalid message length: {total_length}")
            if headers_length > MAX_HEADERS_LENGTH:
                raise EventStreamError(f"Headers exceed {MAX_HEADERS_LENGTH} byte limit")
            if headers_length > total_length - _MIN_MESSAGE_LEN:
                raise EventStreamError("Headers length exceeds message body")
            if len(buffer) < total_length:
                break  # wait for more bytes
            message_bytes = bytes(buffer[:total_length])
            del buffer[:total_length]
            yield decode_message(message_bytes)

    if buffer:
        raise EventStreamError(
            f"Stream ended with {len(buffer)} trailing undecoded bytes"
        )


def encode_message(headers: Dict[str, HeaderValue], payload: bytes) -> bytes:
    """
    Encode a message back into event-stream bytes. Only string, int and bool
    headers are supported (sufficient for tests and mock servers).
    """
    header_bytes = bytearray()
    for name, value in headers.items():
        name_encoded = name.encode("utf-8")
        header_bytes.append(len(name_encoded))
        header_bytes.extend(name_encoded)
        if isinstance(value, bool):
            header_bytes.append(_HEADER_BOOL_TRUE if value else _HEADER_BOOL_FALSE)
        elif isinstance(value, int):
            header_bytes.append(_HEADER_LONG)
            header_bytes.extend(struct.pack("!q", value))
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
            header_bytes.append(_HEADER_STRING)
            header_bytes.extend(struct.pack("!H", len(encoded)))
            header_bytes.extend(encoded)
        else:
            raise EventStreamError(f"Cannot encode header value of type {type(value)}")

    headers_length = len(header_bytes)
    total_length = _PRELUDE_LEN + _PRELUDE_CRC_LEN + headers_length + len(payload) + _MESSAGE_CRC_LEN
    prelude = struct.pack("!II", total_length, headers_length)
    prelude_crc = struct.pack("!I", _crc32(prelude))

    message = bytearray()
    message.extend(prelude)
    message.extend(prelude_crc)
    message.extend(header_bytes)
    message.extend(payload)
    message.extend(struct.pack("!I", _crc32(bytes(message))))
    return bytes(message)
