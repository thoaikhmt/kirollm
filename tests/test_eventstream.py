import json

import pytest

from kirollm.eventstream import (
    EventStreamError,
    decode_message,
    encode_message,
    iter_messages,
)


def _event(event_type, payload):
    return encode_message(
        {":message-type": "event", ":event-type": event_type,
         ":content-type": "application/json"},
        json.dumps(payload).encode("utf-8"),
    )


def test_encode_decode_roundtrip():
    raw = _event("assistantResponseEvent", {"content": "hello"})
    msg = decode_message(raw)
    assert msg.message_type == "event"
    assert msg.event_type == "assistantResponseEvent"
    assert json.loads(msg.payload) == {"content": "hello"}


def test_prelude_crc_validation():
    raw = bytearray(_event("assistantResponseEvent", {"content": "x"}))
    raw[5] ^= 0xFF  # corrupt a byte inside the prelude
    with pytest.raises(EventStreamError):
        decode_message(bytes(raw))


def test_message_crc_validation():
    raw = bytearray(_event("assistantResponseEvent", {"content": "x"}))
    raw[-1] ^= 0xFF  # corrupt the trailing message CRC
    with pytest.raises(EventStreamError):
        decode_message(bytes(raw))


def test_iter_messages_reassembles_across_chunks():
    m1 = _event("assistantResponseEvent", {"content": "foo"})
    m2 = _event("assistantResponseEvent", {"content": "bar"})
    stream = m1 + m2

    # Feed the concatenated stream one byte at a time to force reassembly.
    chunks = (stream[i:i + 1] for i in range(len(stream)))
    messages = list(iter_messages(chunks))
    assert len(messages) == 2
    assert json.loads(messages[0].payload)["content"] == "foo"
    assert json.loads(messages[1].payload)["content"] == "bar"


def test_iter_messages_multiple_per_chunk():
    m1 = _event("assistantResponseEvent", {"content": "a"})
    m2 = _event("toolUseEvent", {"toolUseId": "t1", "name": "x", "input": "{}"})
    messages = list(iter_messages(iter([m1 + m2])))
    assert [m.event_type for m in messages] == ["assistantResponseEvent", "toolUseEvent"]


def test_trailing_bytes_raise():
    m1 = _event("assistantResponseEvent", {"content": "a"})
    with pytest.raises(EventStreamError):
        list(iter_messages(iter([m1 + b"\x00\x01"])))


def _raw_message(headers: bytes, payload: bytes = b"") -> bytes:
    import binascii
    import struct

    total = 16 + len(headers) + len(payload)
    prelude = struct.pack("!II", total, len(headers))
    message = prelude + struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message += headers + payload
    return message + struct.pack("!I", binascii.crc32(message) & 0xFFFFFFFF)


def test_crc_valid_truncated_header_raises_typed_error():
    # Claims a five-byte name but only carries one byte.
    raw = _raw_message(b"\x05a")
    with pytest.raises(EventStreamError, match="Truncated event-stream header"):
        decode_message(raw)


def test_headers_cannot_exceed_message_body():
    import binascii
    import struct

    prelude = struct.pack("!II", 16, 1)
    raw = prelude + struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    raw += struct.pack("!I", binascii.crc32(raw) & 0xFFFFFFFF)
    with pytest.raises(EventStreamError, match="Headers length exceeds"):
        decode_message(raw)


def test_oversized_message_rejected_before_buffering():
    import binascii
    import struct

    prelude = struct.pack("!II", 16 * 1024 * 1024 + 1, 0)
    chunk = prelude + struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    with pytest.raises(EventStreamError, match="Invalid message length"):
        list(iter_messages(iter([chunk])))
