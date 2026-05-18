"""Redis frame envelope helpers.

The pipeline has two wire formats during migration:
- legacy raw input: b"SSFM" + version + json length + json + JPEG bytes
- v2 envelope: b"SSF2" + version + json length + json + frame bytes

Callers can parse both formats and may still receive legacy raw BGR bytes with
no envelope on the processed-frame queue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

LEGACY_MAGIC = b"SSFM"
ENVELOPE_MAGIC = b"SSF2"
HEADER_MIN_LEN = 9


@dataclass(frozen=True)
class DecodedFrame:
    metadata: dict[str, Any]
    payload: bytes
    envelope: str


def pack_frame(payload: bytes, metadata: dict[str, Any]) -> bytes:
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return ENVELOPE_MAGIC + bytes([1]) + len(meta_bytes).to_bytes(4, "big") + meta_bytes + payload


def pack_legacy_jpeg_frame(jpeg_bytes: bytes, capture_ts: float, width: int, height: int) -> bytes:
    metadata = {"ts": float(capture_ts), "w": int(width), "h": int(height), "src": "rtsp_to_redis"}
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return LEGACY_MAGIC + bytes([1]) + len(meta_bytes).to_bytes(4, "big") + meta_bytes + jpeg_bytes


def unpack_frame(data: bytes) -> DecodedFrame | None:
    if len(data) < HEADER_MIN_LEN:
        return None

    magic = data[:4]
    if magic not in {LEGACY_MAGIC, ENVELOPE_MAGIC}:
        return None

    meta_len = int.from_bytes(data[5:9], "big")
    meta_start = HEADER_MIN_LEN
    meta_end = meta_start + meta_len
    if meta_len < 0 or meta_end > len(data):
        return None

    try:
        metadata = json.loads(data[meta_start:meta_end].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None

    envelope = "v2" if magic == ENVELOPE_MAGIC else "legacy"
    return DecodedFrame(metadata=metadata, payload=data[meta_end:], envelope=envelope)
