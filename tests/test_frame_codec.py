import cv2
import numpy as np

from stopsign.frame_codec import ENVELOPE_MAGIC
from stopsign.frame_codec import LEGACY_MAGIC
from stopsign.frame_codec import pack_frame
from stopsign.frame_codec import pack_legacy_jpeg_frame
from stopsign.frame_codec import unpack_frame


def test_pack_frame_round_trip():
    payload = b"bgr-bytes"
    metadata = {"format": "bgr24", "width": 1920, "height": 1080, "ts": 123.45}

    packed = pack_frame(payload, metadata)
    decoded = unpack_frame(packed)

    assert packed.startswith(ENVELOPE_MAGIC)
    assert decoded is not None
    assert decoded.envelope == "v2"
    assert decoded.metadata == metadata
    assert decoded.payload == payload


def test_legacy_jpeg_round_trip():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", frame)
    assert ok

    packed = pack_legacy_jpeg_frame(jpeg.tobytes(), capture_ts=123.45, width=20, height=10)
    decoded = unpack_frame(packed)

    assert packed.startswith(LEGACY_MAGIC)
    assert decoded is not None
    assert decoded.envelope == "legacy"
    assert decoded.metadata["ts"] == 123.45
    assert decoded.metadata["w"] == 20
    assert decoded.metadata["h"] == 10
    assert decoded.payload == jpeg.tobytes()


def test_plain_bytes_are_not_an_envelope():
    assert unpack_frame(b"raw-bgr") is None
