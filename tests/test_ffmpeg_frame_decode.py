from stopsign.ffmpeg_service import decode_processed_frame
from stopsign.frame_codec import pack_frame


def test_decode_processed_frame_accepts_legacy_raw_bytes():
    payload = b"raw-bgr"

    decoded, shape = decode_processed_frame(payload)

    assert decoded == payload
    assert shape is None


def test_decode_processed_frame_accepts_envelope():
    payload = b"raw-bgr"
    packed = pack_frame(payload, {"format": "bgr24", "width": 30, "height": 20})

    decoded, shape = decode_processed_frame(packed)

    assert decoded == payload
    assert shape == (30, 20)
