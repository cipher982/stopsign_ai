"""Unit tests for the ONNX detector shim and preprocessing."""

import numpy as np
import pytest

from stopsign.onnx_detector import DetectionBox
from stopsign.onnx_detector import _Indexable
from stopsign.onnx_detector import _letterbox
from stopsign.onnx_detector import _preprocess
from stopsign.onnx_detector import _TensorLike

# ---------------------------------------------------------------------------
# _TensorLike
# ---------------------------------------------------------------------------


class TestTensorLike:
    def test_item_returns_int(self):
        t = _TensorLike(42)
        assert t.item() == 42

    def test_repr(self):
        assert "42" in repr(_TensorLike(42))


# ---------------------------------------------------------------------------
# _Indexable
# ---------------------------------------------------------------------------


class TestIndexable:
    def test_index_zero(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        idx = _Indexable(arr)
        np.testing.assert_array_equal(idx[0], arr)

    def test_index_nonzero_raises(self):
        idx = _Indexable(np.array([1.0]))
        with pytest.raises(IndexError):
            _ = idx[1]


# ---------------------------------------------------------------------------
# DetectionBox shim
# ---------------------------------------------------------------------------


class TestDetectionBox:
    def _make_box(self, track_id=1, cls=2):
        return DetectionBox(
            _track_id=track_id,
            _xywh=np.array([100.0, 200.0, 50.0, 60.0], dtype=np.float32),
            _xyxy=np.array([75.0, 170.0, 125.0, 230.0], dtype=np.float32),
            _cls=cls,
        )

    def test_id_with_track_id(self):
        box = self._make_box(track_id=7)
        assert box.id is not None
        assert box.id.item() == 7

    def test_id_none(self):
        box = self._make_box(track_id=None)
        assert box.id is None

    def test_xywh_indexable(self):
        box = self._make_box()
        x, y, w, h = box.xywh[0]
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)
        assert w == pytest.approx(50.0)
        assert h == pytest.approx(60.0)

    def test_xyxy_indexable(self):
        box = self._make_box()
        x1, y1, x2, y2 = box.xyxy[0]
        assert x1 == pytest.approx(75.0)
        assert y1 == pytest.approx(170.0)
        assert x2 == pytest.approx(125.0)
        assert y2 == pytest.approx(230.0)

    def test_cls(self):
        box = self._make_box(cls=5)
        assert box.cls == 5

    def test_tracking_contract(self):
        """Verify the exact code path from tracking.py:328-340 works."""
        box = self._make_box(track_id=3)

        # tracking.py line 328-329
        assert box.id is not None
        car_id = int(box.id.item())
        assert car_id == 3

        # tracking.py line 333
        x, y, w, h = box.xywh[0]
        location = (float(x), float(y))
        assert location == (100.0, 200.0)

        # video_analyzer.py line 920 (draw_box)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        assert (x1, y1, x2, y2) == (75, 170, 125, 230)

        # video_analyzer.py line 780 (cls filter)
        assert box.cls == 2


# ---------------------------------------------------------------------------
# Letterbox preprocessing
# ---------------------------------------------------------------------------


class TestLetterbox:
    def test_square_input(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, scale, (px, py) = _letterbox(frame)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)
        assert px == 0
        assert py == 0

    def test_landscape_input(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, scale, (px, py) = _letterbox(frame)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)
        assert px == 0
        assert py == 80  # (640-480)//2

    def test_portrait_input(self):
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        padded, scale, (px, py) = _letterbox(frame)
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)
        assert px == 80  # (640-480)//2
        assert py == 0

    def test_large_input_downscaled(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        padded, scale, (px, py) = _letterbox(frame)
        assert padded.shape == (640, 640, 3)
        expected_scale = 640 / 1920
        assert scale == pytest.approx(expected_scale, rel=1e-3)


class TestPreprocess:
    def test_output_shape(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        blob, scale, padding = _preprocess(frame)
        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32

    def test_pixel_range(self):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        blob, _, _ = _preprocess(frame)
        assert blob.min() >= 0.0
        assert blob.max() <= 1.0
