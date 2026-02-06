"""ONNX Runtime YOLO detector with supervision ByteTrack.

Drop-in replacement for ultralytics YOLO — exposes the same DetectionBox
interface that tracking.py:328-340 consumes:
    box.id          → _TensorLike | None  (.item() → int)
    box.xywh[0]     → np.ndarray  [x, y, w, h]
    box.xyxy[0]     → np.ndarray  [x1, y1, x2, y2]
    box.cls         → int
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DetectionBox shim — mirrors the ultralytics box interface
# ---------------------------------------------------------------------------


class _TensorLike:
    """Mimics a single-element tensor with an .item() method."""

    __slots__ = ("_value",)

    def __init__(self, value: int):
        self._value = value

    def item(self) -> int:
        return self._value

    def __repr__(self) -> str:
        return f"_TensorLike({self._value})"


class _Indexable:
    """list-like wrapper so that obj[0] returns the inner array."""

    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx != 0:
            raise IndexError("DetectionBox coordinate arrays only support [0]")
        return self._arr


@dataclass(slots=True)
class DetectionBox:
    """Compatibility shim matching the ultralytics box interface."""

    _track_id: Optional[int]
    _xywh: np.ndarray  # [x_center, y_center, w, h]
    _xyxy: np.ndarray  # [x1, y1, x2, y2]
    _cls: int

    @property
    def id(self) -> Optional[_TensorLike]:
        return _TensorLike(self._track_id) if self._track_id is not None else None

    @property
    def xywh(self) -> _Indexable:
        return _Indexable(self._xywh)

    @property
    def xyxy(self) -> _Indexable:
        return _Indexable(self._xyxy)

    @property
    def cls(self) -> int:
        return self._cls


# ---------------------------------------------------------------------------
# ONNX YOLO detector
# ---------------------------------------------------------------------------

# YOLO model output shape: (1, 84, 8400) for 80-class COCO at 640x640
# Transpose to (8400, 84): first 4 values = cx, cy, w, h; remaining 80 = class scores
_INPUT_SIZE = 640
_PAD_VALUE = 114


def _letterbox(
    frame: np.ndarray,
    target_size: int = _INPUT_SIZE,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize *frame* preserving aspect ratio and pad to square.

    Returns (padded_image, scale, (pad_x, pad_y)).
    """
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded = np.full((target_size, target_size, 3), _PAD_VALUE, dtype=np.uint8)
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return padded, scale, (pad_x, pad_y)


def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox + BGR→RGB + normalise + HWC→CHW + batch dim."""
    padded, scale, (pad_x, pad_y) = _letterbox(frame)
    blob = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, [0,1]
    blob = blob.transpose(2, 0, 1)  # HWC→CHW
    blob = np.expand_dims(blob, 0)  # add batch
    return blob, scale, (pad_x, pad_y)


class OnnxYoloDetector:
    """ONNX Runtime YOLO detector with ByteTrack tracking."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf_thresh: float = 0.25,
        nms_iou_thresh: float = 0.7,
    ):
        # Build provider list based on device
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thresh = conf_thresh
        self.nms_iou_thresh = nms_iou_thresh

        # ByteTrack — params match the ultralytics bytetrack.yaml defaults
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=15,
        )

        logger.info(
            "OnnxYoloDetector ready — providers: %s, conf=%.2f, nms_iou=%.2f",
            self.session.get_providers(),
            conf_thresh,
            nms_iou_thresh,
        )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        padding: tuple[int, int],
        orig_shape: tuple[int, int],
        class_filter: Optional[Sequence[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse raw ONNX output → (xyxy, confidences, class_ids) in original coords.

        *output* shape is (1, 84, 8400). Returns arrays suitable for sv.Detections.
        """
        # (1, 84, 8400) → (8400, 84)
        pred = output[0].T

        # Split box coords and class scores
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        class_scores = pred[:, 4:]  # (8400, 80)

        # Best class per detection
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Confidence filter
        mask = confidences >= self.conf_thresh
        if class_filter is not None:
            class_set = set(class_filter)
            class_mask = np.array([cid in class_set for cid in class_ids])
            mask = mask & class_mask

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(confidences) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        # xywh → xyxy in letterboxed coords
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # NMS
        boxes_for_nms = np.stack([x1, y1, w, h], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            confidences.tolist(),
            self.conf_thresh,
            self.nms_iou_thresh,
        )
        if len(indices) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        indices = np.array(indices).flatten()

        x1, y1, x2, y2 = x1[indices], y1[indices], x2[indices], y2[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]

        # Rescale to original frame coords
        pad_x, pad_y = padding
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clip to frame
        orig_h, orig_w = orig_shape
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        return xyxy, confidences.astype(np.float32), class_ids.astype(np.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_track(
        self,
        frame: np.ndarray,
        class_filter: Optional[Sequence[int]] = None,
    ) -> List[DetectionBox]:
        """Run ONNX inference + NMS + ByteTrack on *frame*.

        Returns a list of DetectionBox objects compatible with tracking.py.
        """
        orig_h, orig_w = frame.shape[:2]
        blob, scale, padding = _preprocess(frame)

        # Inference
        outputs = self.session.run(None, {self.input_name: blob})
        raw_output = outputs[0]  # (1, 84, 8400)

        # Post-process
        xyxy, confidences, class_ids = self._postprocess(raw_output, scale, padding, (orig_h, orig_w), class_filter)

        if len(xyxy) == 0:
            return []

        # Build supervision Detections for ByteTrack
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids,
        )
        detections = self.tracker.update_with_detections(detections)

        # Convert to DetectionBox shim objects
        result: List[DetectionBox] = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else None
            cid = int(detections.class_id[i]) if detections.class_id is not None else 0

            result.append(
                DetectionBox(
                    _track_id=tid,
                    _xywh=np.array([cx, cy, w, h], dtype=np.float32),
                    _xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    _cls=cid,
                )
            )

        return result
