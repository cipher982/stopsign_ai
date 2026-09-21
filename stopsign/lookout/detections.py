"""Detection summarisation for Lookout.

Lookout consumes *fresh* detections with confidence, not the stop pipeline's
tracked boxes: a retained tracker id is not evidence that something is still
present, and the pipeline's exported boxes drop confidence. Callers hand us
whatever the detector produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Sequence

import numpy as np

from stopsign.lookout.types import Box

# COCO classes the analyzer's YOLO model can produce.
COCO_LABELS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
}

# Coarse subject groups. A watch armed on a car is satisfied by car/truck/bus
# detections, because those are interchangeable evidence of "something is there".
SUBJECT_GROUPS = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "person": "person",
    "bicycle": "bicycle",
    "cat": "animal",
    "dog": "animal",
}


def subject_group(label: str) -> str:
    return SUBJECT_GROUPS.get(label, label or "")


def labels_match(watch_subject: str, detection_label: str) -> bool:
    """True when a detection counts as evidence for the watched subject."""
    if not watch_subject:
        return True
    if watch_subject == detection_label:
        return True
    return subject_group(watch_subject) == subject_group(detection_label)


@dataclass(slots=True)
class Detection:
    box: Box
    label: str
    conf: float


def detections_from_arrays(
    xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    *,
    min_conf: float,
    class_filter: Optional[Sequence[int]] = None,
) -> list[Detection]:
    """Build detections from raw detector arrays (pre-tracker, conf retained)."""
    allowed = set(class_filter) if class_filter is not None else None
    out: list[Detection] = []
    for index in range(len(xyxy)):
        conf = float(confidences[index])
        if conf < min_conf:
            continue
        class_id = int(class_ids[index])
        if allowed is not None and class_id not in allowed:
            continue
        x1, y1, x2, y2 = (float(value) for value in xyxy[index])
        out.append(Detection(box=(x1, y1, x2, y2), label=COCO_LABELS.get(class_id, f"class_{class_id}"), conf=conf))
    return out


def detections_from_boxes(boxes: Iterable[Any], *, min_conf: float) -> list[Detection]:
    """Build detections from the analyzer's DetectionBox shim (no confidence)."""
    out: list[Detection] = []
    for item in boxes:
        box = getattr(item, "xyxy", None)
        if box is None:
            continue
        x1, y1, x2, y2 = (float(value) for value in np.asarray(box).reshape(-1)[:4])
        class_id = _class_id(item)
        out.append(
            Detection(
                box=(x1, y1, x2, y2),
                label=COCO_LABELS.get(class_id, f"class_{class_id}"),
                conf=max(min_conf, _confidence(item)),
            )
        )
    return out


def _class_id(item: Any) -> int:
    raw = getattr(item, "cls", 0)
    try:
        return int(raw.item()) if hasattr(raw, "item") else int(raw)
    except (TypeError, ValueError):
        return 0


def _confidence(item: Any) -> float:
    for attribute in ("conf", "confidence"):
        raw = getattr(item, attribute, None)
        if raw is None:
            continue
        try:
            return float(raw.item()) if hasattr(raw, "item") else float(raw)
        except (TypeError, ValueError):
            continue
    return 0.5
