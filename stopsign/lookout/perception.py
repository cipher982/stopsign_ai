"""Measurement layer: what does this frame actually show for this watch?

Nothing here decides anything. Perception reports coverage by the watched
subject, coverage by *other* objects (possible occlusion), appearance
similarity to the arming reference, and frame-level statistics. The decider
turns those measurements into a typed answer.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from stopsign.lookout.detections import Detection
from stopsign.lookout.detections import labels_match
from stopsign.lookout.types import Box
from stopsign.lookout.types import area
from stopsign.lookout.types import intersection_area


class RegionMetrics:
    """Per-region measurements for one frame."""

    __slots__ = ("subject_cover", "subject_conf", "subject_label", "subject_box", "occluder_cover", "occluder_label")

    def __init__(
        self,
        subject_cover: float = 0.0,
        subject_conf: float = 0.0,
        subject_label: Optional[str] = None,
        subject_box: Optional[Box] = None,
        occluder_cover: float = 0.0,
        occluder_label: Optional[str] = None,
    ) -> None:
        self.subject_cover = subject_cover
        self.subject_conf = subject_conf
        self.subject_label = subject_label
        self.subject_box = subject_box
        self.occluder_cover = occluder_cover
        self.occluder_label = occluder_label


def measure_region(region: Box, detections: list[Detection], watch_subject: str) -> RegionMetrics:
    """Coverage of *region* by the watched subject and by anything else.

    Coverage is summed per detection rather than unioned, which can exceed the
    true covered fraction when two boxes overlap the same pixels. It is an upper
    bound used only for thresholding, and it is reported as such in evidence.
    """
    region_area = max(1.0, area(region))
    metrics = RegionMetrics()
    for det in detections:
        overlap = intersection_area(region, det.box)
        if overlap <= 0.0:
            continue
        cover = overlap / region_area
        if labels_match(watch_subject, det.label):
            metrics.subject_cover += cover
            if det.conf >= metrics.subject_conf:
                metrics.subject_conf = det.conf
                metrics.subject_label = det.label
                metrics.subject_box = det.box
        else:
            metrics.occluder_cover += cover
            if metrics.occluder_label is None:
                metrics.occluder_label = det.label
    metrics.subject_cover = min(1.0, metrics.subject_cover)
    metrics.occluder_cover = min(1.0, metrics.occluder_cover)
    return metrics


def region_histogram(frame: np.ndarray, box: Box, *, bins: int = 16) -> list[float]:
    """Normalised HSV histogram of a region — the watch's appearance reference."""
    patch = crop(frame, box)
    if patch.size == 0:
        return []
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256]).astype(np.float32)
    total = float(hist.sum())
    if total <= 0:
        return []
    hist /= total
    return hist.reshape(-1).tolist()


def appearance_similarity(
    frame: np.ndarray,
    box: Box,
    reference: list[float] | np.ndarray,
    *,
    bins: int = 16,
) -> float:
    """Correlation between the region's histogram and the arming reference (0-1).

    A patch that is too small to histogram returns 0.0 — no evidence, rather than
    a confident match.
    """
    current = region_histogram(frame, box, bins=bins)
    if not current or reference is None or len(reference) == 0:
        return 0.0
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    cur = np.asarray(current, dtype=np.float32).reshape(-1)
    if ref.shape != cur.shape:
        return 0.0
    corr = float(cv2.compareHist(ref.reshape(-1, 1), cur.reshape(-1, 1), cv2.HISTCMP_CORREL))
    if not np.isfinite(corr):
        return 0.0
    return max(0.0, min(1.0, corr))


def crop(frame: np.ndarray, box: Box) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(round(box[0]))))
    y1 = max(0, min(height, int(round(box[1]))))
    x2 = max(0, min(width, int(round(box[2]))))
    y2 = max(0, min(height, int(round(box[3]))))
    if x2 <= x1 or y2 <= y1:
        return frame[0:0, 0:0]
    return frame[y1:y2, x1:x2]


def frame_statistics(frame: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of frame brightness, from a 64x64 thumbnail."""
    thumb = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)), float(np.std(gray))


class ZoneBackground:
    """Slow background model of a static region.

    A static camera plus a slow exponential update gives a second, purely visual
    occupancy signal that does not depend on the detector's class list — it sees
    a parked delivery box or a standing person the same way. The update is slow
    on purpose: things that were already there stop counting as new, but a
    vehicle dwelling for a minute still reads as foreground.
    """

    def __init__(self, region: Box, *, alpha: float, diff_threshold: int, max_width: int = 160) -> None:
        self.region = region
        self.alpha = alpha
        self.diff_threshold = diff_threshold
        self.max_width = max_width
        self._background: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> float:
        """Return the fraction of the region that differs from the background."""
        patch = _small_gray(frame, self.region, self.max_width)
        if patch is None:
            return 0.0
        if self._background is None or self._background.shape != patch.shape:
            self._background = patch
            return 0.0
        diff = cv2.absdiff(patch, self._background)
        ratio = float(np.count_nonzero(diff > self.diff_threshold)) / float(diff.size)
        # Update only where the frame looks like background, so a dwelling object
        # is not immediately absorbed into the model.
        quiet = diff <= self.diff_threshold
        self._background = np.where(quiet, (1 - self.alpha) * self._background + self.alpha * patch, self._background)
        self._background = self._background.astype(np.uint8)
        return ratio


def _small_gray(frame: np.ndarray, box: Box, max_width: int) -> np.ndarray | None:
    patch = crop(frame, box)
    if patch.size == 0:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        gray = cv2.resize(gray, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return gray
