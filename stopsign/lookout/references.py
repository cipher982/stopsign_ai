"""Two-reference presence model.

The perceptual question is not "does this patch match the template I saved" but
"**is this region closer to how it looked when the subject was there, or to how
it looked when the subject was not there?**"

Two stored references turn that into a three-way answer without thresholds
invented from nowhere:

* close to the armed reference → the region still looks like the armed state
* close to the opposite one    → the state flipped
* close to neither            → we cannot tell (an occluder, glare, or something
  else entirely) — reported as ``decisive = False`` rather than as a false
  all-clear

The opposite reference is learned the first time the opposite state is observed
with confidence, and both references drift slowly while they are confidently
nominal, so dusk and cloud do not read as state changes.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from stopsign.lookout.types import Box

THUMB = 64
HIST_BINS = 16

# Mean absolute difference between z-scored thumbnails at which two regions
# count as unrelated. A region whose average pixel has moved more than its own
# standard deviation is not the same region.
STRUCTURE_DIFF_SCALE = 1.2


@dataclass(slots=True)
class Reference:
    """A remembered appearance of one region: thumbnail + colour histogram."""

    patch: np.ndarray  # float32 (THUMB, THUMB), z-scored
    histogram: np.ndarray  # float32 (HIST_BINS * HIST_BINS,), L1-normalised
    updated_at: float = 0.0
    samples: int = 0

    def to_storable(self) -> dict[str, object]:
        gray = _from_zscore(self.patch)
        ok, buf = cv2.imencode(".png", gray)
        return {
            "patch_png": base64.b64encode(buf.tobytes()).decode("ascii") if ok else "",
            "histogram": [round(float(v), 6) for v in self.histogram],
            "updated_at": self.updated_at,
            "samples": self.samples,
        }

    @classmethod
    def from_storable(cls, payload: object) -> Optional[Reference]:
        if not isinstance(payload, dict):
            return None
        try:
            hist = np.asarray(payload.get("histogram") or [], dtype=np.float32)
            encoded = str(payload.get("patch_png") or "")
            if hist.size == 0 or not encoded:
                return None
            buf = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
            gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return None
            patch = _to_zscore(gray.astype(np.float32))
            return cls(
                patch=patch,
                histogram=hist,
                updated_at=float(payload.get("updated_at") or 0.0),
                samples=int(payload.get("samples") or 0),
            )
        except (ValueError, TypeError, cv2.error):
            return None


@dataclass(slots=True)
class ReferenceScore:
    similarity_armed: float
    similarity_opposite: float
    present_prob: float
    # False when the region resembles no known state well enough to choose, or
    # when nothing could be measured at all. Callers must never read a False here
    # as evidence of absence.
    decisive: bool


class ReferenceModel:
    """Holds both references for one watch and scores new patches against them."""

    def __init__(
        self,
        *,
        armed_occupied: bool,
        drift: float = 0.02,
        decisive_similarity: float = 0.55,
        present_floor: float = 0.72,
        absent_ceiling: float = 0.35,
        min_opposite_samples: int = 3,
    ) -> None:
        self.armed_occupied = armed_occupied
        self.drift = drift
        self.decisive_similarity = decisive_similarity
        # Only a confident reading may teach the model: mid-band readings are
        # exactly the ones where we do not know what we are looking at.
        self.present_floor = present_floor
        self.absent_ceiling = absent_ceiling
        # A second reference only earns the right to make a reading
        # inconclusive once several confident observations built it.
        self.min_opposite_samples = min_opposite_samples
        self.armed: Optional[Reference] = None
        self.opposite: Optional[Reference] = None

    @classmethod
    def from_armed_patch(
        cls,
        patch_bgr: Optional[np.ndarray],
        *,
        armed_occupied: bool,
        drift: float = 0.02,
        decisive_similarity: float = 0.55,
        present_floor: float = 0.72,
        absent_ceiling: float = 0.35,
        min_opposite_samples: int = 3,
    ) -> ReferenceModel:
        model = cls(
            armed_occupied=armed_occupied,
            drift=drift,
            decisive_similarity=decisive_similarity,
            present_floor=present_floor,
            absent_ceiling=absent_ceiling,
            min_opposite_samples=min_opposite_samples,
        )
        if patch_bgr is not None:
            patch, hist = describe(patch_bgr)
            if patch is not None and hist is not None:
                model.armed = Reference(patch=patch, histogram=hist)
        return model

    def restore(self, armed: Optional[Reference], opposite: Optional[Reference]) -> None:
        if armed is not None:
            self.armed = armed
        if opposite is not None:
            self.opposite = opposite

    @property
    def ready(self) -> bool:
        return self.armed is not None

    def score(self, patch_bgr: Optional[np.ndarray]) -> ReferenceScore:
        """Similarity to each reference, plus whether either is a real match."""
        patch, hist = describe(patch_bgr) if patch_bgr is not None else (None, None)
        if patch is None or hist is None or self.armed is None:
            return ReferenceScore(similarity_armed=0.0, similarity_opposite=0.0, present_prob=0.5, decisive=False)

        sim_armed = _similarity(self.armed, patch, hist)
        sim_opposite = _similarity(self.opposite, patch, hist) if self.opposite is not None else None
        # One usable reference: a *low* similarity is not an inconclusive answer,
        # it is the answer. The armed reference describes what the operator saw;
        # once the region stops resembling it, that is a measurement, not a
        # failure to measure. Whether the change is the state they asked about is
        # decided by the band, not by withholding the reading.
        if sim_opposite is None or self.opposite.samples < self.min_opposite_samples:
            present_prob = sim_armed if self.armed_occupied else 1.0 - sim_armed
            return ReferenceScore(
                similarity_armed=float(sim_armed),
                similarity_opposite=float(sim_opposite) if sim_opposite is not None else 0.0,
                present_prob=float(_clamp(present_prob)),
                decisive=True,
            )

        # Two established references are only a better answer when one of them
        # actually matches. Resembling neither is not blindness — it is a
        # measurement of unfamiliar content — so the ratio is used when a
        # reference matches, and the single-reference reading stands otherwise.
        # Real blindness is reported by the caller: no measurement, stale
        # detections, a blocked view, or a large illumination change.
        if max(sim_armed, sim_opposite) >= self.decisive_similarity:
            positive = sim_armed if self.armed_occupied else sim_opposite
            total = sim_armed + sim_opposite
            present_prob = positive / total if total > 1e-6 else 0.5
        else:
            present_prob = sim_armed if self.armed_occupied else 1.0 - sim_armed
        return ReferenceScore(
            similarity_armed=float(sim_armed),
            similarity_opposite=float(sim_opposite),
            present_prob=float(_clamp(present_prob)),
            decisive=True,
        )

    def observe(self, patch_bgr: Optional[np.ndarray], *, present_prob: float, ts: float) -> None:
        """Learn the opposite reference, and let both drift while confidently nominal."""
        if patch_bgr is None:
            return
        patch, hist = describe(patch_bgr)
        if patch is None or hist is None or self.armed is None:
            return
        if present_prob >= self.present_floor:
            self._drift_toward(self.armed_state_reference(), patch, hist, present_prob, ts)
        elif present_prob <= self.absent_ceiling:
            self._drift_toward(self.opposite_state_reference(), patch, hist, 1.0 - present_prob, ts)

    def armed_state_reference(self) -> Optional[Reference]:
        """The reference describing the state the watch was armed in."""
        if self.armed_occupied:
            return self.armed
        return self.opposite if self.opposite is not None else self.armed

    def opposite_state_reference(self) -> Optional[Reference]:
        return self.opposite if self.armed_occupied else self.armed

    def _drift_toward(
        self,
        target: Optional[Reference],
        patch: np.ndarray,
        hist: np.ndarray,
        confidence: float,
        ts: float,
    ) -> None:
        if target is None:
            # First confident observation of the opposite state: remember it.
            if self.opposite is None:
                self.opposite = Reference(patch=patch.copy(), histogram=hist.copy(), updated_at=ts, samples=1)
            return
        rate = self.drift * float(confidence)
        target.patch = (1.0 - rate) * target.patch + rate * patch
        target.histogram = (1.0 - rate) * target.histogram + rate * hist
        total = float(target.histogram.sum())
        if total > 0:
            target.histogram /= total
        target.updated_at = ts
        target.samples += 1


def describe(patch_bgr: Optional[np.ndarray]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Reduce a region to (z-scored 64x64 thumbnail, L1-normalised HSV histogram)."""
    if patch_bgr is None or patch_bgr.size == 0:
        return None, None
    if min(patch_bgr.shape[:2]) < 4:
        return None, None
    resized = cv2.resize(patch_bgr, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [HIST_BINS, HIST_BINS], [0, 180, 0, 256]).astype(np.float32).reshape(-1)
    total = float(hist.sum())
    if total <= 0:
        return None, None
    hist /= total
    return _to_zscore(gray), hist


def patch_for(frame: np.ndarray, box: Box) -> Optional[np.ndarray]:
    """Crop a region, tolerating boxes that hang off the frame."""
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(round(box[0]))))
    y1 = max(0, min(height - 1, int(round(box[1]))))
    x2 = max(x1 + 1, min(width, int(round(box[2]))))
    y2 = max(y1 + 1, min(height, int(round(box[3]))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _to_zscore(gray: np.ndarray) -> np.ndarray:
    """Remove mean brightness and contrast so dusk is not a state change."""
    centered = gray - float(gray.mean())
    std = float(centered.std())
    if std < 1e-3:
        return centered
    return centered / std


def _from_zscore(patch: np.ndarray) -> np.ndarray:
    """Map a z-scored patch back to displayable 8-bit for storage and the UI."""
    return np.clip(patch * 32.0 + 128.0, 0, 255).astype(np.uint8)


def _similarity(reference: Reference, patch: np.ndarray, hist: np.ndarray) -> float:
    """Similarity in 0..1: how much of the region still matches the reference.

    The primary signal is the z-scored absolute difference, not correlation.
    Calibrated on real footage (a pickup leaving its stopped position): the same
    region measures <= 0.4 while the vehicle is there and >= 0.75 once it has
    gone. Correlation cannot express that — two unrelated patches of road
    correlate only weakly, and mapping [-1, 1] onto [0, 1] reports that as "half
    similar", which is precisely the band where no decision is possible.
    """
    diff = float(np.mean(np.abs(reference.patch - patch)))
    if not math.isfinite(diff):
        return 0.0
    structure = 1.0 - min(1.0, diff / STRUCTURE_DIFF_SCALE)
    colour = float(cv2.compareHist(reference.histogram.reshape(-1, 1), hist.reshape(-1, 1), cv2.HISTCMP_CORREL))
    if not math.isfinite(colour):
        colour = 0.0
    return float(_clamp(0.8 * structure + 0.2 * _clamp(colour)))
