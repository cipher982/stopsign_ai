"""Frame freeze detection helpers for camera ingest pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FreezeDetectionEvent:
    """Result of a single freeze-detector update."""

    frozen: bool
    freeze_age_sec: float
    mad: float
    incident_started: bool
    incident_resolved: bool


class FrameFreezeDetector:
    """Detect frozen streams using mean absolute frame difference (MAD)."""

    def __init__(
        self,
        freeze_detect_sec: float,
        mad_threshold: float,
        sample_width: int = 160,
        sample_height: int = 90,
    ) -> None:
        self.freeze_detect_sec = max(0.0, float(freeze_detect_sec))
        self.mad_threshold = max(0.0, float(mad_threshold))
        self.sample_width = max(8, int(sample_width))
        self.sample_height = max(8, int(sample_height))

        self._prev_sample: np.ndarray | None = None
        self._last_motion_ts: float | None = None
        self._frozen = False

    def reset(self) -> None:
        self._prev_sample = None
        self._last_motion_ts = None
        self._frozen = False

    def update(self, frame: np.ndarray, now_ts: float) -> FreezeDetectionEvent:
        if self.freeze_detect_sec <= 0:
            return FreezeDetectionEvent(
                frozen=False,
                freeze_age_sec=0.0,
                mad=0.0,
                incident_started=False,
                incident_resolved=False,
            )

        sample = cv2.resize(frame, (self.sample_width, self.sample_height), interpolation=cv2.INTER_AREA)
        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

        if self._prev_sample is None:
            self._prev_sample = sample_gray
            self._last_motion_ts = float(now_ts)
            return FreezeDetectionEvent(
                frozen=False,
                freeze_age_sec=0.0,
                mad=0.0,
                incident_started=False,
                incident_resolved=False,
            )

        mad = float(cv2.absdiff(self._prev_sample, sample_gray).mean())
        self._prev_sample = sample_gray

        if self._last_motion_ts is None or mad > self.mad_threshold:
            self._last_motion_ts = float(now_ts)

        freeze_age_sec = max(0.0, float(now_ts) - float(self._last_motion_ts or now_ts))
        frozen_now = freeze_age_sec >= self.freeze_detect_sec

        incident_started = frozen_now and not self._frozen
        incident_resolved = (not frozen_now) and self._frozen
        self._frozen = frozen_now

        return FreezeDetectionEvent(
            frozen=frozen_now,
            freeze_age_sec=freeze_age_sec,
            mad=mad,
            incident_started=incident_started,
            incident_resolved=incident_resolved,
        )
