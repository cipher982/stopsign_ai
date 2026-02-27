"""Unit tests for frame freeze detection."""

import numpy as np

from stopsign.freeze_detector import FrameFreezeDetector


def test_freeze_detector_opens_incident_after_threshold():
    detector = FrameFreezeDetector(
        freeze_detect_sec=2.0,
        mad_threshold=0.1,
        sample_width=32,
        sample_height=18,
    )
    static = np.zeros((120, 160, 3), dtype=np.uint8)

    event = detector.update(static, now_ts=100.0)
    assert event.frozen is False

    detector.update(static, now_ts=100.5)
    detector.update(static, now_ts=101.0)
    detector.update(static, now_ts=101.5)
    event = detector.update(static, now_ts=102.1)

    assert event.frozen is True
    assert event.incident_started is True
    assert event.freeze_age_sec >= 2.0


def test_freeze_detector_resolves_on_motion():
    detector = FrameFreezeDetector(
        freeze_detect_sec=1.5,
        mad_threshold=0.1,
        sample_width=32,
        sample_height=18,
    )
    static = np.zeros((120, 160, 3), dtype=np.uint8)
    moving = static.copy()
    moving[20:80, 40:120] = 255

    detector.update(static, now_ts=200.0)
    detector.update(static, now_ts=200.8)
    detector.update(static, now_ts=201.6)  # opens incident
    event = detector.update(moving, now_ts=201.8)

    assert event.frozen is False
    assert event.incident_resolved is True
    assert event.mad > 0.1


def test_freeze_detector_disabled():
    detector = FrameFreezeDetector(
        freeze_detect_sec=0.0,
        mad_threshold=0.1,
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    event = detector.update(frame, now_ts=300.0)
    assert event.frozen is False
    assert event.freeze_age_sec == 0.0
