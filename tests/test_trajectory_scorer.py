import numpy as np
import pytest

from stopsign.trajectory_scorer import score_raw_payload
from stopsign.trajectory_scorer import score_samples

SCHEMA = ["t", "x", "y", "x1", "y1", "x2", "y2", "raw_speed", "speed"]
STOP_ZONE = np.array([[900, 700], [1150, 700], [1150, 860], [900, 860]], dtype=np.float32)
PRE_STOP_LINE = np.array([[1660, 650], [1660, 900]], dtype=np.float32)


def _sample(t, x, y, bbox, raw_speed=80.0):
    x1, y1, x2, y2 = bbox
    return [t, x, y, x1, y1, x2, y2, raw_speed, raw_speed]


def test_scores_completed_approach_pass():
    samples = [
        _sample(1000.0, 1320.0, 740.0, (1260.0, 690.0, 1380.0, 790.0)),
        _sample(1000.1, 1200.0, 740.0, (1140.0, 690.0, 1260.0, 790.0)),
        _sample(1000.2, 1100.0, 750.0, (1040.0, 705.0, 1160.0, 805.0), raw_speed=45.0),
        _sample(1000.4, 1030.0, 760.0, (970.0, 710.0, 1090.0, 810.0), raw_speed=10.0),
        _sample(1000.6, 950.0, 760.0, (890.0, 710.0, 1010.0, 810.0), raw_speed=55.0),
    ]

    score = score_samples(
        samples,
        SCHEMA,
        STOP_ZONE,
        PRE_STOP_LINE,
        stop_speed_threshold=20.0,
        min_zone_time_sec=0.2,
    )

    assert score.would_record_pass is True
    assert score.reason == "pass"
    assert score.time_in_zone == pytest.approx(0.5)
    assert score.entry_time == pytest.approx(1000.1)
    assert score.exit_time == pytest.approx(1000.6)
    assert score.min_speed is not None
    assert score.stop_duration > 0
    assert score.approach_progress > 20.0


def test_rejects_parked_jitter_in_zone():
    samples = [
        _sample(1000.0, 1040.0, 750.0, (980.0, 710.0, 1100.0, 800.0), raw_speed=1.0),
        _sample(1000.2, 1041.0, 750.0, (980.0, 710.0, 1100.0, 800.0), raw_speed=1.0),
        _sample(1000.4, 1039.0, 750.0, (980.0, 710.0, 1100.0, 800.0), raw_speed=1.0),
    ]

    score = score_samples(
        samples,
        SCHEMA,
        STOP_ZONE,
        PRE_STOP_LINE,
        stop_speed_threshold=20.0,
        min_zone_time_sec=0.2,
    )

    assert score.would_record_pass is False
    assert score.reason == "insufficient_progress"


def test_scores_raw_payload_with_raw_geometry_conversion():
    raw_payload = {
        "sample_schema": SCHEMA,
        "samples": [
            _sample(1000.0, 1320.0, 740.0, (1260.0, 690.0, 1380.0, 790.0)),
            _sample(1000.2, 1100.0, 750.0, (1040.0, 705.0, 1160.0, 805.0)),
            _sample(1000.4, 1030.0, 760.0, (970.0, 710.0, 1090.0, 810.0)),
        ],
        "dimensions": {"raw": {"width": 1800, "height": 900}},
        "config_snapshot": {
            "video_processing": {"scale": 1.0, "crop_top": 0.0, "crop_side": 0.0},
            "stopsign_detection": {
                "stop_zone": STOP_ZONE.tolist(),
                "pre_stop_line": PRE_STOP_LINE.tolist(),
                "stop_speed_threshold": 20.0,
                "in_zone_time_threshold": 0.1,
            },
        },
    }

    score = score_raw_payload(raw_payload)

    assert score.would_record_pass is True
