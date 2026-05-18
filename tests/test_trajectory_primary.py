from unittest.mock import MagicMock

import numpy as np

from stopsign.tracking import Car
from stopsign.tracking import StopDetector


def _make_detector(mock_config, mock_database):
    mock_config.in_zone_frame_threshold = 2
    mock_config.out_zone_frame_threshold = 2
    mock_config.in_zone_time_threshold = 0.1
    mock_config.out_zone_time_threshold = 0.1
    mock_config.stop_speed_threshold = 20.0
    mock_database.add_vehicle_pass.return_value = 123

    detector = StopDetector(mock_config, mock_database)
    detector.stop_zone = np.array([[900, 700], [1150, 700], [1150, 860], [900, 860]], dtype=np.float32)
    detector.pre_stop_line_proc = np.array([[1660, 650], [1660, 900]], dtype=np.float32)
    detector.capture_line_proc = np.array([[1460, 650], [1460, 900]], dtype=np.float32)
    detector.capture_car_image = MagicMock(
        side_effect=lambda car, timestamp, frame: (
            setattr(car.state.capture, "image_captured", True),
            setattr(car.state.capture, "image_path", "local://test.jpg"),
        )
    )
    return detector


def _update_car(car, detector, timestamp, location, bbox, frame):
    prev_timestamp = car.update(location, timestamp, bbox)
    car.state.raw_speed = car.state.speed
    detector.update_car_stop_status(car, timestamp, frame, prev_timestamp=prev_timestamp)


def test_trajectory_primary_records_late_track(mock_config, mock_database):
    detector = _make_detector(mock_config, mock_database)
    car = Car(id=42, config=mock_config)
    frame = np.zeros((900, 1800, 3), dtype=np.uint8)
    base = 1000.0

    for idx, (location, bbox) in enumerate(
        [
            ((1185.0, 735.0), (1160.0, 670.0, 1260.0, 780.0)),
            ((1120.0, 735.0), (1040.0, 690.0, 1200.0, 780.0)),
            ((1060.0, 740.0), (980.0, 700.0, 1140.0, 790.0)),
            ((1000.0, 745.0), (920.0, 705.0, 1080.0, 795.0)),
            ((850.0, 745.0), (780.0, 705.0, 880.0, 795.0)),
            ((760.0, 745.0), (700.0, 705.0, 800.0, 795.0)),
            ((680.0, 745.0), (620.0, 705.0, 720.0, 795.0)),
        ]
    ):
        _update_car(car, detector, base + idx * 0.1, location, bbox, frame)

    assert mock_database.add_vehicle_pass.called
    _, kwargs = mock_database.add_vehicle_pass.call_args
    assert kwargs["raw_payload"]["raw_complete"] is True
    assert kwargs["image_path"] == "local://test.jpg"
    assert kwargs["time_in_zone"] == kwargs["raw_payload"]["summary"]["time_in_zone"]


def test_trajectory_primary_rejects_backward_artifact_even_if_old_gate_passed(mock_config, mock_database):
    detector = _make_detector(mock_config, mock_database)
    car = Car(id=99, config=mock_config)
    car.state.zone.passed_pre_stop = True
    frame = np.zeros((900, 1800, 3), dtype=np.uint8)
    base = 1000.0

    for idx, (location, bbox) in enumerate(
        [
            ((980.0, 745.0), (920.0, 705.0, 1040.0, 795.0)),
            ((1040.0, 745.0), (980.0, 705.0, 1100.0, 795.0)),
            ((1100.0, 745.0), (1040.0, 705.0, 1160.0, 795.0)),
            ((1210.0, 745.0), (1160.0, 705.0, 1260.0, 795.0)),
            ((1300.0, 745.0), (1250.0, 705.0, 1350.0, 795.0)),
        ]
    ):
        _update_car(car, detector, base + idx * 0.1, location, bbox, frame)

    assert mock_database.add_vehicle_pass.call_count == 0
    assert detector.capture_car_image.call_count == 0
