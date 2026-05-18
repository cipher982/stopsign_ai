from unittest.mock import MagicMock

import numpy as np

from stopsign.tracking import Car
from stopsign.tracking import StopDetector
from stopsign.trajectory_scorer import TrajectoryScore


def test_shadow_candidate_scoring_is_throttled(mock_config, mock_database):
    detector = StopDetector(mock_config, mock_database)
    detector.stop_zone = np.array([[900, 700], [1150, 700], [1150, 860], [900, 860]], dtype=np.float32)
    detector.pre_stop_line_proc = np.array([[1660, 650], [1660, 900]], dtype=np.float32)
    detector.capture_line_proc = np.array([[1460, 650], [1460, 900]], dtype=np.float32)
    detector._score_current_trajectory = MagicMock(
        return_value=TrajectoryScore(False, "never_in_zone"),
    )

    car = Car(id=1, config=mock_config)
    car.state.last_update_time = 1000.0
    detector._log_trajectory_shadow_candidate(car)
    car.state.last_update_time = 1000.5
    detector._log_trajectory_shadow_candidate(car)
    car.state.last_update_time = 1001.1
    detector._log_trajectory_shadow_candidate(car)

    assert detector._score_current_trajectory.call_count == 2
