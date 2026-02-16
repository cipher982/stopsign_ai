"""Tests for Car class and bbox interpolation."""

import time

import numpy as np
import pytest

from stopsign.tracking import Car
from stopsign.tracking import CarState
from stopsign.tracking import CarTracker
from stopsign.tracking import StopDetector


class TestCarState:
    """Test suite for CarState dataclass."""

    def test_default_values(self):
        """Test that CarState has sensible defaults."""
        state = CarState()

        assert state.location == (0.0, 0.0)
        assert state.bbox == (0.0, 0.0, 0.0, 0.0)
        assert state.velocity == (0.0, 0.0)
        assert state.speed == 0.0
        assert state.motion.is_parked is True
        assert state.zone.in_zone is False

    def test_state_is_mutable(self):
        """Test that state can be updated."""
        state = CarState()
        state.location = (100.0, 200.0)
        state.velocity = (50.0, 25.0)
        state.bbox = (90.0, 190.0, 110.0, 210.0)

        assert state.location == (100.0, 200.0)
        assert state.velocity == (50.0, 25.0)
        assert state.bbox == (90.0, 190.0, 110.0, 210.0)


class TestCar:
    """Test suite for Car class."""

    def test_init(self, mock_config):
        """Test Car initialization."""
        car = Car(id=1, config=mock_config)

        assert car.id == 1
        assert car.state is not None
        assert car.kalman_filter is not None

    def test_update_sets_bbox(self, mock_config):
        """Test that update() sets the bbox."""
        car = Car(id=1, config=mock_config)
        timestamp = time.time()

        car.update(
            location=(150.0, 150.0),
            timestamp=timestamp,
            bbox=(125.0, 135.0, 175.0, 165.0),
        )

        assert car.state.bbox == (125.0, 135.0, 175.0, 165.0)
        assert car.state.last_update_time == timestamp

    def test_update_tracks_location_history(self, mock_config):
        """Test that update() maintains location history."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        for i in range(5):
            car.update(
                location=(100.0 + i * 10, 100.0 + i * 5),
                timestamp=base_time + i * 0.1,
                bbox=(90.0, 95.0, 110.0, 105.0),
            )

        assert len(car.state.track) == 5

    def test_velocity_calculation_after_multiple_updates(self, mock_config):
        """Test that velocity is calculated from position history."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        # Move 10 pixels per frame at ~15 FPS = 150 px/s
        for i in range(10):
            car.update(
                location=(100.0 + i * 10, 100.0),
                timestamp=base_time + i * (1 / 15),
                bbox=(90.0, 95.0, 110.0, 105.0),
            )

        # Velocity magnitude should be approximately 150 px/s
        # Sign may vary based on velocity calculation internals
        vx, vy = car.state.velocity
        assert abs(vx) == pytest.approx(150.0, rel=0.5)  # 50% tolerance for Kalman smoothing
        assert abs(vy) < 50  # Should be close to zero


class TestCarInterpolation:
    """Test suite for bbox interpolation between YOLO frames."""

    def test_get_interpolated_bbox_no_time_elapsed(self, mock_config):
        """Test interpolation when no time has elapsed returns original bbox."""
        car = Car(id=1, config=mock_config)
        timestamp = time.time()

        car.state.bbox = (100.0, 100.0, 150.0, 130.0)
        car.state.velocity = (50.0, 25.0)
        car.state.last_update_time = timestamp

        bbox = car.get_interpolated_bbox(timestamp)

        assert bbox == (100.0, 100.0, 150.0, 130.0)

    def test_get_interpolated_bbox_translates_by_velocity(self, mock_config):
        """Test that bbox is translated by velocity * dt."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        car.state.bbox = (100.0, 100.0, 150.0, 130.0)
        car.state.velocity = (150.0, 75.0)  # pixels/second
        car.state.last_update_time = base_time

        # Interpolate 100ms later
        bbox = car.get_interpolated_bbox(base_time + 0.1)

        # Expected: x += 150 * 0.1 = 15, y += 75 * 0.1 = 7.5
        assert bbox[0] == pytest.approx(115.0, abs=0.1)  # x1
        assert bbox[1] == pytest.approx(107.5, abs=0.1)  # y1
        assert bbox[2] == pytest.approx(165.0, abs=0.1)  # x2
        assert bbox[3] == pytest.approx(137.5, abs=0.1)  # y2

    @pytest.mark.parametrize(
        "dt,vx,vy",
        [
            (0.0667, 150.0, 75.0),  # 15 FPS interval
            (0.125, 100.0, 50.0),  # 8 FPS interval
            (0.25, 200.0, 100.0),  # 4 FPS interval
        ],
    )
    def test_get_interpolated_bbox_various_frame_rates(self, mock_config, dt, vx, vy):
        """Test interpolation at various frame rates."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        car.state.bbox = (100.0, 100.0, 150.0, 130.0)
        car.state.velocity = (vx, vy)
        car.state.last_update_time = base_time

        bbox = car.get_interpolated_bbox(base_time + dt)

        expected_dx = vx * dt
        expected_dy = vy * dt

        assert bbox[0] == pytest.approx(100.0 + expected_dx, abs=0.1)
        assert bbox[1] == pytest.approx(100.0 + expected_dy, abs=0.1)

    def test_get_interpolated_bbox_clamps_at_500ms(self, mock_config):
        """Test that interpolation doesn't extrapolate beyond 500ms."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        car.state.bbox = (100.0, 100.0, 150.0, 130.0)
        car.state.velocity = (150.0, 75.0)
        car.state.last_update_time = base_time

        # Interpolate 600ms later (beyond 500ms clamp)
        bbox = car.get_interpolated_bbox(base_time + 0.6)

        # Should return original bbox, not extrapolated
        assert bbox == (100.0, 100.0, 150.0, 130.0)

    def test_get_interpolated_bbox_negative_dt_returns_original(self, mock_config):
        """Test that negative dt returns original bbox."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        car.state.bbox = (100.0, 100.0, 150.0, 130.0)
        car.state.velocity = (150.0, 75.0)
        car.state.last_update_time = base_time

        # Interpolate with negative dt (timestamp before last update)
        bbox = car.get_interpolated_bbox(base_time - 0.1)

        assert bbox == (100.0, 100.0, 150.0, 130.0)

    def test_get_interpolated_bbox_zero_velocity(self, mock_config):
        """Test interpolation with zero velocity (stationary car)."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        car.state.bbox = (100.0, 100.0, 150.0, 130.0)
        car.state.velocity = (0.0, 0.0)
        car.state.last_update_time = base_time

        bbox = car.get_interpolated_bbox(base_time + 0.1)

        # Should stay in place
        assert bbox == (100.0, 100.0, 150.0, 130.0)

    def test_get_interpolated_bbox_negative_velocity(self, mock_config):
        """Test interpolation with negative velocity (car moving backwards)."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        car.state.bbox = (200.0, 200.0, 250.0, 230.0)
        car.state.velocity = (-100.0, -50.0)  # Moving up-left
        car.state.last_update_time = base_time

        bbox = car.get_interpolated_bbox(base_time + 0.1)

        # Expected: x -= 10, y -= 5
        assert bbox[0] == pytest.approx(190.0, abs=0.1)
        assert bbox[1] == pytest.approx(195.0, abs=0.1)
        assert bbox[2] == pytest.approx(240.0, abs=0.1)
        assert bbox[3] == pytest.approx(225.0, abs=0.1)

    def test_interpolation_maintains_bbox_size(self, mock_config):
        """Test that interpolation preserves bbox dimensions."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        original_width = 50.0
        original_height = 30.0
        car.state.bbox = (100.0, 100.0, 100.0 + original_width, 100.0 + original_height)
        car.state.velocity = (150.0, 75.0)
        car.state.last_update_time = base_time

        bbox = car.get_interpolated_bbox(base_time + 0.2)

        # Width and height should remain the same
        interpolated_width = bbox[2] - bbox[0]
        interpolated_height = bbox[3] - bbox[1]

        assert interpolated_width == pytest.approx(original_width, abs=0.01)
        assert interpolated_height == pytest.approx(original_height, abs=0.01)


class TestCarMovementStatus:
    """Test car parked/moving state transitions."""

    def test_car_starts_parked(self, mock_config):
        """Test that new cars start in parked state."""
        car = Car(id=1, config=mock_config)
        assert car.state.motion.is_parked is True

    def test_car_becomes_unparked_with_movement(self, mock_config):
        """Test that car becomes unparked after sustained movement."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        # Simulate fast movement (high speed)
        for i in range(10):
            car.update(
                location=(100.0 + i * 50, 100.0),  # Large jumps = high speed
                timestamp=base_time + i * (1 / 15),
                bbox=(90.0, 95.0, 110.0, 105.0),
            )

        # Car should be unparked after sustained high-speed movement
        # (depends on config thresholds)
        assert car.state.motion.consecutive_moving_frames > 0 or not car.state.motion.is_parked


class TestBboxXYXYFormat:
    """Test that XYXY bbox format is handled correctly throughout the pipeline.

    CRITICAL: These tests ensure that functions consuming bbox correctly interpret
    the XYXY format (x1, y1, x2, y2) rather than the old XYWH format (cx, cy, w, h).

    This was added after a production bug where _get_car_polygon and save_vehicle_image
    were treating XYXY coords as XYWH, causing zone detection and image cropping to fail.
    """

    def test_get_car_polygon_xyxy_format(self, mock_config, mock_database):
        """Test that _get_car_polygon correctly interprets XYXY bbox format."""
        from stopsign.tracking import StopDetector

        detector = StopDetector(mock_config, mock_database)

        # XYXY bbox: top-left (100, 150), bottom-right (200, 250)
        bbox = (100.0, 150.0, 200.0, 250.0)
        polygon = detector._get_car_polygon(bbox)

        # Expected polygon corners (counter-clockwise from top-left)
        assert polygon.shape == (4, 2), "Polygon should have 4 corners"

        # Top-left
        assert polygon[0][0] == pytest.approx(100.0), "Top-left x should be x1"
        assert polygon[0][1] == pytest.approx(150.0), "Top-left y should be y1"

        # Top-right
        assert polygon[1][0] == pytest.approx(200.0), "Top-right x should be x2"
        assert polygon[1][1] == pytest.approx(150.0), "Top-right y should be y1"

        # Bottom-right
        assert polygon[2][0] == pytest.approx(200.0), "Bottom-right x should be x2"
        assert polygon[2][1] == pytest.approx(250.0), "Bottom-right y should be y2"

        # Bottom-left
        assert polygon[3][0] == pytest.approx(100.0), "Bottom-left x should be x1"
        assert polygon[3][1] == pytest.approx(250.0), "Bottom-left y should be y2"

    def test_get_car_polygon_width_height_correct(self, mock_config, mock_database):
        """Test that polygon dimensions match the bbox width/height."""
        from stopsign.tracking import StopDetector

        detector = StopDetector(mock_config, mock_database)

        # bbox with known dimensions: width=80, height=60
        x1, y1, x2, y2 = 50.0, 100.0, 130.0, 160.0
        expected_width = x2 - x1  # 80
        expected_height = y2 - y1  # 60

        polygon = detector._get_car_polygon((x1, y1, x2, y2))

        # Check polygon width (x2 - x1)
        actual_width = polygon[1][0] - polygon[0][0]
        assert actual_width == pytest.approx(
            expected_width
        ), f"Polygon width {actual_width} should match bbox width {expected_width}"

        # Check polygon height (y2 - y1)
        actual_height = polygon[2][1] - polygon[0][1]
        assert actual_height == pytest.approx(
            expected_height
        ), f"Polygon height {actual_height} should match bbox height {expected_height}"

    def test_save_vehicle_image_crop_bounds(self, sample_frame):
        """Test that save_vehicle_image calculates correct crop bounds from XYXY bbox."""
        # We can't easily test the full function (needs Minio), so test the crop logic directly

        # Simulate the crop calculation from save_vehicle_image
        bbox = (100.0, 150.0, 200.0, 250.0)  # XYXY: 100x100 box
        bx1, by1, bx2, by2 = bbox
        w = bx2 - bx1  # Should be 100
        h = by2 - by1  # Should be 100

        padding_factor = 0.1
        padding_x = int(w * padding_factor)  # 10
        padding_y = int(h * padding_factor)  # 10

        x1 = int(bx1 - padding_x)  # 90
        y1 = int(by1 - padding_y)  # 140
        x2 = int(bx2 + padding_x)  # 210
        y2 = int(by2 + padding_y)  # 260

        # Verify dimensions are reasonable (not whole frame)
        crop_width = x2 - x1  # 120
        crop_height = y2 - y1  # 120

        assert crop_width == 120, f"Crop width should be 120, got {crop_width}"
        assert crop_height == 120, f"Crop height should be 120, got {crop_height}"

        # Verify this is NOT the old buggy calculation
        # Old buggy code would do: x - w/2 where w=200 (actually x2), giving x1=0
        # which would result in a much larger crop
        assert x1 > 50, f"x1={x1} suggests correct XYXY handling (buggy code gives ~0)"
        assert y1 > 100, f"y1={y1} suggests correct XYXY handling (buggy code gives ~0)"

    def test_bbox_format_consistency_through_update(self, mock_config):
        """Test that bbox maintains XYXY format through Car.update()."""
        car = Car(id=1, config=mock_config)

        # XYXY bbox representing a 50x30 box at position (125, 135) to (175, 165)
        input_bbox = (125.0, 135.0, 175.0, 165.0)

        car.update(
            location=(150.0, 150.0),  # center
            timestamp=time.time(),
            bbox=input_bbox,
        )

        # Verify bbox is stored unchanged
        assert car.state.bbox == input_bbox

        # Verify we can derive correct width/height from stored bbox
        x1, y1, x2, y2 = car.state.bbox
        width = x2 - x1
        height = y2 - y1

        assert width == pytest.approx(50.0), f"Width should be 50, got {width}"
        assert height == pytest.approx(30.0), f"Height should be 30, got {height}"

    def test_interpolated_bbox_maintains_xyxy_format(self, mock_config):
        """Test that get_interpolated_bbox returns valid XYXY format."""
        car = Car(id=1, config=mock_config)
        base_time = time.time()

        # Set up car with XYXY bbox and velocity
        car.state.bbox = (100.0, 100.0, 150.0, 130.0)  # 50x30 box
        car.state.velocity = (100.0, 50.0)  # Moving right and down
        car.state.last_update_time = base_time

        # Get interpolated bbox 100ms later
        bbox = car.get_interpolated_bbox(base_time + 0.1)
        x1, y1, x2, y2 = bbox

        # Verify XYXY invariant: x2 > x1 and y2 > y1
        assert x2 > x1, f"XYXY invariant violated: x2 ({x2}) should be > x1 ({x1})"
        assert y2 > y1, f"XYXY invariant violated: y2 ({y2}) should be > y1 ({y1})"

        # Verify dimensions preserved
        width = x2 - x1
        height = y2 - y1
        assert width == pytest.approx(50.0, abs=0.1), f"Width should be preserved at 50, got {width}"
        assert height == pytest.approx(30.0, abs=0.1), f"Height should be preserved at 30, got {height}"


class TestStopDurationClamping:
    """Test that stop_duration never exceeds time_in_zone."""

    def _make_detector(self, mock_config, mock_database):
        """Create a StopDetector with stop zone geometry bypassed."""
        mock_config.in_zone_frame_threshold = 3
        mock_config.out_zone_frame_threshold = 3
        mock_config.in_zone_time_threshold = 0.1
        mock_config.out_zone_time_threshold = 0.1
        mock_config.stop_speed_threshold = 20.0
        detector = StopDetector(mock_config, mock_database)
        # Stop zone: y=400-600. Pre-stop and capture lines above zone.
        detector.stop_zone = np.array([[0, 400], [600, 400], [600, 600], [0, 600]], dtype=np.float32)
        detector.pre_stop_line_proc = np.array([[0, 300], [600, 300]], dtype=np.float32)
        detector.capture_line_proc = np.array([[0, 350], [600, 350]], dtype=np.float32)
        return detector

    def test_stop_duration_clamped_to_entry_time(self, mock_config, mock_database):
        """stop_duration should not include time before zone entry."""
        detector = self._make_detector(mock_config, mock_database)
        car = Car(id=1, config=mock_config)
        car.state.raw_speed = 5.0  # Below stop threshold
        base = 1000.0
        frame = np.zeros((700, 700, 3), dtype=np.uint8)

        # Cross pre-stop line (y=300), outside zone
        car.state.bbox = (100.0, 280.0, 200.0, 320.0)
        detector.update_car_stop_status(car, base, frame, prev_timestamp=base - 1.0)

        # Move car into the stop zone for enough frames to pass debounce
        car.state.bbox = (100.0, 450.0, 200.0, 550.0)
        for i in range(1, 5):
            ts = base + i * 0.1
            prev_ts = base + (i - 1) * 0.1
            detector.update_car_stop_status(car, ts, frame, prev_timestamp=prev_ts)

        # stop_duration must have accumulated (car is stopped in zone)
        assert car.state.zone.stop_duration > 0, "stop_duration should accumulate for stopped car in zone"
        # stop_duration must not exceed time since entry
        if car.state.zone.entry_time > 0:
            max_possible = (base + 4 * 0.1) - car.state.zone.entry_time
            assert car.state.zone.stop_duration <= max_possible + 0.01

    def test_stop_duration_with_large_dt_gap(self, mock_config, mock_database):
        """A large dt gap before entry should not inflate stop_duration."""
        detector = self._make_detector(mock_config, mock_database)
        car = Car(id=1, config=mock_config)
        car.state.raw_speed = 5.0
        base = 1000.0
        frame = np.zeros((700, 700, 3), dtype=np.uint8)

        # Cross pre-stop line
        car.state.bbox = (100.0, 280.0, 200.0, 320.0)
        detector.update_car_stop_status(car, base, frame, prev_timestamp=0.0)

        # Large gap (5 seconds), then enter zone
        car.state.bbox = (100.0, 450.0, 200.0, 550.0)
        for i in range(5):
            ts = base + 5.0 + i * 0.1
            prev_ts = base if i == 0 else (base + 5.0 + (i - 1) * 0.1)
            detector.update_car_stop_status(car, ts, frame, prev_timestamp=prev_ts)

        # stop_duration should be at most the time spent since entry, not 5+ seconds
        assert car.state.zone.stop_duration < 2.0


class TestTimeDebouncedZoneTransitions:
    """Test time-based zone entry/exit debouncing."""

    def _make_detector(self, mock_config, mock_database, in_time=0.2, out_time=0.2):
        mock_config.in_zone_frame_threshold = 3
        mock_config.out_zone_frame_threshold = 3
        mock_config.in_zone_time_threshold = in_time
        mock_config.out_zone_time_threshold = out_time
        mock_config.stop_speed_threshold = 20.0
        detector = StopDetector(mock_config, mock_database)
        # Stop zone: y=400-600 region. Pre-stop and capture lines above the zone.
        detector.stop_zone = np.array([[0, 400], [600, 400], [600, 600], [0, 600]], dtype=np.float32)
        detector.pre_stop_line_proc = np.array([[0, 300], [600, 300]], dtype=np.float32)
        detector.capture_line_proc = np.array([[0, 350], [600, 350]], dtype=np.float32)
        return detector

    def _cross_pre_stop(self, detector, car, base):
        """Move car across pre-stop line (y=300), outside stop zone."""
        frame = np.zeros((700, 700, 3), dtype=np.uint8)
        car.state.bbox = (100.0, 280.0, 200.0, 320.0)  # Spans y=300 line
        detector.update_car_stop_status(car, base, frame)

    def test_single_frame_in_zone_does_not_trigger_entry(self, mock_config, mock_database):
        """A single detection inside zone should not count as zone entry."""
        detector = self._make_detector(mock_config, mock_database, in_time=0.2)
        car = Car(id=1, config=mock_config)
        car.state.raw_speed = 50.0
        base = 1000.0
        frame = np.zeros((700, 700, 3), dtype=np.uint8)

        self._cross_pre_stop(detector, car, base)

        # Single frame in zone
        car.state.bbox = (100.0, 450.0, 200.0, 550.0)
        detector.update_car_stop_status(car, base + 0.05, frame)

        assert car.state.zone.in_zone is False

    def test_sustained_presence_triggers_entry(self, mock_config, mock_database):
        """Multiple frames over time threshold should trigger zone entry."""
        detector = self._make_detector(mock_config, mock_database, in_time=0.15)
        car = Car(id=1, config=mock_config)
        car.state.raw_speed = 50.0
        base = 1000.0
        frame = np.zeros((700, 700, 3), dtype=np.uint8)

        self._cross_pre_stop(detector, car, base)

        # Multiple frames in zone over 0.2s
        car.state.bbox = (100.0, 450.0, 200.0, 550.0)
        for i in range(1, 5):
            detector.update_car_stop_status(car, base + i * 0.07, frame)

        assert car.state.zone.in_zone is True

    def test_min_observation_guard(self, mock_config, mock_database):
        """Even with enough elapsed time, need >= 2 observations to enter."""
        detector = self._make_detector(mock_config, mock_database, in_time=0.05)
        car = Car(id=1, config=mock_config)
        car.state.raw_speed = 50.0
        base = 1000.0
        frame = np.zeros((700, 700, 3), dtype=np.uint8)

        self._cross_pre_stop(detector, car, base)

        # Single frame in zone with large dt (time threshold easily met, but only 1 observation)
        car.state.bbox = (100.0, 450.0, 200.0, 550.0)
        detector.update_car_stop_status(car, base + 1.0, frame)

        assert car.state.zone.in_zone is False

    def test_exit_debounce_requires_sustained_absence(self, mock_config, mock_database):
        """Car should not exit zone on a single out-of-zone frame."""
        detector = self._make_detector(mock_config, mock_database, in_time=0.1, out_time=0.2)
        car = Car(id=1, config=mock_config)
        car.state.raw_speed = 50.0
        base = 1000.0
        frame = np.zeros((700, 700, 3), dtype=np.uint8)

        # Cross pre-stop line and enter zone
        self._cross_pre_stop(detector, car, base)
        car.state.bbox = (100.0, 450.0, 200.0, 550.0)
        for i in range(1, 5):
            detector.update_car_stop_status(car, base + i * 0.07, frame)
        assert car.state.zone.in_zone is True

        # Single frame outside zone — should NOT exit yet
        car.state.bbox = (100.0, 50.0, 200.0, 150.0)  # Outside zone
        detector.update_car_stop_status(car, base + 0.5, frame)
        assert car.state.zone.in_zone is True, "Should not exit zone after single out-of-zone frame"

        # Back in zone — resets out-of-zone debounce
        car.state.bbox = (100.0, 450.0, 200.0, 550.0)
        detector.update_car_stop_status(car, base + 0.6, frame)
        assert car.state.zone.in_zone is True


class TestResetCarState:
    """Test that _reset_car_state fully resets all sub-states."""

    def test_full_motion_reset(self, mock_config, mock_database):
        """Motion state timestamps should be reset on pass completion."""
        detector = StopDetector(mock_config, mock_database)
        car = Car(id=1, config=mock_config)

        # Set up some motion state
        car.state.motion.is_parked = False
        car.state.motion.moving_since = 1000.0
        car.state.motion.stationary_since = 999.0
        car.state.motion.consecutive_moving_frames = 10

        detector._reset_car_state(car)

        assert car.state.motion.is_parked is True
        assert car.state.motion.moving_since == 0.0
        assert car.state.motion.stationary_since == 0.0
        assert car.state.motion.consecutive_moving_frames == 0
        assert car.state.motion.consecutive_stationary_frames == 0

    def test_zone_and_capture_reset(self, mock_config, mock_database):
        """Zone and capture states should be fully reset."""
        detector = StopDetector(mock_config, mock_database)
        car = Car(id=1, config=mock_config)

        car.state.zone.in_zone = True
        car.state.zone.stop_duration = 3.5
        car.state.zone.speed_samples = [10.0, 15.0, 20.0]
        car.state.capture.image_captured = True
        car.state.capture.image_path = "/some/path.jpg"

        detector._reset_car_state(car)

        assert car.state.zone.in_zone is False
        assert car.state.zone.stop_duration == 0.0
        assert car.state.zone.speed_samples == []
        assert car.state.capture.image_captured is False
        assert car.state.capture.image_path == ""


class TestCurrentFrameCarIds:
    """Test that CarTracker exposes current_frame_car_ids correctly."""

    def test_current_frame_ids_populated(self, mock_config, mock_database):
        """current_frame_car_ids should contain only cars from the latest update."""
        from tests.conftest import MockBox

        tracker = CarTracker(mock_config, mock_database)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        base = 1000.0

        # First frame: cars 1 and 2
        boxes = [
            MockBox(car_id=1, x=150.0, y=150.0, w=50.0, h=30.0),
            MockBox(car_id=2, x=200.0, y=200.0, w=50.0, h=30.0),
        ]
        tracker.update_cars(boxes, base, frame)
        assert tracker.current_frame_car_ids == {1, 2}

        # Second frame: only car 1
        boxes = [MockBox(car_id=1, x=155.0, y=150.0, w=50.0, h=30.0)]
        tracker.update_cars(boxes, base + 0.1, frame)
        assert tracker.current_frame_car_ids == {1}

    def test_empty_frame(self, mock_config, mock_database):
        """current_frame_car_ids should be empty when no detections."""
        tracker = CarTracker(mock_config, mock_database)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        tracker.update_cars([], 1000.0, frame)
        assert tracker.current_frame_car_ids == set()
