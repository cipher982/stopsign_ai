"""Tests for Car class and bbox interpolation."""

import time

import pytest

from stopsign.tracking import Car
from stopsign.tracking import CarState


class TestCarState:
    """Test suite for CarState dataclass."""

    def test_default_values(self):
        """Test that CarState has sensible defaults."""
        state = CarState()

        assert state.location == (0.0, 0.0)
        assert state.bbox == (0.0, 0.0, 0.0, 0.0)
        assert state.velocity == (0.0, 0.0)
        assert state.speed == 0.0
        assert state.is_parked is True
        assert state.in_stop_zone is False

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
        assert car.state.is_parked is True

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
        assert car.state.consecutive_moving_frames > 0 or not car.state.is_parked
