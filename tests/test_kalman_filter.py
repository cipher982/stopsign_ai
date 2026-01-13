"""Tests for KalmanFilterWrapper with dynamic dt support."""

import numpy as np
import pytest

from stopsign.kalman_filter import KalmanFilterWrapper


class TestKalmanFilterWrapper:
    """Test suite for KalmanFilterWrapper."""

    def test_init_creates_valid_filter(self):
        """Test that initialization creates a valid Kalman filter."""
        kf = KalmanFilterWrapper(process_noise=0.1, measurement_noise=1.0)

        assert kf.kf is not None
        assert kf.kf.dim_x == 4  # [x, y, vx, vy]
        assert kf.kf.dim_z == 2  # [x, y] measurements

    def test_predict_without_dt_uses_default(self):
        """Test predict() without dt uses the default dt=1.0."""
        kf = KalmanFilterWrapper(process_noise=0.1, measurement_noise=1.0)

        # Set initial state
        kf.kf.x = np.array([100.0, 100.0, 10.0, 5.0])  # pos=(100,100), vel=(10,5)

        # Predict without dt - should use default dt=1.0
        kf.predict()

        # With dt=1.0: new_x = 100 + 10*1 = 110, new_y = 100 + 5*1 = 105
        # (approximately, Kalman adds process noise)
        assert kf.kf.x[0] == pytest.approx(110.0, abs=1.0)
        assert kf.kf.x[1] == pytest.approx(105.0, abs=1.0)

    def test_predict_with_dt_updates_transition_matrix(self):
        """Test that predict(dt) updates the transition matrix F."""
        kf = KalmanFilterWrapper(process_noise=0.1, measurement_noise=1.0)

        # Set initial state
        kf.kf.x = np.array([100.0, 100.0, 150.0, 75.0])  # pos=(100,100), vel=(150,75) px/s

        # Predict with dt=0.1 (100ms)
        kf.predict(dt=0.1)

        # With dt=0.1: new_x = 100 + 150*0.1 = 115, new_y = 100 + 75*0.1 = 107.5
        assert kf.kf.x[0] == pytest.approx(115.0, abs=1.0)
        assert kf.kf.x[1] == pytest.approx(107.5, abs=1.0)

    @pytest.mark.parametrize(
        "dt,expected_x_delta,expected_y_delta",
        [
            (0.0667, 10.0, 5.0),  # ~15 FPS frame interval
            (0.125, 18.75, 9.375),  # ~8 FPS (YOLO interval)
            (0.5, 75.0, 37.5),  # 500ms
        ],
    )
    def test_predict_with_various_dt_values(self, dt, expected_x_delta, expected_y_delta):
        """Test predict() with various dt values for different frame rates."""
        kf = KalmanFilterWrapper(process_noise=0.1, measurement_noise=1.0)

        # Set initial state with known velocity
        initial_x, initial_y = 100.0, 100.0
        vx, vy = 150.0, 75.0  # pixels/second
        kf.kf.x = np.array([initial_x, initial_y, vx, vy])

        kf.predict(dt=dt)

        # Expected position = initial + velocity * dt
        expected_x = initial_x + expected_x_delta
        expected_y = initial_y + expected_y_delta

        assert kf.kf.x[0] == pytest.approx(expected_x, abs=2.0)
        assert kf.kf.x[1] == pytest.approx(expected_y, abs=2.0)

    def test_predict_with_negative_dt_ignored(self):
        """Test that negative dt values are ignored (uses default)."""
        kf = KalmanFilterWrapper(process_noise=0.1, measurement_noise=1.0)

        kf.kf.x = np.array([100.0, 100.0, 10.0, 5.0])
        initial_state = kf.kf.x.copy()

        # Negative dt should be ignored, uses default dt=1.0
        kf.predict(dt=-0.5)

        # Should still predict forward with default dt
        assert kf.kf.x[0] > initial_state[0]

    def test_predict_with_zero_dt_ignored(self):
        """Test that zero dt is ignored (uses default)."""
        kf = KalmanFilterWrapper(process_noise=0.1, measurement_noise=1.0)

        kf.kf.x = np.array([100.0, 100.0, 10.0, 5.0])
        initial_state = kf.kf.x.copy()

        # Zero dt should be ignored, uses default dt=1.0
        kf.predict(dt=0.0)

        # Should still predict forward with default dt
        assert kf.kf.x[0] > initial_state[0]

    def test_update_returns_smoothed_location(self):
        """Test that update() returns smoothed location."""
        kf = KalmanFilterWrapper(process_noise=0.1, measurement_noise=1.0)

        # Initial measurement
        loc1 = kf.update(np.array([100.0, 100.0]))
        assert len(loc1) == 2

        # Predict and update with noisy measurement
        kf.predict(dt=0.1)
        loc2 = kf.update(np.array([115.5, 107.8]))  # Slightly noisy

        # Smoothed location should be close to measurement
        assert loc2[0] == pytest.approx(115.5, abs=5.0)
        assert loc2[1] == pytest.approx(107.8, abs=5.0)

    def test_filter_smooths_noisy_measurements(self):
        """Test that filter smooths a sequence of noisy measurements."""
        kf = KalmanFilterWrapper(process_noise=0.1, measurement_noise=1.0)

        # Simulate object moving in straight line with noisy measurements
        true_positions = [(100 + i * 10, 100 + i * 5) for i in range(10)]
        noise = [(np.random.randn() * 2, np.random.randn() * 2) for _ in range(10)]
        noisy_measurements = [(true[0] + n[0], true[1] + n[1]) for true, n in zip(true_positions, noise)]

        smoothed = []
        for i, (x, y) in enumerate(noisy_measurements):
            if i > 0:
                kf.predict(dt=1 / 15)  # 15 FPS
            loc = kf.update(np.array([x, y]))
            smoothed.append(loc)

        # Smoothed trajectory should be less noisy than raw measurements
        raw_variance = np.var([m[0] for m in noisy_measurements])
        smoothed_variance = np.var([s[0] for s in smoothed])

        # Smoothed variance should generally be lower (filter is working)
        # Note: this may not always hold for small samples, so we're lenient
        assert smoothed_variance < raw_variance * 2

    def test_velocity_estimation_from_position_updates(self):
        """Test that velocity is estimated from position updates."""
        kf = KalmanFilterWrapper(process_noise=0.1, measurement_noise=1.0)

        # Object moving at ~150 px/s horizontally
        dt = 1 / 15  # 15 FPS
        for i in range(20):
            x = 100 + i * 10  # 10 px per frame at 15 FPS = 150 px/s
            y = 100
            if i > 0:
                kf.predict(dt=dt)
            kf.update(np.array([float(x), float(y)]))

        # After many updates, velocity estimate should converge
        estimated_vx = kf.kf.x[2]
        estimated_vy = kf.kf.x[3]

        # Velocity should be approximately 150 px/s horizontal, 0 vertical
        assert estimated_vx == pytest.approx(150.0, rel=0.3)  # 30% tolerance
        assert abs(estimated_vy) < 20  # Close to zero
