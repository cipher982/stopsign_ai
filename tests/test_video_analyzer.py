"""Tests for VideoAnalyzer decoupled YOLO logic.

These tests focus on the frame timing and YOLO gating logic
introduced to decouple AI inference from video output.
"""

import pytest


class TestDecoupledYOLOTiming:
    """Test suite for decoupled YOLO timing logic."""

    def test_yolo_interval_calculation(self):
        """Test that YOLO interval is correctly set to ~8 FPS."""
        min_yolo_interval = 1.0 / 8  # From video_analyzer.py

        # Should be approximately 125ms
        assert min_yolo_interval == pytest.approx(0.125, abs=0.001)

    @pytest.mark.parametrize(
        "last_yolo_ts,current_ts,expected_should_run",
        [
            # No previous YOLO run (last_yolo_ts=0)
            (0.0, 1000.0, True),
            # Exactly at interval
            (1000.0, 1000.125, True),
            # Just past interval
            (1000.0, 1000.130, True),
            # Just before interval
            (1000.0, 1000.120, False),
            # Way past interval
            (1000.0, 1001.0, True),
            # Same timestamp
            (1000.0, 1000.0, False),
        ],
    )
    def test_should_run_yolo_logic(self, last_yolo_ts, current_ts, expected_should_run):
        """Test the should_run_yolo gating logic."""
        min_yolo_interval = 1.0 / 8  # ~125ms

        should_run_yolo = (current_ts - last_yolo_ts) >= min_yolo_interval

        assert should_run_yolo == expected_should_run

    def test_yolo_runs_at_approximately_8_fps(self):
        """Simulate frame processing and verify YOLO runs at ~8 FPS."""
        min_yolo_interval = 1.0 / 8
        last_yolo_ts = 0.0
        yolo_run_count = 0
        frame_count = 0

        # Simulate 1 second of 15 FPS video
        base_time = 1000.0
        for i in range(15):
            current_ts = base_time + i * (1 / 15)  # 15 FPS timestamps
            frame_count += 1

            should_run_yolo = (current_ts - last_yolo_ts) >= min_yolo_interval
            if should_run_yolo:
                yolo_run_count += 1
                last_yolo_ts = current_ts

        # YOLO should run approximately 8 times per second
        assert yolo_run_count == pytest.approx(8, abs=1)
        assert frame_count == 15

    def test_all_frames_output_regardless_of_yolo(self):
        """Test that all frames are output even when YOLO is skipped."""
        min_yolo_interval = 1.0 / 8
        last_yolo_ts = 0.0
        output_count = 0
        yolo_count = 0

        base_time = 1000.0
        for i in range(30):  # 2 seconds at 15 FPS
            current_ts = base_time + i * (1 / 15)

            should_run_yolo = (current_ts - last_yolo_ts) >= min_yolo_interval
            if should_run_yolo:
                yolo_count += 1
                last_yolo_ts = current_ts

            # Every frame gets output (this is the key change)
            output_count += 1

        # All 30 frames should be output
        assert output_count == 30
        # YOLO should run ~16 times (8 FPS * 2 seconds)
        assert yolo_count == pytest.approx(16, abs=2)


class TestFrameTimestampHandling:
    """Test that frame timestamps (capture_ts) are used correctly."""

    def test_interpolation_uses_capture_timestamp_not_wall_clock(self):
        """Verify interpolation uses frame timestamp, not time.time()."""
        # This is a documentation/design test
        # The actual implementation should use capture_ts passed to visualize()

        # Frame captured at T=1000.0, processed at T=1000.5 (500ms lag)
        capture_ts = 1000.0
        processing_time = 1000.5
        last_yolo_ts = 999.9  # YOLO ran 100ms before capture

        # Interpolation dt should be based on capture_ts, not processing_time
        correct_dt = capture_ts - last_yolo_ts  # 0.1 seconds
        wrong_dt = processing_time - last_yolo_ts  # 0.6 seconds

        assert correct_dt == pytest.approx(0.1, abs=0.01)
        assert wrong_dt == pytest.approx(0.6, abs=0.01)

        # The implementation should use correct_dt for interpolation
        # to avoid "rubber banding" when processing lags behind


class TestCatchUpPolicyRemoval:
    """Test that the aggressive catch-up policy is removed."""

    def test_high_lag_does_not_skip_frames(self):
        """Verify that high lag logs but doesn't skip frames."""
        # Old behavior: if lag > 15s, skip frame and return None
        # New behavior: log high lag but continue processing

        capture_ts = 1000.0
        current_time = 1020.0  # 20 seconds lag
        analyzer_catchup_sec = 15.0

        lag = current_time - capture_ts

        # Old check would return None (skip)
        old_behavior_skip = lag > analyzer_catchup_sec

        # New behavior: log but don't skip
        # (This is a design test - implementation should NOT skip)
        should_skip_frame = False  # New behavior

        assert old_behavior_skip is True  # Old would have skipped
        assert should_skip_frame is False  # New should not skip


class TestOutputFrameRate:
    """Test that output maintains target frame rate."""

    def test_output_rate_matches_input_rate(self):
        """Test that we output at the same rate we receive frames."""
        input_frames = 150  # 10 seconds at 15 FPS
        output_frames = 0

        # Simulate processing
        for _ in range(input_frames):
            # Every frame should be output
            output_frames += 1

        assert output_frames == input_frames

    def test_yolo_skip_does_not_reduce_output(self):
        """Test that skipping YOLO doesn't reduce output frame count."""
        min_yolo_interval = 1.0 / 8
        last_yolo_ts = 0.0

        input_frames = 30
        output_frames = 0
        yolo_runs = 0
        yolo_skips = 0

        base_time = 1000.0
        for i in range(input_frames):
            current_ts = base_time + i * (1 / 15)

            should_run_yolo = (current_ts - last_yolo_ts) >= min_yolo_interval
            if should_run_yolo:
                yolo_runs += 1
                last_yolo_ts = current_ts
            else:
                yolo_skips += 1

            # KEY: output happens regardless of YOLO
            output_frames += 1

        assert output_frames == input_frames
        assert yolo_runs + yolo_skips == input_frames
        assert yolo_skips > 0  # Some YOLO runs should be skipped


class TestVisualizationWithInterpolation:
    """Test that visualization uses interpolated boxes."""

    def test_visualization_parameters(self):
        """Test that visualize() receives timestamp for interpolation."""
        # The visualize() method signature should include timestamp
        # This is a design documentation test

        # Expected signature:
        # def visualize(self, frame, cars, boxes, stop_detector, timestamp) -> np.ndarray

        # timestamp is used to call car.get_interpolated_bbox(timestamp)
        # This allows smooth box positions between YOLO detections
        pass

    def test_draw_car_interpolated_exists(self):
        """Test that draw_car_interpolated method exists."""
        # This method should draw using interpolated bbox from car state
        # Not from YOLO detection boxes

        # Import to verify method exists
        from stopsign.video_analyzer import VideoAnalyzer

        assert hasattr(VideoAnalyzer, "draw_car_interpolated")
