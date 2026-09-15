"""Tests for analyzer overlays, frame geometry, and frame decoding."""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest


class TestVisualizationWithInterpolation:
    """Retained track history must not appear as a vehicle on an empty road."""

    @pytest.fixture
    def analyzer(self, mock_config, mock_database):
        from stopsign.tracking import CarTracker
        from stopsign.tracking import StopDetector
        from stopsign.video_analyzer import VideoAnalyzer

        analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
        analyzer.car_tracker = CarTracker(mock_config, mock_database)
        analyzer.stop_detector = StopDetector(mock_config, mock_database)
        analyzer.stop_detector.stop_zone = np.array([(400, 300), (500, 300), (500, 400), (400, 400)])
        analyzer.frame_count = 1
        analyzer.debug_mode = False
        return analyzer

    def render(self, analyzer, frame, timestamp, cars=None):
        return analyzer.visualize(
            frame.copy(),
            analyzer.car_tracker.cars if cars is None else cars,
            [],
            analyzer.stop_detector,
            timestamp,
        )

    def test_lost_track_clears_overlays_but_preserves_history(self, analyzer, black_frame, mock_yolo_boxes):
        tracker = analyzer.car_tracker
        tracker.update_cars(mock_yolo_boxes, 1000.0, black_frame)
        car = tracker.cars[1]
        car.state.motion.is_parked = False
        car.state.zone.in_zone = True
        car.state.zone.stop_duration = 1.25
        car.state.track = [((100.0, 250.0), 999.9), ((200.0, 250.0), 1000.0)]

        visible = self.render(analyzer, black_frame, 1000.0)
        np.testing.assert_array_equal(visible[135, 150], (0, 255, 0))  # bbox
        np.testing.assert_array_equal(visible[250, 150], (255, 0, 0))  # trail
        np.testing.assert_array_equal(visible[350, 420], (0, 76, 0))  # occupied zone

        # One car disappears while another is still detected.
        tracker.update_cars(mock_yolo_boxes[1:], 1000.1, black_frame)
        lost = self.render(analyzer, black_frame, 1000.1)
        remaining = self.render(analyzer, black_frame, 1000.1, cars={2: tracker.cars[2]})
        np.testing.assert_array_equal(lost, remaining)

        # An entirely empty detection frame must clear even parked-car overlays.
        tracker.update_cars([], 1000.2, black_frame)
        empty = self.render(analyzer, black_frame, 1000.2, cars={})
        np.testing.assert_array_equal(self.render(analyzer, black_frame, 1000.2), empty)

        # Reacquisition keeps accumulated stop evidence rather than creating a new car.
        tracker.update_cars(mock_yolo_boxes[:1], 1000.3, black_frame)
        assert tracker.cars[1] is car
        assert car.state.zone.stop_duration == 1.25
        reacquired = self.render(analyzer, black_frame, 1000.3)
        np.testing.assert_array_equal(reacquired[135, 150], (0, 255, 0))

    def test_skipped_inference_interpolates_briefly_then_clears(self, analyzer, black_frame, single_car_box):
        tracker = analyzer.car_tracker
        tracker.update_cars([single_car_box], 1000.0, black_frame)
        car = tracker.cars[1]
        car.state.motion.is_parked = False
        car.state.zone.in_zone = True
        car.state.velocity = (100.0, 0.0)

        # No tracker update: inference was skipped, not an empty detection result.
        between_detections = self.render(analyzer, black_frame, 1000.25)
        np.testing.assert_array_equal(between_detections[150, 150], (0, 255, 0))
        np.testing.assert_array_equal(between_detections[150, 125], (0, 0, 0))
        at_horizon = self.render(analyzer, black_frame, 1000.5)
        np.testing.assert_array_equal(at_horizon[150, 175], (0, 255, 0))

        # Prolonged inference skipping must not freeze a box at its old location.
        stale = self.render(analyzer, black_frame, 1000.501)
        empty = self.render(analyzer, black_frame, 1000.501, cars={})
        np.testing.assert_array_equal(stale, empty)


class TestRawDimensionSetup:
    """Test raw frame dimension setup without raw-frame overlay copies."""

    def test_ensure_raw_dimensions_initializes_geometry(self):
        from stopsign.video_analyzer import VideoAnalyzer

        analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
        analyzer.raw_width = None
        analyzer.raw_height = None
        analyzer.frame_dimensions = None
        analyzer.stop_detector = MagicMock()
        analyzer._update_coordinate_system = MagicMock()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        analyzer.ensure_raw_dimensions(frame)

        assert analyzer.raw_width == 1280
        assert analyzer.raw_height == 720
        assert analyzer.frame_dimensions == (1280, 720)
        analyzer._update_coordinate_system.assert_called_once()
        analyzer.stop_detector.set_video_analyzer.assert_called_once_with(analyzer)

    def test_ensure_raw_dimensions_is_noop_when_shape_unchanged(self):
        from stopsign.video_analyzer import VideoAnalyzer

        analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
        analyzer.raw_width = 1280
        analyzer.raw_height = 720
        analyzer.frame_dimensions = (1280, 720)
        analyzer.stop_detector = MagicMock()
        analyzer._update_coordinate_system = MagicMock()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        analyzer.ensure_raw_dimensions(frame)

        analyzer._update_coordinate_system.assert_not_called()
        analyzer.stop_detector.set_video_analyzer.assert_not_called()


class TestFrameEnvelopeHandling:
    def test_parse_raw_frame_accepts_legacy_jpeg_envelope(self):
        from stopsign.frame_codec import pack_legacy_jpeg_frame
        from stopsign.video_analyzer import VideoAnalyzer

        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        frame[:, :, 1] = 255
        ok, jpeg = cv2.imencode(".jpg", frame)
        assert ok

        analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
        decoded, capture_ts = analyzer._parse_raw_frame(
            pack_legacy_jpeg_frame(jpeg.tobytes(), capture_ts=123.45, width=30, height=20)
        )

        assert capture_ts == pytest.approx(123.45)
        assert decoded is not None
        assert decoded.shape == frame.shape

    def test_store_frame_data_writes_self_describing_processed_frame(self):
        from stopsign.frame_codec import unpack_frame
        from stopsign.video_analyzer import VideoAnalyzer

        pipeline = MagicMock()
        redis_client = MagicMock()
        redis_client.pipeline.return_value = pipeline
        analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
        analyzer.redis_client = redis_client
        analyzer.frame_buffer_size = 10
        analyzer.frame_count = 5
        analyzer.redis_op_latency = MagicMock()

        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        analyzer.store_frame_data(frame, {"capture_timestamp": 123.45})

        pushed = pipeline.lpush.call_args.args[1]
        decoded = unpack_frame(pushed)
        assert decoded is not None
        assert decoded.metadata["format"] == "bgr24"
        assert decoded.metadata["width"] == 30
        assert decoded.metadata["height"] == 20
        assert decoded.metadata["ts"] == pytest.approx(123.45)
        assert decoded.payload == frame.tobytes()
