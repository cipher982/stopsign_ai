"""Shared test fixtures for stopsign_ai tests."""

import os
import time
from typing import Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Set required environment variables BEFORE importing stopsign modules
# ---------------------------------------------------------------------------
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("RAW_FRAME_KEY", "raw_frames")
os.environ.setdefault("PROCESSED_FRAME_KEY", "processed_frames")
os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "test_access_key")
os.environ.setdefault("MINIO_SECRET_KEY", "test_secret_key")
os.environ.setdefault("MINIO_BUCKET", "test-bucket")
os.environ.setdefault("DB_URL", "postgresql://test:test@localhost:5432/test")


# ---------------------------------------------------------------------------
# Mock Config Fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_config():
    """Create a mock Config object with typical values."""
    config = MagicMock()
    config.version = "1.0.0"

    # Vehicle tracking settings
    config.max_movement_speed = 5.0
    config.unparked_frame_threshold = 3
    config.unparked_speed_threshold = 10.0
    config.parked_frame_threshold = 30
    config.parked_time_threshold = 4.0
    config.unparked_time_threshold = 1.33
    config.frame_buffer_size = 500
    config.grid_size = 50
    config.draw_grid = False

    # Stop zone (quadrilateral)
    config.stop_zone = [(100, 100), (200, 100), (200, 200), (100, 200)]
    config.pre_stop_line = [(50, 150), (250, 150)]
    config.capture_line = [(50, 180), (250, 180)]

    # Stream settings
    config.stream_settings = {"width": 640, "height": 480}

    return config


# ---------------------------------------------------------------------------
# Mock Database Fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_database():
    """Create a mock Database object."""
    db = MagicMock()
    db.save_vehicle_pass = MagicMock(return_value=True)
    db.save_car_state_history = MagicMock(return_value=True)
    return db


# ---------------------------------------------------------------------------
# Frame Generation Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_frame():
    """Generate a sample BGR frame (640x480)."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def black_frame():
    """Generate a black BGR frame (640x480)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Car State Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def car_track_history():
    """Generate a sample car track history with timestamps."""
    base_time = time.time()
    # Car moving from (100, 100) to (200, 200) over 10 frames at 15 FPS
    track = []
    for i in range(10):
        x = 100 + i * 10
        y = 100 + i * 10
        ts = base_time + i * (1 / 15)  # 15 FPS
        track.append(((float(x), float(y)), ts))
    return track


@pytest.fixture
def stationary_car_track():
    """Generate a stationary car track history."""
    base_time = time.time()
    track = []
    for i in range(10):
        ts = base_time + i * (1 / 15)
        track.append(((100.0, 100.0), ts))
    return track


# ---------------------------------------------------------------------------
# YOLO Detection Mock Fixtures
# ---------------------------------------------------------------------------
class MockTensor:
    """Mock for PyTorch tensor with .item() method."""

    def __init__(self, value):
        self.value = value

    def item(self):
        return int(self.value)


class MockBox:
    """Mock for YOLO detection box."""

    def __init__(
        self,
        car_id: int,
        x: float,
        y: float,
        w: float,
        h: float,
        conf: float = 0.9,
    ):
        self.id = MockTensor(car_id)
        self.xywh = [np.array([x, y, w, h])]
        self.xyxy = [np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])]
        self.conf = MockTensor(conf)


@pytest.fixture
def mock_yolo_boxes():
    """Create a list of mock YOLO detection boxes."""
    return [
        MockBox(car_id=1, x=150.0, y=150.0, w=50.0, h=30.0),
        MockBox(car_id=2, x=300.0, y=200.0, w=60.0, h=35.0),
    ]


@pytest.fixture
def single_car_box():
    """Create a single mock YOLO detection box."""
    return MockBox(car_id=1, x=150.0, y=150.0, w=50.0, h=30.0)


# ---------------------------------------------------------------------------
# Bbox Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_bbox() -> Tuple[float, float, float, float]:
    """Sample bounding box (x1, y1, x2, y2)."""
    return (100.0, 100.0, 150.0, 130.0)


@pytest.fixture
def sample_velocity() -> Tuple[float, float]:
    """Sample velocity in pixels/second."""
    return (50.0, 25.0)  # Moving right and down
