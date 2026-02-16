import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
from minio import Minio

from stopsign.config import Config
from stopsign.database import Database
from stopsign.kalman_filter import KalmanFilterWrapper
from stopsign.settings import BREMEN_MINIO_ACCESS_KEY
from stopsign.settings import BREMEN_MINIO_BUCKET
from stopsign.settings import BREMEN_MINIO_ENDPOINT
from stopsign.settings import BREMEN_MINIO_SECRET_KEY
from stopsign.settings import LOCAL_IMAGE_DIR
from stopsign.settings import LOCAL_IMAGE_MAX_COUNT

Point = Tuple[float, float]
Line = Tuple[Point, Point]

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level upload queue and worker state
_upload_queue: queue.Queue = queue.Queue(maxsize=100)
_worker_started = False
_worker_lock = threading.Lock()


def _start_upload_worker():
    """Start the background upload worker thread if not already running."""
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            thread = threading.Thread(target=_bremen_upload_worker, daemon=True)
            thread.start()
            _worker_started = True
            logger.info("Bremen upload worker thread started")


def _bremen_upload_worker():
    """Background worker that uploads images to Bremen MinIO with retry logic."""
    while True:
        try:
            local_path, object_name, db = _upload_queue.get()

            # Skip if Bremen credentials not configured
            if not BREMEN_MINIO_SECRET_KEY:
                logger.debug(f"Bremen MinIO not configured, skipping archive of {object_name}")
                _upload_queue.task_done()
                continue

            # Retry up to 3 times with exponential backoff
            for attempt in range(3):
                try:
                    client = Minio(
                        BREMEN_MINIO_ENDPOINT,
                        access_key=BREMEN_MINIO_ACCESS_KEY,
                        secret_key=BREMEN_MINIO_SECRET_KEY,
                        secure=False,  # Bremen is on local network
                    )

                    # Upload the file
                    client.fput_object(
                        BREMEN_MINIO_BUCKET,
                        object_name,
                        local_path,
                        content_type="image/jpeg",
                    )
                    logger.debug(f"Archived {object_name} to Bremen MinIO")

                    # Flip DB path from local:// to bremen://
                    if db is not None:
                        try:
                            rows = db.update_image_path(
                                f"local://{object_name}",
                                f"bremen://{object_name}",
                            )
                            if rows:
                                logger.info(f"Updated DB path for {object_name}: local:// -> bremen://")
                        except Exception as db_err:
                            logger.warning(f"Failed to update DB path for {object_name}: {db_err}")

                    break
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Failed to archive {object_name} after 3 attempts: {e}")
                    else:
                        logger.warning(f"Bremen upload attempt {attempt + 1} failed for {object_name}: {e}")
                        time.sleep(2**attempt)  # Exponential backoff: 1s, 2s

            _upload_queue.task_done()
        except Exception as e:
            logger.error(f"Unexpected error in Bremen upload worker: {e}")


def _prune_old_images():
    """Remove oldest images when count exceeds LOCAL_IMAGE_MAX_COUNT."""
    try:
        image_dir = Path(LOCAL_IMAGE_DIR)
        if not image_dir.exists():
            return

        # Get all jpg files with their modification times
        images = list(image_dir.glob("*.jpg"))

        if len(images) <= LOCAL_IMAGE_MAX_COUNT:
            return

        # Sort by modification time (oldest first)
        images.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest images to get back under limit
        to_remove = len(images) - LOCAL_IMAGE_MAX_COUNT
        for img_path in images[:to_remove]:
            try:
                img_path.unlink()
                logger.debug(f"Pruned old image: {img_path.name}")
            except Exception as e:
                logger.warning(f"Failed to prune {img_path.name}: {e}")

        logger.info(f"Pruned {to_remove} old images to maintain {LOCAL_IMAGE_MAX_COUNT} limit")
    except Exception as e:
        logger.error(f"Error pruning old images: {e}")


@dataclass
class CarState:
    location: Point = field(default_factory=lambda: (0.0, 0.0))
    bbox: Tuple[float, float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))
    raw_speed: float = 0.0
    speed: float = 0.0
    prev_speed: float = 0.0
    velocity: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    direction: float = 0.0
    is_parked: bool = True
    consecutive_moving_frames: int = 0
    consecutive_stationary_frames: int = 0
    stationary_since: float = 0.0
    moving_since: float = 0.0
    track: List[Tuple[Point, float]] = field(default_factory=list)
    last_update_time: float = 0.0
    prev_update_time: float = 0.0
    in_stop_zone: bool = False
    passed_pre_stop_zone: bool = False
    entry_time: float = 0.0
    exit_time: float = 0.0
    time_in_zone: float = 0.0
    consecutive_in_zone_frames: int = 0
    consecutive_out_zone_frames: int = 0
    min_speed_in_zone: float = float("inf")
    speed_samples_in_zone: List[float] = field(default_factory=list)
    stop_duration: float = 0.0
    stop_position: Point = field(default_factory=lambda: (0.0, 0.0))
    image_captured: bool = False
    image_path: str = ""


class Car:
    def __init__(self, id: int, config: Config):
        self.id = id
        self.state = CarState()
        self.kalman_filter = KalmanFilterWrapper(process_noise=0.1, measurement_noise=1.0)
        self.config = config

    def update(
        self,
        location: Tuple[float, float],
        timestamp: float,
        bbox: Tuple[float, float, float, float],
    ) -> None:
        """Update the car's state with new location data."""
        self._update_location(location, timestamp)
        self._update_speed(timestamp)
        self._update_direction()
        self._update_movement_status()
        self._update_parked_status()
        self.state.bbox = bbox

    def _update_location(self, location: Tuple[float, float], timestamp: float) -> None:
        """Update the car's location using Kalman filter."""
        # Calculate dt for accurate Kalman prediction
        dt = timestamp - self.state.last_update_time if self.state.last_update_time > 0 else None
        self.kalman_filter.predict(dt)
        smoothed_location = self.kalman_filter.update(np.array(location))
        self.state.location = (float(smoothed_location[0]), float(smoothed_location[1]))
        self.state.track.append((location, timestamp))
        self.state.prev_update_time = self.state.last_update_time
        self.state.last_update_time = timestamp

    def _update_speed(self, current_timestamp: float) -> None:
        """Calculate and update the car's velocity, raw speed, and smoothed speed."""
        history_length = min(len(self.state.track), 10)
        if history_length > 2:
            recent_track = self.state.track[-history_length:]
            time_diffs = [current_timestamp - t for _, t in recent_track]
            positions = np.array([pos for pos, _ in recent_track])

            # Calculate velocity vector
            velocities = np.diff(positions, axis=0) / np.diff(time_diffs)[:, np.newaxis]
            median_velocity = np.median(velocities, axis=0)

            self.state.velocity = (float(median_velocity[0]), float(median_velocity[1]))
            self.state.speed = float(np.linalg.norm(median_velocity))

            # Calculate raw speed using the last 6 frames (median filters bbox jitter)
            if history_length >= 6:
                last_n_positions = positions[-6:]
                last_n_times = time_diffs[-6:]
                raw_velocities = np.diff(last_n_positions, axis=0) / np.diff(last_n_times)[:, np.newaxis]
                raw_speed = float(np.median(np.linalg.norm(raw_velocities, axis=1)))

                # Apply light smoothing to raw speed
                raw_alpha = 0.5
                self.state.raw_speed = raw_alpha * raw_speed + (1 - raw_alpha) * self.state.raw_speed
        else:
            self.state.velocity = (0.0, 0.0)
            self.state.speed = 0.0
            self.state.raw_speed = 0.0

        # Apply low-pass filter to smoothed speed
        alpha = 0.3
        self.state.speed = alpha * self.state.speed + (1 - alpha) * self.state.prev_speed
        self.state.prev_speed = self.state.speed

    def _update_movement_status(self) -> None:
        """Update the car's movement status based on its speed (time-based)."""
        now = self.state.last_update_time
        if abs(self.state.speed) < self.config.max_movement_speed:
            self.state.consecutive_moving_frames = 0
            self.state.consecutive_stationary_frames += 1
            if self.state.stationary_since == 0.0:
                self.state.stationary_since = now
            self.state.moving_since = 0.0
        else:
            self.state.consecutive_moving_frames += 1
            self.state.consecutive_stationary_frames = 0
            if self.state.moving_since == 0.0:
                self.state.moving_since = now
            self.state.stationary_since = 0.0

    def _update_direction(self) -> None:
        """Calculate the direction based on recent trajectory using linear regression."""
        history_length = min(len(self.state.track), 10)
        if history_length > 2:
            recent_track = self.state.track[-history_length:]
            positions = np.array([pos for pos, _ in recent_track])

            # Use linear regression to find the best fit line
            x = positions[:, 0]
            y = positions[:, 1]
            A = np.vstack([x, np.ones(len(x))]).T
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]

            # Determine the direction of movement along X
            dx_total = x[-1] - x[0]
            sign_dx = np.sign(dx_total) if dx_total != 0 else 1.0

            # Calculate unit direction vector from the slope
            dx = sign_dx / np.sqrt(1 + m**2)
            dy = (m * sign_dx) / np.sqrt(1 + m**2)

            # Calculate total movement
            total_movement = abs(dx) + abs(dy)

            if total_movement > 0:
                # Calculate direction value
                self.state.direction = dx / total_movement
            else:
                self.state.direction = 0.0
        else:
            self.state.direction = 0.0

    def _update_parked_status(self) -> None:
        """Update the car's parked status based on its movement and speed (time-based)."""
        now = self.state.last_update_time
        # Time-based thresholds with frame-count fallback for backward compat
        parked_time = getattr(self.config, "parked_time_threshold", None) or 4.0
        unparked_time = getattr(self.config, "unparked_time_threshold", None) or 1.33

        if self.state.is_parked:
            moving_elapsed = (now - self.state.moving_since) if self.state.moving_since > 0 else 0.0
            if moving_elapsed >= unparked_time or self.state.speed > self.config.unparked_speed_threshold:
                self.state.is_parked = False
                self.state.consecutive_stationary_frames = 0
                self.state.stationary_since = 0.0
        else:
            stationary_elapsed = (now - self.state.stationary_since) if self.state.stationary_since > 0 else 0.0
            if stationary_elapsed >= parked_time:
                self.state.is_parked = True
                self.state.consecutive_moving_frames = 0
                self.state.moving_since = 0.0

    def get_interpolated_bbox(self, current_ts: float) -> Tuple[float, float, float, float]:
        """Predict bbox position based on velocity and time since last YOLO update.

        Used for smooth visualization between detection frames. The velocity
        is in pixels/second, so we multiply by dt (seconds) to get pixel offset.
        """
        dt = current_ts - self.state.last_update_time
        # Don't extrapolate if no time has passed, or too far into future
        if dt <= 0 or dt > 0.5:
            return self.state.bbox

        vx, vy = self.state.velocity
        x1, y1, x2, y2 = self.state.bbox
        # Translate bbox by velocity * time
        return (x1 + vx * dt, y1 + vy * dt, x2 + vx * dt, y2 + vy * dt)

    def __repr__(self) -> str:
        return (
            f"Car {self.id} @ ({self.state.location[0]:.2f}, {self.state.location[1]:.2f}) "
            f"(Speed: {self.state.speed:.1f}px/s, Parked: {self.state.is_parked})"
        )


class CarTracker:
    def __init__(self, config: Config, db: Database):
        self.cars: Dict[int, Car] = {}
        self.config = config
        self.db = db
        self.last_seen: Dict[int, float] = {}
        self.persistence_threshold = 10.0  # seconds

        # Initialize tracer once for better performance
        from stopsign.telemetry import get_tracer

        self._tracer = get_tracer("stopsign.vehicle_tracking")

    def update_cars(self, boxes: List, timestamp: float, frame: np.ndarray) -> None:
        current_car_ids = set()

        for box in boxes:
            if box.id is None:
                continue
            try:
                car_id = int(box.id.item())
                current_car_ids.add(car_id)
                x, y, w, h = box.xywh[0]
                location = (float(x), float(y))
                # Convert XYWH (center) to XYXY (corners) for drawing
                x1 = float(x - w / 2)
                y1 = float(y - h / 2)
                x2 = float(x + w / 2)
                y2 = float(y + h / 2)
                bbox = (x1, y1, x2, y2)

                if car_id not in self.cars:
                    self.cars[car_id] = Car(id=car_id, config=self.config)

                    # Emit telemetry for new vehicle tracking started
                    try:
                        with self._tracer.start_as_current_span("vehicle_tracking_started") as span:
                            span.set_attribute("vehicle.id", car_id)
                            span.set_attribute("vehicle.initial_location_x", location[0])
                            span.set_attribute("vehicle.initial_location_y", location[1])
                            span.set_attribute("vehicle.bbox_width", bbox[2])
                            span.set_attribute("vehicle.bbox_height", bbox[3])
                    except Exception as e:
                        logger.warning(f"Failed to emit vehicle_tracking_started telemetry: {e}")

                self.cars[car_id].update(location, timestamp, bbox)
                self.last_seen[car_id] = timestamp
            except Exception as e:
                logger.error(f"Error updating car {box.id}: {str(e)}")

        # Handle cars no longer tracked
        for car_id in list(self.cars.keys()):
            if car_id not in current_car_ids:
                if timestamp - self.last_seen[car_id] > self.persistence_threshold:
                    self.persist_and_remove_car(car_id)

    def persist_and_remove_car(self, car_id: int) -> None:
        car = self.cars[car_id]
        self.db.save_car_state(car)
        del self.cars[car_id]
        del self.last_seen[car_id]

    def get_cars(self) -> Dict[int, Car]:
        return self.cars


class StopDetector:
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.stop_zone: Optional[np.ndarray] = None  # Will be set when dimensions are known
        self.pre_stop_line_proc: Optional[np.ndarray] = None
        self.capture_line_proc: Optional[np.ndarray] = None
        self.in_zone_frame_threshold = config.in_zone_frame_threshold
        self.out_zone_frame_threshold = config.out_zone_frame_threshold
        self.stop_speed_threshold = config.stop_speed_threshold
        self._video_analyzer = None  # Will be set by video analyzer

        # Initialize tracer once for better performance
        from stopsign.telemetry import get_tracer

        self._tracer = get_tracer("stopsign.vehicle_tracking")

    def set_video_analyzer(self, video_analyzer):
        """Set reference to video analyzer for coordinate conversion."""
        self._video_analyzer = video_analyzer
        try:
            self._initialize_geometry()
            print(f"Stop geometry initialized with {len(self.config.stop_zone)} points")
        except ValueError as e:
            # Video dimensions not available yet - will be created when first frame is processed
            print(f"Stop geometry initialization deferred: {e}")
            self._reset_geometry()

    def _create_stop_zone(self) -> np.ndarray:
        # Get processing coordinates from video analyzer
        if self._video_analyzer is None:
            raise ValueError("Video analyzer not set. Call set_video_analyzer first.")

        if not self.config.stop_zone:
            raise ValueError("No stop zone configured in config")

        if len(self.config.stop_zone) != 4:
            raise ValueError(f"Stop zone must have exactly four points, got {len(self.config.stop_zone)}")

        # Check if video analyzer has dimensions set yet
        if self._video_analyzer.raw_width is None or self._video_analyzer.raw_height is None:
            raise ValueError(
                "Video dimensions not yet detected. Stop zone will be created after video dimensions are available."
            )

        stop_zone_polygon = []
        for i, point in enumerate(self.config.stop_zone):
            if point is None:
                raise ValueError(f"Stop zone point {i+1} is None. Stop zone: {self.config.stop_zone}")
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"Stop zone point {i+1} must be a [x, y] pair, got: {point}")
            x, y = point
            proc_x, proc_y = self._video_analyzer.raw_to_processing_coordinates(x, y)
            stop_zone_polygon.append([proc_x, proc_y])

        return np.array(stop_zone_polygon, dtype=np.float32)

    def _initialize_geometry(self) -> None:
        if self._video_analyzer is None:
            raise ValueError("Video analyzer not set. Call set_video_analyzer first.")

        self.stop_zone = self._create_stop_zone()
        self.pre_stop_line_proc = self._convert_line_to_processing(self.config.pre_stop_line)
        self.capture_line_proc = self._convert_line_to_processing(self.config.capture_line)

    def _reset_geometry(self) -> None:
        self.stop_zone = None
        self.pre_stop_line_proc = None
        self.capture_line_proc = None

    def _convert_line_to_processing(self, raw_line: List[Point]) -> np.ndarray:
        if self._video_analyzer is None:
            raise ValueError("Video analyzer not set. Cannot convert line to processing coordinates.")

        if self._video_analyzer.raw_width is None or self._video_analyzer.raw_height is None:
            raise ValueError("Video dimensions not yet detected. Line conversion deferred.")

        processed: list[list[float]] = []
        for raw_x, raw_y in raw_line:
            proc_x, proc_y = self._video_analyzer.raw_to_processing_coordinates(raw_x, raw_y)
            processed.append([proc_x, proc_y])

        if processed[0] == processed[1]:
            raise ValueError("Detection line points collapse to zero length after coordinate conversion")

        return np.array(processed, dtype=np.float32)

    def _get_car_polygon(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        # bbox is XYXY format (x1, y1, x2, y2) - corner coordinates
        x1, y1, x2, y2 = bbox
        bbox_polygon = np.array(
            [
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2],  # Bottom-left
            ],
            dtype=np.float32,
        )
        return bbox_polygon

    def _check_polygon_intersection(self, poly1: np.ndarray, poly2: np.ndarray) -> bool:
        # Ensure polygons are convex and in the correct format
        poly1 = poly1.astype(np.float32).reshape(-1, 2)
        poly2 = poly2.astype(np.float32).reshape(-1, 2)

        # Check if polygons have at least 3 points
        if poly1.shape[0] < 3 or poly2.shape[0] < 3:
            logger.warning(f"Invalid polygon shape: poly1 {poly1.shape}, poly2 {poly2.shape}")
            return False

        try:
            retval, _ = cv2.intersectConvexConvex(poly1, poly2)
            return retval > 0
        except cv2.error as e:
            logger.error(f"OpenCV error in intersectConvexConvex: {str(e)}")
            logger.debug(f"poly1: {poly1}")
            logger.debug(f"poly2: {poly2}")
            return False

    @staticmethod
    def _polygon_crosses_line(polygon: np.ndarray, line: Optional[np.ndarray]) -> bool:
        if line is None:
            return False

        reshaped = polygon.reshape(-1, 2)
        if reshaped.size == 0:
            return False

        (x1, y1), (x2, y2) = line
        dx = x2 - x1
        dy = y2 - y1

        sides: list[float] = []
        for vx, vy in reshaped:
            value = dx * (vy - y1) - dy * (vx - x1)
            if abs(value) < 1e-6:
                # Treat touching the line as a crossing
                return True
            sides.append(value)

        has_positive = any(value > 0 for value in sides)
        has_negative = any(value < 0 for value in sides)

        return has_positive and has_negative

    def _update_stop_duration(self, car: Car, timestamp: float) -> None:
        if car.state.raw_speed <= self.stop_speed_threshold:
            if car.state.stop_position == (0.0, 0.0):
                car.state.stop_position = car.state.location
            # Use prev_update_time since last_update_time is already set to
            # current timestamp by car.update() before stop detection runs.
            dt = timestamp - car.state.prev_update_time if car.state.prev_update_time > 0 else 0.0
            car.state.stop_duration += dt
        else:
            car.state.stop_position = (0.0, 0.0)

    def update_car_stop_status(self, car: Car, timestamp: float, frame: np.ndarray) -> None:
        # Lazy initialization of geometry once video dimensions are available
        if self._video_analyzer is not None and (
            self.stop_zone is None or self.pre_stop_line_proc is None or self.capture_line_proc is None
        ):
            if self._video_analyzer.raw_width is not None and self._video_analyzer.raw_height is not None:
                try:
                    self._initialize_geometry()
                    print(f"Stop geometry initialized with {len(self.config.stop_zone)} points and detection lines")
                except ValueError as e:
                    print(f"Failed to initialize stop geometry: {e}")
                    self._reset_geometry()
                    return
            else:
                return  # Still waiting for video dimensions

        # Skip if geometry still not available
        if self.stop_zone is None or self.pre_stop_line_proc is None or self.capture_line_proc is None:
            return

        car_polygon = self._get_car_polygon(car.state.bbox)

        crossed_pre_stop_line = self._polygon_crosses_line(car_polygon, self.pre_stop_line_proc)

        # Check if car is in stop zone
        in_stop_zone = self._check_polygon_intersection(car_polygon, self.stop_zone)

        # Update pre-stop zone flag only if not already in stop zone
        if crossed_pre_stop_line and not in_stop_zone and not car.state.passed_pre_stop_zone:
            car.state.passed_pre_stop_zone = True

        # Check if car crosses the capture line (only capture if pre-stop line was crossed first)
        # This ensures we only capture cars going the correct direction (right-to-left)
        if (
            car.state.passed_pre_stop_zone
            and not car.state.image_captured
            and self._polygon_crosses_line(car_polygon, self.capture_line_proc)
        ):
            self.capture_car_image(car, timestamp, frame)

        # Only proceed with stop zone logic if pre-stop zone was passed
        if not car.state.passed_pre_stop_zone:
            return

        # Update consecutive frame counters
        if in_stop_zone:
            car.state.consecutive_in_zone_frames += 1
            car.state.consecutive_out_zone_frames = 0
        else:
            car.state.consecutive_out_zone_frames += 1
            car.state.consecutive_in_zone_frames = 0

        # State transitions based on debounce counters
        if car.state.consecutive_in_zone_frames >= self.in_zone_frame_threshold:
            if not car.state.in_stop_zone:
                car.state.in_stop_zone = True
                car.state.entry_time = timestamp
                logger.debug(f"Car {car.id} has entered the stop zone at {timestamp}")

                # Emit telemetry for vehicle entering stop zone
                try:
                    with self._tracer.start_as_current_span("vehicle_zone_entered") as span:
                        span.set_attribute("vehicle.id", car.id)
                        span.set_attribute("vehicle.entry_time", timestamp)
                        span.set_attribute("vehicle.speed_at_entry", car.state.raw_speed)
                        span.set_attribute("vehicle.location_x", car.state.location[0])
                        span.set_attribute("vehicle.location_y", car.state.location[1])
                except Exception as e:
                    logger.warning(f"Failed to emit vehicle_zone_entered telemetry: {e}")
        elif car.state.consecutive_out_zone_frames >= self.out_zone_frame_threshold:
            if car.state.in_stop_zone:
                car.state.in_stop_zone = False
                car.state.exit_time = timestamp
                car.state.time_in_zone = car.state.exit_time - car.state.entry_time
                logger.debug(f"Car {car.id} has exited the stop zone at {timestamp}")

                # Emit business telemetry for completed vehicle pass
                try:
                    with self._tracer.start_as_current_span("vehicle_pass_completed") as span:
                        span.set_attribute("vehicle.id", car.id)
                        span.set_attribute("vehicle.time_in_zone", car.state.time_in_zone)
                        span.set_attribute("vehicle.stop_duration", car.state.stop_duration)
                        span.set_attribute("vehicle.min_speed", car.state.min_speed_in_zone)
                        span.set_attribute("vehicle.has_image", bool(car.state.image_path))
                        span.set_attribute("vehicle.entry_time", car.state.entry_time)
                        span.set_attribute("vehicle.exit_time", car.state.exit_time)
                except Exception as e:
                    logger.warning(f"Failed to emit vehicle_pass_completed telemetry: {e}")

                # Compute robust min speed from collected samples (5th percentile)
                if len(car.state.speed_samples_in_zone) >= 3:
                    car.state.min_speed_in_zone = float(np.percentile(car.state.speed_samples_in_zone, 5))
                elif car.state.speed_samples_in_zone:
                    car.state.min_speed_in_zone = min(car.state.speed_samples_in_zone)

                # Save data to the database
                self.db.add_vehicle_pass(
                    vehicle_id=car.id,
                    time_in_zone=car.state.time_in_zone,
                    stop_duration=car.state.stop_duration,
                    min_speed=car.state.min_speed_in_zone,
                    image_path=car.state.image_path,
                    entry_time=car.state.entry_time,
                    exit_time=car.state.exit_time,
                )
                logger.info(
                    f"Vehicle pass recorded: ID={car.id}, "
                    f"Time in zone={car.state.time_in_zone:.2f}s, "
                    f"Stop duration={car.state.stop_duration:.2f}s, "
                    f"Min speed={car.state.min_speed_in_zone:.2f}px/s "
                    f"Timestamp={datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}"
                )
                # Reset state
                self._reset_car_state(car)

        if car.state.in_stop_zone:
            self._update_stop_duration(car, timestamp)
            car.state.speed_samples_in_zone.append(car.state.raw_speed)

    def _reset_car_state(self, car: Car) -> None:
        # Reset stop zone related attributes
        car.state.in_stop_zone = False
        car.state.passed_pre_stop_zone = False
        car.state.entry_time = 0.0
        car.state.exit_time = 0.0
        car.state.time_in_zone = 0.0
        car.state.consecutive_in_zone_frames = 0
        car.state.consecutive_out_zone_frames = 0
        car.state.min_speed_in_zone = float("inf")
        car.state.speed_samples_in_zone = []
        car.state.stop_duration = 0.0
        car.state.stop_position = (0.0, 0.0)

        # Reset image capture related attributes
        car.state.image_captured = False
        car.state.image_path = ""

        # Optionally, you might want to reset some movement-related attributes
        car.state.consecutive_moving_frames = 0
        car.state.consecutive_stationary_frames = 0

        logger.debug(f"Reset state for Car {car.id}")

    def is_in_capture_zone(self, bbox: Tuple[float, float, float, float]) -> bool:
        if self.capture_line_proc is None:
            return False

        car_polygon = self._get_car_polygon(bbox)
        return self._polygon_crosses_line(car_polygon, self.capture_line_proc)

    def capture_car_image(self, car: Car, timestamp: float, frame: np.ndarray) -> None:
        image_path = save_vehicle_image(
            frame=frame,
            timestamp=timestamp,
            bbox=car.state.bbox,
            db=self.db,
        )
        car.state.image_captured = True
        car.state.image_path = image_path


def save_vehicle_image(
    frame: np.ndarray,
    timestamp: float,
    bbox: Tuple[float, float, float, float],
    db: Optional[Database] = None,
) -> str:
    """Save vehicle image locally and queue upload to Bremen MinIO for archival.

    Returns local:// path for fast serving from cube.
    """
    # Generate a random UUID for the filename
    file_id = uuid.uuid4().hex
    filename = f"vehicle_{file_id}_{int(timestamp)}.jpg"

    # Crop the image - bbox is XYXY format (x1, y1, x2, y2)
    bx1, by1, bx2, by2 = bbox
    w = bx2 - bx1
    h = by2 - by1
    padding_factor = 0.1
    padding_x = int(w * padding_factor)
    padding_y = int(h * padding_factor)
    x1, y1 = int(bx1 - padding_x), int(by1 - padding_y)
    x2, y2 = int(bx2 + padding_x), int(by2 + padding_y)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    cropped_image = frame[y1:y2, x1:x2]

    # Ensure local image directory exists
    image_dir = Path(LOCAL_IMAGE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)

    local_path = image_dir / filename

    # Save to local filesystem (synchronous, ensure file is fully written)
    try:
        _, img_encoded = cv2.imencode(".jpg", cropped_image)
        with open(local_path, "wb") as f:
            f.write(img_encoded.tobytes())
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        logger.debug(f"Saved vehicle image locally: {filename}")

        # Start upload worker if not already running
        _start_upload_worker()

        # Queue upload to Bremen MinIO (non-blocking)
        try:
            _upload_queue.put_nowait((str(local_path), filename, db))
        except queue.Full:
            logger.warning(f"Upload queue full, skipping archive of {filename}")

        # Prune old images to maintain disk space
        _prune_old_images()

        return f"local://{filename}"

    except Exception as e:
        logger.error(f"Failed to save vehicle image locally: {str(e)}", exc_info=True)
        return ""
