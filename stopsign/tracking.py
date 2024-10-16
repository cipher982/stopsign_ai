import logging
import os
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Dict
from typing import List
from typing import Tuple

import cv2
import numpy as np

from stopsign.config import Config
from stopsign.database import Database
from stopsign.kalman_filter import KalmanFilterWrapper

Point = Tuple[float, float]
Line = Tuple[Point, Point]

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_env(key: str) -> str:
    value = os.getenv(key)
    assert value is not None, f"{key} is not set"
    return value


VEHICLE_IMAGE_DIR = "/app/data/vehicle_images"


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
    track: List[Tuple[Point, float]] = field(default_factory=list)
    last_update_time: float = 0.0
    in_stop_zone: bool = False
    passed_pre_stop_zone: bool = False
    entry_time: float = 0.0
    exit_time: float = 0.0
    time_in_zone: float = 0.0
    consecutive_in_zone_frames: int = 0
    consecutive_out_zone_frames: int = 0
    min_speed_in_zone: float = float("inf")
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
        self.kalman_filter.predict()
        smoothed_location = self.kalman_filter.update(np.array(location))
        self.state.location = (float(smoothed_location[0]), float(smoothed_location[1]))
        self.state.track.append((location, timestamp))
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

            # Calculate raw speed using the last 3 frames
            last_3_positions = positions[-3:]
            last_3_times = time_diffs[-3:]
            raw_velocities = np.diff(last_3_positions, axis=0) / np.diff(last_3_times)[:, np.newaxis]
            raw_speed = float(np.mean(np.linalg.norm(raw_velocities, axis=1)))

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
        """Update the car's movement status based on its speed."""
        if abs(self.state.speed) < self.config.max_movement_speed:
            self.state.consecutive_moving_frames = 0
            self.state.consecutive_stationary_frames += 1
        else:
            self.state.consecutive_moving_frames += 1
            self.state.consecutive_stationary_frames = 0

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
        """Update the car's parked status based on its movement and speed."""
        if self.state.is_parked:
            if (
                self.state.consecutive_moving_frames >= self.config.unparked_frame_threshold
                or self.state.speed > self.config.unparked_speed_threshold
            ):
                self.state.is_parked = False
                self.state.consecutive_stationary_frames = 0
        else:
            if self.state.consecutive_stationary_frames >= self.config.parked_frame_threshold:
                self.state.is_parked = True
                self.state.consecutive_moving_frames = 0

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
                bbox = (float(x), float(y), float(w), float(h))

                if car_id not in self.cars:
                    self.cars[car_id] = Car(id=car_id, config=self.config)

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
        self.stop_zone = self._create_stop_zone()
        self.pre_stop_zone = self.config.pre_stop_zone
        self.image_capture_zone = self.config.image_capture_zone
        self.in_zone_frame_threshold = config.in_zone_frame_threshold
        self.out_zone_frame_threshold = config.out_zone_frame_threshold
        self.stop_speed_threshold = config.stop_speed_threshold

    def _create_stop_zone(self) -> np.ndarray:
        (x1, y1), (x2, y2) = self.config.stop_line
        stop_box_tolerance = self.config.stop_box_tolerance

        # Calculate tolerances
        left_tolerance = stop_box_tolerance[0]
        right_tolerance = stop_box_tolerance[1]

        # Extend the stop line on the x-axis
        left_x1 = x1 - left_tolerance
        left_x2 = x2 - left_tolerance
        right_x1 = x1 + right_tolerance
        right_x2 = x2 + right_tolerance

        # Create the polygon corners
        stop_zone_polygon = np.array(
            [
                [left_x1, y1],  # Top-left
                [right_x1, y1],  # Top-right
                [right_x2, y2],  # Bottom-right
                [left_x2, y2],  # Bottom-left
            ],
            dtype=np.float32,
        )

        return stop_zone_polygon

    def _get_car_polygon(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        x, y, w, h = bbox
        x1, y1 = x - w / 2, y - h / 2  # Top-left corner
        x2, y2 = x + w / 2, y + h / 2  # Bottom-right corner
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

    def _update_stop_duration(self, car: Car, timestamp: float) -> None:
        if car.state.raw_speed <= self.stop_speed_threshold:
            if car.state.stop_position == (0.0, 0.0):
                car.state.stop_position = car.state.location
            car.state.stop_duration += timestamp - car.state.last_update_time
        else:
            car.state.stop_position = (0.0, 0.0)

    def update_car_stop_status(self, car: Car, timestamp: float, frame: np.ndarray) -> None:
        car_polygon = self._get_car_polygon(car.state.bbox)
        car_x = car.state.location[0]

        # Check if car is in pre-stop zone
        in_pre_stop_zone = self.pre_stop_zone[0] <= car_x <= self.pre_stop_zone[1]

        # Check if car is in stop zone
        in_stop_zone = self._check_polygon_intersection(car_polygon, self.stop_zone)

        # Update pre-stop zone flag only if not already in stop zone
        if in_pre_stop_zone and not in_stop_zone and not car.state.passed_pre_stop_zone:
            car.state.passed_pre_stop_zone = True

        # Check if car is in image capture zone
        in_image_capture_zone = self.image_capture_zone[0] <= car_x <= self.image_capture_zone[1]
        if not car.state.image_captured and in_image_capture_zone:
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
        elif car.state.consecutive_out_zone_frames >= self.out_zone_frame_threshold:
            if car.state.in_stop_zone:
                car.state.in_stop_zone = False
                car.state.exit_time = timestamp
                car.state.time_in_zone = car.state.exit_time - car.state.entry_time
                logger.debug(f"Car {car.id} has exited the stop zone at {timestamp}")
                # Save data to the database
                self.db.add_vehicle_pass(
                    vehicle_id=car.id,
                    time_in_zone=car.state.time_in_zone,
                    stop_duration=car.state.stop_duration,
                    min_speed=car.state.min_speed_in_zone,
                    image_path=car.state.image_path,
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
            car.state.min_speed_in_zone = min(car.state.min_speed_in_zone, car.state.raw_speed)

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
        car.state.stop_duration = 0.0
        car.state.stop_position = (0.0, 0.0)

        # Reset image capture related attributes
        car.state.image_captured = False
        car.state.image_path = ""

        # Optionally, you might want to reset some movement-related attributes
        car.state.consecutive_moving_frames = 0
        car.state.consecutive_stationary_frames = 0

        logger.debug(f"Reset state for Car {car.id}")

    def is_in_capture_zone(self, x: float) -> bool:
        return self.image_capture_zone[0] <= x <= self.image_capture_zone[1]

    def capture_car_image(self, car: Car, timestamp: float, frame: np.ndarray) -> None:
        image_path = save_vehicle_image(
            frame=frame,
            timestamp=timestamp,
            bbox=car.state.bbox,
        )
        car.state.image_captured = True
        car.state.image_path = image_path


def save_vehicle_image(
    frame: np.ndarray,
    timestamp: float,
    bbox: Tuple[float, float, float, float],
) -> str:
    image_dir = VEHICLE_IMAGE_DIR
    os.makedirs(image_dir, exist_ok=True)

    # Generate a random UUID for the filename
    file_id = uuid.uuid4().hex
    filename = f"{image_dir}/vehicle_{file_id}_{int(timestamp)}.jpg"

    # Crop and save the image
    x, y, w, h = bbox
    padding_factor = 0.1
    padding_x = int(w * padding_factor)
    padding_y = int(h * padding_factor)
    x1, y1 = int(x - w / 2 - padding_x), int(y - h / 2 - padding_y)
    x2, y2 = int(x + w / 2 + padding_x), int(y + h / 2 + padding_y)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    cropped_image = frame[y1:y2, x1:x2]
    cv2.imwrite(filename, cropped_image)

    return filename
