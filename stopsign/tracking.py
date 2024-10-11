import logging
import os
import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Tuple
from typing import cast

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


@dataclass
class CarState:
    location: Point = field(default_factory=lambda: (0.0, 0.0))
    bbox: Tuple[float, float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))
    speed: float = 0.0
    prev_speed: float = 0.0
    velocity: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    direction: float = 0.0
    is_parked: bool = False
    consecutive_moving_frames: int = 0
    consecutive_stationary_frames: int = 0
    track: List[Tuple[Point, float]] = field(default_factory=list)
    last_update_time: float = 0.0
    stop_score: float = 0
    scored: bool = False
    # stop detection
    stop_zone_state: str = "APPROACHING"
    in_stop_zone: bool = False
    entry_time: float = 0.0
    exit_time: float = 0.0
    min_speed_in_zone: float = float("inf")
    time_at_zero: float = 0.0
    stop_position: Point = field(default_factory=lambda: (0.0, 0.0))
    image_captured: bool = False
    image_path: str = ""


class StopZone:
    def __init__(self, stop_line: Line, stop_box_tolerance: int, min_stop_duration: float):
        self.stop_line = stop_line
        self.stop_box_tolerance = stop_box_tolerance
        self.min_stop_duration = min_stop_duration
        self.zone_length = 200
        self._calculate_geometry()

    def update_configuration(self, new_config):
        self.stop_line = new_config["stop_line"]
        self.stop_box_tolerance = new_config["stop_box_tolerance"]
        self.min_stop_duration = new_config["min_stop_duration"]
        self._calculate_geometry()

    def _calculate_geometry(self):
        self.midpoint = self._midpoint(self.stop_line)
        self._angle = self._calculate_angle()
        self.zone_width = np.linalg.norm(np.array(self.stop_line[1]) - np.array(self.stop_line[0]))
        self.corners = self._calculate_corners()
        self.entry, self.exit = self._calculate_entry_exit()
        self.stop_box = self._calculate_stop_box()

    def _midpoint(self, line: Line) -> Point:
        mid = (np.array(line[0]) + np.array(line[1])) / 2
        return float(mid[0]), float(mid[1])

    def _calculate_angle(self) -> float:
        dx, dy = np.array(self.stop_line[1]) - np.array(self.stop_line[0])
        return np.arctan2(dy, dx)

    def _calculate_corners(self) -> np.ndarray:
        perp_vector = np.array([-np.sin(self.angle), np.cos(self.angle)])
        direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        half_width, half_length = self.zone_width / 2, self.zone_length / 2

        corners = [
            self.midpoint + half_width * direction + half_length * perp_vector,
            self.midpoint - half_width * direction + half_length * perp_vector,
            self.midpoint - half_width * direction - half_length * perp_vector,
            self.midpoint + half_width * direction - half_length * perp_vector,
        ]
        return np.array(corners, dtype=np.int32)

    def _calculate_stop_box(self) -> List[Point]:
        # Get the start and end points of the stop line
        (x1, y1), (x2, y2) = self.stop_line

        # Calculate tolerances
        left_tolerance = self.stop_box_tolerance
        right_tolerance = self.stop_box_tolerance * 1.5

        # Extend the stop line on the x-axis
        left_x1 = x1 - left_tolerance
        left_x2 = x2 - left_tolerance
        right_x1 = x1 + right_tolerance
        right_x2 = x2 + right_tolerance

        # Create the box corners without altering the y-axis
        top_left = (left_x1, y1)
        top_right = (right_x1, y1)
        bottom_right = (right_x2, y2)
        bottom_left = (left_x2, y2)

        return [top_left, top_right, bottom_right, bottom_left]

    @property
    def angle(self) -> float:
        dx, dy = np.array(self.stop_line[1]) - np.array(self.stop_line[0])
        return np.arctan2(dy, dx)

    def _calculate_entry_exit(self) -> Tuple[Line, Line]:
        entry_offset, exit_offset = 100, 50
        perp_vector = np.array([-np.sin(self.angle), np.cos(self.angle)])
        direction = np.array([np.cos(self.angle), np.sin(self.angle)])

        entry_mid = self.midpoint - entry_offset * perp_vector
        exit_mid = self.midpoint + exit_offset * perp_vector

        half_width = self.zone_width / 2 + 50

        entry: Line = (
            (
                float(entry_mid[0] - half_width * direction[0]),
                float(entry_mid[1] - half_width * direction[1]),
            ),
            (
                float(entry_mid[0] + half_width * direction[0]),
                float(entry_mid[1] + half_width * direction[1]),
            ),
        )
        exit: Line = (
            (
                float(exit_mid[0] - half_width * direction[0]),
                float(exit_mid[1] - half_width * direction[1]),
            ),
            (
                float(exit_mid[0] + half_width * direction[0]),
                float(exit_mid[1] + half_width * direction[1]),
            ),
        )
        return entry, exit

    def _calculate_bounding_box(self) -> Tuple[Point, Point]:
        x_coords, y_coords = self.corners[:, 0], self.corners[:, 1]
        return ((min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)))

    def is_in_stop_zone(self, point: Point) -> bool:
        return cv2.pointPolygonTest(np.array(self.stop_box, dtype=np.int32), point, False) >= 0

    def is_crossing_line(self, prev_point: Point, current_point: Point, line: Line) -> bool:
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        (x1, y1), (x2, y2) = line
        return ccw(prev_point, (x1, y1), (x2, y2)) != ccw(current_point, (x1, y1), (x2, y2)) and ccw(
            prev_point, current_point, (x1, y1)
        ) != ccw(prev_point, current_point, (x2, y2))

    def min_distance_to_line(self, point: Point, line: Line) -> float:
        x, y = point
        (x1, y1), (x2, y2) = line

        d1 = ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
        d2 = ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5

        line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if line_length == 0:
            return min(d1, d2)

        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length**2)))
        projection = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        d_line = ((x - projection[0]) ** 2 + (y - projection[1]) ** 2) ** 0.5

        return min(d1, d2, d_line)


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
        frame: np.ndarray,
    ) -> None:
        """Update the car's state with new location data."""
        self._update_location(location, timestamp)
        self._update_speed(timestamp)
        self._update_movement_status()
        self._update_parked_status()
        self.state.bbox = bbox
        self.update_direction()

    def _update_location(self, location: Tuple[float, float], timestamp: float) -> None:
        """Update the car's location using Kalman filter."""
        self.kalman_filter.predict()
        smoothed_location = self.kalman_filter.update(np.array(location))
        self.state.location = (float(smoothed_location[0]), float(smoothed_location[1]))
        self.state.track.append((location, timestamp))
        self.state.last_update_time = timestamp

    def _update_speed(self, current_timestamp: float) -> None:
        """Calculate and update the car's velocity and speed."""
        history_length = min(len(self.state.track), 10)
        if history_length > 1:
            recent_track = self.state.track[-history_length:]
            time_diffs = [current_timestamp - t for _, t in recent_track]
            positions = np.array([pos for pos, _ in recent_track])

            # Calculate velocity vector
            velocities = np.diff(positions, axis=0) / np.diff(time_diffs)[:, np.newaxis]
            median_velocity = np.median(velocities, axis=0)

            self.state.velocity = (float(median_velocity[0]), float(median_velocity[1]))
            self.state.speed = float(np.linalg.norm(median_velocity))
        else:
            self.state.velocity = (0.0, 0.0)
            self.state.speed = 0.0

        # Apply low-pass filter to speed
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

    def _update_parked_status(self) -> None:
        """Update the car's parked status based on its movement."""
        if not self.state.is_parked:
            if self.state.consecutive_stationary_frames >= self.config.parked_frame_threshold:
                self.state.is_parked = True
                self.state.consecutive_moving_frames = 0
        else:
            if self.state.consecutive_moving_frames >= self.config.unparked_frame_threshold:
                self.state.is_parked = False
                self.state.consecutive_stationary_frames = 0

    def update_direction(self) -> None:
        """Calculate the direction based on the velocity."""
        vx, vy = self.state.velocity
        self.state.direction = np.arctan2(vy, vx)

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

                self.cars[car_id].update(location, timestamp, bbox, frame)
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
        self.image_capture_zone = self.config.image_capture_zone

    def _create_stop_zone(self) -> StopZone:
        stop_line: Line = (
            cast(Point, tuple(self.config.stop_line[0])),  # First point
            cast(Point, tuple(self.config.stop_line[1])),  # Second point
        )
        return StopZone(
            stop_line=stop_line,
            stop_box_tolerance=self.config.stop_box_tolerance,
            min_stop_duration=self.config.min_stop_time,
        )

    def update_stop_zone(self, new_config):
        self.stop_zone.update_configuration(new_config)

    def update_car_stop_status(self, car: Car, timestamp: float, frame: np.ndarray) -> None:
        if car.state.direction >= 0:
            return  # moving left to right, ignore

        # Use bottom center of bounding box
        x, y, w, h = car.state.bbox
        car_box = np.array(
            [
                [x - w / 2, y - h / 2],  # Top-left
                [x + w / 2, y - h / 2],  # Top-right
                [x + w / 2, y + h / 2],  # Bottom-right
                [x - w / 2, y + h / 2],  # Bottom-left
            ],
            dtype=np.int32,
        )

        stop_box = np.array(self.stop_zone._calculate_stop_box(), dtype=np.int32)

        car.state.in_stop_zone = (
            cv2.rotatedRectangleIntersection(cv2.minAreaRect(car_box), cv2.minAreaRect(stop_box))[0]
            != cv2.INTERSECT_NONE
        )

        if car.state.in_stop_zone:
            self._handle_car_in_stop_zone(car, timestamp)
        else:
            self._handle_car_outside_stop_zone(car, timestamp, frame)

        if not car.state.image_captured and self.is_in_capture_zone(car.state.location[0]):
            self.capture_car_image(car, timestamp, frame)

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

    def _handle_car_in_stop_zone(self, car: Car, timestamp: float) -> None:
        if car.state.stop_zone_state == "APPROACHING":
            car.state.stop_zone_state = "ENTERED"
            car.state.entry_time = timestamp
        elif car.state.stop_zone_state == "ENTERED":
            car.state.min_speed_in_zone = min(car.state.min_speed_in_zone, abs(car.state.speed))
            if car.state.speed == 0:
                car.state.time_at_zero += timestamp - car.state.last_update_time
            if self._is_crossing_stop_line(car):
                car.state.stop_zone_state = "EXITING"
                car.state.exit_time = timestamp
                car.state.stop_position = car.state.location

    def _handle_car_outside_stop_zone(self, car: Car, timestamp: float, frame: np.ndarray) -> None:
        if car.state.stop_zone_state in ["ENTERED", "EXITING"]:
            car.state.stop_zone_state = "EXITED"
            if not car.state.scored and car.state.entry_time > 0:
                car.state.stop_score = self.calculate_stop_score(car)
                car.state.scored = True

                # Capture an image now if the capture area did not work
                if not car.state.image_path:
                    self.capture_car_image(car, timestamp, frame)

                # Save data to database
                self.db.add_vehicle_pass(
                    vehicle_id=car.id,
                    stop_score=car.state.stop_score,
                    stop_duration=car.state.time_at_zero,
                    min_speed=car.state.min_speed_in_zone,
                    image_path=car.state.image_path,
                )
        elif car.state.stop_zone_state == "EXITED":
            # Reset for next approach
            car.state.stop_zone_state = "APPROACHING"
            car.state.min_speed_in_zone = float("inf")
            car.state.time_at_zero = 0.0
            car.state.entry_time = 0.0
            car.state.exit_time = 0.0
            car.state.scored = False
            car.state.stop_position = (0.0, 0.0)
            car.state.image_captured = False
            car.state.image_path = ""

    def _is_crossing_stop_line(self, car: Car) -> bool:
        if car.state.direction >= 0:
            return False  # Moving left to right, not in the correct lane
        prev_point, _ = car.state.track[-2]
        current_point = car.state.location
        return self.stop_zone.is_crossing_line(prev_point, current_point, self.stop_zone.stop_line)

    def calculate_stop_score(self, car: Car) -> float:
        # Time spent in zone (normalized to a 0-1 range)
        time_in_zone = car.state.exit_time - car.state.entry_time
        max_expected_time = 10.0  # Adjust this value as needed
        time_score = min(time_in_zone / max_expected_time, 1.0)

        # Use absolute speed for speed score
        speed_score = 1.0 - min(abs(car.state.min_speed_in_zone) / self.config.max_movement_speed, 1.0)

        # Combine time and speed scores (equal weight)
        raw_score = (time_score + speed_score) / 2

        return raw_score

    def _get_speed_at_line_crossing(self, car: Car) -> float:
        for i in range(1, len(car.state.track)):
            prev_point, prev_time = car.state.track[i - 1]
            curr_point, curr_time = car.state.track[i]
            if self.stop_zone.is_crossing_line(prev_point, curr_point, self.stop_zone.stop_line):
                return car.state.speed  # Assuming speed is updated at each track point
        return 0.0  # Return 0 if line crossing not found (shouldn't happen in normal operation)

    def _calculate_smoothness(self, track: List[Tuple[Tuple[float, float], float]]) -> float:
        if len(track) < 3:
            return 1.0  # Not enough data for smoothness calculation

        accelerations = []
        for i in range(1, len(track) - 1):
            prev_pos, prev_time = track[i - 1]
            curr_pos, curr_time = track[i]
            next_pos, next_time = track[i + 1]

            prev_speed = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos)) / (curr_time - prev_time)
            next_speed = np.linalg.norm(np.array(next_pos) - np.array(curr_pos)) / (next_time - curr_time)

            acceleration = (next_speed - prev_speed) / (next_time - prev_time)
            accelerations.append(acceleration)

        # Calculate smoothness as the inverse of acceleration variance
        acceleration_variance = np.var(accelerations)
        smoothness = float(1 / (1 + acceleration_variance))

        return smoothness


def save_vehicle_image(
    frame: np.ndarray,
    timestamp: float,
    bbox: Tuple[float, float, float, float],
) -> str:
    image_dir = str(os.getenv("VEHICLE_IMAGE_DIR"))
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
