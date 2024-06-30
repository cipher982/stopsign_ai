import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Tuple

import cv2
import dotenv
import numpy as np
import yaml
from ultralytics import YOLO

from stopsign.kalman_filter import KalmanFilterWrapper
from stopsign.utils.video import crop_scale_frame
from stopsign.utils.video import draw_box
from stopsign.utils.video import draw_gridlines
from stopsign.utils.video import open_rtsp_stream
from stopsign.utils.video import signal_handler

dotenv.load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
SAMPLE_FILE_PATH = os.getenv("SAMPLE_FILE_PATH")
OUTPUT_VIDEO_DIR = os.getenv("OUTPUT_VIDEO_DIR")

os.environ["DISPLAY"] = ":0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


Point = Tuple[float, float]
Line = Tuple[Point, Point]


class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Video processing
        self.scale = config["video_processing"]["scale"]
        self.crop_top = config["video_processing"]["crop_top"]
        self.crop_side = config["video_processing"]["crop_side"]
        self.frame_buffer_size = config["video_processing"]["frame_buffer_size"]

        # Stop sign detection
        stop_sign = config["stopsign_detection"]
        self.stop_line = tuple(tuple(i) for i in stop_sign["stop_line"])
        self.stop_box_tolerance = stop_sign["stop_box_tolerance"]
        self.min_stop_time = stop_sign["min_stop_time"]
        self.max_movement_speed = stop_sign["max_movement_speed"]
        self.parked_frame_threshold = stop_sign["parked_frame_threshold"]
        self.unparked_frame_threshold = stop_sign["unparked_frame_threshold"]

        # Tracking
        self.use_kalman_filter = config["tracking"]["use_kalman_filter"]

        # Output
        self.save_video = config["output"]["save_video"]

        # Visualization
        self.draw_grid = config["debugging_visualization"]["draw_grid"]
        self.grid_size = config["debugging_visualization"]["grid_size"]

        # Stream settings
        self.fps = config["stream_settings"]["fps"]
        self.vehicle_classes = config["stream_settings"]["vehicle_classes"]


@dataclass
class CarState:
    location: Point = field(default_factory=lambda: (0.0, 0.0))
    speed: float = 0.0
    prev_speed: float = 0.0
    is_parked: bool = True
    consecutive_moving_frames: int = 0
    consecutive_stationary_frames: int = 0
    track: List[Tuple[Point, float]] = field(default_factory=list)
    last_update_time: float = 0.0
    stop_score: int = 0
    scored: bool = False
    # stop detection
    stop_zone_state: str = "APPROACHING"
    entry_time: float = 0.0
    exit_time: float = 0.0
    min_speed_in_zone: float = float("inf")
    time_at_zero: float = 0.0
    stop_position: Point = field(default_factory=lambda: (0.0, 0.0))


@dataclass
class StopZone:
    stop_line: Line
    stop_box_tolerance: int
    min_stop_duration: float
    zone_length: int = 200

    def __post_init__(self):
        self._calculate_geometry()

    def _calculate_geometry(self):
        self.midpoint = self._midpoint(self.stop_line)
        self.zone_width = np.linalg.norm(np.array(self.stop_line[1]) - np.array(self.stop_line[0]))
        self.corners = self._calculate_corners()
        self.entry, self.exit = self._calculate_entry_exit()
        self.stop_box = self._calculate_stop_box()
        self.bounding_box = self._calculate_bounding_box()

    @property
    def angle(self) -> float:
        dx, dy = np.array(self.stop_line[1]) - np.array(self.stop_line[0])
        return np.arctan2(dy, dx)

    def _midpoint(self, line: Line) -> Point:
        mid = (np.array(line[0]) + np.array(line[1])) / 2
        return float(mid[0]), float(mid[1])

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

    def _calculate_entry_exit(self) -> Tuple[Line, Line]:
        entry_offset, exit_offset = 100, 50
        perp_vector = np.array([-np.sin(self.angle), np.cos(self.angle)])
        direction = np.array([np.cos(self.angle), np.sin(self.angle)])

        entry_mid = self.midpoint - entry_offset * perp_vector
        exit_mid = self.midpoint + exit_offset * perp_vector

        half_width = self.zone_width / 2 + 50

        entry: Line = (
            (float(entry_mid[0] - half_width * direction[0]), float(entry_mid[1] - half_width * direction[1])),
            (float(entry_mid[0] + half_width * direction[0]), float(entry_mid[1] + half_width * direction[1])),
        )
        exit: Line = (
            (float(exit_mid[0] - half_width * direction[0]), float(exit_mid[1] - half_width * direction[1])),
            (float(exit_mid[0] + half_width * direction[0]), float(exit_mid[1] + half_width * direction[1])),
        )
        return entry, exit

    def _calculate_stop_box(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        left_x = min(self.stop_line[0][0], self.stop_line[1][0]) - self.stop_box_tolerance
        right_x = max(self.stop_line[0][0], self.stop_line[1][0]) + self.stop_box_tolerance
        top_y = min(self.stop_line[0][1], self.stop_line[1][1])
        bottom_y = max(self.stop_line[0][1], self.stop_line[1][1])
        return ((int(left_x), int(top_y)), (int(right_x), int(bottom_y)))

    def _calculate_bounding_box(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        x_coords, y_coords = self.corners[:, 0], self.corners[:, 1]
        return ((min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)))

    def is_in_stop_zone(self, point: Tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(self.corners, point, False) >= 0


class Car:
    def __init__(self, id: int, config: Config):
        self.id = id
        self.state = CarState()
        self.kalman_filter = KalmanFilterWrapper(process_noise=10, measurement_noise=10)
        self.config = config

    def update(self, location: Tuple[float, float], timestamp: float, stop_zone: StopZone):
        self.kalman_filter.predict()
        smoothed_location = self.kalman_filter.update(np.array(location))
        self.state.location = (float(smoothed_location[0]), float(smoothed_location[1]))
        self.state.track.append((location, timestamp))

        self._update_speed(timestamp)
        self._update_movement_status()
        self._update_parked_status()
        self._update_stop_zone(stop_zone, timestamp)

        self.state.last_update_time = timestamp

    def _update_stop_zone(self, stop_zone: StopZone, current_time: float):
        prev_location = self.state.track[-2][0] if len(self.state.track) > 1 else self.state.location

        if self.state.stop_zone_state == "APPROACHING":
            if is_crossing_line(prev_location, self.state.location, stop_zone.stop_line):
                self.state.stop_zone_state = "IN_STOP_ZONE"
                self.state.entry_time = current_time

        elif self.state.stop_zone_state == "IN_STOP_ZONE":
            self.state.min_speed_in_zone = min(self.state.min_speed_in_zone, self.state.speed)

            if self.state.speed == 0:
                self.state.time_at_zero += current_time - self.state.last_update_time
                if self.state.stop_position == (0.0, 0.0):
                    self.state.stop_position = self.state.location

            if is_crossing_line(prev_location, self.state.location, stop_zone.exit):
                self.state.stop_zone_state = "EXITING"
                self.state.exit_time = current_time

        elif self.state.stop_zone_state == "EXITING":
            if not is_point_in_rectangle(self.state.location, stop_zone.bounding_box):
                self.state.stop_zone_state = "SCORED"
                self.state.stop_score = calculate_stop_score(self, stop_zone, self.config)
                self.state.scored = True

    def set_stop_score(self, score: int):
        self.state.stop_score = score
        self.state.scored = True

    def _update_speed(self, timestamp: float):
        history_length = min(len(self.state.track), 10)
        if history_length > 1:
            past_positions = np.array([pos for pos, _ in self.state.track[-history_length:]])
            past_times = np.array([t for _, t in self.state.track[-history_length:]])

            time_diffs = np.diff(past_times)
            position_diffs = np.diff(past_positions, axis=0)

            speeds = np.linalg.norm(position_diffs, axis=1) / time_diffs

            # Apply moving average filter
            window_size = min(10, len(speeds))
            smoothed_speed = np.convolve(speeds, np.ones(window_size) / window_size, mode="valid")

            # Calculate median speed
            median_speed = np.median(speeds)

            # Use median speed if available, otherwise use smoothed speed
            if len(speeds) > 0:
                self.state.speed = float(median_speed)
            elif len(smoothed_speed) > 0:
                self.state.speed = float(smoothed_speed[-1])
            else:
                self.state.speed = 0.0
        else:
            self.state.speed = 0.0

        # Ensure speed is non-negative
        self.state.speed = abs(self.state.speed)

        # Add a maximum speed limit to filter out unrealistic spikes
        max_speed_limit = 200  # pixels per second
        self.state.speed = min(self.state.speed, max_speed_limit)

        # Apply a low-pass filter
        alpha = 0.2
        prev_speed = getattr(self.state, "prev_speed", self.state.speed)
        self.state.speed = alpha * self.state.speed + (1 - alpha) * prev_speed
        self.state.prev_speed = self.state.speed

        # Set a minimum speed threshold
        min_speed_threshold = 1.0  # pixels per second
        if self.state.speed < min_speed_threshold:
            self.state.speed = 0.0

    def _update_movement_status(self):
        if self.state.speed < self.config.max_movement_speed:
            self.state.consecutive_moving_frames = 0
            self.state.consecutive_stationary_frames += 1
        else:
            self.state.consecutive_moving_frames += 1
            self.state.consecutive_stationary_frames = 0

    def _update_parked_status(self):
        if self.state.is_parked:
            if self.state.consecutive_moving_frames >= self.config.unparked_frame_threshold:
                self.state.is_parked = False
                self.state.consecutive_stationary_frames = 0
        else:
            if self.state.consecutive_stationary_frames >= self.config.parked_frame_threshold:
                self.state.is_parked = True
                self.state.consecutive_moving_frames = 0

    def __repr__(self):
        return (
            f"Car {self.id} @ ({self.state.location[0]:.2f}, {self.state.location[1]:.2f}) "
            f"(Speed: {self.state.speed:.1f}px/s, Parked: {self.state.is_parked})"
        )


class CarTracker:
    def __init__(self, config: Config):
        self.cars: Dict[int, Car] = {}
        self.config = config

    def update_cars(self, boxes: List, timestamp: float, stop_zone: StopZone) -> None:
        for box in boxes:
            try:
                car_id = int(box.id.item())
                x, y, w, h = box.xywh[0]
                location = (float(x), float(y))

                if car_id not in self.cars:
                    self.cars[car_id] = Car(id=car_id, config=self.config)

                self.cars[car_id].update(location, timestamp, stop_zone)
            except Exception as e:
                logger.error(f"Error updating car {box.id.item()}: {str(e)}")

    def get_cars(self) -> Dict[int, Car]:
        return self.cars


def is_point_in_rectangle(point: Point, rectangle: Tuple[Point, Point]) -> bool:
    x, y = point
    (x1, y1), (x2, y2) = rectangle
    return x1 <= x <= x2 and y1 <= y <= y2


def is_crossing_line(
    prev_point: Point,
    current_point: Point,
    line: Line,
) -> bool:
    (x1, y1), (x2, y2) = line

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Check if the line segment between prev_point and current_point intersects with the given line
    intersects = ccw(prev_point, (x1, y1), (x2, y2)) != ccw(current_point, (x1, y1), (x2, y2)) and ccw(
        prev_point, current_point, (x1, y1)
    ) != ccw(prev_point, current_point, (x2, y2))

    # If not intersecting, check if the current point is very close to the line
    if not intersects:
        distance = min_distance_to_line(current_point, line)
        return distance < 5  # Adjust this threshold as needed

    return True


# def update_car_in_stop_zone(car: Car, stop_zone: StopZone, current_time: float):
#     prev_location = car.state.track[-2][0] if len(car.state.track) > 1 else car.state.location

#     if car.state.stop_zone_state == "APPROACHING":
#         if is_crossing_line(prev_location, car.state.location, stop_zone.stop_line):
#             car.state.stop_zone_state = "IN_STOP_ZONE"
#             car.state.entry_time = current_time

#     elif car.state.stop_zone_state == "IN_STOP_ZONE":
#         car.state.min_speed_in_zone = min(car.state.min_speed_in_zone, car.state.speed)

#         if car.state.speed == 0:
#             car.state.time_at_zero += current_time - car.state.last_update_time
#             if car.state.stop_position == (0.0, 0.0):
#                 car.state.stop_position = car.state.location

#         if is_crossing_line(prev_location, car.state.location, stop_zone.exit):
#             car.state.stop_zone_state = "EXITING"
#             car.state.exit_time = current_time

#     elif car.state.stop_zone_state == "EXITING":
#         if not is_point_in_rectangle(car.state.location, stop_zone.bounding_box):
#             car.state.stop_zone_state = "SCORED"
#             car.state.stop_score = calculate_stop_score(car, stop_zone, car.config)
#             car.state.scored = True

#     if car.id == 3:
#         print(f"{car.id}: {car.state.stop_zone_state}")


def calculate_stop_score(car: Car, stop_zone: StopZone, config: Config) -> int:
    """
    Calculate the stop score for a car based on its behavior in the stop zone.
    """
    base_score = 10

    # Stopping
    if car.state.min_speed_in_zone == 0:
        full_stop_bonus = 3
        rolling_stop_penalty = 0
    else:
        full_stop_bonus = 0
        rolling_stop_penalty = min(car.state.min_speed_in_zone * 2, 3)

    # Stop position
    if car.state.stop_position != (0.0, 0.0):
        distance_from_line = min_distance_to_line(car.state.stop_position, stop_zone.stop_line)
        position_penalty = min(distance_from_line / config.stop_box_tolerance, 2)
    else:
        position_penalty = 2

    # Stop duration
    stop_duration_score = min(car.state.time_at_zero / config.min_stop_time, 1) * 3

    # Speed at stop line crossing
    stop_line_crossing_speed = get_speed_at_line_crossing(car, stop_zone.stop_line)
    line_crossing_penalty = min(stop_line_crossing_speed / config.max_movement_speed, 2)

    # Smooth deceleration and acceleration
    smoothness_score = calculate_smoothness(car.state.track) * 2

    final_score = (
        base_score
        + full_stop_bonus
        - rolling_stop_penalty
        - position_penalty
        + stop_duration_score
        - line_crossing_penalty
        + smoothness_score
    )

    normalized_score = max(min(round(final_score), 10), 0)  # Ensure score is between 0 and 10
    return normalized_score


def min_distance_to_line(point: Point, line: Line) -> float:
    """
    Calculate the minimum distance from a point to a line segment.
    """
    x, y = point
    (x1, y1), (x2, y2) = line

    # Calculate the distance to each endpoint
    d1 = ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    d2 = ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5

    # Calculate the distance to the line itself
    line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if line_length == 0:
        return min(d1, d2)

    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length**2)))
    projection = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    d_line = ((x - projection[0]) ** 2 + (y - projection[1]) ** 2) ** 0.5

    return min(d1, d2, d_line)


def get_speed_at_line_crossing(car: Car, line: Line) -> float:
    """
    Get the speed of the car when it crossed the stop line.
    """
    for i in range(1, len(car.state.track)):
        prev_point, prev_time = car.state.track[i - 1]
        curr_point, curr_time = car.state.track[i]
        if is_crossing_line(prev_point, curr_point, line):
            return car.state.speed  # Assuming speed is updated at each track point
    return 0.0  # Return 0 if line crossing not found (shouldn't happen in normal operation)


def calculate_smoothness(track: List[Tuple[Tuple[float, float], float]]) -> float:
    """
    Calculate the smoothness of the car's movement through the stop zone.
    """
    if len(track) < 3:
        return 1.0  # Not enough data for smoothness calculation

    accelerations = []
    for i in range(1, len(track) - 1):
        prev_pos, prev_time = track[i - 1]
        curr_pos, curr_time = track[i]
        next_pos, next_time = track[i + 1]

        prev_speed = distance(prev_pos, curr_pos) / (curr_time - prev_time)
        next_speed = distance(curr_pos, next_pos) / (next_time - curr_time)

        acceleration = (next_speed - prev_speed) / (next_time - prev_time)
        accelerations.append(abs(acceleration))

    avg_acceleration = sum(accelerations) / len(accelerations)
    smoothness = 1 / (1 + avg_acceleration)  # Normalize to 0-1 range
    return smoothness


def distance(point1: Point, point2: Point) -> float:
    """
    Calculate the Euclidean distance between two points.
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def process_frame(
    model: YOLO,
    frame: np.ndarray,
    scale: float,
    crop_top_ratio: float,
    crop_side_ratio: float,
    vehicle_classes: list,
) -> tuple:
    # Initial frame preprocessing
    frame = crop_scale_frame(frame, scale, crop_top_ratio, crop_side_ratio)

    # Run YOLO inference
    results = model.track(
        source=frame,
        tracker="./trackers/bytetrack.yaml",
        stream=False,
        persist=True,
        classes=vehicle_classes,
    )

    # Filter out non-vehicle classes
    boxes = results[0].boxes
    if boxes:
        boxes = [obj for obj in boxes if obj.cls in vehicle_classes]
    else:
        boxes = []
    return frame, boxes


def visualize(frame, cars, boxes, stop_zone, n_frame) -> np.ndarray:
    # Draw the stop box as a semi-transparent rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, stop_zone.stop_box[0], stop_zone.stop_box[1], (0, 0, 255), -1)
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Plot the stop sign line
    cv2.line(frame, stop_zone.stop_line[0], stop_zone.stop_line[1], (0, 0, 255), 2)

    # Draw boxes for each car
    for box in boxes:
        car = cars[int(box.id.item())]
        if car.state.is_parked:
            draw_box(frame, car, box, color=(255, 255, 255), thickness=1)  # parked cars
        else:
            draw_box(frame, car, box, color=(0, 255, 0), thickness=2)  # moving cars

    # Path tracking code
    for id in cars:
        car = cars[id]
        if cars[id].state.is_parked:
            continue
        locations = [loc for loc, _ in car.state.track]  # Extract locations from track
        points = np.array(locations, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

    # Display the frame number on image
    cv2.putText(frame, f"Frame: {n_frame}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def main(input_source: str, config: Config):
    global cap, video_writer, max_x

    if input_source == "live":
        if not RTSP_URL:
            print("Error: RTSP_URL environment variable is not set.")
            sys.exit(1)
        print(f"Opening RTSP stream: {RTSP_URL}")
        cap = open_rtsp_stream(RTSP_URL)
    elif input_source == "file":
        print(f"Opening video file: {SAMPLE_FILE_PATH}")
        cap = cv2.VideoCapture(SAMPLE_FILE_PATH)  # type: ignore
    else:
        print("Error: Invalid input source")
        sys.exit(1)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        sys.exit()

    if not MODEL_PATH:
        print("Error: YOLO_MODEL_PATH environment variable is not set.")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    max_x = w

    if config.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        output_file_name = f"{OUTPUT_VIDEO_DIR}/output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_writer = cv2.VideoWriter(
            filename=output_file_name,
            apiPreference=cv2.CAP_FFMPEG,
            fourcc=fourcc,
            fps=config.fps,
            frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )

    signal.signal(signal.SIGINT, signal_handler)

    # Create the model
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")

    stop_zone = StopZone(
        stop_line=config.stop_line,  # type: ignore
        stop_box_tolerance=config.stop_box_tolerance,
        min_stop_duration=config.min_stop_time,
    )

    car_tracker = CarTracker(config)

    # Begin streaming loop
    print("Streaming...")
    cars = {}
    frame_count = 0
    frame_buffer = []
    buffer_size = 5
    prev_frame_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video file reached.")
                break

            # Get the timestamp for the frame
            if input_source == "live":
                current_time = time.time()
                timestamp = current_time - prev_frame_time
                prev_frame_time = current_time
            else:  # Video file
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                timestamp = frame_number / fps

            # Add the frame to the buffer
            frame_buffer.append((frame, timestamp))
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)

            # Crop, Scale, Model the frame
            frame, boxes = process_frame(
                model=model,
                frame=frame,
                scale=config.scale,
                crop_top_ratio=config.crop_top,
                crop_side_ratio=config.crop_side,
                vehicle_classes=config.vehicle_classes,
            )

            # Update the car tracker
            car_tracker.update_cars(boxes, timestamp, stop_zone)

            # Visualize only non-parked cars
            try:
                annotated_frame = visualize(
                    frame,
                    cars,
                    boxes,
                    stop_zone,
                    frame_count,
                )
            except Exception as e:
                annotated_frame = frame
                print(f"Visualization failed: {e}")

            # Draw the gridlines for debugging
            if config.draw_grid:
                draw_gridlines(annotated_frame, config.grid_size)

            cv2.imshow("Output", annotated_frame)
            cv2.waitKey(1)

            if frame_count == 25:
                print("Pausing...")

            if config.save_video:
                assert frame.size > 0, "Error: Frame is empty"
                video_writer.write(annotated_frame)

            frame_count += 1

            # time.sleep(0.3)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

    finally:
        cap.release()
        if config.save_video:
            print(f"Output video saved to: {output_file_name}")  # type: ignore
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection on live RTSP stream or video file.")
    parser.add_argument(
        "input_source", choices=["live", "file"], help="Input source type (live RTSP stream or video file)"
    )
    args = parser.parse_args()

    config = Config("./config.yaml")

    main(
        input_source=args.input_source,
        config=config,
    )
