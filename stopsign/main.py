import argparse
import asyncio
import base64
import contextlib
import io
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
from typing import cast

import cv2
import dotenv
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from stopsign.kalman_filter import KalmanFilterWrapper
from stopsign.stream_out import VideoStreamer
from stopsign.utils.video import crop_scale_frame
from stopsign.utils.video import draw_box
from stopsign.utils.video import draw_gridlines
from stopsign.utils.video import open_rtsp_stream

exit_flag = False


def signal_handler(signum, frame):
    global exit_flag
    exit_flag = True
    print("Exiting gracefully...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

dotenv.load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
SAMPLE_FILE_PATH = os.getenv("SAMPLE_FILE_PATH")
STREAM_BUFFER_DIR = os.path.join(os.path.dirname(__file__), "tmp_stream_buffer")

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


class StopZone:
    def __init__(self, stop_line: Line, stop_box_tolerance: int, min_stop_duration: float):
        self.stop_line = stop_line
        self.stop_box_tolerance = stop_box_tolerance
        self.min_stop_duration = min_stop_duration
        self.zone_length = 200
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
        perp_vector = np.array([-np.sin(self.angle), np.cos(self.angle)])
        direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        half_width = self.zone_width / 2 + self.stop_box_tolerance
        half_length = self.stop_box_tolerance

        corners = [
            self.midpoint + half_width * direction + half_length * perp_vector,
            self.midpoint - half_width * direction + half_length * perp_vector,
            self.midpoint - half_width * direction - half_length * perp_vector,
            self.midpoint + half_width * direction - half_length * perp_vector,
        ]
        return [cast(Point, tuple(corner)) for corner in corners]

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
            (float(entry_mid[0] - half_width * direction[0]), float(entry_mid[1] - half_width * direction[1])),
            (float(entry_mid[0] + half_width * direction[0]), float(entry_mid[1] + half_width * direction[1])),
        )
        exit: Line = (
            (float(exit_mid[0] - half_width * direction[0]), float(exit_mid[1] - half_width * direction[1])),
            (float(exit_mid[0] + half_width * direction[0]), float(exit_mid[1] + half_width * direction[1])),
        )
        return entry, exit

    def _calculate_bounding_box(self) -> Tuple[Point, Point]:
        x_coords, y_coords = self.corners[:, 0], self.corners[:, 1]
        return ((min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)))

    def is_in_stop_zone(self, point: Point) -> bool:
        return cv2.pointPolygonTest(self.corners, point, False) >= 0

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
        self.kalman_filter = KalmanFilterWrapper(process_noise=10, measurement_noise=10)
        self.config = config

    def update(self, location: Tuple[float, float], timestamp: float) -> None:
        """Update the car's state with new location data."""
        self._update_location(location, timestamp)
        self._update_speed(timestamp)
        self._update_movement_status()
        self._update_parked_status()

    def _update_location(self, location: Tuple[float, float], timestamp: float) -> None:
        """Update the car's location using Kalman filter."""
        self.kalman_filter.predict()
        smoothed_location = self.kalman_filter.update(np.array(location))
        self.state.location = (float(smoothed_location[0]), float(smoothed_location[1]))
        self.state.track.append((location, timestamp))
        self.state.last_update_time = timestamp

    def _update_speed(self, current_timestamp: float) -> None:
        """Calculate and update the car's speed."""
        history_length = min(len(self.state.track), 10)
        if history_length > 1:
            # Use only the last 10 positions (or less if not available)
            recent_track = self.state.track[-history_length:]

            # Calculate time differences and distances
            time_diffs = [current_timestamp - t for _, t in recent_track]
            positions = np.array([pos for pos, _ in recent_track])
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)

            # Calculate speeds
            speeds = distances / np.diff(time_diffs)

            # Use median speed to reduce impact of outliers
            median_speed = np.median(speeds)
            self.state.speed = float(median_speed) if len(speeds) > 0 else 0.0
        else:
            self.state.speed = 0.0

        self.state.speed = abs(self.state.speed)
        self.state.speed = min(self.state.speed, 200)  # Max speed limit

        # Apply low-pass filter
        alpha = 0.2
        self.state.speed = alpha * self.state.speed + (1 - alpha) * self.state.prev_speed
        self.state.prev_speed = self.state.speed

        # Set minimum speed threshold
        self.state.speed = 0.0 if self.state.speed < 1.0 else self.state.speed

    def _update_movement_status(self) -> None:
        """Update the car's movement status based on its speed."""
        if self.state.speed < self.config.max_movement_speed:
            self.state.consecutive_moving_frames = 0
            self.state.consecutive_stationary_frames += 1
        else:
            self.state.consecutive_moving_frames += 1
            self.state.consecutive_stationary_frames = 0

    def _update_parked_status(self) -> None:
        """Update the car's parked status based on its movement."""
        if self.state.is_parked:
            if self.state.consecutive_moving_frames >= self.config.unparked_frame_threshold:
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
    def __init__(self, config: Config):
        self.cars: Dict[int, Car] = {}
        self.config = config

    def update_cars(self, boxes: List, timestamp: float) -> None:
        for box in boxes:
            if box.id is None:
                logger.warning("Skipping box without ID")
                continue
            try:
                car_id = int(box.id.item())
                x, y, w, h = box.xywh[0]
                location = (float(x), float(y))

                if car_id not in self.cars:
                    self.cars[car_id] = Car(id=car_id, config=self.config)

                self.cars[car_id].update(location, timestamp)
            except Exception as e:
                logger.error(f"Error updating car {box.id}: {str(e)}")

    def get_cars(self) -> Dict[int, Car]:
        return self.cars


class StopDetector:
    def __init__(self, config: Config):
        self.config = config
        self.stop_zone = self._create_stop_zone()

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

    def update_car_stop_status(self, car: Car, timestamp: float) -> None:
        if self.stop_zone.is_in_stop_zone(car.state.location):
            self._handle_car_in_stop_zone(car, timestamp)
        else:
            self._handle_car_outside_stop_zone(car, timestamp)

    def _handle_car_in_stop_zone(self, car: Car, timestamp: float) -> None:
        if car.state.stop_zone_state == "APPROACHING":
            car.state.stop_zone_state = "ENTERED"
            car.state.entry_time = timestamp
        elif car.state.stop_zone_state == "ENTERED":
            car.state.min_speed_in_zone = min(car.state.min_speed_in_zone, car.state.speed)
            if car.state.speed == 0:
                car.state.time_at_zero += timestamp - car.state.last_update_time
            if self._is_crossing_stop_line(car):
                car.state.stop_zone_state = "EXITING"
                car.state.exit_time = timestamp
                car.state.stop_position = car.state.location

    def _handle_car_outside_stop_zone(self, car: Car, timestamp: float) -> None:
        if car.state.stop_zone_state in ["ENTERED", "EXITING"]:
            car.state.stop_zone_state = "EXITED"
            if not car.state.scored and car.state.entry_time > 0:
                car.state.stop_score = self.calculate_stop_score(car)
                car.state.scored = True
                print(f"Car {car.id} stop score: {car.state.stop_score}")  # Debug print
        elif car.state.stop_zone_state == "EXITED":
            # Reset for next approach
            car.state.stop_zone_state = "APPROACHING"
            car.state.min_speed_in_zone = float("inf")
            car.state.time_at_zero = 0.0
            car.state.entry_time = 0.0
            car.state.exit_time = 0.0
            car.state.scored = False
            car.state.stop_position = (0.0, 0.0)

    def _is_crossing_stop_line(self, car: Car) -> bool:
        if len(car.state.track) < 2:
            return False
        prev_point, _ = car.state.track[-2]
        current_point = car.state.location
        return self.stop_zone.is_crossing_line(prev_point, current_point, self.stop_zone.stop_line)

    def calculate_stop_score(self, car: Car) -> int:
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
            distance_from_line = self.stop_zone.min_distance_to_line(car.state.stop_position, self.stop_zone.stop_line)
            position_penalty = min(distance_from_line / self.config.stop_box_tolerance, 2)
        else:
            position_penalty = 2

        # Stop duration
        stop_duration_score = min(car.state.time_at_zero / self.config.min_stop_time, 1) * 3

        # Speed at stop line crossing
        stop_line_crossing_speed = self._get_speed_at_line_crossing(car)
        line_crossing_penalty = min(stop_line_crossing_speed / self.config.max_movement_speed, 2)

        # Smooth deceleration and acceleration
        smoothness_score = self._calculate_smoothness(car.state.track) * 2

        final_score = (
            base_score
            + full_stop_bonus
            - rolling_stop_penalty
            - position_penalty
            + stop_duration_score
            - line_crossing_penalty
            + smoothness_score
        )

        return max(min(round(final_score), 10), 0)  # Ensure score is between 0 and 10

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
    with contextlib.redirect_stdout(io.StringIO()):
        results = model.track(
            source=frame,
            tracker="./trackers/bytetrack.yaml",
            stream=False,
            persist=True,
            classes=vehicle_classes,
            verbose=False,
        )

    # Filter out non-vehicle classes
    boxes = results[0].boxes
    if boxes:
        boxes = [obj for obj in boxes if obj.cls in vehicle_classes]
    else:
        boxes = []
    return frame, boxes


def visualize(frame, cars: Dict[int, Car], boxes: List, stop_zone: StopZone, n_frame: int) -> np.ndarray:
    # Create a copy of the frame for the overlay
    overlay = frame.copy()

    # Check if any moving car is inside the stop zone
    car_in_stop_zone = any(
        stop_zone.is_in_stop_zone(car.state.location) for car in cars.values() if not car.state.is_parked
    )

    # Set the color based on whether a moving car is in the stop zone
    color = (0, 255, 0) if car_in_stop_zone else (255, 255, 255)  # Green if car inside, white else

    # Draw the stop box as a semi-transparent rectangle
    stop_box_corners = np.array(stop_zone._calculate_stop_box(), dtype=np.int32)
    cv2.fillPoly(overlay, [stop_box_corners], color)
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Plot the stop sign line
    start_point = tuple(map(int, stop_zone.stop_line[0]))
    end_point = tuple(map(int, stop_zone.stop_line[1]))
    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

    # Draw boxes for each car
    for box in boxes:
        if box.id is None:
            logger.warning("Skipping box without ID in visualize function")
            continue
        try:
            car_id = int(box.id.item())
            if car_id in cars:
                car = cars[car_id]
                if car.state.is_parked:
                    draw_box(frame, car, box, color=(255, 255, 255), thickness=1)  # parked cars
                else:
                    draw_box(frame, car, box, color=(0, 255, 0), thickness=2)  # moving cars
            else:
                logger.warning(f"Car with ID {car_id} not found in cars dictionary")
        except Exception as e:
            logger.error(f"Error processing box in visualize function: {str(e)}")

    # Path tracking code
    for car in cars.values():
        if car.state.is_parked:
            continue
        locations = [loc for loc, _ in car.state.track]  # Extract locations from track
        points = np.array(locations, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

    # Display the frame number on image
    cv2.putText(frame, f"Frame: {n_frame}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


async def process_frames():
    global frame_count
    while True:
        annotated_frame = process_and_annotate_frame()
        if annotated_frame is None:
            break
        await asyncio.sleep(1 / config.fps)  # Adjust sleep time based on desired frame rate


async def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()


app = FastAPI()
os.makedirs(STREAM_BUFFER_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.abspath(STREAM_BUFFER_DIR)), name="static")


@app.get("/")
async def get():
    return HTMLResponse(content=open("stopsign/index.html", "r").read())


def process_frame_task():
    return process_and_annotate_frame()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    frame_interval = max(1, int(config.fps / 10))
    frame_count = 0
    last_send_time = time.time()

    while True:
        frame_count += 1
        annotated_frame = await asyncio.to_thread(process_frame_task)

        if annotated_frame is None:
            break

        current_time = time.time()
        if frame_count % frame_interval == 0 and current_time - last_send_time >= 0.1:
            _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(jpg_as_text)
            last_send_time = current_time

        await asyncio.sleep(0.001)


def initialize_video_capture(input_source):
    if input_source == "live":
        if not RTSP_URL:
            print("Error: RTSP_URL environment variable is not set.")
            sys.exit(1)
        print(f"Opening RTSP stream: {RTSP_URL}")
        return open_rtsp_stream(RTSP_URL)
    elif input_source == "file":
        print(f"Opening video file: {SAMPLE_FILE_PATH}")
        return cv2.VideoCapture(SAMPLE_FILE_PATH)  # type: ignore
    else:
        print("Error: Invalid input source")
        sys.exit(1)


def initialize_components(config: Config, debug_mode: bool) -> None:
    global model, car_tracker, stop_detector, streamer

    if not MODEL_PATH:
        print("Error: YOLO_MODEL_PATH environment variable is not set.")
        sys.exit(1)
    model = YOLO(MODEL_PATH, verbose=False)
    print("Model loaded successfully")

    car_tracker = CarTracker(config)
    stop_detector = StopDetector(config)

    # Initialize VideoStreamer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    streamer = VideoStreamer(STREAM_BUFFER_DIR, config.fps, width, height)
    streamer.debug_mode = debug_mode
    streamer.start()
    print(f"VideoStreamer initialized with output directory: {STREAM_BUFFER_DIR}")


def process_and_annotate_frame():
    global frame_count
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached.")
        return None

    frame, boxes = process_frame(
        model=model,
        frame=frame,
        scale=config.scale,
        crop_top_ratio=config.crop_top,
        crop_side_ratio=config.crop_side,
        vehicle_classes=config.vehicle_classes,
    )

    car_tracker.update_cars(boxes, time.time())
    for car in car_tracker.get_cars().values():
        if not car.state.is_parked:
            stop_detector.update_car_stop_status(car, time.time())

    annotated_frame = visualize(
        frame,
        car_tracker.cars,
        boxes,
        stop_detector.stop_zone,
        frame_count,
    )

    if config.draw_grid:
        draw_gridlines(annotated_frame, config.grid_size)

    # Ensure the frame is in BGR format
    if len(annotated_frame.shape) == 2:  # If grayscale
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2BGR)
    elif annotated_frame.shape[2] == 4:  # If RGBA
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGBA2BGR)
        print("Converted RGBA to BGR")

    streamer.add_frame(annotated_frame)

    frame_count += 1
    return annotated_frame


def cleanup():
    global cap, streamer
    cap.release()
    cv2.destroyAllWindows()
    streamer.stop()


def main(input_source: str, config: Config, web_mode: bool, debug_mode: bool):
    global cap, frame_count, streamer, exit_flag

    os.makedirs(STREAM_BUFFER_DIR, exist_ok=True)
    cap = initialize_video_capture(input_source)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        sys.exit()

    initialize_components(config, debug_mode)
    frame_count = 0

    if web_mode:
        loop = asyncio.get_event_loop()
        loop.create_task(process_frames())
        try:
            loop.run_until_complete(run_server())
        except KeyboardInterrupt:
            print("Keyboard interrupt received, shutting down...")
        finally:
            exit_flag = True
            cleanup()
    else:
        try:
            while not exit_flag:
                annotated_frame = process_and_annotate_frame()
                if annotated_frame is None:
                    break

                cv2.imshow("Output", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            cleanup()

    # Ensure the streamer is stopped
    if streamer:
        streamer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection on live RTSP stream or video file.")
    parser.add_argument(
        "input_source", choices=["live", "file"], help="Input source type (live RTSP stream or video file)"
    )
    parser.add_argument("--web", action="store_true", help="Run in web mode")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with static image")
    args = parser.parse_args()

    config = Config("./config.yaml")

    main(input_source=args.input_source, config=config, web_mode=args.web, debug_mode=args.debug)
