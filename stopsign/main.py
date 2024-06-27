import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from dataclasses import field
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
    location: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    speed: float = 0.0
    is_parked: bool = True
    consecutive_moving_frames: int = 0
    consecutive_stationary_frames: int = 0
    track: List[Tuple[Tuple[float, float], float]] = field(default_factory=list)
    last_update_time: float = 0.0
    stop_score: int = 0
    scored: bool = False
    # stop detection
    stop_zone_state: str = "APPROACHING"
    entry_time: float = 0.0
    exit_time: float = 0.0
    min_speed_in_zone: float = float("inf")
    time_at_zero: float = 0.0
    stop_position: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


class StopZone:
    def __init__(self, config):
        self.stop_line = config.stop_line
        self.stop_box_tolerance = config.stop_box_tolerance
        self.min_stop_duration = config.min_stop_time

        # Calculate entry, exit, and stop box
        self.entry = (
            (self.stop_line[0][0] - 100, self.stop_line[0][1]),
            (self.stop_line[1][0] + 100, self.stop_line[1][1]),
        )
        self.exit = (
            (self.stop_line[0][0] - 50, self.stop_line[0][1] + 50),
            (self.stop_line[1][0] + 50, self.stop_line[1][1] + 50),
        )
        self._calculate_stop_box()
        self._calculate_bounding_box()

    def _calculate_stop_box(self):
        left_x = min(self.stop_line[0][0], self.stop_line[1][0]) - self.stop_box_tolerance
        right_x = max(self.stop_line[0][0], self.stop_line[1][0]) + self.stop_box_tolerance
        top_y = min(self.stop_line[0][1], self.stop_line[1][1])
        bottom_y = max(self.stop_line[0][1], self.stop_line[1][1])
        self.stop_box = ((left_x, top_y), (right_x, bottom_y))

    def _calculate_bounding_box(self):
        x_coords = [
            self.entry[0][0],
            self.entry[1][0],
            self.stop_line[0][0],
            self.stop_line[1][0],
            self.exit[0][0],
            self.exit[1][0],
        ]
        y_coords = [
            self.entry[0][1],
            self.entry[1][1],
            self.stop_line[0][1],
            self.stop_line[1][1],
            self.exit[0][1],
            self.exit[1][1],
        ]
        self.bounding_box = ((min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)))


class Car:
    def __init__(self, id: int, config: Config):
        self.id = id
        self.state = CarState()
        self.kalman_filter = KalmanFilterWrapper(process_noise=10, measurement_noise=10)
        self.config = config

    def update(self, location: Tuple[float, float], timestamp: float):
        self.kalman_filter.predict()
        smoothed_location = self.kalman_filter.update(np.array(location))
        self.state.location = (float(smoothed_location[0]), float(smoothed_location[1]))
        self.state.track.append((location, timestamp))

        self._update_speed(timestamp)
        self._update_movement_status()
        self._update_parked_status()

        self.state.last_update_time = timestamp

    def set_stop_score(self, score: int):
        self.state.stop_score = score
        self.state.scored = True

    def _update_speed(self, timestamp: float):
        history_length = min(len(self.state.track), 10)
        if history_length > 1:
            past_positions = np.array([pos for pos, _ in self.state.track[-history_length:]])
            avg_past_position = np.mean(past_positions, axis=0)
            distance = np.linalg.norm(np.array(self.state.location) - avg_past_position)
            time_diff = timestamp - self.state.track[-history_length][1]
            self.state.speed = float(distance / time_diff) if time_diff > 0 else 0.0
        else:
            self.state.speed = 0

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


def is_point_in_rectangle(
    point: Tuple[float, float], rectangle: Tuple[Tuple[float, float], Tuple[float, float]]
) -> bool:
    x, y = point
    (x1, y1), (x2, y2) = rectangle
    return x1 <= x <= x2 and y1 <= y <= y2


def is_crossing_line(
    prev_point: Tuple[float, float],
    current_point: Tuple[float, float],
    line: Tuple[Tuple[float, float], Tuple[float, float]],
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


def update_car_in_stop_zone(car: Car, stop_zone: StopZone, current_time: float):
    prev_location = car.state.track[-2][0] if len(car.state.track) > 1 else car.state.location

    if car.state.stop_zone_state == "APPROACHING":
        if is_crossing_line(prev_location, car.state.location, stop_zone.stop_line):
            car.state.stop_zone_state = "IN_STOP_ZONE"
            car.state.entry_time = current_time

    elif car.state.stop_zone_state == "IN_STOP_ZONE":
        car.state.min_speed_in_zone = min(car.state.min_speed_in_zone, car.state.speed)

        if car.state.speed == 0:
            car.state.time_at_zero += current_time - car.state.last_update_time
            if car.state.stop_position == (0.0, 0.0):
                car.state.stop_position = car.state.location

        if is_crossing_line(prev_location, car.state.location, stop_zone.exit):
            car.state.stop_zone_state = "EXITING"
            car.state.exit_time = current_time

    elif car.state.stop_zone_state == "EXITING":
        if not is_point_in_rectangle(car.state.location, stop_zone.bounding_box):
            car.state.stop_zone_state = "SCORED"
            car.state.stop_score = calculate_stop_score(car, stop_zone, car.config)
            car.state.scored = True

    if car.id == 3:
        print(f"{car.id}: {car.state.stop_zone_state}")


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


def min_distance_to_line(point: Tuple[float, float], line: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
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


def get_speed_at_line_crossing(car: Car, line: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
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


def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def update_car(car: Car, location: Tuple[float, float], current_time: float, stop_zone: StopZone):
    car.update(location, current_time)
    update_car_in_stop_zone(car, stop_zone, current_time)


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
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    stop_zone = StopZone(config)

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

            # print(f"Frame {frame_count} - Timestamp: {timestamp:.2f}s")

            # Crop, Scale, Model the frame
            frame, boxes = process_frame(
                model=model,
                frame=frame,
                scale=config.scale,
                crop_top_ratio=config.crop_top,
                crop_side_ratio=config.crop_side,
                vehicle_classes=config.vehicle_classes,
            )

            # Update or create car objects
            for box in boxes:
                try:
                    track_id = int(box.id.item())
                    x, y, w, h = box.xywh[0]  # type: ignore
                    location = (float(x), float(y))

                    if track_id in cars:
                        car = cars[track_id]
                    else:
                        car = Car(id=track_id, config=config)
                        cars[track_id] = car

                    update_car(car, location, timestamp, stop_zone)
                except Exception:
                    pass

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
