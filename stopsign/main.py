import argparse
import os
import signal
import sys
import time

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

        self.scale = config["video_processing"]["scale"]
        self.crop_top_ratio = config["video_processing"]["crop_top_ratio"]
        self.crop_side_ratio = config["video_processing"]["crop_side_ratio"]
        self.buffer_size = config["video_processing"]["buffer_size"]

        self.stopsign_line = tuple(tuple(i) for i in config["stopsign_detection"]["stopsign_line"])
        self.stop_box_tolerance = config["stopsign_detection"]["stop_box_tolerance"]
        self.min_stop_duration = config["stopsign_detection"]["min_stop_duration"]
        self.movement_allowance = config["stopsign_detection"]["movement_allowance"]
        self.frames_before_parked = config["stopsign_detection"]["frames_before_parked"]

        self.save_video = config["output"]["save_video"]

        self.draw_grid = config["debugging_visualization"]["draw_grid"]
        self.grid_increment = config["debugging_visualization"]["grid_increment"]

        self.fps = config["stream_settings"]["fps"]
        self.vehicle_classes = config["stream_settings"]["vehicle_classes"]


class Car:
    def __init__(
        self,
        id: int,
        config: Config,
    ):
        self.id = id
        self.location = (0, 0)
        self.speed = 0
        self.is_parked = False
        self.frames_parked = 0
        self.track = []  # Store track history
        self.movement_allowance = config.movement_allowance
        self.frames_before_parked = config.frames_before_parked

        # Initialize Kalman filter
        self.kalman_filter = KalmanFilterWrapper()

    def update(self, location: tuple, speed: float):
        self.kalman_filter.predict()
        self.location = self.kalman_filter.update(location)
        self.speed = speed
        self.track.append(tuple(self.location))  # Update track history

        if self.speed < self.movement_allowance:
            self.frames_parked += 1
        else:
            self.frames_parked = 0
            self.is_parked = False  # Reset parked status if the car moves

        if self.frames_parked >= self.frames_before_parked:
            self.is_parked = True

        pass

    def __repr__(self):
        x, y = self.location
        return f"Car {self.id} @ ({x:.2f}, {y:.2f}) (Speed: {self.speed:.1f}px/s, Parked: {self.is_parked})"


class Stopsign:
    def __init__(
        self,
        stopsign_line: tuple,
        stop_box_tolerance: int,
        min_stop_duration: int,
    ):
        # Some constants for stop sign detection
        self.stopsign_line = stopsign_line
        self.stop_box_tolerance = stop_box_tolerance  # pixels
        self.min_stop_duration = min_stop_duration  # seconds

        # Calculate stop box coordinates
        left_x = min(self.stopsign_line[0][0], self.stopsign_line[1][0]) - self.stop_box_tolerance
        right_x = max(self.stopsign_line[0][0], self.stopsign_line[1][0]) + self.stop_box_tolerance
        top_y = min(self.stopsign_line[0][1], self.stopsign_line[1][1])
        bottom_y = max(self.stopsign_line[0][1], self.stopsign_line[1][1])
        self.stop_box = ((left_x, top_y), (right_x, bottom_y))


def stop_score(car: Car, stop_box: tuple, min_stop_frames: int, fps: int) -> int:
    """
    Calculate a stop score for a car based on its behavior at a stop sign.

    Score is determined by an algorithm considering speed and stop duration,
    ranging from 1 (no stop) to 10 (perfect stop).

    Args:
        car (Car): The car object.
        stop_box (tuple): Tuple defining the top-left and bottom-right corners of the stop box.
        min_stop_frames (int): Minimum number of frames stopped to be considered a complete stop.
        fps (int): Frames per second of the video.

    Returns:
        int: A score from 1 to 10 based on the car's stopping behavior.
    """

    left_x, top_y = stop_box[0]
    right_x, bottom_y = stop_box[1]

    # Check if the car is within the stop box
    if left_x <= car.location[0] <= right_x and top_y <= car.location[1] <= bottom_y:
        if car.speed == 0:
            car.frames_parked += 1
        else:
            car.frames_parked = 0

        # Score algorithm:
        # - Full points if stopped for minimum duration
        # - Deductions based on speed (higher speed = lower score)
        # - Minimum score of 1
        stop_duration_score = min(car.frames_parked / min_stop_frames, 1) * 10
        speed_score = max(10 - int(car.speed), 1)  # Assuming speed is in pixels/frame, adjust as needed
        score = int(stop_duration_score * 0.7 + speed_score * 0.3)  # Weighting towards stop duration
        return min(score, 10)  # Cap at 10
    else:
        car.frames_parked = 0
        return 1  # Not at the stop sign yet


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


def visualize(frame, cars, boxes, stopsign_line, n_frame) -> np.ndarray:
    # Plot the stop sign line
    cv2.line(frame, stopsign_line[0], stopsign_line[1], (0, 0, 255), 2)

    # Draw boxes for each car
    for box in boxes:
        car = cars[int(box.id.item())]
        if car.is_parked:
            draw_box(frame, car, box, color=(255, 255, 255), thickness=1)  # parked cars
        else:
            draw_box(frame, car, box, color=(0, 255, 0), thickness=2)  # moving cars

    # Path tracking code
    for id in cars:
        car = cars[id]
        if cars[id].is_parked:
            continue
        points = np.array(car.track, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

    # Display the frame number on image
    cv2.putText(frame, f"Frame: {n_frame}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def main(input_source, config: Config):
    global cap, video_writer

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

    stopsign = Stopsign(
        stopsign_line=config.stopsign_line,
        stop_box_tolerance=config.stop_box_tolerance,
        min_stop_duration=config.min_stop_duration,
    )

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

            print(f"Frame {frame_count} - Timestamp: {timestamp:.2f}s")

            # Crop, Scale, Model the frame
            frame, boxes = process_frame(
                model=model,
                frame=frame,
                scale=config.scale,
                crop_top_ratio=config.crop_top_ratio,
                crop_side_ratio=config.crop_side_ratio,
                vehicle_classes=config.vehicle_classes,
            )

            # Update or create car objects
            for box in boxes:
                track_id = int(box.id.item())
                x, y, w, h = box.xywh[0]  # type: ignore
                location = (float(x), float(y))

                if track_id in cars:
                    car = cars[track_id]
                else:
                    car = Car(id=track_id, config=config)
                    cars[track_id] = car

                # Calculate speed (using car's track history)
                if car.track:
                    previous_point = car.track[-1]
                    distance = np.linalg.norm(np.array(location) - np.array(previous_point))
                    speed = float(distance / timestamp if timestamp > 0 else 0)
                    print(f"Car {track_id} - dist: {distance:.2f}px, speed: {speed:.2f}px/s")
                else:
                    speed = 0.0

                car.update(location, speed)

            # Visualize only non-parked cars
            annotated_frame = visualize(
                frame,
                cars,
                boxes,
                stopsign.stopsign_line,
                frame_count,
            )

            # Draw the gridlines for debugging
            if config.draw_grid:
                draw_gridlines(annotated_frame, config.grid_increment)

            cv2.imshow("Output", annotated_frame)
            cv2.waitKey(1)

            if frame_count == 100:
                print("Pausing...")

            if config.save_video:
                assert frame.size > 0, "Error: Frame is empty"
                video_writer.write(annotated_frame)

            frame_count += 1

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
