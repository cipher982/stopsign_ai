import argparse
import os
import signal
import sys
import time
from collections import defaultdict

import cv2
import dotenv
import numpy as np
from ultralytics import YOLO

from stopsign.utils.video import draw_boxes
from stopsign.utils.video import draw_gridlines
from stopsign.utils.video import open_rtsp_stream
from stopsign.utils.video import preprocess_frame
from stopsign.utils.video import signal_handler

dotenv.load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
OUTPUT_VIDEO_DIR = os.getenv("OUTPUT_VIDEO_DIR")
SAMPLE_FILE_PATH = os.getenv("SAMPLE_FILE_PATH")
SAVE_VIDEO = False

os.environ["DISPLAY"] = ":0"

vehicle_classes = [1, 2, 3, 5, 6, 7]

if SAVE_VIDEO:
    output_video_path = str(OUTPUT_VIDEO_DIR) + f"/output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

if not RTSP_URL:
    print("Error: RTSP_URL environment variable is not set.")
    sys.exit(1)

if not MODEL_PATH:
    print("Error: YOLO_MODEL_PATH environment variable is not set.")
    sys.exit(1)


class Car:
    def __init__(
        self,
        id: int,
        parked_threshold: int,
        parked_frames_threshold: int,
    ):
        self.id = id
        self.location = (0, 0)
        self.speed = 0
        self.is_parked = False
        self.frames_parked = 0
        self.track = []  # Store track history
        self.parked_threshold = parked_threshold
        self.parked_frames_threshold = parked_frames_threshold

    def update(
        self,
        location: tuple,
        speed: float,
    ):
        self.location = location
        self.speed = speed
        self.track.append(location)  # Update track history

        if self.speed < self.parked_threshold:
            self.frames_parked += 1
        else:
            self.frames_parked = 0

        if self.frames_parked >= self.parked_frames_threshold:
            self.is_parked = True


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
    frame = preprocess_frame(frame, scale, crop_top_ratio, crop_side_ratio)

    # Run YOLO inference
    results = model.track(
        source=frame,
        tracker="./trackers/bytetrack.yaml",
        stream=False,
        persist=True,
        classes=vehicle_classes,
    )

    boxes = results[0].boxes
    if boxes:
        boxes = [obj for obj in boxes if obj.cls in vehicle_classes]
    else:
        boxes = []
    return frame, boxes


def visualize(frame, tracked_cars, track_history, stopsign_line, boxes) -> np.ndarray:
    # Plot the stop sign line
    cv2.line(frame, stopsign_line[0], stopsign_line[1], (0, 0, 255), 2)

    # Filter out parked cars from boxes before plotting
    non_parked_boxes = [box for box in boxes if box.id.item() in tracked_cars]

    # Plot the boxes on the frame
    annotated_frame = draw_boxes(frame, non_parked_boxes)

    # Path tracking code
    for track_id in tracked_cars:
        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)
    return annotated_frame


def main(input_source, draw_grid=False, grid_increment=100, scale=1.0, crop_top_ratio=0.5, crop_side_ratio=1 / 6):
    global cap, video_writer

    if input_source == "live":
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

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if SAVE_VIDEO:
        assert OUTPUT_VIDEO_DIR, "Error: OUTPUT_VIDEO_PATH environment variable is not set."
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        video_writer = cv2.VideoWriter(
            filename=output_video_path,  # type: ignore
            apiPreference=cv2.CAP_FFMPEG,
            fourcc=fourcc,
            fps=fps,
            frameSize=(w, h),
        )

    signal.signal(signal.SIGINT, signal_handler)

    # Create the model
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")

    stopsign_line = ((650, 450), (500, 500))
    parked_threshold = 500  # pixels
    parked_frames_threshold = 150
    # parked_timeout = 100
    # speed_threshold = 2  # pixels per frame
    # exclusion_radius = 50  # pixels
    # parked_buffer_frames = 10  # Buffer period for parked detection

    stop_box_tolerance = 50  # pixels
    min_stop_duration = 2  # seconds

    stopsign = Stopsign(
        stopsign_line=stopsign_line,
        stop_box_tolerance=stop_box_tolerance,
        min_stop_duration=min_stop_duration,
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

            frame, boxes = process_frame(
                model,
                frame,
                scale,
                crop_top_ratio,
                crop_side_ratio,
                vehicle_classes,
            )

            # Update or create car objects
            for box in boxes:
                track_id = int(box.id.item())
                x, y, w, h = box.xywh[0]  # type: ignore
                location = (float(x), float(y))

                if track_id in cars:
                    car = cars[track_id]
                else:
                    car = Car(track_id, parked_threshold, parked_frames_threshold)
                    cars[track_id] = car

                # Calculate speed (using car's track history)
                if car.track:
                    previous_point = car.track[-1]
                    distance = np.linalg.norm(np.array(location) - np.array(previous_point))
                    speed = float(distance / timestamp if timestamp > 0 else 0)
                else:
                    speed = 0.0

                car.update(location, speed)

            # Visualize only non-parked cars
            non_parked_cars = [car for car in cars.values() if not car.is_parked]
            annotated_frame = visualize(
                frame,
                [car.id for car in non_parked_cars],
                {car.id: car.track for car in non_parked_cars},
                stopsign.stopsign_line,
                [box for box in boxes if int(box.id.item()) in [car.id for car in non_parked_cars]],
            )

            # Draw the gridlines for debugging
            if draw_grid:
                draw_gridlines(annotated_frame, grid_increment)

            cv2.imshow("Output", annotated_frame)
            cv2.waitKey(1)

            if SAVE_VIDEO:
                print(f"Frame shape: {frame.shape}")
                assert frame.size > 0, "Error: Frame is empty"
                video_writer.write(annotated_frame)
                print(f"Frame {frame_count} written")

            frame_count += 1

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        cap.release()
        if SAVE_VIDEO:
            print(f"Output video saved to: {output_video_path}")  # type: ignore
            video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection on live RTSP stream or video file.")
    parser.add_argument(
        "input_source", choices=["live", "file"], help="Input source type (live RTSP stream or video file)"
    )
    args = parser.parse_args()

    main(
        input_source=args.input_source,
        draw_grid=True,
        grid_increment=100,
        crop_top_ratio=0,
        crop_side_ratio=0,
        scale=0.75,
    )
