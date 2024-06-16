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


def open_rtsp_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
    return cap


def preprocess_frame(frame: np.ndarray, scale: float, crop_top_ratio: float, crop_side_ratio: float) -> np.ndarray:
    """
    Preprocess the input frame by resizing and cropping.
    I want to just focus on area around the stop sign.
    """
    h, w = frame.shape[:2]
    resized_w = int(w * scale)
    resized_h = int(h * scale)
    resized_frame = cv2.resize(frame, (resized_w, resized_h))

    crop_top = int(resized_h * crop_top_ratio)
    crop_side = int(resized_w * crop_side_ratio)
    cropped_frame = resized_frame[crop_top:, crop_side : resized_w - crop_side]
    return np.ascontiguousarray(cropped_frame)


def draw_gridlines(frame: np.ndarray, grid_increment: int) -> None:
    """
    Draws gridlines on the given frame.
    Helpful for debugging locations for development.
    """
    h, w = frame.shape[:2]
    for x in range(0, w, grid_increment):
        cv2.line(frame, (x, 0), (x, h), (128, 128, 128), 1)
        cv2.putText(frame, str(x), (x + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, str(x), (x + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    for y in range(0, h, grid_increment):
        cv2.line(frame, (0, y), (w, y), (128, 128, 128), 1)
        cv2.putText(frame, str(y), (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, str(y), (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)


def draw_boxes(frame, boxes, color=(0, 255, 0), thickness=2) -> np.ndarray:
    frame_with_boxes = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, thickness)
        label = f"{int(box.id.item())}: {box.conf.item():.2f}"
        cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
    return frame_with_boxes


def signal_handler(sig, frame):
    print("Interrupt signal received. Cleaning up...")
    sys.exit(0)


class CarTracker:
    def __init__(
        self,
        stopsign_line: tuple,
        parked_threshold: int,
        parked_frames_threshold: int,
        parked_timeout: int,
        speed_threshold: int,
        exclusion_radius: int,
        parked_buffer_frames: int,
    ):
        # Parked car detection parameters
        self.parked_threshold = parked_threshold  # pixels
        self.parked_frames_threshold = parked_frames_threshold
        self.parked_timeout = parked_timeout
        self.speed_threshold = speed_threshold  # pixels per frame
        self.exclusion_radius = exclusion_radius  # pixels
        self.parked_buffer_frames = parked_buffer_frames  # Buffer period for parked detection
        self.stopsign_line = stopsign_line

        # Store the track history, previous positions, and parked car information
        self.track_history = defaultdict(lambda: [])
        self.previous_positions = {}
        self.previous_timestamps = {}
        self.parked_cars = {}

    def update(self, boxes, timestamp):
        for box in boxes:
            x, y, w, h = box.xywh[0]  # type: ignore
            track_id = int(box.id.item())
            track = self.track_history[track_id]
            current_point = (float(x), float(y))
            track.append(current_point)  # x, y center point
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Parked car detection
            if track_id in self.previous_positions:
                previous_point = self.previous_positions[track_id]
                previous_timestamp = self.previous_timestamps[track_id]
                distance = np.linalg.norm(np.array(current_point) - np.array(previous_point))
                time_diff = timestamp - previous_timestamp
                speed = distance / time_diff if time_diff > 0 else 0
                sign_distances = [
                    np.linalg.norm(np.array(current_point) - np.array(stop_point)) for stop_point in self.stopsign_line
                ]
                sign_distance = min(sign_distances)  # type: ignore

                if (
                    sign_distance > self.exclusion_radius
                    and distance < self.parked_threshold
                    and speed < self.speed_threshold
                ):
                    if track_id in self.parked_cars:
                        self.parked_cars[track_id]["frames_parked"] += 1
                    else:
                        self.parked_cars[track_id] = {"frames_parked": 1, "timeout": 0}
                else:
                    if track_id in self.parked_cars:
                        self.parked_cars[track_id]["frames_parked"] -= self.parked_buffer_frames
                        if self.parked_cars[track_id]["frames_parked"] <= 0:
                            del self.parked_cars[track_id]

            self.previous_positions[track_id] = current_point
            self.previous_timestamps[track_id] = timestamp

        # Update timeout for parked cars
        for track_id in list(self.parked_cars.keys()):
            parked_car = self.parked_cars[track_id]

            # Check if the car has been parked for the required number of frames
            if parked_car["frames_parked"] >= self.parked_frames_threshold:
                # Increment the timeout counter for the parked car
                parked_car["timeout"] += 1

                # If the timeout exceeds the allowed limit, remove the car from parked cars
                if parked_car["timeout"] > self.parked_timeout:
                    del self.parked_cars[track_id]

    def get_tracked_cars(self):
        return [track_id for track_id in self.track_history if track_id not in self.parked_cars]

    def calculate_stop_duration(
        self,
        track_id: int,
        fps: int,
        stop_box: tuple,
        min_stop_duration: int,
    ) -> float:
        # Pass min_stop_duration as an argument
        left_x, top_y = stop_box[0]
        right_x, bottom_y = stop_box[1]
        # Calculate stop duration and speed
        stop_duration = 0
        stopped_frames = 0
        previous_point = None
        track = self.track_history[track_id]
        for point in track:
            if left_x <= point[0] <= right_x and top_y <= point[1] <= bottom_y:
                stopped_frames += 1
                if stopped_frames >= min_stop_duration * fps:  # Use the passed min_stop_duration
                    stop_duration += 1 / fps
            else:
                stopped_frames = 0

            if previous_point is not None:
                distance = np.linalg.norm(np.array(point) - np.array(previous_point))
                speed = distance / (1 / fps)
                # TODO - implement scoring system
            previous_point = point
        return stop_duration


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


def visualize(frame, tracked_cars, track_history, stopsign_line, boxes):
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
    parked_timeout = 100
    speed_threshold = 2  # pixels per frame
    exclusion_radius = 50  # pixels
    parked_buffer_frames = 10  # Buffer period for parked detection

    parked_tracker = CarTracker(
        stopsign_line=stopsign_line,
        parked_threshold=parked_threshold,
        parked_frames_threshold=parked_frames_threshold,
        parked_timeout=parked_timeout,
        speed_threshold=speed_threshold,
        exclusion_radius=exclusion_radius,
        parked_buffer_frames=parked_buffer_frames,
    )

    stop_box_tolerance = 50  # pixels
    min_stop_duration = 2  # seconds

    stopsign = Stopsign(
        stopsign_line=stopsign_line,
        stop_box_tolerance=stop_box_tolerance,
        min_stop_duration=min_stop_duration,
    )

    # Begin streaming loop
    print("Streaming...")
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

            # Update track history and detect parked cars
            parked_tracker.update(boxes, timestamp)
            tracked_cars = parked_tracker.get_tracked_cars()

            # Calculate stop duration for each car
            for track_id in tracked_cars:
                stop_duration = parked_tracker.calculate_stop_duration(
                    track_id, fps, stopsign.stop_box, stopsign.min_stop_duration
                )  # Pass min_stop_duration here
                if stop_duration >= stopsign.min_stop_duration:
                    print(f"car {track_id} stopped for {stop_duration:.2f} seconds.")
                else:
                    print(f"car {track_id} did not stop completely.")

            annotated_frame = visualize(
                frame,
                tracked_cars,
                parked_tracker.track_history,
                stopsign.stopsign_line,
                boxes,
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
