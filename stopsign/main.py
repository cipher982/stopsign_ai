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
SAVE_VIDEO = True

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


def signal_handler(sig, frame):
    print("Interrupt signal received. Cleaning up...")
    sys.exit(0)


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

    track_history = defaultdict(lambda: [])

    # virtual stop sign line
    stopsign_line = (650, 450), (500, 500)

    # Begin streaming loop
    print("Streaming...")
    frame_count = 0
    frame_buffer = []
    buffer_size = 5
    prev_frame_time = time.time()
    inference_times = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video file reached.")
                break

            frame = preprocess_frame(frame, scale, crop_top_ratio, crop_side_ratio)

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

            # Run YOLO inference
            start_time = time.time()
            results = model.track(
                source=frame,
                tracker="./trackers/bytetrack.yaml",
                stream=False,
                persist=True,
                classes=vehicle_classes,
            )
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            boxes = results[0].boxes
            if boxes:
                boxes = [obj for obj in boxes if obj.cls in vehicle_classes]
            else:
                boxes = []
            print(f"Frame {frame_count}: {len(boxes)} vehicles detected")

            # Plot the stop sign line
            cv2.line(frame, stopsign_line[0], stopsign_line[1], (0, 0, 255), 2)

            # Update track history
            for box in boxes:
                x, y, w, h = box.xywh[0]  # type: ignore
                track_id = int(box.id.item())
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:
                    track.pop(0)

            # Draw the vehicles IDs and bounding boxes
            frame = results[0].plot()

            # Plot tracking lines
            for track_id, track in track_history.items():
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

            # Draw the gridlines for debugging
            if draw_grid:
                draw_gridlines(frame, grid_increment)

            cv2.imshow("Output", frame)
            cv2.waitKey(1)

            if SAVE_VIDEO:
                print(f"Frame shape: {frame.shape}")
                assert frame.size > 0, "Error: Frame is empty"
                video_writer.write(frame)
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

        if len(inference_times) > 0:
            mean_inference_time = sum(inference_times) / len(inference_times)
            median_inference_time = sorted(inference_times)[len(inference_times) // 2]

            print(f"Mean inference time: {mean_inference_time * 1000:.2f} ms")
            print(f"Median inference time: {median_inference_time * 1000:.2f} ms")


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
