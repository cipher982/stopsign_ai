import argparse
import os
import signal
import sys
import time

import cv2
import dotenv
import numpy as np
from ultralytics import YOLO

dotenv.load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
OUTPUT_VIDEO_PATH = os.getenv("OUTPUT_VIDEO_PATH")
SAMPLE_FILE_PATH = os.getenv("SAMPLE_FILE_PATH")
SAVE_VIDEO = False

os.environ["DISPLAY"] = ":0"

vehicle_classes = [1, 2, 3, 5, 6, 7]

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
        cv2.putText(frame, str(x), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for y in range(0, h, grid_increment):
        cv2.line(frame, (0, y), (w, y), (128, 128, 128), 1)
        cv2.putText(frame, str(y), (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def signal_handler(sig, frame):
    print("Interrupt signal received. Cleaning up...")
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
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
        assert OUTPUT_VIDEO_PATH, "Error: OUTPUT_VIDEO_PATH environment variable is not set."
        video_writer = cv2.VideoWriter(
            filename=OUTPUT_VIDEO_PATH,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            fps=fps,
            framesize=(w, h),
        )  # type: ignore

    signal.signal(signal.SIGINT, signal_handler)

    # Create the model
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")

    # Begin streaming loop
    print("Streaming...")
    frame_count = 0
    inference_times = []
    try:
        while True:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                continue

            ret, frame = cap.read()
            if not ret:
                print("Stream read failed, attempting to reconnect...")
                cap.release()
                cap = open_rtsp_stream(RTSP_URL)
                continue

            frame = preprocess_frame(frame, scale, crop_top_ratio, crop_side_ratio)

            start_time = time.time()
            results = model.track(
                source=frame,
                stream=False,
                persist=True,
                classes=vehicle_classes,
                show=False,
            )
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            annotated_frame = results[0].plot()

            if draw_grid:
                draw_gridlines(annotated_frame, grid_increment)

            cv2.imshow("Output", frame)
            cv2.waitKey(1)

            if SAVE_VIDEO:
                video_writer.write(frame)  # type: ignore

            frame_count += 1

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        cap.release()
        if SAVE_VIDEO:
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
