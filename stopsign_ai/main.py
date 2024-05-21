import os
import sys
import time
import signal

import cv2
import dotenv
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics import solutions

dotenv.load_dotenv()

rtsp_url = os.getenv("RTSP_URL")
model_path = os.getenv("YOLO_MODEL_PATH")
output_video_path = os.getenv("OUTPUT_VIDEO_PATH")

if not rtsp_url:
    print("Error: RTSP_URL environment variable is not set.")
    sys.exit(1)

def open_rtsp_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
    return cap

def preprocess_frame(frame, scale, crop_top_ratio, crop_side_ratio):
    h, w = frame.shape[:2]
    resized_w = int(w * scale)
    resized_h = int(h * scale)
    resized_frame = cv2.resize(frame, (resized_w, resized_h))

    crop_top = int(resized_h * crop_top_ratio)
    crop_side = int(resized_w * crop_side_ratio)
    cropped_frame = resized_frame[crop_top:, crop_side:resized_w - crop_side]
    return np.ascontiguousarray(cropped_frame)

def draw_gridlines(frame, grid_increment):
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

def main(draw_grid=False, grid_increment=100, scale=1.0, crop_top_ratio=0.5, crop_side_ratio=1/6):
    global cap, video_writer

    cap = open_rtsp_stream(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        sys.exit()

    model = YOLO(model_path)
    names = model.model.names

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    line_pts = [(600, 800), (700, 600)]

    speed_obj = solutions.SpeedEstimator(
        reg_pts=line_pts,
        names=names,
        view_img=True,
    )

    vehicle_classes = [1, 2, 3, 5, 6, 7]

    signal.signal(signal.SIGINT, signal_handler)

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
                cap = open_rtsp_stream(rtsp_url)
                continue

            processed_frame = preprocess_frame(frame, scale, crop_top_ratio, crop_side_ratio)
            pil_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

            start_time = time.time()
            tracks = model.track(
                source=pil_img,
                stream=False,
                persist=True,
                classes=vehicle_classes
            )
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            if draw_grid:
                draw_gridlines(processed_frame, grid_increment)

            cv2.line(processed_frame, line_pts[0], line_pts[1], (0, 255, 0), 2)

            processed_frame = speed_obj.estimate_speed(processed_frame, tracks)

            video_writer.write(processed_frame)

            frame_count += 1

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        if len(inference_times) > 0:
            mean_inference_time = sum(inference_times) / len(inference_times)
            median_inference_time = sorted(inference_times)[len(inference_times) // 2]

            print(f"Mean inference time: {mean_inference_time * 1000:.2f} ms")
            print(f"Median inference time: {median_inference_time * 1000:.2f} ms")

if __name__ == "__main__":
    main(draw_grid=True, grid_increment=100, scale=0.75)