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

# Constants
rtsp_url = os.getenv("RTSP_URL")
model_path = os.getenv("YOLO_MODEL_PATH")
output_video_path = os.getenv("OUTPUT_VIDEO_PATH")

if not rtsp_url:
    print("Error: RTSP_URL environment variable is not set.")
    sys.exit(1)

# Function to open the RTSP stream
def open_rtsp_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set buffer size
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)  # Set timeout to 60 seconds
    return cap


def draw_gridlines(frame, w, h, grid_increment):
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


def main(draw_grid=False, grid_increment=100, scale=1.0):
    global cap, video_writer

    # Open the RTSP stream
    cap = open_rtsp_stream(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        sys.exit()

    # Load the YOLO model
    model = YOLO(model_path)
    names = model.model.names

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Resize dimensions
    resized_w = int(w * scale)
    resized_h = int(h * scale)

    # Calculate crop dimensions
    crop_top = resized_h // 2
    crop_left = resized_w // 6
    crop_right = resized_w - crop_left

    # Video writer with new dimensions
    cropped_w = crop_right - crop_left
    cropped_h = resized_h - crop_top
    video_writer = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (cropped_w, cropped_h)
    )

    # Define line points for speed estimation
    line_pts = [(600, 800), (700, 600)]

    # Initialize speed estimation object
    speed_obj = solutions.SpeedEstimator(
        reg_pts=line_pts,
        names=names,
        view_img=True,
    )

    # vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "train", "truck"]
    vehicle_classes = [1, 2, 3, 5, 6, 7]  # Indices of vehicle classes

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Main loop
    frame_count = 0
    inference_times = []  # Store inference times

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

            # Resize the frame
            resized_frame = cv2.resize(frame, (resized_w, resized_h))

            # Crop the frame
            cropped_frame = resized_frame[crop_top:resized_h, crop_left:crop_right]
            cropped_frame = np.ascontiguousarray(cropped_frame)

            # Convert frame to PIL image
            pil_img = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

            # Perform tracking and measure inference time
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
                draw_gridlines(cropped_frame, cropped_w, cropped_h, grid_increment)

            # Draw the current line_pts on the frame
            cv2.line(cropped_frame, line_pts[0], line_pts[1], (0, 255, 0), 2)

            # Estimate speed and draw on frame
            cropped_frame = speed_obj.estimate_speed(cropped_frame, tracks)

            # Write the frame to the output video
            video_writer.write(cropped_frame)

            # Display the frame
            # cv2.imshow("Frame", cropped_frame)
            frame_count += 1

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Cleanup actions
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        # Calculate and print mean and median inference times
        if len(inference_times) > 0:
            mean_inference_time = sum(inference_times) / len(inference_times)
            median_inference_time = sorted(inference_times)[len(inference_times) // 2]

            print(f"Mean inference time: {mean_inference_time * 1000:.2f} ms")
            print(f"Median inference time: {median_inference_time * 1000:.2f} ms")

if __name__ == "__main__":
    main(draw_grid=True, grid_increment=100, scale=0.75)