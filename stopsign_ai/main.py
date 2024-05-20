import os
import sys
import time

import cv2
import dotenv
from PIL import Image
from ultralytics import YOLO
from ultralytics import solutions

dotenv.load_dotenv()

# Constants
rtsp_url = os.getenv("RTSP_URL")
model_path = os.getenv("YOLO_MODEL_PATH")
output_video_path = os.getenv("OUTPUT_VIDEO_PATH")


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


def main(draw_grid=False, grid_increment=100):
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

    # Video writer
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Define line points for speed estimation
    line_pts = [(900, 700), (1200, 600)]

    # Initialize speed estimation object
    speed_obj = solutions.SpeedEstimator(
        reg_pts=line_pts,
        names=names,
        view_img=True,
    )

    # vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "train", "truck"]
    vehicle_classes = [1, 2, 3, 5, 6, 7]  # Indices of vehicle classes

    # Main loop
    frame_count = 0
    inference_times = []  # Store inference times

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
            continue

        ret, frame = cap.read()
        if not ret:
            print("Stream read failed, attempting to reconnect...")
            cap.release()
            cap = open_rtsp_stream(rtsp_url)
            continue

        # Convert frame to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform tracking and measure inference time
        start_time = time.time()
        tracks = model.track(source=pil_img, persist=True, classes=vehicle_classes)
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        # Estimate speed and draw on frame
        frame = speed_obj.estimate_speed(frame, tracks)

        # Write the frame to the output video
        video_writer.write(frame)

        if draw_grid:
            draw_gridlines(frame, w, h, grid_increment)

        # Draw the current line_pts on the frame
        cv2.line(frame, line_pts[0], line_pts[1], (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Frame", frame)
        frame_count += 1

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture, video writer, and close any OpenCV windows
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Calculate and print mean and median inference times
    mean_inference_time = sum(inference_times) / len(inference_times)
    median_inference_time = sorted(inference_times)[len(inference_times) // 2]

    print(f"Mean inference time: {mean_inference_time * 1000:.2f} ms")
    print(f"Median inference time: {median_inference_time * 1000:.2f} ms")


if __name__ == "__main__":
    main(draw_grid=True, grid_increment=100)
