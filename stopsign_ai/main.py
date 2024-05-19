import os
import sys
import time

import cv2
import dotenv
from PIL import Image
from ultralytics import YOLO

dotenv.load_dotenv()

# Constants
rtsp_url = os.getenv("RTSP_URL")
model_path = os.getenv("YOLO_MODEL_PATH")


# Function to open the RTSP stream
def open_rtsp_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Set buffer size
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)  # Set timeout to 60 seconds
    return cap


# Open the RTSP stream
cap = open_rtsp_stream(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    sys.exit()

# Load the YOLO model
model = YOLO(model_path)

# Main loop
frame_count = 0
last_boxes = []  # Store the last set of bounding boxes
inference_times = []  # Store inference times

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream read failed, attempting to reconnect...")
        cap.release()
        cap = open_rtsp_stream(rtsp_url)
        continue

    # Run AI model every 4th frame
    if frame_count % 2 == 0:
        # Convert frame to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform detection and measure inference time
        start_time = time.time()
        results = model.predict(source=pil_img)
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        last_boxes = []  # Clear the last boxes
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates and class label
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]

                # Filter objects to track only vehicles
                if label in ["car", "motorcycle", "bus", "truck"]:
                    last_boxes.append((x1, y1, x2, y2, label))

    # Draw the last set of bounding boxes
    for x1, y1, x2, y2, label in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    frame_count += 1

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Calculate and print mean and median inference times
mean_inference_time = sum(inference_times) / len(inference_times)
median_inference_time = sorted(inference_times)[len(inference_times) // 2]

print(f"Mean inference time: {mean_inference_time * 1000:.2f} ms")
print(f"Median inference time: {median_inference_time * 1000:.2f} ms")
