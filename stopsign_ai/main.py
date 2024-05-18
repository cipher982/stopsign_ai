import os

import cv2
import dotenv

dotenv.load_dotenv()

# RTSP stream URL
rtsp_url = os.getenv("RTSP_URL")

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with your AI model
    # For example, you can use a pre-trained YOLO model
    # detections = your_ai_model.detect(frame)

    # Display the frame (optional)
    cv2.imshow("Frame", frame)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()