import sys

import cv2
import numpy as np


def signal_handler(sig, frame):
    print("Interrupt signal received. Cleaning up...")
    sys.exit(0)


def open_rtsp_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
    return cap


def crop_scale_frame(frame: np.ndarray, scale: float, crop_top_ratio: float, crop_side_ratio: float) -> np.ndarray:
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


def draw_box(frame, car, box, color=(0, 255, 0), thickness=2) -> None:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"{int(box.id.item())}: {car.speed:.1f} px/s, ({int(box.conf.item() * 100)}%)"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
