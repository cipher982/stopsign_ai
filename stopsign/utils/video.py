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


def apply_perspective_transform(frame, src_corners):
    """
    Apply a perspective transform to the frame using source corners.
    :param frame: The image frame to transform.
    :param src_corners: A list of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] representing the source corners.
    """
    # Ensure src_corners has exactly four points
    if len(src_corners) != 4:
        raise ValueError("src_corners must contain exactly four coordinate pairs")

    # Points selected from the original image
    src_points = np.float32(src_corners)

    # Calculate dst_points based on src_corners and desired output aspect ratio
    width = np.linalg.norm(np.array(src_corners[0]) - np.array(src_corners[1]))
    height = np.linalg.norm(np.array(src_corners[0]) - np.array(src_corners[3]))
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])  # type: ignore

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)  # type: ignore

    # Apply the perspective warp
    warped_frame = cv2.warpPerspective(frame, matrix, (int(width), int(height)))

    return warped_frame


def select_points(frame):
    """
    Display the frame and allow the user to select points.
    :param frame: The image frame to select points from.
    :return: A list of selected points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    """
    points = []

    # Function to capture mouse click events
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Points", frame)

    # Display the frame and set the mouse callback function
    cv2.imshow("Select Points", frame)
    cv2.setMouseCallback("Select Points", click_event)

    # Wait until four points are selected
    while len(points) < 4:
        cv2.waitKey(1)

    cv2.destroyWindow("Select Points")
    return points
