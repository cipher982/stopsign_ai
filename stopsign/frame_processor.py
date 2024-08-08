import base64
import contextlib
import io
import logging
import os
import threading
import time
from collections import deque
from multiprocessing import Queue
from typing import Dict
from typing import List
from typing import Tuple

import cv2
import dotenv
import numpy as np
from ultralytics import YOLO

from stopsign.config import Config
from stopsign.config import shutdown_flag
from stopsign.tracking import Car
from stopsign.tracking import CarTracker
from stopsign.tracking import StopDetector
from stopsign.tracking import StopZone

# Load environment variables
dotenv.load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
SAMPLE_FILE_PATH = os.getenv("SAMPLE_FILE_PATH")
STREAM_BUFFER_DIR = os.path.join(os.path.dirname(__file__), "tmp_stream_buffer")

car_tracker = None
stop_detector = None
model = None
cap = None
frame_count = 0
frame_buffer = None


logger = logging.getLogger(__name__)


class FrameBuffer:
    def __init__(self, maxsize=30):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()

    def put(self, frame):
        with self.lock:
            self.buffer.append(frame)
        logger.info(f"Frame buffer size: {self.qsize()}")

    def get(self):
        with self.lock:
            if len(self.buffer) > 0:
                return self.buffer.popleft()
        return None

    def qsize(self):
        with self.lock:
            return len(self.buffer)


def crop_scale_frame(frame: np.ndarray, scale: float, crop_top_ratio: float, crop_side_ratio: float) -> np.ndarray:
    height, width = frame.shape[:2]
    crop_top = int(height * crop_top_ratio)
    crop_side = int(width * crop_side_ratio)

    cropped = frame[crop_top:, crop_side : width - crop_side]

    if scale != 1.0:
        new_width = int(cropped.shape[1] * scale)
        new_height = int(cropped.shape[0] * scale)
        return cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return cropped


def draw_box(frame: np.ndarray, car, box, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f"ID: {car.id}, Speed: {car.state.speed:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_gridlines(frame: np.ndarray, grid_size: int):
    height, width = frame.shape[:2]

    for x in range(0, width, grid_size):
        cv2.line(frame, (x, 0), (x, height), (128, 128, 128), 1)

    for y in range(0, height, grid_size):
        cv2.line(frame, (0, y), (width, y), (128, 128, 128), 1)


def open_rtsp_stream(url: str):
    return cv2.VideoCapture(url)


def initialize_video_capture(input_source: str, fps: int):
    global cap
    if input_source == "live":
        if not RTSP_URL:
            logger.error("Error: RTSP_URL environment variable is not set.")
            return None, None, None
        logger.info(f"Opening RTSP stream: {RTSP_URL}")
        cap = open_rtsp_stream(RTSP_URL)
    elif input_source == "file":
        logger.info(f"Opening video file: {SAMPLE_FILE_PATH}")
        cap = cv2.VideoCapture(SAMPLE_FILE_PATH)  # type: ignore
    else:
        logger.error("Error: Invalid input source")
        return None, None, None

    if not cap.isOpened():
        logger.error("Error: Could not open video stream")
        return None, None, None

    cap.set(cv2.CAP_PROP_FPS, fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height


def initialize_components(config: Config):
    global model, car_tracker, stop_detector, frame_buffer

    if not MODEL_PATH:
        logger.error("Error: YOLO_MODEL_PATH environment variable is not set.")
        return False

    model = YOLO(MODEL_PATH, verbose=False)
    logger.info("Model loaded successfully")

    car_tracker = CarTracker(config)
    stop_detector = StopDetector(config)
    frame_buffer = FrameBuffer(maxsize=30)
    return True


def process_frame(frame: np.ndarray, config: Config) -> Tuple[np.ndarray, List]:
    global model
    assert model is not None
    with contextlib.redirect_stdout(io.StringIO()):
        results = model.track(
            source=frame,
            tracker="./trackers/bytetrack.yaml",
            stream=False,
            persist=True,
            classes=config.vehicle_classes,
            verbose=False,
        )

    boxes = results[0].boxes
    if boxes:
        boxes = [obj for obj in boxes if obj.cls in config.vehicle_classes]
    else:
        boxes = []

    return frame, boxes


def visualize(frame, cars: Dict[int, Car], boxes: List, stop_zone: StopZone, n_frame: int) -> np.ndarray:
    overlay = frame.copy()

    car_in_stop_zone = any(
        stop_zone.is_in_stop_zone(car.state.location) for car in cars.values() if not car.state.is_parked
    )

    color = (0, 255, 0) if car_in_stop_zone else (255, 255, 255)

    stop_box_corners = np.array(stop_zone._calculate_stop_box(), dtype=np.int32)
    cv2.fillPoly(overlay, [stop_box_corners], color)
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    start_point = tuple(map(int, stop_zone.stop_line[0]))
    end_point = tuple(map(int, stop_zone.stop_line[1]))
    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

    for box in boxes:
        if box.id is None:
            continue
        try:
            car_id = int(box.id.item())
            if car_id in cars:
                car = cars[car_id]
                if car.state.is_parked:
                    draw_box(frame, car, box, color=(255, 255, 255), thickness=1)
                else:
                    draw_box(frame, car, box, color=(0, 255, 0), thickness=2)
        except Exception as e:
            logger.error(f"Error processing box in visualize function: {str(e)}")

    for car in cars.values():
        if car.state.is_parked:
            continue
        locations = [loc for loc, _ in car.state.track]
        points = np.array(locations, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

    cv2.putText(frame, f"Frame: {n_frame}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def process_and_annotate_frame(frame: np.ndarray, config: Config) -> np.ndarray:
    global frame_count, model, car_tracker, stop_detector
    assert model is not None
    assert car_tracker is not None
    assert stop_detector is not None

    if frame is None or frame.size == 0:
        raise ValueError("Received empty frame")

    frame = crop_scale_frame(frame, config.scale, config.crop_top, config.crop_side)
    processed_frame, boxes = process_frame(frame, config)

    car_tracker.update_cars(boxes, time.time())
    for car in car_tracker.get_cars().values():
        if not car.state.is_parked:
            stop_detector.update_car_stop_status(car, time.time())

    annotated_frame = visualize(
        processed_frame,
        car_tracker.cars,
        boxes,
        stop_detector.stop_zone,
        frame_count,
    )

    if config.draw_grid:
        draw_gridlines(annotated_frame, config.grid_size)

    frame_count += 1
    return annotated_frame


def frame_producer(output_queue: Queue, config: Config):
    global cap
    frame_time = 1 / config.fps
    last_frame_time = time.time()

    logger.info("Starting frame producer")
    while not shutdown_flag.is_set():
        if cap is None or not cap.isOpened():
            break

        current_time = time.time()
        elapsed_time = current_time - last_frame_time

        if elapsed_time >= frame_time:
            ret, frame = cap.read()
            if not ret:
                continue

            processed_frame = process_and_annotate_frame(frame, config)

            _, buffer = cv2.imencode(
                ext=".jpg", img=processed_frame, params=[cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
            )
            encoded_frame = base64.b64encode(buffer).decode("utf-8")

            output_queue.put(encoded_frame)
            last_frame_time = current_time
        else:
            time.sleep(frame_time - elapsed_time)


def cleanup() -> None:
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()


def main(input_source: str, output_queue: Queue, config: Config):
    global cap

    cap, width, height = initialize_video_capture(input_source, config.fps)
    if cap is None:
        raise ValueError("Failed to initialize video capture")

    initialize_components(config)
    logger.info("Components initialized")

    try:
        frame_producer(output_queue, config)
    except Exception as e:
        logger.error(f"Error in frame producer: {str(e)}")
        raise
    finally:
        cleanup()


if __name__ == "__main__":
    # This block is mainly for testing the frame processor independently
    from multiprocessing import Queue

    test_queue = Queue()

    config = Config("./config.yaml")
    main("live", test_queue, config)  # Use "live" for RTSP stream or "file"
