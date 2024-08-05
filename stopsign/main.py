import argparse
import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from threading import Thread
from typing import Dict
from typing import List

import cv2
import dotenv
import numpy as np
from fasthtml import Body
from fasthtml import Div
from fasthtml import FastHTML
from fasthtml import Head
from fasthtml import Html
from fasthtml import Img
from fasthtml import Script
from fasthtml import Title
from ultralytics import YOLO

from stopsign.config import Config
from stopsign.tracking import Car
from stopsign.tracking import CarTracker
from stopsign.tracking import StopDetector
from stopsign.tracking import StopZone
from stopsign.utils import crop_scale_frame
from stopsign.utils import draw_box
from stopsign.utils import draw_gridlines
from stopsign.utils import open_rtsp_stream

# Set environment variables
dotenv.load_dotenv()
RTSP_URL = os.getenv("RTSP_URL")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
SAMPLE_FILE_PATH = os.getenv("SAMPLE_FILE_PATH")
STREAM_BUFFER_DIR = os.path.join(os.path.dirname(__file__), "tmp_stream_buffer")

# Initialize config
config = Config("./config.yaml")
FRAME_INTERVAL = 1 / config.fps

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
frame_count = 0
original_width = None
original_height = None
frame_interval = 1 / config.fps

# Start the server
app = FastHTML(ws_hdr=True)


async def frame_loop(send):
    global frame_count
    while not shutdown_flag.is_set():
        try:
            frame = process_frame_task()
            if frame is not None:
                await send(f"data:image/jpeg;base64,{frame}")
                await send(f"Frame {frame_count} sent")
            else:
                logger.info("No frame available")
                await send("No frame available")
        except Exception as e:
            logger.error(f"Error in frame loop: {str(e)}")
            break
        await asyncio.sleep(FRAME_INTERVAL)


async def on_connect(send):
    logger.info("WebSocket connected")
    dimensions = {"type": "dimensions", "width": original_width, "height": original_height}
    await send(json.dumps(dimensions))
    # Start the message loop
    asyncio.create_task(frame_loop(send))


async def on_disconnect():
    logger.info("WebSocket disconnected")


@app.ws("/ws", conn=on_connect, disconn=on_disconnect)
async def ws_handler(websocket):
    # global frame_count
    logger.info("WebSocket handler started")


@app.get("/")  # type: ignore
def home():
    return (
        Title("Stop Sign Compliance"),
        Html(
            Head(
                Script(src="https://unpkg.com/htmx.org@1.9.4"),
                Script("""
                    document.addEventListener('DOMContentLoaded', function() {
                        var socket = new WebSocket('ws://' + window.location.host + '/ws');
                        socket.onopen = function(event) {
                            console.log("WebSocket connection established");
                        };
                        socket.onmessage = function(event) {
                            var data = event.data;
                            if (data.startsWith('data:image')) {
                                document.getElementById('videoFrame').src = data;
                            } else {
                                try {
                                    var message = JSON.parse(data);
                                    if (message.type === 'status') {
                                        document.getElementById('status').innerText = message.content;
                                    } else if (message.type === 'dimensions') {
                                        console.log("Received dimensions:", message);
                                    }
                                } catch (e) {
                                    console.error("Error parsing message:", e);
                                }
                            }
                        };
                        socket.onerror = function(error) {
                            console.error("WebSocket error:", error);
                        };
                    });
                """),
            ),
            Body(
                Img(id="videoFrame", style="max-width: 100%; height: auto; border: 1px solid black;"),
                Div(id="status"),
            ),
        ),
    )


class FrameBuffer:
    def __init__(self, maxsize=30):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def put(self, frame):
        with self.lock:
            self.buffer.append(frame)

    def get(self):
        with self.lock:
            if len(self.buffer) > 0:
                frame = self.buffer.popleft()
                return frame
        return None

    def qsize(self):
        with self.lock:
            return len(self.buffer)


frame_buffer = FrameBuffer()
shutdown_flag = threading.Event()


def frame_producer():
    global cap
    frame_time = 1 / config.fps
    last_frame_time = time.time()
    while not shutdown_flag.is_set():
        if cap is None or not cap.isOpened():
            break

        current_time = time.time()
        elapsed_time = current_time - last_frame_time

        if elapsed_time >= frame_time:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_buffer.put(frame)
            last_frame_time = current_time
        else:
            time.sleep(frame_time - elapsed_time)


def process_frame(
    model: YOLO,
    frame: np.ndarray,
    scale: float,
    crop_top_ratio: float,
    crop_side_ratio: float,
    vehicle_classes: list,
) -> tuple:
    # Initial frame preprocessing
    frame = crop_scale_frame(frame, scale, crop_top_ratio, crop_side_ratio)

    # Run YOLO inference
    with contextlib.redirect_stdout(io.StringIO()):
        results = model.track(
            source=frame,
            tracker="./trackers/bytetrack.yaml",
            stream=False,
            persist=True,
            classes=vehicle_classes,
            verbose=False,
        )

    # Filter out non-vehicle classes
    boxes = results[0].boxes
    if boxes:
        boxes = [obj for obj in boxes if obj.cls in vehicle_classes]
    else:
        boxes = []
    return frame, boxes


def visualize(frame, cars: Dict[int, Car], boxes: List, stop_zone: StopZone, n_frame: int) -> np.ndarray:
    # Create a copy of the frame for the overlay
    overlay = frame.copy()

    # Check if any moving car is inside the stop zone
    car_in_stop_zone = any(
        stop_zone.is_in_stop_zone(car.state.location) for car in cars.values() if not car.state.is_parked
    )

    # Set the color based on whether a moving car is in the stop zone
    color = (0, 255, 0) if car_in_stop_zone else (255, 255, 255)  # Green if car inside, white else

    # Draw the stop box as a semi-transparent rectangle
    stop_box_corners = np.array(stop_zone._calculate_stop_box(), dtype=np.int32)
    cv2.fillPoly(overlay, [stop_box_corners], color)
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Plot the stop sign line
    start_point = tuple(map(int, stop_zone.stop_line[0]))
    end_point = tuple(map(int, stop_zone.stop_line[1]))
    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

    # Draw boxes for each car
    for box in boxes:
        if box.id is None:
            logger.debug("Skipping box without ID in visualize function")
            continue
        try:
            car_id = int(box.id.item())
            if car_id in cars:
                car = cars[car_id]
                if car.state.is_parked:
                    draw_box(frame, car, box, color=(255, 255, 255), thickness=1)  # parked cars
                else:
                    draw_box(frame, car, box, color=(0, 255, 0), thickness=2)  # moving cars
            else:
                logger.warning(f"Car with ID {car_id} not found in cars dictionary")
        except Exception as e:
            logger.error(f"Error processing box in visualize function: {str(e)}")

    # Path tracking code
    for car in cars.values():
        if car.state.is_parked:
            continue
        locations = [loc for loc, _ in car.state.track]  # Extract locations from track
        points = np.array(locations, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

    # Display the frame number on image
    cv2.putText(frame, f"Frame: {n_frame}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def process_frame_task():
    try:
        frame = frame_buffer.get()
        if frame is not None:
            result = process_and_annotate_frame(frame)
            return result
        return None
    except Exception as e:
        logger.error(f"Error in process_frame_task: {str(e)}")
        return None


def initialize_video_capture(input_source):
    global original_width, original_height
    if input_source == "live":
        if not RTSP_URL:
            print("Error: RTSP_URL environment variable is not set.")
            sys.exit(1)
        print(f"Opening RTSP stream: {RTSP_URL}")
        cap = open_rtsp_stream(RTSP_URL)
    elif input_source == "file":
        print(f"Opening video file: {SAMPLE_FILE_PATH}")
        cap = cv2.VideoCapture(SAMPLE_FILE_PATH)  # type: ignore
    else:
        print("Error: Invalid input source")
        sys.exit(1)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS, config.fps)  # Set FPS here
    return cap


def initialize_components(config: Config) -> None:
    global model, car_tracker, stop_detector, frame_count

    if not MODEL_PATH:
        print("Error: YOLO_MODEL_PATH environment variable is not set.")
        sys.exit(1)
    model = YOLO(MODEL_PATH, verbose=False)
    print("Model loaded successfully")

    car_tracker = CarTracker(config)
    stop_detector = StopDetector(config)
    frame_count = 0


def process_and_annotate_frame(frame):
    global frame_count

    def process_frame_wrapper(frame):
        try:
            processed_frame, boxes = process_frame(
                model=model,
                frame=frame,
                scale=config.scale,
                crop_top_ratio=config.crop_top,
                crop_side_ratio=config.crop_side,
                vehicle_classes=config.vehicle_classes,
            )
            return processed_frame, boxes
        except Exception as e:
            logger.error(f"Error in process_frame_wrapper: {str(e)}")
            raise

    def update_tracking(boxes):
        try:
            car_tracker.update_cars(boxes, time.time())
            for car in car_tracker.get_cars().values():
                if not car.state.is_parked:
                    stop_detector.update_car_stop_status(car, time.time())
        except Exception as e:
            logger.error(f"Error in update_tracking: {str(e)}")
            raise

    def create_annotated_frame(processed_frame, boxes):
        try:
            annotated_frame = visualize(
                processed_frame,
                car_tracker.cars,
                boxes,
                stop_detector.stop_zone,
                frame_count,
            )
            if config.draw_grid:
                draw_gridlines(annotated_frame, config.grid_size)
            return ensure_bgr_format(annotated_frame)
        except Exception as e:
            logger.error(f"Error in create_annotated_frame: {str(e)}")
            raise

    def ensure_bgr_format(frame):
        if len(frame.shape) == 2:  # If grayscale
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # If RGBA
            logger.info("Converting RGBA to BGR")
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame

    def encode_frame(frame):
        try:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
            encoded_frame = base64.b64encode(buffer).decode("utf-8")
            return encoded_frame
        except Exception as e:
            logger.error(f"Error in encode_frame: {str(e)}")
            raise

    # Capture frame with timeout
    try:
        raw_frame = frame  # Use the frame passed as an argument
        if raw_frame is None:
            raise ValueError("Failed to capture frame within timeout")
    except Exception as e:
        logger.error(f"Error in frame_buffer.get(): {str(e)}")
        raise

    # Process frame
    try:
        processed_frame, boxes = process_frame_wrapper(raw_frame)
    except Exception as e:
        logger.error(f"Error in process_frame_wrapper: {str(e)}")
        raise

    # Update tracking
    try:
        update_tracking(boxes)
    except Exception as e:
        logger.error(f"Error in update_tracking: {str(e)}")
        raise

    # Annotate frame
    try:
        annotated_frame = create_annotated_frame(processed_frame, boxes)
    except Exception as e:
        logger.error(f"Error in create_annotated_frame: {str(e)}")
        raise

    # Encode frame
    try:
        encoded_frame = encode_frame(annotated_frame)
    except Exception as e:
        logger.error(f"Error in encode_frame: {str(e)}")
        raise

    frame_count += 1
    return encoded_frame


def cleanup():
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()


def main(input_source: str):
    global cap, frame_count

    cap = initialize_video_capture(input_source)
    if not cap.isOpened():
        logger.error("Error: Could not open video stream")
        return

    initialize_components(config)
    frame_count = 0

    # Start the frame producer thread
    producer_thread = Thread(target=frame_producer, daemon=True)
    producer_thread.start()

    # Run the FastHTML server
    import uvicorn

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        shutdown_flag.set()  # Signal the producer thread to stop
        producer_thread.join(timeout=5)  # Wait for the producer thread to finish
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection on live RTSP stream or video file.")
    parser.add_argument(
        "input_source", choices=["live", "file"], help="Input source type (live RTSP stream or video file)"
    )
    args = parser.parse_args()
    main(input_source=args.input_source)
