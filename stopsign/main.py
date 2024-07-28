import argparse
import asyncio
import base64
import contextlib
import io
import logging
import os
import signal
import sys
import time
from typing import Dict
from typing import List

import cv2
import dotenv
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

from stopsign.config import Config
from stopsign.tracking import Car
from stopsign.tracking import CarTracker
from stopsign.tracking import StopDetector
from stopsign.tracking import StopZone
from stopsign.utils.video import crop_scale_frame
from stopsign.utils.video import draw_box
from stopsign.utils.video import draw_gridlines
from stopsign.utils.video import open_rtsp_stream

# Set environment variables
dotenv.load_dotenv()
RTSP_URL = os.getenv("RTSP_URL")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
SAMPLE_FILE_PATH = os.getenv("SAMPLE_FILE_PATH")
STREAM_BUFFER_DIR = os.path.join(os.path.dirname(__file__), "tmp_stream_buffer")

# Create FastAPI app
app = FastAPI()

# Global flag to signal shutdown
shutdown_event = asyncio.Event()

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.warning("Skipping box without ID in visualize function")
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


async def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # Setup signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig, server)))

    try:
        await server.serve()
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        await server.shutdown()


@app.get("/")
async def get():
    return HTMLResponse(content=open("stopsign/index.html", "r").read())


def process_frame_task():
    return process_and_annotate_frame()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_counter = 0
    try:
        while not shutdown_event.is_set():
            try:
                result = await asyncio.wait_for(asyncio.to_thread(process_frame_task), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if result is None:
                if not attempt_reconnection():
                    break
                continue

            frame_counter += 1
            if frame_counter % config.frame_skip == 0:
                await websocket.send_text(result)
                print(f"Sent frame {frame_counter}")

            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        print("WebSocket connection closed")
    finally:
        await websocket.close()


def initialize_video_capture(input_source):
    if input_source == "live":
        if not RTSP_URL:
            print("Error: RTSP_URL environment variable is not set.")
            sys.exit(1)
        print(f"Opening RTSP stream: {RTSP_URL}")
        cap = open_rtsp_stream(RTSP_URL)
        cap.set(cv2.CAP_PROP_FPS, config.fps)  # Set FPS here
        return cap
    elif input_source == "file":
        print(f"Opening video file: {SAMPLE_FILE_PATH}")
        cap = cv2.VideoCapture(SAMPLE_FILE_PATH)  # type: ignore
        cap.set(cv2.CAP_PROP_FPS, config.fps)  # Set FPS here
        return cap
    else:
        print("Error: Invalid input source")
        sys.exit(1)


def initialize_components(config: Config) -> None:
    global model, car_tracker, stop_detector

    if not MODEL_PATH:
        print("Error: YOLO_MODEL_PATH environment variable is not set.")
        sys.exit(1)
    model = YOLO(MODEL_PATH, verbose=False)
    print("Model loaded successfully")

    car_tracker = CarTracker(config)
    stop_detector = StopDetector(config)


def process_and_annotate_frame():
    global frame_count, cap, previous_frame

    if not cap or not cap.isOpened():
        cap = initialize_video_capture("live")
    ret, frame = cap.read()
    if not ret:
        logger.warning("Failed to read frame. Attempting to reconnect...")
        cap.release()
        cap = initialize_video_capture("live")
        return None

    frame, boxes = process_frame(
        model=model,
        frame=frame,
        scale=config.scale,
        crop_top_ratio=config.crop_top,
        crop_side_ratio=config.crop_side,
        vehicle_classes=config.vehicle_classes,
    )

    car_tracker.update_cars(boxes, time.time())
    for car in car_tracker.get_cars().values():
        if not car.state.is_parked:
            stop_detector.update_car_stop_status(car, time.time())

    annotated_frame = visualize(
        frame,
        car_tracker.cars,
        boxes,
        stop_detector.stop_zone,
        frame_count,
    )

    if config.draw_grid:
        draw_gridlines(annotated_frame, config.grid_size)

    # Ensure the frame is in BGR format
    if len(annotated_frame.shape) == 2:  # If grayscale
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2BGR)
    elif annotated_frame.shape[2] == 4:  # If RGBA
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGBA2BGR)
        print("Converted RGBA to BGR")

    frame_count += 1

    # Always send full frame for the first frame
    if frame_count == 1:
        _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        result = f"full:{jpg_as_text}"
        process_and_annotate_frame.previous_frame = annotated_frame.copy()
    else:
        # Delta encoding for subsequent frames
        frame_diff = cv2.absdiff(annotated_frame, process_and_annotate_frame.previous_frame)
        _, buffer = cv2.imencode(".jpg", frame_diff, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        result = f"delta:{jpg_as_text}"
        process_and_annotate_frame.previous_frame = annotated_frame.copy()

    return result


def attempt_reconnection(max_attempts=5, delay=5):
    global cap
    for attempt in range(max_attempts):
        logger.info(f"Attempting to reconnect (attempt {attempt + 1}/{max_attempts})...")
        cap.release()
        cap = initialize_video_capture("live")
        if cap.isOpened():
            logger.info("Reconnection successful")
            return True
        time.sleep(delay)
    logger.error("Failed to reconnect after maximum attempts")
    return False


def cleanup():
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()


async def shutdown(sig: signal.Signals, server: uvicorn.Server):
    logger.info(f"Received exit signal {sig.name}...")
    shutdown_event.set()  # Set the shutdown event
    await server.shutdown()
    cleanup()  # Call cleanup function
    sys.exit(0)  # Force exit


async def main_async(input_source: str, config: Config):
    global cap, frame_count

    cap = initialize_video_capture(input_source)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    initialize_components(config)
    frame_count = 0

    try:
        await run_server()
    finally:
        cleanup()


def main(input_source: str, config: Config):
    try:
        asyncio.run(main_async(input_source, config))
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
    finally:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection on live RTSP stream or video file.")
    parser.add_argument(
        "input_source", choices=["live", "file"], help="Input source type (live RTSP stream or video file)"
    )
    args = parser.parse_args()
    config = Config("./config.yaml")
    main(input_source=args.input_source, config=config)
