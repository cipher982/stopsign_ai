import base64
import contextlib
import io
import json
import logging
import os
import threading
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import dotenv
import numpy as np
import redis
from ultralytics import YOLO

from stopsign.config import Config
from stopsign.tracking import Car
from stopsign.tracking import CarTracker
from stopsign.tracking import StopDetector
from stopsign.tracking import StopZone

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


MIN_BUFFER_LENGTH = 30
MAX_ERRORS = 100


class StreamProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
        )
        self.model = self.initialize_model()
        self.car_tracker = CarTracker(config)
        self.stop_detector = StopDetector(config)
        self.cap = None
        self.frame_count = 0
        self.frame_buffer_size = self.config.frame_buffer_size
        self.last_processed_time = time.time()
        self.frames_processed = 0
        self.error_count = 0
        self.last_successful_read_time: Optional[float] = None
        self.read_timeout = 3  # 5 seconds timeout for reading a frame
        self.stats_update_interval = 300  # Update stats every 5 minutes
        self._start_stats_update_thread()

    def _start_stats_update_thread(self):
        def update_stats_periodically():
            while True:
                time.sleep(self.stats_update_interval)
                self.stop_detector.db.update_daily_statistics()

        thread = threading.Thread(target=update_stats_periodically, daemon=True)
        thread.start()

    def initialize_model(self):
        model_path = os.getenv("YOLO_MODEL_PATH")
        if not model_path:
            raise ValueError("Error: YOLO_MODEL_PATH environment variable is not set.")
        model = YOLO(model_path, verbose=False)
        logger.info("Model loaded successfully")
        return model

    def initialize_capture(self):
        if self.cap:
            self.cap.release()

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                if self.config.input_source == "live":
                    rtsp_url = os.getenv("RTSP_URL")
                    if not rtsp_url:
                        raise ValueError("Error: RTSP_URL environment variable is not set.")
                    self.cap = cv2.VideoCapture(rtsp_url)
                elif self.config.input_source == "file":
                    sample_file_path = os.getenv("SAMPLE_FILE_PATH")
                    if not sample_file_path:
                        raise ValueError("Error: SAMPLE_FILE_PATH environment variable is not set.")
                    self.cap = cv2.VideoCapture(sample_file_path)
                else:
                    raise ValueError("Error: Invalid input source")

                if not self.cap.isOpened():
                    raise ValueError("Error: Could not open video stream")

                self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                logger.info("Video capture initialized successfully")
                return
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                time.sleep(1)  # Short wait before retrying

        raise ValueError("Failed to initialize video capture after multiple attempts")

    def process_stream(self):
        frame_time = 1 / self.config.fps
        last_frame_time = time.time()

        logger.info("Starting frame processing")
        while True:
            current_time = time.time()
            elapsed_time = current_time - last_frame_time

            if elapsed_time >= frame_time:
                if self.cap is None or (
                    self.last_successful_read_time
                    and (current_time - self.last_successful_read_time > self.read_timeout)
                ):
                    logger.warning("Video capture not initialized or read timeout. Reinitializing...")
                    self.initialize_capture()
                    last_frame_time = time.time()
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame. Reinitializing capture.")
                    self.initialize_capture()
                    last_frame_time = time.time()
                    continue

                self.last_successful_read_time = current_time
                processed_frame, metadata = self.process_frame(frame)
                self.store_frame_data(processed_frame, metadata)

                last_frame_time = current_time
                self.frame_count += 1

                # Update monitoring metrics
                self.frames_processed += 1
                if current_time - self.last_processed_time >= 60:  # Calculate metrics every minute
                    self.log_metrics()
            else:
                time.sleep(frame_time - elapsed_time)

    def log_metrics(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_processed_time
        fps = self.frames_processed / elapsed_time
        buffer_length = self.redis_client.llen("frame_buffer")
        logger.info(f"Processing rate: {fps:.2f} fps, Buffer length: {buffer_length}")

        if buffer_length < MIN_BUFFER_LENGTH:  # type: ignore
            logger.warning(f"Buffer length ({buffer_length}) below minimum threshold ({MIN_BUFFER_LENGTH}).")

        self.frames_processed = 0
        self.last_processed_time = current_time

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        frame = self.crop_scale_frame(frame)
        processed_frame, boxes = self.detect_objects(frame)

        self.car_tracker.update_cars(boxes, time.time())
        for car in self.car_tracker.get_cars().values():
            if not car.state.is_parked:
                self.stop_detector.update_car_stop_status(car, time.time(), frame)

        annotated_frame = self.visualize(
            processed_frame,
            self.car_tracker.cars,
            boxes,
            self.stop_detector.stop_zone,
        )

        if self.config.draw_grid:
            self.draw_gridlines(annotated_frame)

        metadata = self.create_metadata()

        return annotated_frame, metadata

    def crop_scale_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        crop_top = int(height * self.config.crop_top)
        crop_side = int(width * self.config.crop_side)

        cropped = frame[crop_top:, crop_side : width - crop_side]

        if self.config.scale != 1.0:
            new_width = int(cropped.shape[1] * self.config.scale)
            new_height = int(cropped.shape[0] * self.config.scale)
            return cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return cropped

    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        with contextlib.redirect_stdout(io.StringIO()):
            results = self.model.track(
                source=frame,
                tracker="./trackers/bytetrack.yaml",
                stream=False,
                persist=True,
                classes=self.config.vehicle_classes,
                verbose=False,
            )

        boxes = results[0].boxes
        if boxes:
            boxes = [obj for obj in boxes if obj.cls in self.config.vehicle_classes]
        else:
            boxes = []

        return frame, boxes

    def visualize(self, frame, cars: Dict[int, Car], boxes: List, stop_zone: StopZone) -> np.ndarray:
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
                        self.draw_box(frame, car, box, color=(255, 255, 255), thickness=1)
                    else:
                        self.draw_box(frame, car, box, color=(0, 255, 0), thickness=2)
            except Exception as e:
                logger.error(f"Error processing box in visualize function: {str(e)}")

        for car in cars.values():
            if car.state.is_parked:
                continue
            locations = [loc for loc, _ in car.state.track]
            points = np.array(locations, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

        cv2.putText(frame, f"Frame: {self.frame_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def draw_box(self, frame: np.ndarray, car, box, color=(0, 255, 0), thickness=2):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"ID: {car.id}, Speed: {car.state.speed:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_gridlines(self, frame: np.ndarray):
        height, width = frame.shape[:2]

        for x in range(0, width, self.config.grid_size):
            cv2.line(frame, (x, 0), (x, height), (128, 128, 128), 1)

        for y in range(0, height, self.config.grid_size):
            cv2.line(frame, (0, y), (width, y), (128, 128, 128), 1)

    def create_metadata(self) -> Dict:
        return {
            "timestamp": time.time(),
            "frame_count": self.frame_count,
            "cars": [
                {
                    "id": car.id,
                    "location": car.state.location,
                    "speed": car.state.speed,
                    "is_parked": car.state.is_parked,
                    "stop_score": car.state.stop_score,
                    "stop_zone_state": car.state.stop_zone_state,
                }
                for car in self.car_tracker.get_cars().values()
            ],
        }

    def store_frame_data(self, frame: np.ndarray, metadata: Dict):
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        encoded_frame = base64.b64encode(buffer).decode("utf-8")

        timestamp = time.time()
        frame_data = json.dumps({"frame": encoded_frame, "metadata": metadata, "timestamp": timestamp})

        pipeline = self.redis_client.pipeline()
        pipeline.lpush("frame_buffer", frame_data)
        pipeline.ltrim("frame_buffer", 0, self.frame_buffer_size - 1)
        pipeline.execute()

        logger.debug(f"Frame {self.frame_count} stored in Redis")

        # Monitor buffer length after each store operation
        buffer_length = self.redis_client.llen("frame_buffer")
        if buffer_length < MIN_BUFFER_LENGTH:  # type: ignore
            logger.warning(f"Buffer ({buffer_length}) below threshold ({MIN_BUFFER_LENGTH})")

    def run(self):
        while True:
            try:
                self.initialize_capture()
                self.process_stream()
            except Exception as e:
                logger.error(f"Error in stream processor: {str(e)}")
                time.sleep(1)  # Short wait before restarting
            finally:
                if self.cap:
                    self.cap.release()
                cv2.destroyAllWindows()
                logger.info("Stream processor restarting...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.DEBUG)

    config = Config("./config.yaml")
    processor = StreamProcessor(config)
    processor.run()
