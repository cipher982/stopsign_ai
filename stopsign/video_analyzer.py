import contextlib
import io
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import psutil
import pytz
import redis
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import start_http_server
from redis import Redis
from redis import exceptions as redis_exceptions
from ultralytics import YOLO

from stopsign.config import Config
from stopsign.coordinate_transform import Resolution
from stopsign.database import Database
from stopsign.tracking import Car
from stopsign.tracking import CarTracker
from stopsign.tracking import StopDetector

# Load environment variables
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_env(key: str) -> str:
    value = os.getenv(key)
    assert value is not None, f"{key} is not set"
    logger.info(f"Loaded env var {key}: {value}")
    return value


MIN_BUFFER_LENGTH = 30
MAX_ERRORS = 100
PROMETHEUS_PORT: int = int(get_env("PROMETHEUS_PORT"))
REDIS_URL = get_env("REDIS_URL")
RAW_FRAME_KEY = get_env("RAW_FRAME_KEY")
PROCESSED_FRAME_KEY = get_env("PROCESSED_FRAME_KEY")
DB_URL = get_env("DB_URL")
YOLO_MODEL_PATH = os.path.join("/app/models", get_env("YOLO_MODEL_NAME"))


class VideoAnalyzer:
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.redis_client: Redis = redis.from_url(REDIS_URL)
        self.initialize_metrics()
        self.model = self.initialize_model()
        self.car_tracker = CarTracker(config, self.db)
        self.stop_detector = StopDetector(config, db)
        self.frame_rate = 15
        self.frame_count = 0
        self.fps_frame_count = 0
        self.frame_buffer_size = self.config.frame_buffer_size
        self.last_processed_time = time.time()
        self.last_fps_update = time.time()
        self.stats_update_interval = 300  # Update stats every 5 minutes
        self.stats_queue = queue.Queue()
        self.frame_dimensions = self.get_frame_dimensions()

        # Initialize coordinate system tracking
        self.current_stream_resolution = None
        self.coordinate_transformer = None
        self._update_coordinate_system()

        # Initialize FPS counters for each stage
        self.incoming_fps_count = 0
        self.object_detection_fps_count = 0
        self.car_tracking_fps_count = 0
        self.visualization_fps_count = 0
        self.output_fps_count = 0

        # Set the interval for FPS logging (e.g., every 5 seconds)
        self.fps_log_interval = 5  # seconds
        self.last_fps_log_time = time.time()

        self._start_stats_update_thread()
        self._start_fps_logging_thread()  # Start FPS logging thread
        start_http_server(PROMETHEUS_PORT)

    def _start_stats_update_thread(self):
        def schedule_stats_update():
            while True:
                time.sleep(self.stats_update_interval)
                self.stats_queue.put("update_stats")

        thread = threading.Thread(target=schedule_stats_update, daemon=True)
        thread.start()

    def _start_fps_logging_thread(self):
        def log_fps():
            while True:
                time.sleep(self.fps_log_interval)
                current_time = time.time()
                elapsed = current_time - self.last_fps_log_time
                self.last_fps_log_time = current_time

                incoming_fps = self.incoming_fps_count / elapsed
                object_detection_fps = self.object_detection_fps_count / elapsed
                car_tracking_fps = self.car_tracking_fps_count / elapsed
                visualization_fps = self.visualization_fps_count / elapsed
                output_fps = self.output_fps_count / elapsed

                # Update Prometheus metrics
                self.analyzer_incoming_fps.set(incoming_fps)
                self.object_detection_fps.set(object_detection_fps)
                self.car_tracking_fps.set(car_tracking_fps)
                self.visualization_fps.set(visualization_fps)
                self.analyzer_output_fps.set(output_fps)

                # Reset counters after logging
                self.incoming_fps_count = 0
                self.object_detection_fps_count = 0
                self.car_tracking_fps_count = 0
                self.visualization_fps_count = 0
                self.output_fps_count = 0

        thread = threading.Thread(target=log_fps, daemon=True)
        thread.start()

    def initialize_metrics(self):
        # Frame processing metrics
        self.frames_processed = Counter("frames_processed", "Number of frames processed")
        self.frame_processing_time = Histogram("frame_processing_time_seconds", "Time taken to process each frame")
        self.total_frame_time = Gauge("total_frame_time_seconds", "Total time taken for each frame")
        self.fps = Gauge("fps", "Frames processed per second")
        self.processed_buffer_size = Gauge("processed_frame_buffer_size", "Current size of the processed frame buffer")

        # Error and performance metrics
        self.exception_counter = Counter("exceptions_total", "Total number of exceptions", ["type", "method"])
        self.current_memory_usage = Gauge("current_memory_usage_bytes", "Current memory usage of the process")
        self.current_cpu_usage = Gauge("current_cpu_usage_percent", "Current CPU usage percentage of the process")

        # Redis metrics
        self.redis_op_latency = Histogram("redis_op_latency_sec", "Latency of Redis operations in seconds")

        # Model metrics
        self.model_inference_latency = Histogram("model_inference_latency_seconds", "Model latency")

        # Object tracking metrics
        self.cars_tracked = Gauge("cars_tracked", "Number of cars currently being tracked")
        self.cars_in_stop_zone = Gauge("cars_in_stop_zone", "Number of cars in the stop zone")

        # System metrics
        self.cpu_package_temp = Gauge("cpu_package_temp", "CPU package temperature")
        self.cpu_core_temp_avg = Gauge("cpu_core_temp_avg", "Average CPU core temperature")
        self.nvme_temp_avg = Gauge("nvme_temp_avg", "Average NVMe temperature")
        self.acpitz_temp = Gauge("acpitz_temp", "ACPI thermal zone temperature")

        # Image metrics
        self.avg_brightness = Gauge("avg_frame_brightness", "Average brightness of processed frames")
        self.contrast = Gauge("frame_contrast", "Contrast of processed frames")

        # Timing specific metrics
        self.visualization_time = Histogram("visualization_time_seconds", "Time taken to visualize the frame")
        self.object_detection_time = Histogram("object_detection_time_seconds", "Time taken for object detection")
        self.car_tracking_time = Histogram("car_tracking_time_seconds", "Time taken to update car tracking")
        self.stop_detection_time = Histogram("stop_detection_time_seconds", "Time to update stop detection")
        self.metadata_creation_time = Histogram("metadata_creation_time_seconds", "Time taken to create metadata")
        self.frame_encoding_time = Histogram("frame_encoding_time_seconds", "Time taken to encode the frame")

        # FPS metrics
        self.analyzer_incoming_fps = Gauge("analyzer_incoming_fps", "Analyzer incoming frames per second")
        self.object_detection_fps = Gauge("object_detection_fps", "Object detection frames per second")
        self.car_tracking_fps = Gauge("car_tracking_fps", "Car tracking frames per second")
        self.visualization_fps = Gauge("visualization_fps", "Visualization frames per second")
        self.analyzer_output_fps = Gauge("analyzer_output_fps", "Analyzed to Redis buffer frames per second")

    def update_temp_metrics(self):
        temps = psutil.sensors_temperatures()

        if "coretemp" in temps:
            cpu_temps = temps["coretemp"]
            self.cpu_package_temp.set(cpu_temps[0].current)  # Package id 0
            core_temps = [t.current for t in cpu_temps[1:]]  # All core temperatures
            if core_temps:
                self.cpu_core_temp_avg.set(sum(core_temps) / len(core_temps))

        if "nvme" in temps:
            nvme_temps = [t.current for t in temps["nvme"] if t.label == "Composite"]
            if nvme_temps:
                self.nvme_temp_avg.set(sum(nvme_temps) / len(nvme_temps))

        if "acpitz" in temps:
            if temps["acpitz"]:
                self.acpitz_temp.set(temps["acpitz"][0].current)

    def increment_exception_counter(self, exception_type: str, method: str):
        self.exception_counter.labels(type=exception_type, method=method).inc()

    def initialize_model(self):
        """Load YOLO on the best available device.

        Priority:
        1. YOLO_DEVICE env var ("cpu" / "cuda" / "cuda:0", etc.)
        2. Auto-detect CUDA, otherwise fall back to CPU.
        This lets the same codebase run on a GPU server and on a
        CUDA-less laptop without edits.
        """

        logger.info("Loading YOLO model from %s", YOLO_MODEL_PATH)

        # Resolve device --------------------------------------------------
        forced = os.getenv("YOLO_DEVICE")
        if forced:
            device = forced
            logger.info("YOLO_DEVICE override detected: %s", device)
        else:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        # Instantiate model ----------------------------------------------
        model = YOLO(YOLO_MODEL_PATH, task="detect").to(device)
        logger.info("Model ready on device: %s", device)
        return model

    def get_frame_from_redis(self, key: str) -> Optional[np.ndarray]:
        try:
            # Use BLPOP to block until a frame is available or timeout after 1 second
            frame_data = self.redis_client.blpop([key], timeout=1)
            if frame_data:
                _, data = frame_data  # type: ignore
                nparr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.error("Failed to decode frame data")
                return frame
            else:
                return None
        except Exception as e:
            logger.error(f"Error retrieving frame from Redis: {str(e)}")
            return None

    def process_stream(self):
        logger.info("Starting frame processing")
        self.last_fps_update = time.time()
        self.start_time = time.time()

        logger.info(
            f"Connecting to Redis at {self.redis_client.connection_pool.connection_kwargs['host']}:"
            f"{self.redis_client.connection_pool.connection_kwargs['port']}"
        )

        try:
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except redis_exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            return

        last_usage_update = time.time()
        usage_update_interval = 60

        while True:
            try:
                frame = self.get_frame_from_redis(RAW_FRAME_KEY)

                if frame is None:
                    logger.warning("No frame available in Redis. Waiting...")
                    time.sleep(1)
                    continue

                frame_start_time = time.time()

                # Increment incoming FPS counter
                self.incoming_fps_count += 1

                # Timing for frame processing
                start_time = time.time()
                processed_frame, metadata = self.process_frame(frame)
                processing_time = time.time() - start_time
                self.frame_processing_time.observe(processing_time)

                self.store_frame_data(processed_frame, metadata)

                self.processed_buffer_size.set(self.redis_client.llen(PROCESSED_FRAME_KEY))  # type: ignore

                # Increment output FPS counter
                self.output_fps_count += 1

                self.frame_count += 1
                self.fps_frame_count += 1
                self.frames_processed.inc()

                # Update usage metrics less frequently
                current_time = time.time()
                if current_time - last_usage_update >= usage_update_interval:
                    self.update_usage_metrics()
                    last_usage_update = current_time

                while not self.stats_queue.empty():
                    _ = self.stats_queue.get()

                total_frame_time = time.time() - frame_start_time
                self.total_frame_time.set(total_frame_time)

            except Exception as e:
                logger.error(f"Error in stream processing: {str(e)}")
                self.increment_exception_counter(type(e).__name__, "process_stream")
                time.sleep(1)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        start_time = time.time()

        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)  # type: ignore
        contrast = np.std(gray)  # type: ignore
        self.avg_brightness.set(float(avg_brightness))
        self.contrast.set(float(contrast))

        frame = self.crop_scale_frame(frame)

        # Object Detection
        object_detection_start = time.time()
        processed_frame, boxes = self.detect_objects(frame)
        object_detection_time = time.time() - object_detection_start
        self.object_detection_time.observe(object_detection_time)
        self.object_detection_fps_count += 1  # Increment object detection FPS counter

        # Car Tracking
        car_tracking_start = time.time()
        self.car_tracker.update_cars(boxes, time.time(), processed_frame)
        car_tracking_time = time.time() - car_tracking_start
        self.car_tracking_time.observe(car_tracking_time)
        self.car_tracking_fps_count += 1  # Increment car tracking FPS counter

        # Stop Detection
        stop_detection_start = time.time()
        for car in self.car_tracker.get_cars().values():
            if not car.state.is_parked:
                self.stop_detector.update_car_stop_status(car, time.time(), processed_frame)
        stop_detection_time = time.time() - stop_detection_start
        self.stop_detection_time.observe(stop_detection_time)

        # Visualization
        visualization_start = time.time()
        annotated_frame = self.visualize(
            processed_frame,
            self.car_tracker.cars,
            boxes,
            self.stop_detector,
        )
        visualization_time = time.time() - visualization_start
        self.visualization_time.observe(visualization_time)
        self.visualization_fps_count += 1  # Increment visualization FPS counter

        if self.config.draw_grid:
            self.draw_gridlines(annotated_frame)

        # Metadata
        metadata_start = time.time()
        metadata = self.create_metadata()
        self.metadata_creation_time.observe(time.time() - metadata_start)

        self.cars_tracked.set(len(self.car_tracker.get_cars()))
        cars_in_stop_zone = sum(
            1 for car in self.car_tracker.get_cars().values() if car.state.in_stop_zone and not car.state.is_parked
        )
        self.cars_in_stop_zone.set(cars_in_stop_zone)

        # Measure frame processing time
        self.frame_processing_time.observe(time.time() - start_time)

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
        try:
            start_time = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                results = self.model.track(
                    source=frame,
                    tracker="./trackers/bytetrack.yaml",
                    stream=False,
                    persist=True,
                    classes=self.config.vehicle_classes,
                    verbose=False,
                )
            self.model_inference_latency.observe(time.time() - start_time)

            boxes = results[0].boxes
            if boxes:
                boxes = [obj for obj in boxes if obj.cls in self.config.vehicle_classes]
            else:
                boxes = []

            return frame, boxes
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            self.increment_exception_counter(type(e).__name__, "detect_objects")
            raise

    def visualize(self, frame, cars: Dict[int, Car], boxes: List, stop_detector: StopDetector) -> np.ndarray:
        overlay = frame.copy()

        car_in_stop_zone = any(car.state.in_stop_zone for car in cars.values() if not car.state.is_parked)

        # draw stop zone
        color = (0, 255, 0) if car_in_stop_zone else (255, 255, 255)
        stop_box_corners = np.array(stop_detector.stop_zone, dtype=np.int32)
        cv2.fillPoly(overlay, [stop_box_corners], color)
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # draw stop zone midline
        top_mid = (stop_box_corners[0] + stop_box_corners[1]) // 2
        bottom_mid = (stop_box_corners[2] + stop_box_corners[3]) // 2
        cv2.line(frame, tuple(top_mid), tuple(bottom_mid), (0, 0, 255), 2)

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
                self.increment_exception_counter(type(e).__name__, "visualize")

        current_time = time.time()
        for car in cars.values():
            if car.state.is_parked:
                continue
            recent_locations = [(loc, t) for loc, t in car.state.track if current_time - t <= 30]
            if len(recent_locations) > 1:
                points = np.array([loc for loc, _ in recent_locations], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

        cv2.putText(frame, f"Frame: {self.frame_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Add timestamp in top-right corner
        tz = pytz.timezone("America/Chicago")
        utc_dt = datetime.fromtimestamp(time.time(), pytz.UTC)
        local_dt = utc_dt.astimezone(tz)
        current_time = local_dt.strftime("%Y-%m-%d %H:%M:%S")
        text_size, _ = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = frame.shape[1] - text_size[0] - 10
        cv2.putText(frame, current_time, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def draw_box(self, frame: np.ndarray, car, box, color=(0, 255, 0), thickness=2):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # format velocity and direction
        vx, vy = car.state.velocity
        direction = car.state.direction

        label1 = f"ID: {car.id}, Speed: {car.state.speed:.1f}"
        label2 = f"Vel: {int(vx)}, {int(vy)}, Dir: {direction:.2f}"
        cv2.putText(frame, label1, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, label2, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
                    "time_in_zone": car.state.time_in_zone,
                }
                for car in self.car_tracker.get_cars().values()
            ],
        }

    def store_frame_data(self, frame: np.ndarray, metadata: Dict):
        # Ensure the frame is in BGR24 format
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be in BGR24 format (3-channel uint8 numpy array)")

        # Encode the frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        raw_frame_data = buffer.tobytes()

        start_time = time.time()
        pipeline = self.redis_client.pipeline()

        # Store processed frame data
        pipeline.lpush(PROCESSED_FRAME_KEY, raw_frame_data)
        pipeline.ltrim(PROCESSED_FRAME_KEY, 0, self.frame_buffer_size - 1)

        # Store metadata separately
        if metadata:
            metadata_key = f"frame_metadata:{self.frame_count}"
            pipeline.set(metadata_key, json.dumps(metadata))
            pipeline.expire(metadata_key, 300)  # expire after 5 minutes, adjust as needed

        pipeline.execute()
        self.redis_op_latency.observe(time.time() - start_time)

        logger.debug(f"Processed frame {self.frame_count} stored in Redis")
        self.frame_count += 1

    def get_frame_dimensions(self):
        # Attempt to get frame dimensions from Redis
        time.sleep(3)
        max_attempts = 3
        for _ in range(max_attempts):
            frame = self.get_frame_from_redis(RAW_FRAME_KEY)
            if frame is not None:
                return frame.shape[1], frame.shape[0]  # width, height
            time.sleep(0.5)
        # Fallback to default if unable to retrieve from Redis
        logger.warning("Unable to determine frame dimensions from Redis. Using default 1920x1080.")
        return 1920, 1080

    def update_usage_metrics(self):
        process = psutil.Process(os.getpid())
        self.current_memory_usage.set(process.memory_info().rss)
        self.current_cpu_usage.set(process.cpu_percent())
        self.update_temp_metrics()

    def _update_coordinate_system(self):
        """Update coordinate system based on current processed frame dimensions."""
        if hasattr(self, "frame_dimensions") and self.frame_dimensions:
            width, height = self.frame_dimensions
            raw_resolution = Resolution(width, height)

            # Calculate processed resolution after crop/scale
            crop_width = int(width * (1.0 - 2 * self.config.crop_side))
            crop_height = int(height * (1.0 - self.config.crop_top))
            cropped_resolution = Resolution(crop_width, crop_height)

            scaled_width = int(crop_width * self.config.scale)
            scaled_height = int(crop_height * self.config.scale)
            scaled_resolution = Resolution(scaled_width, scaled_height)

            # Stream resolution starts as scaled, may be updated by FFmpeg service
            self.current_stream_resolution = scaled_resolution

            logger.info(f"Coordinate system updated: {raw_resolution} → {scaled_resolution}")

    def get_coordinate_info(self) -> Optional[Dict]:
        """Get current coordinate system information for API consumption."""
        if not hasattr(self, "frame_dimensions") or not self.frame_dimensions:
            return None

        width, height = self.frame_dimensions

        # Calculate all coordinate system dimensions
        raw_resolution = Resolution(width, height)

        crop_width = int(width * (1.0 - 2 * self.config.crop_side))
        crop_height = int(height * (1.0 - self.config.crop_top))
        cropped_resolution = Resolution(crop_width, crop_height)

        scaled_width = int(crop_width * self.config.scale)
        scaled_height = int(crop_height * self.config.scale)
        scaled_resolution = Resolution(scaled_width, scaled_height)

        # Current stream resolution (may differ from scaled if FFmpeg resizes)
        stream_resolution = self.current_stream_resolution or scaled_resolution

        return {
            "raw_resolution": {"width": raw_resolution.width, "height": raw_resolution.height},
            "cropped_resolution": {"width": cropped_resolution.width, "height": cropped_resolution.height},
            "scaled_resolution": {"width": scaled_resolution.width, "height": scaled_resolution.height},
            "stream_resolution": {"width": stream_resolution.width, "height": stream_resolution.height},
            "transform_parameters": {
                "crop_top": self.config.crop_top,
                "crop_side": self.config.crop_side,
                "scale_factor": self.config.scale,
            },
            "current_stop_line": {"coordinates": list(self.config.stop_line), "coordinate_system": "scaled_resolution"},
        }

    def update_stream_resolution(self, width: int, height: int):
        """Update the actual stream resolution (called from FFmpeg service)."""
        self.current_stream_resolution = Resolution(width, height)
        logger.info(f"Stream resolution updated to: {width}x{height}")

    def run(self):
        while True:
            try:
                self.process_stream()
            except Exception as e:
                logger.error(f"Error in stream processor: {str(e)}")
                self.increment_exception_counter(type(e).__name__, "run")
                time.sleep(1)
            finally:
                logger.info("Stream processor restarting...")


if __name__ == "__main__":
    config = Config("./config.yaml")
    db = Database(db_url=DB_URL)
    processor = VideoAnalyzer(config, db)
    processor.run()
