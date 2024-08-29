import base64
import contextlib
import io
import json
import logging
import os
import queue
import threading
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import psutil
import redis
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import start_http_server
from redis import Redis
from ultralytics import YOLO

from stopsign.config import Config
from stopsign.tracking import Car
from stopsign.tracking import CarTracker
from stopsign.tracking import StopDetector
from stopsign.tracking import StopZone

# Load environment variables
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_BUFFER_LENGTH = 30
MAX_ERRORS = 100
PROMETHEUS_PORT = int(os.environ["PROMETHEUS_PORT"])


class StreamProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.redis_client: Redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
        )
        self.initialize_metrics()
        self.model = self.initialize_model()
        self.car_tracker = CarTracker(config)
        self.stop_detector = StopDetector(config)
        self.frame_count = 0
        self.fps_frame_count = 0
        self.frame_buffer_size = self.config.frame_buffer_size
        self.last_processed_time = time.time()
        self.last_fps_update = time.time()
        self.stats_update_interval = 300  # Update stats every 5 minutes
        self.stats_queue = queue.Queue()
        self._start_stats_update_thread()
        start_http_server(PROMETHEUS_PORT)
        logger.info(f"Prometheus server started on port {PROMETHEUS_PORT}")

    def _start_stats_update_thread(self):
        def schedule_stats_update():
            while True:
                time.sleep(self.stats_update_interval)
                self.stats_queue.put("update_stats")

        thread = threading.Thread(target=schedule_stats_update, daemon=True)
        thread.start()

    def initialize_metrics(self):
        # Frame processing metrics
        self.frames_processed = Counter("frames_processed", "Number of frames processed")
        self.frame_processing_time = Histogram("frame_processing_time_seconds", "Time taken to process each frame")
        self.fps = Gauge("fps", "Frames processed per second")
        self.buffer_size = Gauge("buffer_size", "Current size of the frame buffer")

        # Error and performance metrics
        self.exception_counter = Counter("exceptions_total", "Total number of exceptions", ["type", "method"])
        self.current_memory_usage = Gauge("current_memory_usage_bytes", "Current memory usage of the process")
        self.current_cpu_usage = Gauge("current_cpu_usage_percent", "Current CPU usage percentage of the process")

        # Redis metrics
        self.redis_op_latency = Histogram("redis_op_latency_seconds", "Latency of Redis operations in seconds")

        # Model metrics
        self.model_inference_latency = Histogram("model_inference_latency_seconds", "Model latency")

        # Object tracking metrics
        self.cars_tracked = Gauge("cars_tracked", "Number of cars currently being tracked")
        self.cars_in_stop_zone = Gauge("cars_in_stop_zone", "Number of cars in the stop zone")

        # system metrics
        self.cpu_package_temp = Gauge("cpu_package_temp", "CPU package temperature")
        self.cpu_core_temp_avg = Gauge("cpu_core_temp_avg", "Average CPU core temperature")
        self.nvme_temp_avg = Gauge("nvme_temp_avg", "Average NVMe temperature")
        self.acpitz_temp = Gauge("acpitz_temp", "ACPI thermal zone temperature")

        # image metrics
        self.avg_brightness = Gauge("avg_frame_brightness", "Average brightness of processed frames")
        self.contrast = Gauge("frame_contrast", "Contrast of processed frames")

    def update_temp_metrics(self):
        temps = psutil.sensors_temperatures()

        if "coretemp" in temps:
            cpu_temps = temps["coretemp"]
            self.cpu_package_temp.set(cpu_temps[0].current)  # Package id 0
            core_temps = [t.current for t in cpu_temps[1:]]  # All core temperatures
            self.cpu_core_temp_avg.set(sum(core_temps) / len(core_temps))

        if "nvme" in temps:
            nvme_temps = [t.current for t in temps["nvme"] if t.label == "Composite"]
            self.nvme_temp_avg.set(sum(nvme_temps) / len(nvme_temps))

        if "acpitz" in temps:
            self.acpitz_temp.set(temps["acpitz"][0].current)

    def increment_exception_counter(self, exception_type: str, method: str):
        self.exception_counter.labels(type=exception_type, method=method).inc()

    def initialize_model(self):
        model_path = os.getenv("YOLO_MODEL_PATH")
        if not model_path:
            raise ValueError("Error: YOLO_MODEL_PATH environment variable is not set.")
        model = YOLO(model_path, verbose=False)
        logger.info("Model loaded successfully")
        return model

    def get_frame_from_redis(self) -> Optional[np.ndarray]:
        frame_data = self.redis_client.lindex("raw_frame_buffer", 0)
        if frame_data:
            frame_data_str = frame_data.decode("utf-8") if isinstance(frame_data, bytes) else str(frame_data)
            frame_dict = json.loads(frame_data_str)
            encoded_frame = frame_dict["frame"]
            nparr = np.frombuffer(base64.b64decode(encoded_frame), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        return None

    def process_stream(self):
        logger.info("Starting frame processing")
        self.last_fps_update = time.time()
        self.start_time = time.time()

        while True:
            try:
                frame = self.get_frame_from_redis()
                if frame is None:
                    logger.warning("No frame available in Redis. Waiting...")
                    time.sleep(1)
                    continue

                start_time = time.time()
                processed_frame, metadata = self.process_frame(frame)
                processing_time = time.time() - start_time
                self.frame_processing_time.observe(processing_time)

                self.store_frame_data(processed_frame, metadata)

                self.frame_count += 1
                self.fps_frame_count += 1
                self.frames_processed.inc()
                self.buffer_size.set(self.redis_client.llen("processed_frame_buffer"))  # type: ignore

                # Update memory and CPU usage
                process = psutil.Process(os.getpid())
                self.current_memory_usage.set(process.memory_info().rss)
                self.current_cpu_usage.set(process.cpu_percent())
                self.update_temp_metrics()

                current_time = time.time()
                elapsed_time = current_time - self.last_fps_update
                if elapsed_time >= 1:  # Update FPS every second
                    self.fps.set(self.fps_frame_count / elapsed_time)
                    self.fps_frame_count = 0
                    self.last_fps_update = current_time

                while not self.stats_queue.empty():
                    _ = self.stats_queue.get()
                    self.stop_detector.db.update_daily_statistics()

            except Exception as e:
                logger.error(f"Error in stream processing: {str(e)}")
                self.increment_exception_counter(type(e).__name__, "process_stream")
                time.sleep(1)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        start_time = time.time()

        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        contrast = np.std(gray)
        self.avg_brightness.set(float(avg_brightness))
        self.contrast.set(float(contrast))

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

        # Update Prometheus metrics
        self.cars_tracked.set(len(self.car_tracker.get_cars()))
        cars_in_stop_zone = sum(
            1
            for car in self.car_tracker.get_cars().values()
            if self.stop_detector.stop_zone.is_in_stop_zone(car.state.location) and not car.state.is_parked
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

        start_time = time.time()
        pipeline = self.redis_client.pipeline()
        pipeline.lpush("processed_frame_buffer", frame_data)
        pipeline.ltrim("processed_frame_buffer", 0, self.frame_buffer_size - 1)
        pipeline.execute()
        self.redis_op_latency.observe(time.time() - start_time)

        logger.debug(f"Frame {self.frame_count} stored in Redis")

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
    processor = StreamProcessor(config)
    processor.run()
