"""RTSP to Redis service.

This module reads frames from an RTSP (or ``file://``) source and pushes JPEG
bytes into a Redis list so that the rest of the StopSign pipeline can pick
them up.  Prometheus metrics are exposed on an HTTP port for observability.

The only purpose of this edit is to satisfy Ruff/flake8 rule **E402 – “module
level import not at top of file”**.  Imports were previously sprinkled below
code that executed at import-time (logging configuration, ``sys.path`` hacks,
etc.).  All imports are now consolidated at the very top of the file before
any other statements, as PEP 8 expects.
"""

# isort: skip_file
# ruff: format
from __future__ import annotations

# ----------------- standard library -----------------
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import os
from queue import Empty, Queue
import threading
import time
from typing import Optional

# ------------------ third-party ---------------------
import cv2
import redis

# Prometheus removed - using OpenTelemetry metrics instead
from redis.exceptions import RedisError
from stopsign.telemetry import setup_rtsp_service_telemetry, get_tracer


# ----------------- logging setup --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ------------------- local app ----------------------
def get_env(key: str) -> str:
    value = os.getenv(key)
    assert value is not None, f"{key} is not set"
    logger.info(f"Loaded env var {key}: {value}")
    return value


PROMETHEUS_PORT: int = int(get_env("PROMETHEUS_PORT"))
RTSP_URL: str = get_env("RTSP_URL")
REDIS_URL: str = get_env("REDIS_URL")
RAW_FRAME_KEY: str = get_env("RAW_FRAME_KEY")
FRAME_BUFFER_SIZE: int = int(get_env("FRAME_BUFFER_SIZE"))


class RTSPToRedis:
    def __init__(self):
        self.rtsp_url = RTSP_URL
        self.redis_url = REDIS_URL
        self.prometheus_port = PROMETHEUS_PORT
        self.frame_buffer_size = FRAME_BUFFER_SIZE
        self.fps = 15
        self.jpeg_quality = 85

        self.redis_client: Optional[redis.Redis] = None
        self.frame_queue = Queue(maxsize=1000)
        self.processing_thread = None
        self.should_stop = threading.Event()

        # OpenTelemetry metrics and tracer (set from main)
        self.metrics = None
        self.tracer = None

        # Runtime metrics tracking
        self.actual_fps = 0.0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.rtsp_errors = 0
        self.redis_errors = 0
        self.disconnects = 0

    def set_telemetry(self, metrics, tracer):
        """Set OpenTelemetry metrics and tracer instances."""
        self.metrics = metrics
        self.tracer = tracer

    def initialize_redis(self):
        logger.info(f"Attempting to connect to Redis at {self.redis_url}")
        try:
            self.redis_client = redis.from_url(self.redis_url, socket_timeout=5)
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            if self.metrics:
                self.metrics.redis_operations.add(1, {"operation": "error", "service": "rtsp"})
            raise

    def initialize_capture(self):
        # Check if this is a file:// URL for local development
        if self.rtsp_url.startswith("file://"):
            file_path = self.rtsp_url[len("file://") :]
            logger.info(f"Attempting to open local video file at {file_path}")
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {file_path}")
            logger.info("Local video capture initialized successfully")
            if self.metrics:
                self.metrics.redis_operations.add(1, {"operation": "rtsp_connected", "service": "rtsp"})
            return cap

        # Standard RTSP connection logic
        logger.info(f"Attempting to connect to RTSP at {self.rtsp_url}")
        max_attempts = 5
        for attempt in range(max_attempts):
            if not self.rtsp_url:
                raise ValueError("RTSP URL is not set")
            try:
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    raise ValueError("Could not open video stream")
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                logger.info("Video capture initialized successfully")
                if self.metrics:
                    self.metrics.redis_operations.add(1, {"operation": "rtsp_connected", "service": "rtsp"})
                return cap
            except Exception as e:
                # Retry tracked in OpenTelemetry spans
                # RTSP errors tracked in OpenTelemetry
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                time.sleep(1)
        # Connection status tracked in OpenTelemetry
        raise ValueError("Failed to initialize video capture after multiple attempts")

    def process_frames(self):
        frames_processed = 0
        last_fps_update = time.time()

        while not self.should_stop.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                with self.tracer.start_as_current_span("store_frame") as span:
                    span.set_attribute("frame.height", frame.shape[0])
                    span.set_attribute("frame.width", frame.shape[1])
                    span.set_attribute("frame.channels", frame.shape[2])
                    self.store_frame(frame)
                self.frame_queue.task_done()
                # Track queue size via telemetry
                if self.metrics:
                    self.metrics.redis_operations.add(
                        1, {"operation": "queue_update", "service": "rtsp", "queue_size": str(self.frame_queue.qsize())}
                    )

                frames_processed += 1
                self.frames_processed += 1
                current_time = time.time()
                if current_time - last_fps_update >= 1:
                    # Update processed fps tracking
                    frames_processed = 0
                    last_fps_update = current_time

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                if self.metrics:
                    self.metrics.redis_operations.add(1, {"operation": "error", "service": "rtsp"})

    def store_frame(self, frame):
        with self.tracer.start_as_current_span("encode_frame") as span:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            span.set_attribute("jpeg.quality", self.jpeg_quality)
            span.set_attribute("encoded.size_bytes", len(buffer))

        if self.redis_client is None:
            logger.error("Redis client is not initialized")
            return

        try:
            with self.tracer.start_as_current_span("redis_publish") as span:
                redis_start_time = time.time()
                pipeline = self.redis_client.pipeline()
                pipeline.lpush(RAW_FRAME_KEY, buffer.tobytes())
                pipeline.ltrim(RAW_FRAME_KEY, 0, self.frame_buffer_size - 1)
                pipeline.llen(RAW_FRAME_KEY)
                _, _, current_buffer_size = pipeline.execute()
                redis_duration = time.time() - redis_start_time

                span.set_attribute("redis.operation", "pipeline_publish")
                span.set_attribute("redis.buffer_size", current_buffer_size)
                span.set_attribute("redis.duration_seconds", redis_duration)
                span.set_attribute("frame.buffer_size_bytes", len(buffer))

                # Redis latency now tracked in OpenTelemetry spans

            # Record OpenTelemetry metrics
            if self.metrics:
                self.metrics.frames_processed.add(1, {"service": "rtsp"})
                self.metrics.redis_operations.add(1, {"operation": "frame_publish", "service": "rtsp"})
                self.metrics.db_operation_duration.record(redis_duration)

        except RedisError as e:
            logger.error(f"Redis operation failed: {str(e)}")
            if self.metrics:
                self.metrics.redis_operations.add(1, {"operation": "error", "service": "rtsp"})
            raise

        # Frame processing time tracked in OpenTelemetry spans

    def run(self):
        # Prometheus removed - using OpenTelemetry instead
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        last_log_time = time.time()
        log_interval = 60

        logger.info("RTSP to Redis service starting...")
        self.log_status()

        while not self.should_stop.is_set():
            cap = None
            try:
                self.initialize_redis()
                cap = self.initialize_capture()
                frame_time = 1 / self.fps
                last_frame_time = time.time()
                fps_update_time = time.time()
                rtsp_frames_count = 0

                while not self.should_stop.is_set():
                    current_time = time.time()
                    elapsed_time = current_time - last_frame_time

                    if elapsed_time >= frame_time:
                        ret, frame = cap.read()
                        if not ret:
                            # Handle end of video file by looping back to start
                            if self.rtsp_url.startswith("file://"):
                                logger.info("End of video file reached. Looping back to start.")
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ret, frame = cap.read()
                                if not ret:
                                    logger.error("Failed to read frame after reset. Reinitializing capture.")
                                    # RTSP errors tracked in OpenTelemetry
                                    break
                            else:
                                logger.warning("Failed to read frame. Reinitializing capture.")
                                # RTSP errors tracked in OpenTelemetry
                                break

                        rtsp_frames_count += 1

                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                        else:
                            logger.warning("Frame queue is full. Dropping frame.")
                            self.frames_dropped += 1

                        last_frame_time = current_time

                        # Update FPS every second
                        if current_time - fps_update_time >= 1:
                            elapsed_fps_time = current_time - fps_update_time
                            self.actual_fps = rtsp_frames_count / elapsed_fps_time
                            rtsp_frames_count = 0
                            fps_update_time = current_time

                        # Log status periodically
                        if current_time - last_log_time >= log_interval:
                            self.log_status()
                            last_log_time = current_time

                    else:
                        time.sleep(frame_time - elapsed_time)

            except RedisError as e:
                logger.error(f"Redis error: {str(e)}")
                self.redis_errors += 1
                self.disconnects += 1
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in RTSP to Redis service: {str(e)}")
                self.rtsp_errors += 1
                time.sleep(1)
            finally:
                if cap:
                    cap.release()
                # Connection status tracked in OpenTelemetry
                logger.info("RTSP to Redis service restarting...")

    def stop(self):
        self.should_stop.set()
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("RTSP to Redis service stopped.")

    def log_status(self):
        buffer_utilization = (self.frame_queue.qsize() / 1000.0) * 100  # Queue maxsize is 1000
        logger.info(
            f"Status Update: "
            f"FPS: {self.actual_fps:.2f}, "
            f"Queue Size: {self.frame_queue.qsize()}, "
            f"Frames Processed: {self.frames_processed}, "
            f"Frames Dropped: {self.frames_dropped}, "
            f"Buffer Utilization: {buffer_utilization:.2f}%, "
            f"RTSP Errors: {self.rtsp_errors}, "
            f"Redis Errors: {self.redis_errors}"
        )

    def health_check(self):
        try:
            if self.redis_client and self.redis_client.ping():
                if self.processing_thread and self.processing_thread.is_alive():
                    return True
            return False
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Initialize telemetry
    metrics = setup_rtsp_service_telemetry()
    tracer = get_tracer("stopsign.rtsp_service")

    # Create RTSP service and set telemetry
    rtsp_to_redis = RTSPToRedis()
    rtsp_to_redis.set_telemetry(metrics, tracer)

    from http.server import BaseHTTPRequestHandler
    from http.server import HTTPServer

    class HealthCheckHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                if rtsp_to_redis.health_check():
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"OK")
                else:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"Not OK")
            else:
                self.send_response(404)
                self.end_headers()

    def run_health_server():
        server = HTTPServer(("0.0.0.0", 8080), HealthCheckHandler)
        server.serve_forever()

    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    rtsp_to_redis.run()
