import logging
import os
import threading
import time
from queue import Empty
from queue import Queue
from typing import Optional

import cv2
import redis
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import start_http_server
from redis.exceptions import RedisError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_env(key: str) -> str:
    value = os.getenv(key)
    assert value is not None, f"{key} is not set"
    return value


RTSP_URL: str = get_env("RTSP_URL")
REDIS_URL: str = get_env("REDIS_URL")
RAW_FRAME_KEY: str = get_env("RAW_FRAME_KEY")
PROMETHEUS_PORT: int = int(get_env("PROMETHEUS_PORT"))


class RTSPToRedis:
    def __init__(self):
        self.rtsp_url = os.getenv("RTSP_URL")
        self.redis_url = os.getenv("REDIS_URL")
        self.prometheus_port = int(os.getenv("PROMETHEUS_PORT", 8000))
        self.frame_buffer_size = int(os.getenv("FRAME_BUFFER_SIZE", 20))
        self.fps = 15
        self.jpeg_quality = 100

        self.redis_client: Optional[redis.Redis] = None
        self.frame_queue = Queue(maxsize=1000)
        self.processing_thread = None
        self.should_stop = threading.Event()

        self.initialize_metrics()

    def initialize_metrics(self):
        self.frames_processed = Counter("frames_processed", "Number of frames processed")
        self.rtsp_redis_buffer_size = Gauge("rtsp_redis_buffer_size", "Current size of the frame buffer")
        self.retries = Counter("retries", "Number of retries for video capture initialization")
        self.disconnects = Counter("disconnects", "Number of disconnects from Redis")
        self.rtsp_connection_status = Gauge(
            "rtsp_connection_status",
            "RTSP connection status (1 for connected, 0 for disconnected)",
        )
        self.frame_processing_time = Histogram(
            "frame_processing_time_seconds",
            "Time taken to process each frame",
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 5],
        )
        self.redis_operation_latency = Histogram(
            "redis_operation_latency_seconds",
            "Latency of Redis operations",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
        )
        self.actual_fps = Gauge("actual_fps", "Actual frames per second being processed")
        self.rtsp_errors = Counter("rtsp_errors", "Number of RTSP-related errors")
        self.redis_errors = Counter("redis_errors", "Number of Redis-related errors")
        self.buffer_utilization = Gauge("buffer_utilization_percent", "Percentage of buffer utilized")

        self.rtsp_input_fps = Gauge("rtsp_input_fps", "Frames per second received from the RTSP stream")
        self.processed_fps = Gauge("rtsp_to_redis_fps", "Frames per second processed and stored in Redis")

        self.queue_size = Gauge("rtsp_queue_size", "Current size of the RTSP frame processing queue")
        self.frames_dropped = Counter("rtsp_frames_dropped", "Number of frames dropped due to full queue")

    def initialize_redis(self):
        logger.info(f"Attempting to connect to Redis at {self.redis_url}")
        try:
            self.redis_client = redis.from_url(self.redis_url, socket_timeout=5)
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_errors.inc()
            raise

    def initialize_capture(self):
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
                self.rtsp_connection_status.set(1)
                return cap
            except Exception as e:
                self.retries.inc()
                self.rtsp_errors.inc()
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                time.sleep(1)
        self.rtsp_connection_status.set(0)
        raise ValueError("Failed to initialize video capture after multiple attempts")

    def process_frames(self):
        frames_processed = 0
        last_fps_update = time.time()

        while not self.should_stop.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                self.store_frame(frame)
                self.frame_queue.task_done()
                self.queue_size.set(self.frame_queue.qsize())

                frames_processed += 1
                current_time = time.time()
                if current_time - last_fps_update >= 1:
                    self.processed_fps.set(frames_processed / (current_time - last_fps_update))
                    frames_processed = 0
                    last_fps_update = current_time

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                self.redis_errors.inc()

    def store_frame(self, frame):
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        if self.redis_client is None:
            logger.error("Redis client is not initialized")
            return

        try:
            start_time = time.time()
            redis_start_time = time.time()
            pipeline = self.redis_client.pipeline()
            pipeline.lpush(RAW_FRAME_KEY, buffer.tobytes())
            pipeline.ltrim(RAW_FRAME_KEY, 0, self.frame_buffer_size - 1)
            pipeline.llen(RAW_FRAME_KEY)
            _, _, current_buffer_size = pipeline.execute()
            self.redis_operation_latency.observe(time.time() - redis_start_time)

            self.frames_processed.inc()
            self.rtsp_redis_buffer_size.set(float(current_buffer_size))
            self.buffer_utilization.set((float(current_buffer_size) / self.frame_buffer_size) * 100)

        except RedisError as e:
            logger.error(f"Redis operation failed: {str(e)}")
            self.redis_errors.inc()
            raise

        self.frame_processing_time.observe(time.time() - start_time)

    def run(self):
        start_http_server(self.prometheus_port)
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
                            logger.warning("Failed to read frame. Reinitializing capture.")
                            self.rtsp_errors.inc()
                            break

                        rtsp_frames_count += 1

                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                        else:
                            logger.warning("Frame queue is full. Dropping frame.")
                            self.frames_dropped.inc()

                        last_frame_time = current_time

                        # Update FPS every second
                        if current_time - fps_update_time >= 1:
                            elapsed_fps_time = current_time - fps_update_time
                            self.actual_fps.set(rtsp_frames_count / elapsed_fps_time)
                            self.rtsp_input_fps.set(rtsp_frames_count / elapsed_fps_time)
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
                self.disconnects.inc()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in RTSP to Redis service: {str(e)}")
                self.rtsp_errors.inc()
                time.sleep(1)
            finally:
                if cap:
                    cap.release()
                self.rtsp_connection_status.set(0)
                logger.info("RTSP to Redis service restarting...")

    def stop(self):
        self.should_stop.set()
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("RTSP to Redis service stopped.")

    def log_status(self):
        logger.info(
            f"Status Update: "
            f"FPS: {self.actual_fps._value.get():.2f}, "
            f"Queue Size: {self.frame_queue.qsize()}, "
            f"Frames Processed: {self.frames_processed._value.get()}, "
            f"Frames Dropped: {self.frames_dropped._value.get()}, "
            f"Buffer Utilization: {self.buffer_utilization._value.get():.2f}%, "
            f"RTSP Errors: {self.rtsp_errors._value.get()}, "
            f"Redis Errors: {self.redis_errors._value.get()}"
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
    rtsp_to_redis = RTSPToRedis()

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

    import threading

    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    rtsp_to_redis.run()
