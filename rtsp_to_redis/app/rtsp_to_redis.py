import base64
import json
import logging
import os
import time
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


class RTSPToRedis:
    def __init__(self):
        self.rtsp_url = os.getenv("RTSP_URL")
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.prometheus_port = int(os.getenv("PROMETHEUS_PORT", 8000))
        self.frame_buffer_size = int(os.getenv("FRAME_BUFFER_SIZE", 500))
        self.fps = int(os.getenv("FPS", 15))
        self.jpeg_quality = int(os.getenv("JPEG_QUALITY", 95))

        self.redis_client: Optional[redis.Redis] = None
        self.initialize_metrics()

    def initialize_metrics(self):
        self.frames_processed = Counter("frames_processed", "Number of frames processed")
        self.buffer_size = Gauge("buffer_size", "Current size of the frame buffer")
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

    def initialize_redis(self):
        logger.info(f"Attempting to connect to Redis at {self.redis_host}:{self.redis_port}")
        try:
            self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, db=0, socket_timeout=5)
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_errors.inc()
            raise

    def initialize_capture(self):
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

    def store_frame(self, frame):
        start_time = time.time()
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        encoded_frame = base64.b64encode(buffer).decode("utf-8")

        frame_data = json.dumps({"frame": encoded_frame, "timestamp": time.time()})

        if self.redis_client is None:
            logger.error("Redis client is not initialized")
            return

        try:
            redis_start_time = time.time()
            pipeline = self.redis_client.pipeline()
            pipeline.lpush("raw_frame_buffer", frame_data)
            pipeline.ltrim("raw_frame_buffer", 0, self.frame_buffer_size - 1)
            pipeline.llen("raw_frame_buffer")
            _, _, current_buffer_size = pipeline.execute()
            self.redis_operation_latency.observe(time.time() - redis_start_time)

            self.frames_processed.inc()
            self.buffer_size.set(float(current_buffer_size))
            self.buffer_utilization.set((float(current_buffer_size) / self.frame_buffer_size) * 100)
        except RedisError as e:
            logger.error(f"Redis operation failed: {str(e)}")
            self.redis_errors.inc()
            raise

        self.frame_processing_time.observe(time.time() - start_time)

    def run(self):
        start_http_server(self.prometheus_port)
        while True:
            cap = None
            try:
                self.initialize_redis()
                cap = self.initialize_capture()
                frame_time = 1 / self.fps
                last_frame_time = time.time()
                frames_count = 0
                fps_update_time = time.time()

                while True:
                    current_time = time.time()
                    elapsed_time = current_time - last_frame_time

                    if elapsed_time >= frame_time:
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning("Failed to read frame. Reinitializing capture.")
                            self.rtsp_errors.inc()
                            break

                        self.store_frame(frame)
                        last_frame_time = current_time
                        frames_count += 1

                        # Update FPS every second
                        if current_time - fps_update_time >= 1:
                            self.actual_fps.set(frames_count / (current_time - fps_update_time))
                            frames_count = 0
                            fps_update_time = current_time
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


if __name__ == "__main__":
    rtsp_to_redis = RTSPToRedis()
    rtsp_to_redis.run()