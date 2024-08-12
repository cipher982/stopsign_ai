import base64
import json
import logging
import os
import time

import cv2
import redis
from dotenv import load_dotenv
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import start_http_server

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RTSPToRedis:
    def __init__(self):
        self.rtsp_url = os.getenv("RTSP_URL")
        self.redis_host = "localhost"
        self.redis_port = 6379
        self.redis_client = None
        self.frame_buffer_size = 500
        self.fps = 15
        self.jpeg_quality = 100
        self.frames_processed = Counter("frames_processed", "Number of frames processed")
        self.buffer_size = Gauge("buffer_size", "Current size of the frame buffer")
        self.retries = Counter("retries", "Number of retries for video capture initialization")
        self.disconnects = Counter("disconnects", "Number of disconnects from Redis")

    def initialize_redis(self):
        logger.info(f"Attempting to connect to Redis at {self.redis_host}:{self.redis_port}")
        self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, db=0, socket_timeout=5)
        self.redis_client.ping()
        logger.info("Successfully connected to Redis")

    def initialize_capture(self):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                cap = cv2.VideoCapture(self.rtsp_url)  # type: ignore
                if not cap.isOpened():
                    raise ValueError("Could not open video stream")
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                logger.info("Video capture initialized successfully")
                return cap
            except Exception as e:
                self.retries.inc()
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                time.sleep(1)  # Short wait before retrying
        raise ValueError("Failed to initialize video capture after multiple attempts")

    def store_frame(self, frame):
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        encoded_frame = base64.b64encode(buffer).decode("utf-8")

        frame_data = json.dumps({"frame": encoded_frame, "timestamp": time.time()})

        if self.redis_client is None:
            logger.error("Redis client is not initialized")
            return

        pipeline = self.redis_client.pipeline()
        pipeline.lpush("raw_frame_buffer", frame_data)
        pipeline.ltrim("raw_frame_buffer", 0, self.frame_buffer_size - 1)
        pipeline.execute()
        self.frames_processed.inc()
        self.buffer_size.set(self.redis_client.llen("raw_frame_buffer"))

    def run(self):
        start_http_server(8000)  # Start Prometheus metrics server
        while True:
            cap = None
            try:
                self.initialize_redis()
                if not self.redis_client:
                    raise ConnectionError("Failed to establish Redis connection")

                cap = self.initialize_capture()
                frame_time = 1 / self.fps
                last_frame_time = time.time()

                while True:
                    current_time = time.time()
                    elapsed_time = current_time - last_frame_time

                    if elapsed_time >= frame_time:
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning("Failed to read frame. Reinitializing capture.")
                            break

                        self.store_frame(frame)
                        last_frame_time = current_time
                    else:
                        time.sleep(frame_time - elapsed_time)

            except ConnectionError as e:
                logger.error(f"Redis connection error: {str(e)}")
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                logger.error(f"Error in RTSP to Redis service: {str(e)}")
                time.sleep(1)
            finally:
                if cap:
                    cap.release()
                logger.info("RTSP to Redis service restarting...")


if __name__ == "__main__":
    rtsp_to_redis = RTSPToRedis()
    rtsp_to_redis.run()
