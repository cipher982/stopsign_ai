import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer

import cv2
import numpy as np
import redis

from stopsign.telemetry import get_tracer
from stopsign.telemetry import setup_ffmpeg_service_telemetry

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def get_env(key: str) -> str:
    value = os.getenv(key)
    assert value is not None, f"{key} is not set"
    return value


# Environment Variables
REDIS_URL = get_env("REDIS_URL")
PROCESSED_FRAME_KEY = get_env("PROCESSED_FRAME_KEY")
HEALTH_PORT = 8080

STREAM_DIR = "/app/data/stream"
FRAME_RATE = "15"
RESOLUTION = "1920x1080"

# ---------------------------------------------------------------------------
# Run-time configurable encoding parameters.
#   • FFMPEG_ENCODER  – h264_nvenc (GPU) | libx264 (CPU) | …
#   • FFMPEG_PRESET   – p4 for NVENC, veryfast for libx264, etc.
# ---------------------------------------------------------------------------

ENCODER = os.getenv("FFMPEG_ENCODER", "libx264")
PRESET = os.getenv("FFMPEG_PRESET", "veryfast")

# FFmpeg Configuration
HLS_PLAYLIST = os.path.join(STREAM_DIR, "stream.m3u8")

# For monitoring
frames_processed = 0


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()


def start_health_server():
    server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
    logger.info(f"Starting health check server on port {HEALTH_PORT}")
    server.serve_forever()


def create_ffmpeg_cmd(frame_shape: tuple[int, int]) -> list[str]:
    return [
        "ffmpeg",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{frame_shape[0]}x{frame_shape[1]}",
        "-r",
        FRAME_RATE,
        "-i",
        "-",
        "-vf",
        "format=yuv420p",
        "-pix_fmt",
        "yuv420p",
        "-aspect",
        "16:9",
        "-c:v",
        ENCODER,
        "-preset",
        PRESET,
        "-profile:v",
        "main",
        "-level",
        "4.0",
        "-b:v",
        "6M",
        "-maxrate",
        "8M",
        "-bufsize",
        "12M",
        "-g",
        "30",
        "-r",
        FRAME_RATE,
        "-f",
        "hls",
        "-hls_time",
        "0.25",
        "-hls_list_size",
        "20",
        "-hls_flags",
        "delete_segments+program_date_time",
        "-hls_allow_cache",
        "0",
        "-hls_segment_type",
        "mpegts",
        "-loglevel",
        "warning",
        HLS_PLAYLIST,
    ]


def get_frame_shape(r: redis.Redis) -> tuple[int, int] | None:
    """Get the shape of the first frame from Redis."""
    while True:
        task = r.blpop([PROCESSED_FRAME_KEY], timeout=5)
        if task:
            _, data = task  # type: ignore
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return (frame.shape[1], frame.shape[0])


def start_ffmpeg_process(frame_shape):
    """Starts the FFmpeg subprocess with the correct frame shape."""
    logger.info(
        "Starting FFmpeg process with frame shape: %s - encoder: %s / preset: %s",
        frame_shape,
        ENCODER,
        PRESET,
    )
    ffmpeg_cmd = create_ffmpeg_cmd(frame_shape)
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    return process


def log_stream_files():
    files = os.listdir(STREAM_DIR)
    for file in files:
        file_path = os.path.join(STREAM_DIR, file)
        file_size = os.path.getsize(file_path)
        file_mtime = os.path.getmtime(file_path)
        logger.info(f"File: {file}, Size: {file_size} bytes, Last modified: {time.ctime(file_mtime)}")


def main():
    logger.info(f"Starting FFmpeg service with STREAM_DIR: {STREAM_DIR}")

    # Initialize telemetry
    metrics = setup_ffmpeg_service_telemetry()
    tracer = get_tracer("stopsign.ffmpeg_service")

    # Make telemetry available globally
    globals()["metrics"] = metrics
    globals()["tracer"] = tracer

    clean_stream_directory()

    # Start health server in a separate thread
    try:
        logger.info(f"Starting health check server on port {HEALTH_PORT}")
        health_thread = threading.Thread(target=start_health_server, daemon=True)
        health_thread.start()
        logger.info("Health check server thread started")
    except Exception as e:
        logger.error(f"Failed to start health check server: {e}")

    r = redis.from_url(REDIS_URL)
    logger.info(f"Connected to Redis at {REDIS_URL}")
    frame_shape = get_frame_shape(r)
    if frame_shape is None:
        logger.error("Failed to get frame shape")
        return
    else:
        logger.info(f"Frame shape: {frame_shape}")

    ffmpeg_process = start_ffmpeg_process(frame_shape)
    if ffmpeg_process is None or ffmpeg_process.stdin is None:
        logger.error("Failed to start FFmpeg process")
        return

    try:
        logger.info("Starting main loop")
        global frames_processed
        while True:
            task = r.blpop([PROCESSED_FRAME_KEY], timeout=5)
            if task:
                _, data = task  # type: ignore
                if ffmpeg_process.stdin:
                    try:
                        with tracer.start_as_current_span("process_frame") as span:
                            span.set_attribute("frame.size_bytes", len(data))

                            # Decode frame
                            nparr = np.frombuffer(data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            # Resize frame
                            resized_frame = cv2.resize(frame, (frame_shape[0], frame_shape[1]))
                            span.set_attribute("frame.width", frame_shape[0])
                            span.set_attribute("frame.height", frame_shape[1])

                            # Send to FFmpeg
                            raw_frame = resized_frame.tobytes()
                            ffmpeg_process.stdin.write(raw_frame)
                            ffmpeg_process.stdin.flush()

                            frames_processed += 1
                            metrics.frames_processed.add(1, {"service": "ffmpeg"})
                            span.set_attribute("frames.total_processed", frames_processed)

                    except BrokenPipeError:
                        logger.error("FFmpeg process closed unexpectedly. Restarting...")
                        with tracer.start_as_current_span("restart_ffmpeg_process"):
                            ffmpeg_process = start_ffmpeg_process(frame_shape)

            if ffmpeg_process.poll() is not None:
                logger.error("FFmpeg process terminated. Restarting...")
                ffmpeg_process = start_ffmpeg_process(frame_shape)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if ffmpeg_process and ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        if ffmpeg_process:
            ffmpeg_process.terminate()
            ffmpeg_process.wait()


def clean_stream_directory():
    """Clean up the stream directory by removing all files."""
    logger.info("Cleaning up stream directory...")
    os.makedirs(STREAM_DIR, exist_ok=True)
    for filename in os.listdir(STREAM_DIR):
        file_path = os.path.join(STREAM_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    main()
