import logging
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import redis

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

STREAM_DIR = "/app/data/stream"
FRAME_RATE = "15"
RESOLUTION = "1920x1080"

# FFmpeg Configuration
HLS_PLAYLIST = os.path.join(STREAM_DIR, "stream.m3u8")


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
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-b:v",
        "2M",
        "-maxrate",
        "2.5M",
        "-bufsize",
        "4M",
        "-g",
        "30",
        "-r",
        FRAME_RATE,
        "-f",
        "hls",
        "-hls_time",
        "2",
        "-hls_list_size",
        "180",
        "-hls_flags",
        "delete_segments",
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
    logger.info(f"Starting FFmpeg process with frame shape: {frame_shape}")
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
    clean_stream_directory()

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
        while True:
            task = r.blpop([PROCESSED_FRAME_KEY], timeout=5)
            if task:
                _, data = task  # type: ignore
                if ffmpeg_process.stdin:
                    try:
                        nparr = np.frombuffer(data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        resized_frame = cv2.resize(frame, (frame_shape[0], frame_shape[1]))
                        raw_frame = resized_frame.tobytes()
                        ffmpeg_process.stdin.write(raw_frame)
                        ffmpeg_process.stdin.flush()

                    except BrokenPipeError:
                        logger.error("FFmpeg process closed unexpectedly. Restarting...")
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
