import logging
import os
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

# Environment Variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_BUFFER_KEY = "processed_frame_buffer"
STREAM_DIR = "/app/stream"
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
        "libx264",
        "-preset",
        "medium",
        "-b:v",
        "2M",
        "-maxrate",
        "2.5M",
        "-bufsize",
        "4M",
        "-g",
        "30",
        "-f",
        "hls",
        "-hls_time",
        "2",
        "-hls_list_size",
        "10",
        "-hls_flags",
        "delete_segments",
        HLS_PLAYLIST,
    ]


def get_frame_shape(r: redis.Redis) -> tuple[int, int] | None:
    """Get the shape of the first frame from Redis."""
    while True:
        task = r.blpop([REDIS_BUFFER_KEY], timeout=5)
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
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    frame_shape = get_frame_shape(r)
    if frame_shape is None:
        logger.error("Failed to get frame shape")
        return

    ffmpeg_process = start_ffmpeg_process(frame_shape)
    if ffmpeg_process is None or ffmpeg_process.stdin is None:
        logger.error("Failed to start FFmpeg process")
        return

    try:
        while True:
            # Blocking pop with timeout
            task = r.blpop([REDIS_BUFFER_KEY], timeout=5)
            if task:
                _, data = task  # type: ignore
                if ffmpeg_process.stdin:
                    try:
                        # Decode the JPEG data
                        nparr = np.frombuffer(data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        # print the frame shape
                        resized_frame = cv2.resize(frame, (frame_shape[0], frame_shape[1]))
                        raw_frame = resized_frame.tobytes()

                        ffmpeg_process.stdin.write(raw_frame)
                        ffmpeg_process.stdin.flush()
                    except BrokenPipeError:
                        logger.error("FFmpeg process closed unexpectedly. Restarting...")
                        ffmpeg_process = start_ffmpeg_process(frame_shape)

            # Periodically check FFmpeg process
            if ffmpeg_process.poll() is not None:
                logger.error("FFmpeg process terminated. Restarting...")
                ffmpeg_process = start_ffmpeg_process(frame_shape)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if ffmpeg_process:
            ffmpeg_process.stdin.close()  # type: ignore
            ffmpeg_process.terminate()
            ffmpeg_process.wait()


if __name__ == "__main__":
    main()
