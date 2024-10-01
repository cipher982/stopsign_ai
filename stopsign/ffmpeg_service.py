import logging
import os
import subprocess
import sys
import time

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
FFMPEG_CMD = [
    "ffmpeg",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "-s",
    "1920x1080",
    "-r",
    "15",
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


def start_ffmpeg_process():
    """Starts the FFmpeg subprocess."""
    logger.info("Starting FFmpeg process...")
    process = subprocess.Popen(FFMPEG_CMD, stdin=subprocess.PIPE)
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
    ffmpeg_process = start_ffmpeg_process()

    try:
        while True:
            # Blocking pop with timeout
            task = r.blpop([REDIS_BUFFER_KEY], timeout=5)
            if task:
                _, data = task
                try:
                    ffmpeg_process.stdin.write(data)
                    ffmpeg_process.stdin.flush()
                except BrokenPipeError:
                    logger.error("FFmpeg process closed unexpectedly. Restarting...")
                    ffmpeg_process = start_ffmpeg_process()
            else:
                pass

            # Periodically check FFmpeg process
            if ffmpeg_process.poll() is not None:
                logger.error("FFmpeg process terminated. Restarting...")
                ffmpeg_process = start_ffmpeg_process()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if ffmpeg_process:
            ffmpeg_process.stdin.close()
            ffmpeg_process.terminate()
            ffmpeg_process.wait()


if __name__ == "__main__":
    main()
