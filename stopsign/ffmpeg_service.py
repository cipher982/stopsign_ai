import logging
import os
import subprocess
import sys

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
    "-y",  # Overwrite output files without asking
    "-f",
    "rawvideo",
    "-pix_fmt",
    "bgr24",  # Adjust based on frame format (e.g., bgr24, rgb24, yuv420p)
    "-s",
    RESOLUTION,  # Frame size (width x height)
    "-r",
    FRAME_RATE,  # Frame rate
    "-i",
    "-",  # Input from stdin
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-f",
    "hls",
    "-hls_time",
    "4",
    "-hls_playlist_type",
    "event",
    HLS_PLAYLIST,
]


def start_ffmpeg_process():
    """Starts the FFmpeg subprocess."""
    logger.info("Starting FFmpeg process...")
    process = subprocess.Popen(FFMPEG_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def main():
    # Ensure the stream directory exists
    os.makedirs(STREAM_DIR, exist_ok=True)
    frame_count = 0

    # Connect to Redis
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        # Test connection
        r.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)

    # Start FFmpeg subprocess
    ffmpeg_process = start_ffmpeg_process()

    try:
        while True:
            # Blocking pop from the queue with a timeout
            task = r.blpop(REDIS_BUFFER_KEY, timeout=5)
            if task:
                # logger.info(f"Received task on frame {frame_count}.")
                _, data = task
                try:
                    # Write raw frame data to FFmpeg's stdin
                    ffmpeg_process.stdin.write(data)
                    ffmpeg_process.stdin.flush()
                    # logger.info(f"Written {len(data)} bytes on frame {frame_count} to FFmpeg.")
                    frame_count += 1
                except Exception as e:
                    logger.error(f"Error writing to FFmpeg stdin: {e}")
            else:
                logger.info(f"No task received on frame {frame_count}.")
                pass

            # Check FFmpeg process status
            retcode = ffmpeg_process.poll()
            if retcode is not None:
                # FFmpeg process has terminated
                logger.error(f"FFmpeg process terminated with return code {retcode}. Restarting...")
                # Log FFmpeg stderr
                stderr = ffmpeg_process.stderr.read().decode()
                logger.error(f"FFmpeg stderr: {stderr}")
                # Restart FFmpeg
                ffmpeg_process = start_ffmpeg_process()

    except KeyboardInterrupt:
        logger.info("Shutting down FFmpeg service...")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        if ffmpeg_process:
            ffmpeg_process.stdin.close()
            ffmpeg_process.terminate()
            ffmpeg_process.wait()
            logger.info("FFmpeg process terminated.")


if __name__ == "__main__":
    main()
