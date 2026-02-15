import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer

import redis
from redis import exceptions as redis_exceptions

from stopsign.hls_health import parse_hls_playlist
from stopsign.service_status import FFmpegServiceStatusMixin
from stopsign.settings import GRACE_STARTUP_SEC
from stopsign.settings import PROCESSED_FRAME_SHAPE_KEY
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
HEALTH_PORT = int(os.getenv("FFMPEG_HEALTH_PORT", "8080"))

STREAM_DIR = "/app/data/stream"
FRAME_RATE = "15"
RESOLUTION = "1920x1080"
HLS_LIST_SIZE = os.getenv("HLS_LIST_SIZE", "60")

# ---------------------------------------------------------------------------
# Run-time configurable encoding parameters.
#   • FFMPEG_ENCODER  – h264_nvenc (GPU) | libx264 (CPU) | …
#   • FFMPEG_PRESET   – p4 for NVENC, veryfast for libx264, etc.
# ---------------------------------------------------------------------------

ENCODER = os.getenv("FFMPEG_ENCODER", "libx264")
PRESET = os.getenv("FFMPEG_PRESET", "veryfast")

# Watchdog: if no fresh HLS for this period, exit(1) to let the
# orchestrator restart the container. 0 disables.
PIPELINE_WATCHDOG_SEC = float(os.getenv("PIPELINE_WATCHDOG_SEC", "0"))

# Frame consumption stall detector: if we don't process any frames for this
# many seconds, we mark readiness false (and optionally the watchdog may fire)
FRAME_STALL_SEC = float(os.getenv("FRAME_STALL_SEC", "120"))

# Redis reconnect/backoff
REDIS_MAX_BACKOFF_SEC = float(os.getenv("REDIS_MAX_BACKOFF_SEC", "30"))
REDIS_INITIAL_BACKOFF_SEC = float(os.getenv("REDIS_INITIAL_BACKOFF_SEC", "0.5"))

# FFmpeg Configuration
HLS_PLAYLIST = os.path.join(STREAM_DIR, "stream.m3u8")

# For monitoring
frames_processed = 0
START_TIME = time.time()
LAST_FRAME_TS = START_TIME
CONSEC_EMPTY_POLLS = 0
REDIS_CLIENT: redis.Redis | None = None

# Runtime status (human/debug domain)
status = FFmpegServiceStatusMixin()
status.update_status_metric("service_name", "FFmpegService")


def _parse_hls_playlist(path: str) -> dict:
    # Deprecated local parser; delegate to shared helper
    return parse_hls_playlist(path)


def get_hls_freshness() -> dict:
    """Return current HLS freshness information.

    Keys:
      - exists: bool, whether playlist exists
      - playlist_mtime: float|None, last modification epoch seconds
      - age_seconds: float|None, seconds since last update
      - segments_count: int, number of .ts files present (best-effort)
      - threshold_sec: float, derived from playlist window
    """
    info = _parse_hls_playlist(HLS_PLAYLIST)
    return info


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            info = get_hls_freshness()
            # Healthy only if playlist exists and is fresh (or during startup grace)
            now = time.time()
            warming_up = (now - START_TIME) <= GRACE_STARTUP_SEC
            age = info.get("age_seconds")
            threshold = info.get("threshold_sec", 60.0)
            fresh = bool(info.get("exists")) and (age is not None and age <= threshold)
            fresh = fresh or warming_up

            body = (
                f"status={'ok' if fresh else 'stale'}\n"
                f"playlist_exists={info.get('exists')}\n"
                f"age_seconds={info.get('age_seconds')}\n"
                f"threshold_sec={info.get('threshold_sec')}\n"
                f"segments_count={info.get('segments_count')}\n"
            ).encode()

            self.send_response(200 if fresh else 503)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                import json

                payload = {
                    "status": "ok" if fresh else "stale",
                    "exists": bool(info.get("exists")),
                    "age_seconds": info.get("age_seconds"),
                    "threshold_sec": info.get("threshold_sec"),
                    "segments_count": info.get("segments_count"),
                    "redis_connected": status.get_status_snapshot().get("redis_connected", False),
                    "last_frame_age_sec": max(0.0, time.time() - LAST_FRAME_TS),
                    "note": "readiness-like; use /healthz for liveness; /ready for composite readiness",
                }
                self.wfile.write(json.dumps(payload).encode())
            except Exception:
                self.wfile.write(body)
        elif self.path == "/healthz":
            # Simple liveness probe
            # Consider process live if the thread is running. We do NOT
            # gate liveness on HLS freshness. Watchdog handles hard restarts.
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/ready":
            # Composite readiness: HLS fresh AND recent frame processing AND Redis connected
            info = get_hls_freshness()
            age = info.get("age_seconds")
            threshold = info.get("threshold_sec", 60.0)
            hls_ok = bool(info.get("exists")) and (age is not None and age <= threshold)
            redis_ok = status.get_status_snapshot().get("redis_connected", False)
            recent_frame_ok = (time.time() - LAST_FRAME_TS) <= FRAME_STALL_SEC

            ready = hls_ok and redis_ok and recent_frame_ok

            self.send_response(200 if ready else 503)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                import json

                self.wfile.write(
                    json.dumps(
                        {
                            "ready": ready,
                            "hls_ok": hls_ok,
                            "redis_ok": redis_ok,
                            "recent_frame_ok": recent_frame_ok,
                            "hls_age_seconds": age,
                            "hls_threshold_seconds": threshold,
                            "last_frame_age_seconds": max(0.0, time.time() - LAST_FRAME_TS),
                            "consec_empty_polls": CONSEC_EMPTY_POLLS,
                        }
                    ).encode()
                )
            except Exception:
                self.wfile.write(b"ready check error")
        else:
            self.send_response(404)
            self.end_headers()


def start_health_server():
    server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
    logger.info(f"Starting health check server on port {HEALTH_PORT}")
    server.serve_forever()


def start_watchdog_thread():
    if PIPELINE_WATCHDOG_SEC <= 0:
        logger.info("Pipeline watchdog disabled (PIPELINE_WATCHDOG_SEC <= 0)")
        return

    def watchdog_loop():
        logger.info(
            "Starting pipeline watchdog: will terminate if HLS is stale for > %ss",
            PIPELINE_WATCHDOG_SEC,
        )
        last_fresh_ts = time.monotonic()
        while True:
            try:
                info = get_hls_freshness()
                age = info.get("age_seconds")
                threshold = info.get("threshold_sec", 60.0)
                fresh = bool(info.get("exists")) and (age is not None and age <= threshold)
                if fresh:
                    last_fresh_ts = time.monotonic()
                else:
                    stalled_for = time.monotonic() - last_fresh_ts
                    if stalled_for > PIPELINE_WATCHDOG_SEC:
                        logger.error(
                            "Watchdog trip: age=%.1fs threshold=%.1fs stalled_for=%.1fs segments=%s target_dur=%s",
                            (age or -1),
                            threshold,
                            stalled_for,
                            info.get("segments_count"),
                            info.get("target_duration_sec"),
                        )
                        os._exit(1)  # ensure container restart
                time.sleep(10)
            except Exception as e:
                logger.warning(f"Watchdog error: {e}")
                time.sleep(10)

    t = threading.Thread(target=watchdog_loop, daemon=True)
    t.start()


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
        "2",
        "-hls_list_size",
        HLS_LIST_SIZE,
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
        shape_data = r.get(PROCESSED_FRAME_SHAPE_KEY)
        if shape_data:
            try:
                payload = json.loads(shape_data)
                width = int(payload.get("width", 0))
                height = int(payload.get("height", 0))
                if width > 0 and height > 0:
                    return (width, height)
            except Exception as e:
                logger.warning("Failed to parse processed frame shape: %s", e)
        time.sleep(0.5)


def connect_redis_with_backoff(url: str) -> redis.Redis:
    """Connect to Redis with exponential backoff and status updates."""
    backoff = REDIS_INITIAL_BACKOFF_SEC
    while True:
        try:
            # Allow blocking operations (e.g., BRPOP) to manage their own timeout.
            # Using socket_timeout here causes redis-py to raise TimeoutError even when
            # the BRPOP call is behaving normally, which leads to noisy reconnects.
            client = redis.from_url(url, socket_timeout=None, socket_connect_timeout=5)
            client.ping()
            status.update_status_metric("redis_connected", True)
            logger.info("Connected to Redis at %s", url)
            global REDIS_CLIENT
            REDIS_CLIENT = client
            return client
        except redis_exceptions.RedisError as e:
            status.update_status_metric("redis_connected", False)
            logger.error("Redis connection failed: %s (retrying in %.1fs)", e, backoff)
            if "metrics" in globals():
                globals()["metrics"].service_errors.add(1, {"error_type": "redis_connection", "service": "ffmpeg"})
            time.sleep(backoff)
            backoff = min(REDIS_MAX_BACKOFF_SEC, backoff * 2)


def safe_brpop(key: str, timeout: int = 5):
    """BRPOP wrapper with reconnect/backoff and empty poll tracking.

    Returns (key, data) on success or None on timeout.
    """
    global CONSEC_EMPTY_POLLS
    global REDIS_CLIENT
    try:
        if REDIS_CLIENT is None:
            REDIS_CLIENT = connect_redis_with_backoff(REDIS_URL)
        item = REDIS_CLIENT.brpop([key], timeout=timeout)
        metrics_obj = globals().get("metrics")

        if item is None:
            CONSEC_EMPTY_POLLS += 1
            status.update_custom_metric("consec_empty_polls", CONSEC_EMPTY_POLLS)
            if metrics_obj is not None:
                metrics_obj.redis_empty_polls.add(1, {"service": "ffmpeg"})
                metrics_obj.queue_depth.record(0, {"queue": key})
        else:
            CONSEC_EMPTY_POLLS = 0
            status.update_custom_metric("consec_empty_polls", CONSEC_EMPTY_POLLS)
            queue_depth = REDIS_CLIENT.llen(key)
            status.update_custom_metric("queue_depth", queue_depth)
            if metrics_obj is not None:
                metrics_obj.queue_depth.record(queue_depth, {"queue": key})
        return item
    except redis_exceptions.TimeoutError:
        # Idle queues trigger TimeoutError when socket_timeout is set. Treat this
        # as an empty poll instead of a connection failure so the service stays up.
        CONSEC_EMPTY_POLLS += 1
        status.update_custom_metric("consec_empty_polls", CONSEC_EMPTY_POLLS)
        metrics_obj = globals().get("metrics")
        if metrics_obj is not None:
            metrics_obj.redis_empty_polls.add(1, {"service": "ffmpeg"})
            metrics_obj.queue_depth.record(0, {"queue": key})
        return None
    except redis_exceptions.RedisError as e:
        logger.warning("Redis BRPOP error: %s", e)
        status.update_status_metric("redis_connected", False)
        # Attempt reconnect
        REDIS_CLIENT = connect_redis_with_backoff(REDIS_URL)
        return None


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

    # Start watchdog if configured
    start_watchdog_thread()

    r = connect_redis_with_backoff(REDIS_URL)
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

    # Frame pacing: maintain steady 15 FPS output by duplicating frames when input is slow
    # This is the "Carmack stable fix" - fill gaps with duplicate frames
    target_fps = float(FRAME_RATE)
    frame_interval = 1.0 / target_fps  # ~66.7ms at 15 FPS
    expected_frame_bytes = frame_shape[0] * frame_shape[1] * 3
    last_frame_time = time.time()
    last_raw_frame = None  # Cache for frame duplication

    # FPS logging for diagnosis
    fps_frame_count = 0
    fps_last_log_time = time.time()
    new_frame_count = 0
    dup_frame_count = 0

    try:
        logger.info(f"Starting main loop with frame duplication (target {target_fps} FPS)")
        global frames_processed
        while True:
            # Non-blocking pop - RPOP returns immediately with None if queue empty
            # (BRPOP timeout=0 blocks forever, so we use RPOP instead)
            try:
                data = REDIS_CLIENT.rpop(PROCESSED_FRAME_KEY)
                task = (PROCESSED_FRAME_KEY, data) if data else None
            except Exception:
                task = None

            now = time.time()
            time_since_last = now - last_frame_time

            if task:
                # New frame arrived - cache raw bytes
                _, data = task  # type: ignore
                if ffmpeg_process.stdin:
                    try:
                        if len(data) != expected_frame_bytes:
                            logger.warning(
                                "Processed frame size mismatch (got %d bytes, expected %d)",
                                len(data),
                                expected_frame_bytes,
                            )
                            continue
                        last_raw_frame = data
                        new_frame_count += 1
                    except Exception as e:
                        logger.warning(f"Frame read error: {e}")
                        continue

            # Write frame at steady 15 FPS intervals
            if time_since_last >= frame_interval and last_raw_frame is not None:
                if ffmpeg_process.stdin:
                    try:
                        ffmpeg_process.stdin.write(last_raw_frame)
                        ffmpeg_process.stdin.flush()
                        last_frame_time = now
                        frames_processed += 1
                        fps_frame_count += 1

                        # Track duplicates vs new frames
                        if not task:
                            dup_frame_count += 1

                        # Heartbeat and status updates
                        global LAST_FRAME_TS
                        LAST_FRAME_TS = now
                        status.update_status_metric("current_fps", target_fps)
                        status.increment_counter("processed_count", 1)
                        metrics.frames_processed.add(1, {"service": "ffmpeg"})
                    except BrokenPipeError:
                        logger.error("FFmpeg process closed unexpectedly. Restarting...")
                        ffmpeg_process = start_ffmpeg_process(frame_shape)
                        if not ffmpeg_process or not ffmpeg_process.stdin:
                            logger.error("FFmpeg restart failed: stdin is None")

            # Log FPS every 5 seconds
            if now - fps_last_log_time >= 5.0:
                actual_fps = fps_frame_count / (now - fps_last_log_time)
                new_fps = new_frame_count / (now - fps_last_log_time)
                dup_pct = (dup_frame_count / fps_frame_count * 100) if fps_frame_count > 0 else 0
                logger.info(f"FFmpeg output: {actual_fps:.1f} FPS (new: {new_fps:.1f}, dup: {dup_pct:.0f}%)")
                fps_frame_count = 0
                new_frame_count = 0
                dup_frame_count = 0
                fps_last_log_time = now

            # Small sleep to prevent busy-waiting when no frames available
            if not task and time_since_last < frame_interval:
                time.sleep(frame_interval - time_since_last)

            # If FFmpeg died, restart it
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
    """Clean up the stream directory by removing stale HLS files (preserves clips/)."""
    logger.info("Cleaning up stream directory...")
    os.makedirs(STREAM_DIR, exist_ok=True)
    for filename in os.listdir(STREAM_DIR):
        if filename == "clips":
            continue
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
