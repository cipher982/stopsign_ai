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

from stopsign.frame_codec import frame_metadata_error
from stopsign.frame_codec import unpack_frame
from stopsign.hls_health import parse_hls_playlist
from stopsign.service_status import FFmpegServiceStatusMixin
from stopsign.settings import FFMPEG_HEALTH_KEY
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
RELEASE_GENERATION = os.getenv("RELEASE_GENERATION")
if not RELEASE_GENERATION or RELEASE_GENERATION == "unknown":
    RELEASE_GENERATION = os.getenv("SOURCE_COMMIT", "unknown")
PROJECT_IDENTITY = os.getenv("PROJECT_IDENTITY", "stopsign")
RTSP_HEALTH_KEY = os.getenv("RTSP_HEALTH_KEY", "stopsign.rtsp.health")
FFMPEG_HEALTH_TTL_SEC = int(os.getenv("FFMPEG_HEALTH_TTL_SEC", "300"))

STREAM_DIR = os.getenv("STREAM_DIR", "/app/data/stream")
FRAME_RATE = "15"
RESOLUTION = "1920x1080"
HLS_LIST_SIZE = os.getenv("HLS_LIST_SIZE", "450")

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
FFMPEG_POP_MODE = os.getenv("FFMPEG_POP_MODE", "fifo").strip().lower()
if FFMPEG_POP_MODE not in {"latest", "fifo"}:
    logger.warning("Invalid FFMPEG_POP_MODE=%s; defaulting to 'latest'", FFMPEG_POP_MODE)
    FFMPEG_POP_MODE = "latest"
FFMPEG_LIVE_QUEUE_TARGET = max(1, int(os.getenv("FFMPEG_LIVE_QUEUE_TARGET", "3")))
FFMPEG_FIFO_MAX_BACKLOG = max(1, int(os.getenv("FFMPEG_FIFO_MAX_BACKLOG", "45")))

# For monitoring
frames_processed = 0
START_TIME = time.time()
LAST_FRAME_TS = time.monotonic()
CONSEC_EMPTY_POLLS = 0
REDIS_CLIENT: redis.Redis | None = None

# Runtime status (human/debug domain)
LAST_CONSUMED_METADATA: dict = {}
LAST_CONSUMED_CAPTURE_TS: float | None = None
LAST_CONSUMED_SEQ: int | None = None
LAST_CONSUMED_SOURCE_GENERATION: str | None = None
LAST_ENCODED_CAPTURE_TS: float | None = None
LAST_ENCODED_SEQ: int | None = None
LAST_ENCODED_SOURCE_GENERATION: str | None = None
LAST_ENCODED_AT: float | None = None
LAST_HLS_MTIME: float | None = None
SOURCE_GENERATION_REJECTION_REASON: str | None = None
SOURCE_GENERATION_REJECTION_COUNT = 0
LAST_GENERATION_REJECTION_HEALTH_TS = 0.0
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


def _update_encoded_evidence() -> dict:
    """Associate a newly written HLS playlist with the newest consumed source frame.

    Playlist mtime alone is not enough: a wedged encoder can keep touching files
    while the source frame is old.  We only advance encoded evidence when both a
    new playlist mtime and complete source metadata are present.
    """
    global LAST_HLS_MTIME, LAST_ENCODED_CAPTURE_TS, LAST_ENCODED_SEQ
    global LAST_ENCODED_SOURCE_GENERATION, LAST_ENCODED_AT
    info = get_hls_freshness()
    mtime = info.get("playlist_mtime")
    if mtime is not None and (LAST_HLS_MTIME is None or float(mtime) > LAST_HLS_MTIME):
        LAST_HLS_MTIME = float(mtime)
        if LAST_CONSUMED_CAPTURE_TS is not None and LAST_CONSUMED_SEQ is not None:
            LAST_ENCODED_CAPTURE_TS = LAST_CONSUMED_CAPTURE_TS
            LAST_ENCODED_SEQ = LAST_CONSUMED_SEQ
            LAST_ENCODED_SOURCE_GENERATION = LAST_CONSUMED_SOURCE_GENERATION
            LAST_ENCODED_AT = time.time()
    return info


def _live_rtsp_health() -> dict | None:
    """Read the capture heartbeat used to fence pre-restart frames."""
    if REDIS_CLIENT is None:
        return None
    try:
        raw = REDIS_CLIENT.get(RTSP_HEALTH_KEY)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except (redis_exceptions.RedisError, UnicodeDecodeError, TypeError, ValueError):
        return None


def _accept_processed_frame(metadata: dict) -> bool:
    """Accept only current processed-frame evidence with a complete identity."""
    global SOURCE_GENERATION_REJECTION_REASON, SOURCE_GENERATION_REJECTION_COUNT

    metadata_error = frame_metadata_error(metadata)
    if metadata_error:
        SOURCE_GENERATION_REJECTION_REASON = f"processed frame contract invalid: {metadata_error}"
        SOURCE_GENERATION_REJECTION_COUNT += 1
        status.update_custom_metric("stale_generation_drops", SOURCE_GENERATION_REJECTION_COUNT)
        return False

    source_generation = metadata["source_generation"]
    release_generation = metadata.get("release_generation")
    if release_generation and RELEASE_GENERATION not in {"", "unknown"}:
        if release_generation != RELEASE_GENERATION:
            SOURCE_GENERATION_REJECTION_REASON = (
                f"discarded frame from release generation {release_generation!r}; "
                f"current release is {RELEASE_GENERATION!r}"
            )
            SOURCE_GENERATION_REJECTION_COUNT += 1
            status.update_custom_metric("stale_generation_drops", SOURCE_GENERATION_REJECTION_COUNT)
            return False

    heartbeat = _live_rtsp_health()
    live_generation = heartbeat.get("source_generation") if heartbeat else None
    if not live_generation:
        SOURCE_GENERATION_REJECTION_REASON = "waiting for current RTSP source-generation heartbeat"
        SOURCE_GENERATION_REJECTION_COUNT += 1
        status.update_custom_metric("stale_generation_drops", SOURCE_GENERATION_REJECTION_COUNT)
        return False
    if live_generation != source_generation:
        SOURCE_GENERATION_REJECTION_REASON = (
            f"discarded frame from source generation {source_generation!r}; "
            f"current generation is {live_generation!r}"
        )
        SOURCE_GENERATION_REJECTION_COUNT += 1
        status.update_custom_metric("stale_generation_drops", SOURCE_GENERATION_REJECTION_COUNT)
        return False

    SOURCE_GENERATION_REJECTION_REASON = None
    return True


def _readiness_snapshot(info: dict | None = None) -> dict:
    info = info if info is not None else _update_encoded_evidence()
    now = time.time()
    hls_age = info.get("age_seconds")
    hls_threshold = float(info.get("threshold_sec", 60.0))
    hls_fresh = bool(info.get("exists")) and hls_age is not None and hls_age <= hls_threshold
    encoded_age = max(0.0, now - LAST_ENCODED_CAPTURE_TS) if LAST_ENCODED_CAPTURE_TS is not None else None
    encoded_threshold = max(10.0, FRAME_STALL_SEC)
    encoded_fresh = LAST_ENCODED_SEQ is not None and encoded_age is not None and encoded_age <= encoded_threshold
    redis_ok = bool(status.get_status_snapshot().get("redis_connected", False))
    consumed_age = max(0.0, now - LAST_CONSUMED_CAPTURE_TS) if LAST_CONSUMED_CAPTURE_TS is not None else None
    recent_frame_ok = LAST_CONSUMED_SEQ is not None and consumed_age is not None and consumed_age <= encoded_threshold
    reasons = []
    if not redis_ok:
        reasons.append("Redis is unavailable")
    if not recent_frame_ok:
        reasons.append("no fresh consumed capture evidence")
    if not hls_fresh:
        reasons.append("HLS playlist is missing or stale")
    if not encoded_fresh:
        reasons.append("no fresh encoded capture evidence")
    if SOURCE_GENERATION_REJECTION_REASON:
        reasons.append(SOURCE_GENERATION_REJECTION_REASON)
    ready = bool(
        redis_ok and recent_frame_ok and hls_fresh and encoded_fresh and not SOURCE_GENERATION_REJECTION_REASON
    )
    if ready:
        stage_status = "healthy"
        reason = "Redis, source capture, and encoded HLS evidence are current"
    elif SOURCE_GENERATION_REJECTION_REASON:
        stage_status = "deferred"
        reason = "; ".join(reasons)
    elif LAST_CONSUMED_SEQ is None or LAST_ENCODED_SEQ is None:
        stage_status = "deferred"
        reason = "; ".join(reasons) or "waiting for encoded evidence"
    elif not redis_ok:
        stage_status = "unavailable"
        reason = "; ".join(reasons)
    else:
        stage_status = "failed"
        reason = "; ".join(reasons) or "FFmpeg is not ready"
    return {
        "ready": ready,
        "status": stage_status,
        "reason": reason,
        "hls_fresh": hls_fresh,
        "hls_age_seconds": hls_age,
        "hls_threshold_seconds": hls_threshold,
        "encoded_capture_age_seconds": encoded_age,
        "encoded_capture_threshold_seconds": encoded_threshold,
        "encoded_fresh": encoded_fresh,
        "recent_frame_ok": recent_frame_ok,
        "redis_ok": redis_ok,
        "last_consumed_seq": LAST_CONSUMED_SEQ,
        "last_consumed_capture_ts": LAST_CONSUMED_CAPTURE_TS,
        "last_encoded_seq": LAST_ENCODED_SEQ,
        "last_encoded_capture_ts": LAST_ENCODED_CAPTURE_TS,
        "source_generation": LAST_ENCODED_SOURCE_GENERATION or LAST_CONSUMED_SOURCE_GENERATION,
        "segments_count": info.get("segments_count", 0),
    }


def _publish_generation_deferred() -> None:
    """Publish a throttled deferred heartbeat while draining stale frames."""
    global LAST_GENERATION_REJECTION_HEALTH_TS
    now = time.time()
    if now - LAST_GENERATION_REJECTION_HEALTH_TS < 5.0:
        return
    LAST_GENERATION_REJECTION_HEALTH_TS = now
    readiness = _readiness_snapshot()
    redis_set(
        FFMPEG_HEALTH_KEY,
        json.dumps(
            {
                "schema_version": 2,
                "status": "deferred",
                "release_generation": RELEASE_GENERATION,
                "project_identity": PROJECT_IDENTITY,
                "source_generation": readiness["source_generation"],
                "last_consumed_seq": readiness["last_consumed_seq"],
                "last_encoded_seq": readiness["last_encoded_seq"],
                "reason": readiness["reason"],
                "ts": now,
            }
        ),
        ex=FFMPEG_HEALTH_TTL_SEC,
    )


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            # Liveness is intentionally independent of Redis, HLS, or source freshness.
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"OK")
            return

        if self.path in ("/health", "/ready"):
            info = _update_encoded_evidence()
            snapshot = _readiness_snapshot(info)
            payload = {
                **snapshot,
                "schema_version": 2,
                "release_generation": RELEASE_GENERATION,
                "project_identity": PROJECT_IDENTITY,
                "status": snapshot["status"],
                # Legacy fields retained for existing probes.
                "exists": bool(info.get("exists")),
                "age_seconds": info.get("age_seconds"),
                "threshold_sec": info.get("threshold_sec"),
                "segments_count": info.get("segments_count"),
                "redis_connected": snapshot["redis_ok"],
                "hls_ok": snapshot["hls_fresh"],
                "encoded_fresh": snapshot["encoded_fresh"],
                "last_frame_age_seconds": max(0.0, time.monotonic() - LAST_FRAME_TS),
                "consec_empty_polls": CONSEC_EMPTY_POLLS,
                "note": "readiness-like; use /healthz for liveness",
            }
            self.send_response(200 if snapshot["ready"] else 503)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
            return

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
                info = _update_encoded_evidence()
                readiness = _readiness_snapshot(info)
                if readiness["ready"]:
                    last_fresh_ts = time.monotonic()
                else:
                    stalled_for = time.monotonic() - last_fresh_ts
                    if stalled_for > PIPELINE_WATCHDOG_SEC:
                        logger.error(
                            "Watchdog trip: status=%s reason=%s hls_age=%.1fs encoded_age=%s stalled_for=%.1fs",
                            readiness["status"],
                            readiness["reason"],
                            info.get("age_seconds") or -1,
                            readiness["encoded_capture_age_seconds"],
                            stalled_for,
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
        # Do not emit PROGRAM-DATE-TIME tags. FFmpeg derives PDT from internal
        # PTS, which drifts from wall clock in long-running real-time pipelines
        # and can make HLS.js think live edge is stale/frozen.
        "delete_segments",
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

        frame_data = r.lindex(PROCESSED_FRAME_KEY, 0)
        if frame_data:
            decoded = unpack_frame(frame_data)
            if decoded is not None:
                try:
                    width = int(decoded.metadata.get("width", 0))
                    height = int(decoded.metadata.get("height", 0))
                    if width > 0 and height > 0:
                        return (width, height)
                except Exception as e:
                    logger.warning("Failed to parse processed frame envelope shape: %s", e)
        time.sleep(0.5)


def decode_processed_frame(data: bytes, include_metadata: bool = False):
    """Decode processed bytes while retaining the legacy two-value API by default."""
    decoded = unpack_frame(data)
    if decoded is None:
        return (data, None, {}) if include_metadata else (data, None)

    try:
        width = int(decoded.metadata.get("width", 0))
        height = int(decoded.metadata.get("height", 0))
        shape = (width, height) if width > 0 and height > 0 else None
    except Exception:
        shape = None
    if include_metadata:
        return decoded.payload, shape, dict(decoded.metadata)
    return decoded.payload, shape


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
            logger.info("Connected to Redis")
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


def safe_brpop(key: str, timeout: float = 5, pop_left: bool = False):
    """Reconnect-aware queue pop.

    ``pop_left`` selects newest-first for LPUSH queues; FIFO remains the
    historical BRPOP behavior.
    """
    global CONSEC_EMPTY_POLLS
    global REDIS_CLIENT
    try:
        if REDIS_CLIENT is None:
            REDIS_CLIENT = connect_redis_with_backoff(REDIS_URL)
        item = REDIS_CLIENT.blpop([key], timeout=timeout) if pop_left else REDIS_CLIENT.brpop([key], timeout=timeout)
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
        CONSEC_EMPTY_POLLS += 1
        status.update_custom_metric("consec_empty_polls", CONSEC_EMPTY_POLLS)
        metrics_obj = globals().get("metrics")
        if metrics_obj is not None:
            metrics_obj.redis_empty_polls.add(1, {"service": "ffmpeg"})
            metrics_obj.queue_depth.record(0, {"queue": key})
        return None
    except redis_exceptions.RedisError as e:
        logger.warning("Redis queue pop error: %s", e)
        status.update_status_metric("redis_connected", False)
        REDIS_CLIENT = connect_redis_with_backoff(REDIS_URL)
        return None


def redis_queue_depth(key: str) -> int | None:
    """Read queue depth through the same reconnecting client path as pops."""
    global REDIS_CLIENT
    try:
        if REDIS_CLIENT is None:
            REDIS_CLIENT = connect_redis_with_backoff(REDIS_URL)
        return int(REDIS_CLIENT.llen(key))
    except redis_exceptions.RedisError as exc:
        logger.warning("Redis queue depth read failed: %s", exc)
        status.update_status_metric("redis_connected", False)
        REDIS_CLIENT = connect_redis_with_backoff(REDIS_URL)
        return None


def redis_trim_queue(key: str, start: int, end: int) -> bool:
    global REDIS_CLIENT
    try:
        if REDIS_CLIENT is None:
            REDIS_CLIENT = connect_redis_with_backoff(REDIS_URL)
        REDIS_CLIENT.ltrim(key, start, end)
        return True
    except redis_exceptions.RedisError as exc:
        logger.warning("Redis queue trim failed: %s", exc)
        status.update_status_metric("redis_connected", False)
        REDIS_CLIENT = connect_redis_with_backoff(REDIS_URL)
        return False


def start_ffmpeg_process(frame_shape):
    """Starts the FFmpeg subprocess with the correct frame shape."""
    logger.info(
        "Starting FFmpeg process with frame shape: %s - encoder: %s / preset: %s",
        frame_shape,
        ENCODER,
        PRESET,
    )
    ffmpeg_cmd = create_ffmpeg_cmd(frame_shape)
    # bufsize=0: unbuffered stdin — frames go directly to OS pipe, no Python
    # buffering layer. Eliminates flush() overhead and per-frame buffer latency.
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)

    return process


def redis_set(key: str, value: str, ex: int | None = None) -> bool:
    global REDIS_CLIENT
    try:
        if REDIS_CLIENT is None:
            REDIS_CLIENT = connect_redis_with_backoff(REDIS_URL)
        REDIS_CLIENT.set(key, value, ex=ex)
        return True
    except redis_exceptions.RedisError as exc:
        logger.warning("Redis health write failed: %s", exc)
        status.update_status_metric("redis_connected", False)
        REDIS_CLIENT = connect_redis_with_backoff(REDIS_URL)
        return False


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
    expected_frame_bytes = frame_shape[0] * frame_shape[1] * 3

    ffmpeg_process = start_ffmpeg_process(frame_shape)
    if ffmpeg_process is None or ffmpeg_process.stdin is None:
        logger.error("Failed to start FFmpeg process")
        return
    global LAST_HLS_MTIME, LAST_CONSUMED_METADATA, LAST_CONSUMED_CAPTURE_TS
    global LAST_CONSUMED_SEQ, LAST_CONSUMED_SOURCE_GENERATION
    global LAST_ENCODED_CAPTURE_TS, LAST_ENCODED_SEQ, LAST_ENCODED_SOURCE_GENERATION, LAST_ENCODED_AT
    baseline_info = get_hls_freshness()
    LAST_HLS_MTIME = baseline_info.get("playlist_mtime")
    redis_set(
        FFMPEG_HEALTH_KEY,
        json.dumps(
            {
                "schema_version": 2,
                "status": "deferred",
                "release_generation": RELEASE_GENERATION,
                "project_identity": PROJECT_IDENTITY,
                "reason": "FFmpeg started; waiting for encoded capture evidence",
                "ts": time.time(),
            }
        ),
        ex=FFMPEG_HEALTH_TTL_SEC,
    )

    # Accumulating-deadline paced loop (Codex-reviewed, three failure modes addressed):
    #
    #   1. On time / slightly late (normal):
    #      sleep remainder, write at deadline, advance deadline by frame_interval.
    #      No drift because deadline is accumulated, not re-derived from now.
    #
    #   2. Stall (YOLO spike, Redis hiccup):
    #      Redis pop blocks > frame_interval. next_write_t falls behind now.
    #      On recovery, snap next_write_t = now to resume real-time cadence
    #      instead of bursting through the backlog. Queued frames are consumed
    #      but written at steady pace, not all at once.
    #
    #   3. Startup (no frame yet):
    #      Advance deadline but skip write. No write(None) crash.
    target_fps = float(FRAME_RATE)
    frame_interval = 1.0 / target_fps
    last_raw_frame = None
    next_write_t = time.monotonic()

    # FPS logging
    fps_frame_count = 0
    fps_last_log_time = time.monotonic()
    new_frame_count = 0
    starved_frame_count = 0
    snap_count = 0  # times stall-recovery snap fired

    try:
        logger.info(
            (
                "Starting paced frame loop (target %.1f FPS, pop_mode=%s, "
                "live_queue_target=%d, fifo_max_backlog=%d, bufsize=0)"
            ),
            target_fps,
            FFMPEG_POP_MODE,
            FFMPEG_LIVE_QUEUE_TARGET,
            FFMPEG_FIFO_MAX_BACKLOG,
        )
        global frames_processed
        stale_drop_count = 0
        while True:
            try:
                task = safe_brpop(
                    PROCESSED_FRAME_KEY,
                    timeout=frame_interval * 2,
                    pop_left=FFMPEG_POP_MODE == "latest",
                )
            except Exception as e:
                logger.warning("Redis pop error: %s", e)
                task = None

            if task:
                _, data = task
                frame_bytes, envelope_shape, frame_metadata = decode_processed_frame(data, include_metadata=True)
                if not _accept_processed_frame(frame_metadata):
                    # Do not keep writing the previous generation while waiting
                    # for a frame from the current RTSP process.
                    last_raw_frame = None
                    _publish_generation_deferred()
                    continue
                if envelope_shape is not None and envelope_shape != frame_shape:
                    logger.warning(
                        "Processed frame shape changed (got %s, expected %s); restarting ffmpeg",
                        envelope_shape,
                        frame_shape,
                    )
                    if ffmpeg_process and ffmpeg_process.stdin:
                        ffmpeg_process.stdin.close()
                    if ffmpeg_process:
                        ffmpeg_process.terminate()
                        ffmpeg_process.wait()
                    frame_shape = envelope_shape
                    expected_frame_bytes = frame_shape[0] * frame_shape[1] * 3
                    ffmpeg_process = start_ffmpeg_process(frame_shape)
                    last_raw_frame = None
                    continue

                if len(frame_bytes) != expected_frame_bytes:
                    logger.warning(
                        "Processed frame size mismatch (got %d bytes, expected %d)",
                        len(frame_bytes),
                        expected_frame_bytes,
                    )
                else:
                    last_raw_frame = frame_bytes
                    LAST_CONSUMED_METADATA = dict(frame_metadata)
                    try:
                        LAST_CONSUMED_CAPTURE_TS = float(frame_metadata.get("capture_ts", frame_metadata.get("ts")))
                    except (TypeError, ValueError):
                        LAST_CONSUMED_CAPTURE_TS = None
                    try:
                        LAST_CONSUMED_SEQ = int(frame_metadata.get("source_seq"))
                    except (TypeError, ValueError):
                        LAST_CONSUMED_SEQ = None
                    LAST_CONSUMED_SOURCE_GENERATION = frame_metadata.get("source_generation")
                    new_frame_count += 1
                    if FFMPEG_POP_MODE == "latest":
                        # Keep only a tiny backlog so stream time tracks capture time.
                        try:
                            queue_depth = redis_queue_depth(PROCESSED_FRAME_KEY) or 0
                            if queue_depth > FFMPEG_LIVE_QUEUE_TARGET:
                                redis_trim_queue(PROCESSED_FRAME_KEY, 0, FFMPEG_LIVE_QUEUE_TARGET - 1)
                                stale_drop_count += queue_depth - FFMPEG_LIVE_QUEUE_TARGET
                            status.update_custom_metric(
                                "queue_depth",
                                min(queue_depth, FFMPEG_LIVE_QUEUE_TARGET),
                            )
                        except Exception as trim_err:
                            logger.debug("Failed to trim processed frame backlog: %s", trim_err)
                    else:
                        # FIFO mode keeps strict ordering, but if backlog grows too large
                        # we cap it to prevent multi-second stale-video drift.
                        try:
                            queue_depth = redis_queue_depth(PROCESSED_FRAME_KEY) or 0
                            if queue_depth > FFMPEG_FIFO_MAX_BACKLOG:
                                redis_trim_queue(PROCESSED_FRAME_KEY, 0, FFMPEG_FIFO_MAX_BACKLOG - 1)
                                stale_drop_count += queue_depth - FFMPEG_FIFO_MAX_BACKLOG
                                logger.warning(
                                    "FIFO backlog too deep (%d); dropped %d stale frame(s), keeping newest %d",
                                    queue_depth,
                                    queue_depth - FFMPEG_FIFO_MAX_BACKLOG,
                                    FFMPEG_FIFO_MAX_BACKLOG,
                                )
                            status.update_custom_metric(
                                "queue_depth",
                                min(queue_depth, FFMPEG_FIFO_MAX_BACKLOG),
                            )
                        except Exception as trim_err:
                            logger.debug("Failed to cap FIFO processed backlog: %s", trim_err)
            elif last_raw_frame:
                starved_frame_count += 1

            # No frame yet (startup) — advance deadline and wait
            if last_raw_frame is None:
                next_write_t += frame_interval
                continue

            # Stall recovery: if we've fallen more than one interval behind,
            # snap forward instead of bursting to catch up
            now = time.monotonic()
            if next_write_t < now - frame_interval:
                snap_count += 1
                next_write_t = now
            elif next_write_t > now:
                time.sleep(next_write_t - now)

            if ffmpeg_process.stdin:
                try:
                    ffmpeg_process.stdin.write(last_raw_frame)
                    frames_processed += 1
                    fps_frame_count += 1

                    global LAST_FRAME_TS
                    LAST_FRAME_TS = time.monotonic()
                    status.update_status_metric("current_fps", target_fps)
                    status.increment_counter("processed_count", 1)
                    metrics.frames_processed.add(1, {"service": "ffmpeg"})
                except BrokenPipeError:
                    logger.error("FFmpeg process closed unexpectedly. Restarting...")
                    ffmpeg_process = start_ffmpeg_process(frame_shape)
                    if not ffmpeg_process or not ffmpeg_process.stdin:
                        logger.error("FFmpeg restart failed: stdin is None")

            next_write_t += frame_interval  # advance ideal timeline

            # Log FPS every 5 seconds
            now = time.monotonic()
            if now - fps_last_log_time >= 5.0:
                elapsed = now - fps_last_log_time
                actual_fps = fps_frame_count / elapsed
                new_fps = new_frame_count / elapsed
                # Share of output slots written with the previous frame because no new
                # frame was available: an upstream underrun measure, not a check for
                # duplicate *content*.
                starved_pct = (starved_frame_count / fps_frame_count * 100) if fps_frame_count > 0 else 0
                logger.info(
                    "FFmpeg output: %.1f FPS (new: %.1f, starved: %.0f%%, snaps: %d, dropped_stale: %d)",
                    actual_fps,
                    new_fps,
                    starved_pct,
                    snap_count,
                    stale_drop_count,
                )
                # Pipeline-health signal: persist the fresh-vs-starved snapshot so the
                # web /api/pipeline-health endpoint and Sauron can alert without
                # scraping logs. A high starved percentage means no new frame reached
                # the encoder, i.e. the capture->analyzer chain is underrun somewhere
                # upstream (camera link loss, ingest stall, analyzer stall). It is NOT
                # evidence of a frozen picture - a live camera whose WiFi link is
                # dropping frames produces exactly this signature. `dup_pct`/
                # `dup_count` keep their published names for compatibility.
                hls_info = _update_encoded_evidence()
                readiness = _readiness_snapshot(hls_info)
                payload = {
                    "schema_version": 2,
                    "status": readiness["status"],
                    "release_generation": RELEASE_GENERATION,
                    "project_identity": PROJECT_IDENTITY,
                    "source_generation": readiness["source_generation"],
                    "last_consumed_seq": readiness["last_consumed_seq"],
                    "last_consumed_capture_ts": readiness["last_consumed_capture_ts"],
                    "last_encoded_seq": readiness["last_encoded_seq"],
                    "last_encoded_capture_ts": readiness["last_encoded_capture_ts"],
                    "encoded_capture_age_seconds": readiness["encoded_capture_age_seconds"],
                    "hls_age_seconds": readiness["hls_age_seconds"],
                    "hls_fresh": readiness["hls_fresh"],
                    "output_fps": round(actual_fps, 2),
                    "fps": round(actual_fps, 2),
                    "new_fps": round(new_fps, 2),
                    "starved_slot_ratio": round(starved_pct / 100.0, 4),
                    "dup_pct": round(starved_pct, 1),
                    "dup_count": starved_frame_count,
                    "new_count": new_frame_count,
                    "snaps": snap_count,
                    "dropped_stale": stale_drop_count,
                    "queue_depth": redis_queue_depth(PROCESSED_FRAME_KEY),
                    "redis_connected": readiness["redis_ok"],
                    "last_frame_age_sec": round(max(0.0, time.monotonic() - LAST_FRAME_TS), 1),
                    "reason": readiness["reason"],
                    "ts": time.time(),
                }
                redis_set(FFMPEG_HEALTH_KEY, json.dumps(payload), ex=FFMPEG_HEALTH_TTL_SEC)
                fps_frame_count = 0
                new_frame_count = 0
                starved_frame_count = 0
                snap_count = 0
                stale_drop_count = 0
                fps_last_log_time = now

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


HLS_SEGMENT_SECONDS = 2
# The live window ffmpeg itself keeps (hls_list_size x hls_time). Anything older than
# this cannot belong to a playlist a viewer is holding, so it is safe to drop at
# startup; anything younger may still be being watched.
HLS_WINDOW_SECONDS = int(HLS_LIST_SIZE) * HLS_SEGMENT_SECONDS


def clean_stream_directory(max_age_seconds: float = HLS_WINDOW_SECONDS) -> None:
    """Drop HLS files from the stream directory that are older than the live window.

    Deleting the whole directory at startup 404s the public stream for the length of
    the restart: a viewer is holding a playlist whose segments have just been removed.
    ffmpeg manages its own window (``-hls_flags delete_segments``), so the only thing
    a restart has to clear is what an earlier, longer-lived session left behind -
    which is by definition older than the window.
    """
    logger.info("Pruning stream directory files older than %ss...", max_age_seconds)
    os.makedirs(STREAM_DIR, exist_ok=True)
    cutoff = time.time() - max_age_seconds
    for filename in os.listdir(STREAM_DIR):
        if filename == "clips":
            continue
        file_path = os.path.join(STREAM_DIR, filename)
        try:
            if os.path.isdir(file_path) and not os.path.islink(file_path):
                shutil.rmtree(file_path)
                continue
            if os.path.getmtime(file_path) >= cutoff:
                continue
            os.unlink(file_path)
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.error(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    main()
