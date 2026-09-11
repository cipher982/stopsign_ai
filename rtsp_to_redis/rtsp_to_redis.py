"""RTSP to Redis service.

This module reads frames from an RTSP (or ``file://``) source and pushes JPEG
bytes into a Redis list so that the rest of the StopSign pipeline can pick
them up.  Prometheus metrics are exposed on an HTTP port for observability.

The only purpose of this edit is to satisfy Ruff/flake8 rule **E402 – “module
level import not at top of file”**.  Imports were previously sprinkled below
code that executed at import-time (logging configuration, ``sys.path`` hacks,
etc.).  All imports are now consolidated at the very top of the file before
any other statements, as PEP 8 expects.
"""

# isort: skip_file
# ruff: format
from __future__ import annotations

# ----------------- standard library -----------------
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import os
from queue import Empty, Queue
import threading
import time
from typing import Optional
import json

# ------------------ third-party ---------------------
import cv2
import redis
from redis.exceptions import RedisError
from stopsign.frame_codec import pack_legacy_jpeg_frame
from stopsign.freeze_detector import FrameFreezeDetector
from stopsign.telemetry import setup_rtsp_service_telemetry, get_tracer
from stopsign.service_status import RTSPServiceStatusMixin


# ----------------- logging setup --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ------------------- local app ----------------------
def get_env(key: str) -> str:
    value = os.getenv(key)
    assert value is not None, f"{key} is not set"
    if any(sensitive in key.lower() for sensitive in ["password", "secret", "key", "token", "url"]):
        logger.info(f"Loaded env var {key}: [REDACTED]")
    else:
        logger.info(f"Loaded env var {key}: {value}")
    return value


PROMETHEUS_PORT: int = int(get_env("PROMETHEUS_PORT"))
RTSP_URL: str = get_env("RTSP_URL")
REDIS_URL: str = get_env("REDIS_URL")
RAW_FRAME_KEY: str = get_env("RAW_FRAME_KEY")
FRAME_BUFFER_SIZE: int = int(get_env("FRAME_BUFFER_SIZE"))

# Freeze detection knobs. The MAD detector sees a *visually frozen* stream: same
# picture, arriving 15 times a second. It is blind to the other way the camera
# starves this service - live, changing frames that only arrive at 0.2-4 FPS
# because the WiFi link is dropping them. That mode starves every downstream
# stage while looking perfectly healthy here, so arrival rate is tracked
# separately (RTSP_MIN_INPUT_FPS) and escalates on its own.
RTSP_FREEZE_DETECT_SEC: float = float(os.getenv("RTSP_FREEZE_DETECT_SEC", "120"))
RTSP_FREEZE_MAD_THRESHOLD: float = float(os.getenv("RTSP_FREEZE_MAD_THRESHOLD", "0.015"))
RTSP_FREEZE_SAMPLE_WIDTH: int = int(os.getenv("RTSP_FREEZE_SAMPLE_WIDTH", "160"))
RTSP_FREEZE_SAMPLE_HEIGHT: int = int(os.getenv("RTSP_FREEZE_SAMPLE_HEIGHT", "90"))
RTSP_FREEZE_RECONNECT_SEC: float = float(os.getenv("RTSP_FREEZE_RECONNECT_SEC", "180"))
RTSP_FREEZE_RECONNECT_COOLDOWN_SEC: float = float(os.getenv("RTSP_FREEZE_RECONNECT_COOLDOWN_SEC", "60"))

# Ingest-rate guard (0 disables the guard, or the exit stage when 0).
RTSP_MIN_INPUT_FPS: float = float(os.getenv("RTSP_MIN_INPUT_FPS", "8"))
RTSP_LOW_FPS_RECONNECT_SEC: float = float(os.getenv("RTSP_LOW_FPS_RECONNECT_SEC", "120"))
RTSP_LOW_FPS_EXIT_SEC: float = float(os.getenv("RTSP_LOW_FPS_EXIT_SEC", "900"))

# Readiness tolerance for a below-floor input rate, so one quiet second does not
# flap the probe.
READY_LOW_INPUT_FPS_SEC: float = 30.0


class RTSPToRedis(RTSPServiceStatusMixin):
    def __init__(self):
        # Initialize status tracking first
        super().__init__()

        # Service configuration
        self.rtsp_url = RTSP_URL
        self.redis_url = REDIS_URL
        self.prometheus_port = PROMETHEUS_PORT
        self.frame_buffer_size = FRAME_BUFFER_SIZE
        self.fps = 15
        self.jpeg_quality = max(1, min(100, int(os.getenv("RTSP_JPEG_QUALITY", "85"))))

        # Service state
        self.redis_client: Optional[redis.Redis] = None
        self.frame_queue = Queue(maxsize=1000)
        self.processing_thread = None
        self.should_stop = threading.Event()
        self.last_push_ts: Optional[float] = None  # wall-clock time of last successful Redis push

        # Freeze detection and remediation state
        self.freeze_detector = (
            FrameFreezeDetector(
                freeze_detect_sec=RTSP_FREEZE_DETECT_SEC,
                mad_threshold=RTSP_FREEZE_MAD_THRESHOLD,
                sample_width=RTSP_FREEZE_SAMPLE_WIDTH,
                sample_height=RTSP_FREEZE_SAMPLE_HEIGHT,
            )
            if RTSP_FREEZE_DETECT_SEC > 0
            else None
        )
        self.freeze_age_sec = 0.0
        self.freeze_mad = 0.0
        self.freeze_incidents = 0
        self.freeze_active_incident_id = 0
        self.freeze_incident_start_ts: Optional[float] = None
        self.last_reconnect_ts = 0.0

        # Ingest-rate state: when the camera's frame arrival rate first dropped
        # below RTSP_MIN_INPUT_FPS, or None while the rate is healthy.
        self.last_input_fps = 0.0
        self.low_input_fps_since: Optional[float] = None

        # Surface freeze/ingest state in status logs/health output.
        self.update_custom_metric("freeze_age_sec", 0.0)
        self.update_custom_metric("freeze_mad", 0.0)
        self.update_custom_metric("frozen", 0)
        self.update_custom_metric("freeze_incidents", 0)
        self.update_custom_metric("input_fps", 0.0)
        self.update_custom_metric("low_input_fps_sec", 0.0)
        self.update_custom_metric("low_input_fps_events", 0)

        # OpenTelemetry metrics and tracer (set from main)
        self.metrics = None
        self.tracer = None

    def set_telemetry(self, metrics, tracer):
        """Set OpenTelemetry metrics and tracer instances."""
        self.metrics = metrics
        self.tracer = tracer

    def initialize_redis(self):
        logger.info("Attempting to connect to Redis")
        try:
            self.redis_client = redis.from_url(self.redis_url, socket_timeout=5)
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
            # Update connection status
            self.update_status_metric("redis_connected", True)
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.update_status_metric("redis_connected", False)
            if self.metrics:
                self.metrics.redis_operations.add(1, {"operation": "error", "service": "rtsp"})
                self.metrics.service_errors.add(1, {"error_type": "redis_connection", "service": "rtsp"})
            raise

    def initialize_capture(self):
        # Check if this is a file:// URL for local development
        if self.rtsp_url.startswith("file://"):
            file_path = self.rtsp_url[len("file://") :]
            logger.info(f"Attempting to open local video file at {file_path}")
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {file_path}")
            logger.info("Local video capture initialized successfully")
            self.update_status_metric("rtsp_connected", True)
            if self.metrics:
                self.metrics.redis_operations.add(1, {"operation": "rtsp_connected", "service": "rtsp"})
            return cap

        # Standard RTSP connection logic
        logger.info("Attempting to connect to RTSP source")
        max_attempts = 5
        for attempt in range(max_attempts):
            if not self.rtsp_url:
                raise ValueError("RTSP URL is not set")
            try:
                cap = cv2.VideoCapture(self.rtsp_url)
                # Increase timeout for slow network connections (Tailscale routing)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)  # 60 second timeout
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)  # 10 second read timeout
                if not cap.isOpened():
                    raise ValueError("Could not open video stream")
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                logger.info("Video capture initialized successfully")
                self.update_status_metric("rtsp_connected", True)
                if self.metrics:
                    self.metrics.redis_operations.add(1, {"operation": "rtsp_connected", "service": "rtsp"})
                return cap
            except Exception as e:
                # Retry tracked in OpenTelemetry spans
                # RTSP errors tracked in OpenTelemetry
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                time.sleep(1)
        # Connection status tracked in OpenTelemetry
        raise ValueError("Failed to initialize video capture after multiple attempts")

    def _pack_frame(self, jpeg_bytes: bytes, capture_ts: float, width: int, height: int) -> bytes:
        return pack_legacy_jpeg_frame(jpeg_bytes, capture_ts=capture_ts, width=width, height=height)

    def process_frames(self):
        frames_processed = 0
        last_fps_update = time.time()

        while not self.should_stop.is_set():
            try:
                item = self.frame_queue.get(timeout=1)
                if isinstance(item, tuple):
                    frame, capture_ts = item
                else:
                    frame, capture_ts = item, time.time()  # backward safety
                with self.tracer.start_as_current_span("store_frame") as span:
                    span.set_attribute("frame.height", frame.shape[0])
                    span.set_attribute("frame.width", frame.shape[1])
                    span.set_attribute("frame.channels", frame.shape[2])
                    self.store_frame(frame, capture_ts)
                self.frame_queue.task_done()

                # Update runtime status
                self.update_status_metric("queue_size", self.frame_queue.qsize())
                self.increment_counter("processed_count", 1)

                # Record OpenTelemetry business event
                if self.metrics:
                    self.metrics.frames_processed.add(1, {"service": "rtsp"})

                frames_processed += 1
                current_time = time.time()
                if current_time - last_fps_update >= 1:
                    # Update processed fps tracking
                    frames_processed = 0
                    last_fps_update = current_time

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                if self.metrics:
                    self.metrics.redis_operations.add(1, {"operation": "error", "service": "rtsp"})

    def store_frame(self, frame, capture_ts: float):
        with self.tracer.start_as_current_span("encode_frame") as span:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            span.set_attribute("jpeg.quality", self.jpeg_quality)
            span.set_attribute("encoded.size_bytes", len(buffer))

        if self.redis_client is None:
            logger.error("Redis client is not initialized")
            return

        try:
            with self.tracer.start_as_current_span("redis_publish") as span:
                redis_start_time = time.time()
                pipeline = self.redis_client.pipeline()
                packed = self._pack_frame(
                    buffer.tobytes(), capture_ts=capture_ts, width=frame.shape[1], height=frame.shape[0]
                )
                pipeline.lpush(RAW_FRAME_KEY, packed)
                pipeline.ltrim(RAW_FRAME_KEY, 0, self.frame_buffer_size - 1)
                pipeline.llen(RAW_FRAME_KEY)
                _, _, current_buffer_size = pipeline.execute()
                self.last_push_ts = time.time()
                redis_duration = time.time() - redis_start_time

                span.set_attribute("redis.operation", "pipeline_publish")
                span.set_attribute("redis.buffer_size", current_buffer_size)
                span.set_attribute("redis.duration_seconds", redis_duration)
                span.set_attribute("frame.buffer_size_bytes", len(buffer))

                # Redis latency now tracked in OpenTelemetry spans

            # Record OpenTelemetry metrics
            if self.metrics:
                self.metrics.frames_processed.add(1, {"service": "rtsp"})
                self.metrics.redis_operations.add(1, {"operation": "frame_publish", "service": "rtsp"})
                self.metrics.db_operation_duration.record(redis_duration)

        except RedisError as e:
            logger.error(f"Redis operation failed: {str(e)}")
            if self.metrics:
                self.metrics.redis_operations.add(1, {"operation": "error", "service": "rtsp"})
            raise

        # Frame processing time tracked in OpenTelemetry spans

    def _increment_custom_metric(self, key: str, amount: int = 1) -> None:
        custom = self.get_status_snapshot().get("custom_metrics", {})
        current = int(custom.get(key, 0))
        self.update_custom_metric(key, current + amount)

    def _update_freeze_state(self, frame, now_ts: float) -> None:
        if self.freeze_detector is None:
            self.freeze_age_sec = 0.0
            self.freeze_mad = 0.0
            return

        event = self.freeze_detector.update(frame, now_ts)
        self.freeze_age_sec = event.freeze_age_sec
        self.freeze_mad = event.mad
        self.update_custom_metric("freeze_age_sec", round(self.freeze_age_sec, 2))
        self.update_custom_metric("freeze_mad", round(self.freeze_mad, 4))
        self.update_custom_metric("frozen", 1 if event.frozen else 0)

        if event.incident_started:
            self.freeze_incidents += 1
            self.freeze_active_incident_id = self.freeze_incidents
            self.freeze_incident_start_ts = now_ts
            self.update_custom_metric("freeze_incidents", self.freeze_incidents)
            logger.error(
                "Freeze incident #%d opened: freeze_age=%.1fs threshold=%.1fs mad=%.4f (threshold=%.4f)",
                self.freeze_active_incident_id,
                self.freeze_age_sec,
                RTSP_FREEZE_DETECT_SEC,
                self.freeze_mad,
                RTSP_FREEZE_MAD_THRESHOLD,
            )
            if self.metrics:
                self.metrics.service_errors.add(1, {"error_type": "frozen_stream", "service": "rtsp"})

        if event.incident_resolved:
            incident_duration = 0.0
            if self.freeze_incident_start_ts is not None:
                incident_duration = max(0.0, now_ts - self.freeze_incident_start_ts)
            self.freeze_incident_start_ts = None
            logger.info(
                "Freeze incident #%d resolved after %.1fs (mad=%.4f)",
                self.freeze_active_incident_id,
                incident_duration,
                self.freeze_mad,
            )

    def _record_rtsp_reconnect(self) -> None:
        self._increment_custom_metric("rtsp_reconnects", 1)
        self.last_reconnect_ts = time.time()

    def _track_input_fps(self, fps: float, now_ts: float) -> None:
        """Track sustained camera frame arrival rate.

        The freeze detector answers "is the picture moving?"; this answers "are
        frames arriving?". A lossy camera link fails the second question while
        passing the first, and every stage downstream starves.
        """
        self.last_input_fps = fps
        self.update_custom_metric("input_fps", round(fps, 2))
        if RTSP_MIN_INPUT_FPS <= 0:
            return

        if fps >= RTSP_MIN_INPUT_FPS:
            if self.low_input_fps_since is not None:
                logger.info(
                    "Camera input rate recovered: %.2f FPS after %.1fs below %.1f FPS",
                    fps,
                    now_ts - self.low_input_fps_since,
                    RTSP_MIN_INPUT_FPS,
                )
            self.low_input_fps_since = None
            self.update_custom_metric("low_input_fps_sec", 0.0)
            return

        if self.low_input_fps_since is None:
            self.low_input_fps_since = now_ts
            self._increment_custom_metric("low_input_fps_events", 1)
            logger.warning("Camera input rate below floor: %.2f FPS < %.1f FPS", fps, RTSP_MIN_INPUT_FPS)
        self.update_custom_metric("low_input_fps_sec", round(now_ts - self.low_input_fps_since, 1))

    def _reconnect_reason(self, now_ts: float) -> Optional[str]:
        """Why capture should be re-opened right now, or None if it should not.

        Both triggers share one cooldown: a reconnect costs a couple of seconds of
        frames, so it is only worth it once per cooldown either way.
        """
        if now_ts - self.last_reconnect_ts < RTSP_FREEZE_RECONNECT_COOLDOWN_SEC:
            return None

        if self.freeze_detector is not None and RTSP_FREEZE_RECONNECT_SEC > 0:
            if self.freeze_age_sec >= RTSP_FREEZE_RECONNECT_SEC:
                return f"picture frozen for {self.freeze_age_sec:.1f}s"

        if RTSP_LOW_FPS_RECONNECT_SEC > 0 and self.low_input_fps_since is not None:
            below_for = now_ts - self.low_input_fps_since
            if below_for >= RTSP_LOW_FPS_RECONNECT_SEC:
                return f"input rate below {RTSP_MIN_INPUT_FPS:.1f} FPS for {below_for:.1f}s"

        return None

    def _exit_if_ingest_degraded(self, now_ts: float) -> None:
        """Restart the container when the camera has starved us past recovery.

        A fresh process re-opens the RTSP session and clears any wedged decoder
        state; ``restart: always`` brings it straight back. Same idiom as the
        analyzer and ffmpeg watchdogs, and the last resort the old
        remediation-command hook never actually performed.
        """
        if RTSP_LOW_FPS_EXIT_SEC <= 0 or self.low_input_fps_since is None:
            return

        degraded_for = now_ts - self.low_input_fps_since
        if degraded_for < RTSP_LOW_FPS_EXIT_SEC:
            return
        if not self.get_status_snapshot().get("redis_connected", False):
            # Redis is what is actually down; restarting capture would not help.
            return

        logger.error(
            "Ingest degraded for %.1fs (camera below %.1f FPS); exiting so the container restarts",
            degraded_for,
            RTSP_MIN_INPUT_FPS,
        )
        os._exit(1)

    def run(self):
        # Prometheus removed - using OpenTelemetry instead
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        last_log_time = time.time()
        log_interval = 60

        logger.info("RTSP to Redis service starting...")
        self.log_status_summary()

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
                            # Handle end of video file by looping back to start
                            if self.rtsp_url.startswith("file://"):
                                logger.info("End of video file reached. Looping back to start.")
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ret, frame = cap.read()
                                if not ret:
                                    logger.error("Failed to read frame after reset. Reinitializing capture.")
                                    # RTSP errors tracked in OpenTelemetry
                                    break
                            else:
                                logger.warning("Failed to read frame. Reinitializing capture.")
                                # RTSP errors tracked in OpenTelemetry
                                break

                        rtsp_frames_count += 1

                        # Stamp capture moment as close to cap.read() as possible
                        capture_ts = time.time()
                        self._update_freeze_state(frame, capture_ts)
                        self._exit_if_ingest_degraded(capture_ts)

                        reconnect_reason = self._reconnect_reason(capture_ts)
                        if reconnect_reason:
                            logger.error(
                                "Forcing RTSP reconnect: %s (freeze_age=%.1fs, input_fps=%.2f, cooldown=%.0fs)",
                                reconnect_reason,
                                self.freeze_age_sec,
                                self.last_input_fps,
                                RTSP_FREEZE_RECONNECT_COOLDOWN_SEC,
                            )
                            self._record_rtsp_reconnect()
                            break

                        if not self.frame_queue.full():
                            self.frame_queue.put((frame, capture_ts))
                        else:
                            logger.warning("Frame queue is full. Dropping frame.")
                            self.record_frame_drop()

                        last_frame_time = current_time

                        # Update FPS every second
                        if current_time - fps_update_time >= 1:
                            elapsed_fps_time = current_time - fps_update_time
                            calculated_fps = rtsp_frames_count / elapsed_fps_time
                            self.update_rtsp_fps(calculated_fps)
                            self._track_input_fps(calculated_fps, current_time)
                            rtsp_frames_count = 0
                            fps_update_time = current_time

                        # Log status periodically
                        if current_time - last_log_time >= log_interval:
                            self.log_status_summary()
                            last_log_time = current_time

                    else:
                        time.sleep(frame_time - elapsed_time)

            except RedisError as e:
                logger.error(f"Redis error: {str(e)}")
                self.record_redis_error()
                self.increment_counter("disconnects", 1)
                self.update_status_metric("redis_connected", False)
                if self.metrics:
                    self.metrics.service_errors.add(1, {"error_type": "redis", "service": "rtsp"})
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in RTSP to Redis service: {str(e)}")
                self.increment_counter("error_count", 1)
                self.update_status_metric("rtsp_connected", False)
                if self.metrics:
                    self.metrics.service_errors.add(1, {"error_type": "rtsp", "service": "rtsp"})
                time.sleep(1)
            finally:
                if cap:
                    cap.release()
                # Connection status tracked in OpenTelemetry
                logger.info("RTSP to Redis service restarting...")

    def stop(self):
        self.should_stop.set()
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("RTSP to Redis service stopped.")

    def get_readiness_report(self):
        """Composite readiness snapshot for RTSP ingest."""
        # Update buffer utilization
        buffer_util = (self.frame_queue.qsize() / 1000.0) * 100
        self.update_status_metric("buffer_utilization_percent", buffer_util)

        # Basic connectivity checks
        redis_ok = self.redis_client and self.redis_client.ping() if self.redis_client else False
        thread_ok = self.processing_thread and self.processing_thread.is_alive() if self.processing_thread else False
        self.update_status_metric("redis_connected", bool(redis_ok))

        health_status = self.get_health_status()
        uptime = self.get_uptime_seconds()
        grace_sec = float(os.environ.get("GRACE_STARTUP_SEC", "120"))
        warming_up = uptime <= grace_sec

        frame_stale_sec = 10.0
        push_age = None
        push_ok = True
        if not warming_up:
            if self.last_push_ts is None:
                push_ok = False
            else:
                push_age = time.time() - self.last_push_ts
                push_ok = push_age <= frame_stale_sec

        freeze_enabled = self.freeze_detector is not None and RTSP_FREEZE_DETECT_SEC > 0
        freeze_ok = True
        if freeze_enabled and not warming_up:
            freeze_ok = self.freeze_age_sec < RTSP_FREEZE_DETECT_SEC

        # Ingest rate, independent of push_ok: a link that drops most frames still
        # delivers one every few seconds, which keeps push_age under its stale
        # threshold while every downstream stage starves.
        input_fps_ok = True
        low_input_fps_sec = 0.0
        if RTSP_MIN_INPUT_FPS > 0 and self.low_input_fps_since is not None:
            low_input_fps_sec = max(0.0, time.time() - self.low_input_fps_since)
            if not warming_up:
                input_fps_ok = low_input_fps_sec < READY_LOW_INPUT_FPS_SEC

        ready = bool(health_status["healthy"] and redis_ok and thread_ok and push_ok and freeze_ok and input_fps_ok)
        return {
            "ready": ready,
            "warming_up": warming_up,
            "uptime_seconds": round(uptime, 2),
            "grace_startup_seconds": grace_sec,
            "redis_ok": bool(redis_ok),
            "thread_ok": bool(thread_ok),
            "push_ok": push_ok,
            "push_age_seconds": round(push_age, 2) if push_age is not None else None,
            "push_stale_threshold_seconds": frame_stale_sec,
            "freeze_detection_enabled": freeze_enabled,
            "freeze_ok": freeze_ok,
            "freeze_age_seconds": round(self.freeze_age_sec, 2),
            "freeze_threshold_seconds": RTSP_FREEZE_DETECT_SEC,
            "freeze_mad": round(self.freeze_mad, 4),
            "freeze_mad_threshold": RTSP_FREEZE_MAD_THRESHOLD,
            "freeze_incidents": self.freeze_incidents,
            "input_fps": round(self.last_input_fps, 2),
            "input_fps_min": RTSP_MIN_INPUT_FPS,
            "input_fps_ok": input_fps_ok,
            "low_input_fps_seconds": round(low_input_fps_sec, 1),
        }

    def health_check(self):
        """Backwards-compatible bool health check."""
        try:
            return bool(self.get_readiness_report().get("ready"))
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            self.increment_counter("error_count", 1)
            return False


if __name__ == "__main__":
    # Initialize telemetry
    metrics = setup_rtsp_service_telemetry()
    tracer = get_tracer("stopsign.rtsp_service")

    # Create RTSP service and set telemetry
    rtsp_to_redis = RTSPToRedis()
    rtsp_to_redis.set_telemetry(metrics, tracer)

    from http.server import BaseHTTPRequestHandler
    from http.server import HTTPServer

    class HealthCheckHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/healthz":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            elif self.path in ("/ready", "/health"):
                try:
                    report = rtsp_to_redis.get_readiness_report()
                    payload = json.dumps(report).encode()
                    self.send_response(200 if report.get("ready") else 503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as e:
                    logger.error(f"Readiness endpoint failed: {e}")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ready":false,"error":"readiness_exception"}')
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):  # noqa: A003
            return

    def run_health_server():
        server = HTTPServer(("0.0.0.0", 8080), HealthCheckHandler)
        server.serve_forever()

    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    rtsp_to_redis.run()
