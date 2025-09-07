"""Service status tracking mixin for StopSign AI services.

This module provides a standardized way to track runtime status across all services
while maintaining clean separation between OpenTelemetry event recording and
human-readable status logging.

Architecture:
- OpenTelemetry Domain: Business events for monitoring systems (counters, histograms)
- Runtime Status Domain: Current state for logs, debugging, health checks (simple variables)
"""

import logging
import time
from threading import RLock
from typing import Any
from typing import Dict

logger = logging.getLogger(__name__)


class ServiceStatusMixin:
    """Mixin class providing standardized runtime status tracking for StopSign services.

    This mixin handles the "Runtime Status Domain" while services use OpenTelemetry
    for the "Event Recording Domain". Clean separation of concerns:

    - Runtime status: Current FPS, queue sizes, error counts (for logs/debugging)
    - OpenTelemetry: Business events, durations, totals (for monitoring/dashboards)
    """

    def __init__(self):
        # Threading safety for status updates
        self._status_lock = RLock()

        # Core runtime metrics - thread-safe updates via methods
        self._runtime_status = {
            "service_name": self.__class__.__name__,
            "start_time": time.time(),
            "last_status_update": time.time(),
            # Performance metrics
            "current_fps": 0.0,
            "processed_count": 0,
            "error_count": 0,
            "warning_count": 0,
            # Queue/buffer metrics
            "queue_size": 0,
            "buffer_utilization_percent": 0.0,
            # Connection status
            "redis_connected": False,
            "db_connected": False,
            "rtsp_connected": False,
            # Service-specific metrics (extensible)
            "custom_metrics": {},
        }

    def update_status_metric(self, key: str, value: Any) -> None:
        """Thread-safe update of a runtime status metric."""
        with self._status_lock:
            self._runtime_status[key] = value
            self._runtime_status["last_status_update"] = time.time()

    def update_custom_metric(self, key: str, value: Any) -> None:
        """Thread-safe update of service-specific custom metrics."""
        with self._status_lock:
            self._runtime_status["custom_metrics"][key] = value
            self._runtime_status["last_status_update"] = time.time()

    def increment_counter(self, key: str, amount: int = 1) -> None:
        """Thread-safe increment of counter metrics."""
        with self._status_lock:
            current = self._runtime_status.get(key, 0)
            self._runtime_status[key] = current + amount
            self._runtime_status["last_status_update"] = time.time()

    def get_status_snapshot(self) -> Dict[str, Any]:
        """Get thread-safe snapshot of current runtime status."""
        with self._status_lock:
            return self._runtime_status.copy()

    def get_uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self._runtime_status["start_time"]

    def log_status_summary(self, log_level: str = "INFO") -> None:
        """Log standardized status summary across all StopSign services."""
        status = self.get_status_snapshot()
        uptime = self.get_uptime_seconds()

        # Core status line - consistent format across all services
        status_msg = (
            f"[{status['service_name']}] Status: "
            f"Uptime={uptime:.1f}s, "
            f"FPS={status['current_fps']:.2f}, "
            f"Processed={status['processed_count']}, "
            f"Errors={status['error_count']}, "
            f"Queue={status['queue_size']}"
        )

        # Connection status
        connections = []
        if status["redis_connected"]:
            connections.append("Redis")
        if status["db_connected"]:
            connections.append("DB")
        if status["rtsp_connected"]:
            connections.append("RTSP")

        if connections:
            status_msg += f", Connected=[{','.join(connections)}]"

        # Buffer utilization if relevant
        if status["buffer_utilization_percent"] > 0:
            status_msg += f", Buffer={status['buffer_utilization_percent']:.1f}%"

        # Custom metrics if any
        if status["custom_metrics"]:
            custom_parts = [f"{k}={v}" for k, v in status["custom_metrics"].items()]
            status_msg += f", Custom=[{','.join(custom_parts)}]"

        # Log at appropriate level
        log_func = getattr(logger, log_level.lower(), logger.info)
        log_func(status_msg)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health check status based on runtime metrics.

        Returns structured health data for health check endpoints.
        """
        status = self.get_status_snapshot()
        uptime = self.get_uptime_seconds()

        # Basic health logic - services can override for specific needs
        is_healthy = (
            uptime > 5.0  # Running for at least 5 seconds
            and status["error_count"] < 100  # Not too many errors
            and time.time() - status["last_status_update"] < 300  # Updated recently
        )

        return {
            "healthy": is_healthy,
            "uptime_seconds": uptime,
            "service_name": status["service_name"],
            "last_update": status["last_status_update"],
            "current_fps": status["current_fps"],
            "error_count": status["error_count"],
            "connections": {
                "redis": status["redis_connected"],
                "database": status["db_connected"],
                "rtsp": status["rtsp_connected"],
            },
        }


class RTSPServiceStatusMixin(ServiceStatusMixin):
    """RTSP-specific status tracking extensions."""

    def __init__(self):
        super().__init__()
        # RTSP-specific status tracking
        self.update_custom_metric("rtsp_reconnects", 0)
        self.update_custom_metric("frames_dropped", 0)
        self.update_custom_metric("redis_errors", 0)

    def update_rtsp_fps(self, fps: float) -> None:
        """Update RTSP-specific FPS tracking."""
        self.update_status_metric("current_fps", fps)
        self.update_custom_metric("rtsp_input_fps", fps)

    def record_frame_drop(self) -> None:
        """Record dropped frame for RTSP services."""
        self.increment_counter("frames_dropped", 1)
        self.update_custom_metric("frames_dropped", self._runtime_status.get("frames_dropped", 0))

    def record_redis_error(self) -> None:
        """Record Redis error for RTSP services."""
        self.increment_counter("error_count", 1)
        self.update_custom_metric("redis_errors", self._runtime_status.get("error_count", 0))


class VideoAnalyzerStatusMixin(ServiceStatusMixin):
    """Video analyzer-specific status tracking extensions."""

    def __init__(self):
        super().__init__()
        # Video analyzer-specific metrics
        self.update_custom_metric("vehicles_detected", 0)
        self.update_custom_metric("stop_violations", 0)
        self.update_custom_metric("yolo_inference_ms", 0.0)

    def update_yolo_performance(self, inference_time_ms: float) -> None:
        """Update YOLO inference performance metrics."""
        self.update_custom_metric("yolo_inference_ms", inference_time_ms)

    def record_vehicle_detection(self, count: int) -> None:
        """Record vehicle detections."""
        current = self._runtime_status["custom_metrics"].get("vehicles_detected", 0)
        self.update_custom_metric("vehicles_detected", current + count)


class FFmpegServiceStatusMixin(ServiceStatusMixin):
    """FFmpeg service-specific status tracking extensions."""

    def __init__(self):
        super().__init__()
        # FFmpeg-specific metrics
        self.update_custom_metric("hls_segments_generated", 0)
        self.update_custom_metric("segment_generation_ms", 0.0)
        self.update_custom_metric("encoding_errors", 0)

    def record_hls_segment(self, generation_time_ms: float) -> None:
        """Record HLS segment generation."""
        current = self._runtime_status["custom_metrics"].get("hls_segments_generated", 0)
        self.update_custom_metric("hls_segments_generated", current + 1)
        self.update_custom_metric("segment_generation_ms", generation_time_ms)
