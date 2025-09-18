"""
OpenTelemetry configuration for StopSign AI services.
Provides centralized tracing, metrics, and logging setup.
"""

import logging
import os
from typing import Optional

from opentelemetry import metrics
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

# Global telemetry state
_telemetry_initialized = False
_tracer_provider: Optional[TracerProvider] = None
_meter_provider: Optional[MeterProvider] = None


def get_resource_attributes(service_name: str, service_version: str = "1.0.0") -> Resource:
    """Create OpenTelemetry resource with service information."""
    return Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": os.getenv("ENV", "development"),
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
        }
    )


def setup_telemetry(service_name: str, service_version: str = "1.0.0", enable_console_export: bool = False) -> None:
    """
    Initialize OpenTelemetry tracing and metrics for a service.

    Args:
        service_name: Name of the service (e.g., "stopsign-web-server")
        service_version: Version of the service
        enable_console_export: Whether to also export to console (for debugging)
    """
    global _telemetry_initialized, _tracer_provider, _meter_provider

    if _telemetry_initialized:
        logger.info(f"Telemetry already initialized for {service_name}")
        return

    # Get OTLP endpoint from centralized settings (will fail fast if missing)
    from stopsign.settings import OTEL_EXPORTER_OTLP_ENDPOINT

    otlp_endpoint = OTEL_EXPORTER_OTLP_ENDPOINT

    try:
        # Create resource
        resource = get_resource_attributes(service_name, service_version)

        # Setup tracing
        _tracer_provider = TracerProvider(resource=resource)

        # OTLP trace exporter
        otlp_trace_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,
        )
        _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))

        # Console exporter for debugging
        if enable_console_export:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            console_exporter = ConsoleSpanExporter()
            _tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

        trace.set_tracer_provider(_tracer_provider)

        # Setup metrics
        otlp_metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
        metric_reader = PeriodicExportingMetricReader(
            exporter=otlp_metric_exporter,
            export_interval_millis=30000,  # Export every 30 seconds
        )
        _meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(_meter_provider)

        # Auto-instrument common libraries (conditional imports)
        try:
            from opentelemetry.instrumentation.redis import RedisInstrumentor

            RedisInstrumentor().instrument()
        except ImportError:
            pass

        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor

            RequestsInstrumentor().instrument()
        except ImportError:
            pass

        _telemetry_initialized = True
        logger.info(f"OpenTelemetry initialized for {service_name} -> {otlp_endpoint}")

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")


def setup_fastapi_telemetry(app, service_name: str) -> None:
    """Setup FastAPI-specific telemetry instrumentation."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info(f"FastAPI instrumentation enabled for {service_name}")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")


def get_tracer(name: str):
    """Get a tracer for creating custom spans."""
    return trace.get_tracer(name)


def get_meter(name: str):
    """Get a meter for creating custom metrics."""
    return metrics.get_meter(name)


def create_custom_span(tracer, operation_name: str, **attributes):
    """Create a custom span with attributes."""
    span = tracer.start_span(operation_name)
    for key, value in attributes.items():
        span.set_attribute(key, value)
    return span


# Common metric instruments for StopSign AI
class StopSignMetrics:
    """Common metrics used across StopSign AI services.

    ARCHITECTURE: Two-Domain Telemetry Pattern
    ==========================================

    This class handles the "EVENT RECORDING DOMAIN" for business intelligence and monitoring.
    Services should use ServiceStatusMixin for the "RUNTIME STATUS DOMAIN" for logging/debugging.

    EVENT RECORDING DOMAIN (OpenTelemetry - This Class):
    - Business events: "X happened" - frames processed, objects detected, etc.
    - Performance measurements: "X took Y time" - inference duration, DB operations
    - Exported to: Grafana, monitoring dashboards, alerting systems
    - Data flow: Service -> OpenTelemetry -> OTLP -> Monitoring backend

    RUNTIME STATUS DOMAIN (ServiceStatusMixin - stopsign.service_status):
    - Current state: "X is Y right now" - current FPS, queue size, connection status
    - Human-readable: Debug logs, status summaries, health checks
    - Data flow: Service -> Logs/Health endpoints -> Human operators

    Usage Examples:
    --------------
    # EVENT RECORDING (use this class):
    metrics.frames_processed.add(1, {"service": "rtsp"})           # Counter: increment
    metrics.yolo_inference_duration.record(0.025)                 # Histogram: record duration
    metrics.redis_operations.add(1, {"operation": "publish"})     # Counter: business event

    # RUNTIME STATUS (use ServiceStatusMixin):
    self.update_status_metric('current_fps', 15.2)                # Current state
    self.increment_counter('error_count', 1)                      # Error tracking
    self.log_status_summary()                                     # Human logs

    DO NOT MIX DOMAINS:
    - Don't use OpenTelemetry counters for "current state" (use ServiceStatusMixin)
    - Don't log business events without telemetry recording (use both)
    - Don't create gauges in OpenTelemetry (that's Prometheus thinking)
    """

    def __init__(self, service_name: str):
        self.meter = get_meter(f"stopsign.{service_name}")

        # Frame processing metrics
        self.frames_processed = self.meter.create_counter(
            "frames_processed_total", description="Total number of frames processed"
        )

        self.frame_processing_duration = self.meter.create_histogram(
            "frame_processing_duration_seconds", description="Time spent processing each frame"
        )

        # YOLO inference metrics
        self.yolo_inference_duration = self.meter.create_histogram(
            "yolo_inference_duration_seconds", description="Time spent on YOLO inference"
        )

        self.objects_detected = self.meter.create_counter(
            "objects_detected_total", description="Total number of objects detected"
        )

        # Video streaming metrics
        self.segments_generated = self.meter.create_counter(
            "hls_segments_generated_total", description="Total HLS segments generated"
        )

        self.segment_generation_duration = self.meter.create_histogram(
            "hls_segment_generation_duration_seconds", description="Time to generate each HLS segment"
        )

        # Database metrics
        self.db_operations = self.meter.create_counter(
            "database_operations_total", description="Total database operations"
        )

        self.db_operation_duration = self.meter.create_histogram(
            "database_operation_duration_seconds", description="Time spent on database operations"
        )

        # Queue/backpressure metrics
        self.queue_depth = self.meter.create_histogram("frame_queue_depth", description="Observed frame queue depth")
        self.pipeline_lag = self.meter.create_histogram(
            "frame_pipeline_lag_seconds", description="Observed frame lag within the pipeline"
        )
        self.redis_empty_polls = self.meter.create_counter(
            "redis_empty_polls_total", description="Redis BRPOP/BLPOP calls that returned no data"
        )

        # Redis metrics
        self.redis_operations = self.meter.create_counter(
            "redis_operations_total", description="Total Redis operations"
        )

        # Vehicle tracking metrics
        self.vehicles_tracked = self.meter.create_counter(
            "vehicles_tracked_total", description="Total vehicles tracked"
        )

        self.stop_violations = self.meter.create_counter(
            "stop_violations_total", description="Total stop sign violations detected"
        )

        # Error and health metrics (business events, not current state)
        self.service_errors = self.meter.create_counter(
            "service_errors_total", description="Total service errors occurred"
        )

        self.service_restarts = self.meter.create_counter(
            "service_restarts_total", description="Total service restart events"
        )


# Service-specific telemetry setup functions
def setup_web_server_telemetry(app):
    """Setup telemetry for the web server service."""
    setup_telemetry("stopsign-web-server")
    # Note: Using FastHTML, not FastAPI, so no FastAPI instrumentation needed

    # SQLAlchemy instrumentation for web server
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument()
    except ImportError:
        pass

    return StopSignMetrics("web-server")


def setup_video_analyzer_telemetry():
    """Setup telemetry for the video analyzer service."""
    setup_telemetry("stopsign-video-analyzer")
    return StopSignMetrics("video-analyzer")


def setup_ffmpeg_service_telemetry():
    """Setup telemetry for the FFmpeg service."""
    setup_telemetry("stopsign-ffmpeg-service")
    return StopSignMetrics("ffmpeg-service")


def setup_rtsp_service_telemetry():
    """Setup telemetry for the RTSP service."""
    setup_telemetry("stopsign-rtsp-service")
    return StopSignMetrics("rtsp-service")
