"""
StopSign AI Settings Module

Provides clean configuration management for different environments.
Production uses environment variables, local development uses .env files.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Environment detection
ENV = os.getenv("ENV", "prod")


class ConfigError(Exception):
    """Configuration validation error"""

    pass


def get_env(key: str, default: Optional[str] = None, required: bool = True) -> str:
    """Get environment variable with validation and logging."""
    value = os.getenv(key, default)
    if required and value is None:
        raise ConfigError(f"Required environment variable {key} is not set")
    if value:
        # Don't log sensitive values
        if any(sensitive in key.lower() for sensitive in ["password", "secret", "key", "token"]):
            logger.info(f"Loaded env var {key}: [REDACTED]")
        else:
            logger.info(f"Loaded env var {key}: {value}")
    return value


def get_env_int(key: str, default: Optional[int] = None, required: bool = True) -> int:
    """Get environment variable as integer."""
    str_default = str(default) if default is not None else None
    value = get_env(key, str_default, required)
    try:
        return int(value) if value else 0
    except ValueError:
        raise ConfigError(f"Environment variable {key}='{value}' is not a valid integer")


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean."""
    value = get_env(key, str(default).lower(), required=False)
    return value.lower() in ("true", "1", "yes", "on")


class BaseConfig:
    """Base configuration with common settings"""

    def __init__(self):
        pass

    @property
    def is_local(self) -> bool:
        return ENV == "local"

    @property
    def is_production(self) -> bool:
        return ENV == "prod"


class LocalConfig(BaseConfig):
    """Local development configuration"""

    def __init__(self):
        super().__init__()
        # Core infrastructure (with local defaults)
        self.REDIS_URL = get_env("REDIS_URL", "redis://redis:6379/0", required=False)
        self.DB_URL = get_env("DB_URL", "postgresql://postgres:password@postgres:5432/stopsign", required=False)

        # MinIO settings (required - no defaults)
        self.MINIO_ENDPOINT = get_env("MINIO_ENDPOINT")
        self.MINIO_ACCESS_KEY = get_env("MINIO_ACCESS_KEY")
        self.MINIO_SECRET_KEY = get_env("MINIO_SECRET_KEY")
        self.MINIO_BUCKET = get_env("MINIO_BUCKET")
        self.MINIO_PUBLIC_URL = get_env("MINIO_PUBLIC_URL")

        # AI Model (CPU-optimized)
        self.YOLO_MODEL_NAME = get_env("YOLO_MODEL_NAME", "yolov8n.pt", required=False)
        self.YOLO_DEVICE = get_env("YOLO_DEVICE", "cpu", required=False)

        # FFmpeg (CPU encoding)
        self.FFMPEG_ENCODER = get_env("FFMPEG_ENCODER", "libx264", required=False)
        self.FFMPEG_PRESET = get_env("FFMPEG_PRESET", "veryfast", required=False)


class ProductionConfig(BaseConfig):
    """Production configuration with lazy loading"""

    def __init__(self):
        super().__init__()
        self._redis_url = None
        self._db_url = None
        self._minio_endpoint = None
        self._minio_access_key = None
        self._minio_secret_key = None
        self._minio_bucket = None
        self._minio_public_url = None
        self._yolo_model_name = None
        self._yolo_device = None

    @property
    def REDIS_URL(self):
        if self._redis_url is None:
            self._redis_url = get_env("REDIS_URL")  # Required in production
        return self._redis_url

    @property
    def DB_URL(self):
        if self._db_url is None:
            self._db_url = get_env("DB_URL")  # Required in production
        return self._db_url

    @property
    def MINIO_ENDPOINT(self):
        if self._minio_endpoint is None:
            self._minio_endpoint = get_env("MINIO_ENDPOINT")
        return self._minio_endpoint

    @property
    def MINIO_ACCESS_KEY(self):
        if self._minio_access_key is None:
            self._minio_access_key = get_env("MINIO_ACCESS_KEY")
        return self._minio_access_key

    @property
    def MINIO_SECRET_KEY(self):
        if self._minio_secret_key is None:
            self._minio_secret_key = get_env("MINIO_SECRET_KEY")
        return self._minio_secret_key

    @property
    def MINIO_BUCKET(self):
        if self._minio_bucket is None:
            self._minio_bucket = get_env("MINIO_BUCKET")
        return self._minio_bucket

    @property
    def MINIO_PUBLIC_URL(self):
        if self._minio_public_url is None:
            self._minio_public_url = get_env("MINIO_PUBLIC_URL")
        return self._minio_public_url

    @property
    def YOLO_MODEL_NAME(self):
        if self._yolo_model_name is None:
            self._yolo_model_name = get_env("YOLO_MODEL_NAME", "yolov8x.pt", required=False)
        return self._yolo_model_name

    @property
    def YOLO_DEVICE(self):
        if self._yolo_device is None:
            self._yolo_device = get_env("YOLO_DEVICE", "cuda", required=False)
        return self._yolo_device

    @property
    def FFMPEG_ENCODER(self):
        if not hasattr(self, "_ffmpeg_encoder"):
            self._ffmpeg_encoder = get_env("FFMPEG_ENCODER", "h264_nvenc", required=False)
        return self._ffmpeg_encoder

    @property
    def FFMPEG_PRESET(self):
        if not hasattr(self, "_ffmpeg_preset"):
            self._ffmpeg_preset = get_env("FFMPEG_PRESET", "p4", required=False)
        return self._ffmpeg_preset


# Common settings (same for all environments)
RTSP_URL = get_env("RTSP_URL", required=False)
RAW_FRAME_KEY = get_env("RAW_FRAME_KEY", "raw_frames", required=False)
PROCESSED_FRAME_KEY = get_env("PROCESSED_FRAME_KEY", "processed_frames", required=False)
PROCESSED_FRAME_SHAPE_KEY = get_env("PROCESSED_FRAME_SHAPE_KEY", "processed_frame_shape", required=False)
FRAME_METADATA_KEY = get_env("FRAME_METADATA_KEY", "frame_metadata_latest", required=False)
FRAME_BUFFER_SIZE = get_env_int("FRAME_BUFFER_SIZE", 500, required=False)
PROMETHEUS_PORT = get_env_int("PROMETHEUS_PORT", 9100, required=False)
WEB_SERVER_PORT = get_env_int("WEB_SERVER_PORT", 8000, required=False)
GRACE_STARTUP_SEC = get_env_int("GRACE_STARTUP_SEC", 120, required=False)

# Local image storage (fast homepage)
LOCAL_IMAGE_DIR = get_env("LOCAL_IMAGE_DIR", "/app/data/vehicle-images", required=False)

# Bremen MinIO (archive storage)
BREMEN_MINIO_ENDPOINT = get_env("BREMEN_MINIO_ENDPOINT", "100.98.103.56:9000", required=False)
BREMEN_MINIO_ACCESS_KEY = get_env("BREMEN_MINIO_ACCESS_KEY", "root", required=False)
BREMEN_MINIO_SECRET_KEY = get_env("BREMEN_MINIO_SECRET_KEY", "", required=False)
BREMEN_MINIO_BUCKET = get_env("BREMEN_MINIO_BUCKET", "vehicle-images", required=False)

# Local image retention
LOCAL_IMAGE_MAX_COUNT = get_env_int("LOCAL_IMAGE_MAX_COUNT", 500, required=False)

# Telemetry (optional - set to empty string to disable)
OTEL_EXPORTER_OTLP_ENDPOINT = get_env("OTEL_EXPORTER_OTLP_ENDPOINT", required=False) or None

# Select configuration based on environment
if ENV == "local":
    config = LocalConfig()
else:
    config = ProductionConfig()


# Lazy exports for backward compatibility
def __getattr__(name):
    """Lazy loading of module-level settings"""
    if name == "REDIS_URL":
        return config.REDIS_URL
    elif name == "DB_URL":
        return config.DB_URL
    elif name == "YOLO_MODEL_NAME":
        return config.YOLO_MODEL_NAME
    elif name == "YOLO_DEVICE":
        return config.YOLO_DEVICE
    elif name == "MINIO_ENDPOINT":
        return config.MINIO_ENDPOINT
    elif name == "MINIO_ACCESS_KEY":
        return config.MINIO_ACCESS_KEY
    elif name == "MINIO_SECRET_KEY":
        return config.MINIO_SECRET_KEY
    elif name == "MINIO_BUCKET":
        return config.MINIO_BUCKET
    elif name == "MINIO_PUBLIC_URL":
        return config.MINIO_PUBLIC_URL
    elif name == "FFMPEG_ENCODER":
        return config.FFMPEG_ENCODER
    elif name == "FFMPEG_PRESET":
        return config.FFMPEG_PRESET
    elif name == "OTEL_EXPORTER_OTLP_ENDPOINT":
        return OTEL_EXPORTER_OTLP_ENDPOINT
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
