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
        self.validate_environment()

    @property
    def is_local(self) -> bool:
        return ENV == "local"

    @property
    def is_production(self) -> bool:
        return ENV == "prod"

    def validate_environment(self):
        """Validate environment configuration for safety"""
        if self.is_local:
            # Block production URLs in local mode
            redis_url = os.getenv("REDIS_URL", "")
            db_url = os.getenv("DB_URL", "")

            prod_indicators = ["prod-redis", "production", ".internal", "aws.com", "gcp.com"]
            for indicator in prod_indicators:
                if indicator in redis_url or indicator in db_url:
                    raise ConfigError(f"Production URL detected in local environment: ENV={ENV}")

        elif self.is_production:
            # Ensure production has required variables
            required_vars = ["REDIS_URL", "DB_URL"]
            missing = [var for var in required_vars if not os.getenv(var)]
            if missing:
                raise ConfigError(f"Missing required production environment variables: {missing}")


class LocalConfig(BaseConfig):
    """Local development configuration"""

    def __init__(self):
        super().__init__()
        # Core infrastructure (with local defaults)
        self.REDIS_URL = get_env("REDIS_URL", "redis://redis:6379/0", required=False)
        self.DB_URL = get_env("DB_URL", "postgresql://postgres:password@postgres:***REMOVED***/stopsign", required=False)

        # MinIO defaults
        self.MINIO_ENDPOINT = get_env("MINIO_ENDPOINT", "minio:9000", required=False)
        self.MINIO_ACCESS_KEY = get_env("MINIO_ACCESS_KEY", "minioadmin", required=False)
        self.MINIO_SECRET_KEY = get_env("MINIO_SECRET_KEY", "minioadmin", required=False)
        self.MINIO_BUCKET = get_env("MINIO_BUCKET", "stopsign-local", required=False)

        # AI Model (CPU-optimized)
        self.YOLO_MODEL_NAME = get_env("YOLO_MODEL_NAME", "yolov8n.pt", required=False)
        self.YOLO_DEVICE = get_env("YOLO_DEVICE", "cpu", required=False)

        # FFmpeg (CPU encoding)
        self.FFMPEG_ENCODER = get_env("FFMPEG_ENCODER", "libx264", required=False)
        self.FFMPEG_PRESET = get_env("FFMPEG_PRESET", "veryfast", required=False)


class ProductionConfig(BaseConfig):
    """Production configuration"""

    def __init__(self):
        super().__init__()
        # Core infrastructure (required)
        self.REDIS_URL = get_env("REDIS_URL")
        self.DB_URL = get_env("DB_URL")

        # MinIO (required)
        self.MINIO_ENDPOINT = get_env("MINIO_ENDPOINT")
        self.MINIO_ACCESS_KEY = get_env("MINIO_ACCESS_KEY")
        self.MINIO_SECRET_KEY = get_env("MINIO_SECRET_KEY")
        self.MINIO_BUCKET = get_env("MINIO_BUCKET")

        # AI Model (production defaults)
        self.YOLO_MODEL_NAME = get_env("YOLO_MODEL_NAME", "yolov8x.pt", required=False)
        self.YOLO_DEVICE = get_env("YOLO_DEVICE", "cuda", required=False)

        # FFmpeg (GPU encoding)
        self.FFMPEG_ENCODER = get_env("FFMPEG_ENCODER", "h264_nvenc", required=False)
        self.FFMPEG_PRESET = get_env("FFMPEG_PRESET", "p4", required=False)


# Common settings (same for all environments)
RTSP_URL = get_env("RTSP_URL", required=False)
RAW_FRAME_KEY = get_env("RAW_FRAME_KEY", "raw_frames", required=False)
PROCESSED_FRAME_KEY = get_env("PROCESSED_FRAME_KEY", "processed_frames", required=False)
FRAME_BUFFER_SIZE = get_env_int("FRAME_BUFFER_SIZE", 500, required=False)
PROMETHEUS_PORT = get_env_int("PROMETHEUS_PORT", 9100, required=False)
WEB_SERVER_PORT = get_env_int("WEB_SERVER_PORT", 8000, required=False)

# Select configuration based on environment
if ENV == "local":
    config = LocalConfig()
else:
    config = ProductionConfig()

# Export commonly used settings for backward compatibility
REDIS_URL = config.REDIS_URL
DB_URL = config.DB_URL
YOLO_MODEL_NAME = config.YOLO_MODEL_NAME
YOLO_DEVICE = config.YOLO_DEVICE
MINIO_ENDPOINT = config.MINIO_ENDPOINT
MINIO_ACCESS_KEY = config.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = config.MINIO_SECRET_KEY
MINIO_BUCKET = config.MINIO_BUCKET
FFMPEG_ENCODER = config.FFMPEG_ENCODER
FFMPEG_PRESET = config.FFMPEG_PRESET

logger.info(f"Configuration loaded for environment: {ENV}")
logger.info(f"Using config class: {config.__class__.__name__}")
