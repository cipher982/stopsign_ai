#!/usr/bin/env python3
"""
Configuration Migration Script
Migrates config.yaml values to PostgreSQL database.
Safe to run multiple times (idempotent).
"""

import logging
import os
import sys

import yaml
from sqlalchemy import JSON
from sqlalchemy import BigInteger
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class ConfigSetting(Base):
    __tablename__ = "config_settings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    category = Column(String, nullable=False)  # "zones", "video_processing", "tracking", etc.
    key = Column(String, nullable=False)  # "stop_line", "scale", "fps", etc.
    value = Column(JSON, nullable=False)  # The actual config value
    coordinate_system = Column(String)  # "raw", "processing", null for non-zones
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    updated_by = Column(String, default="migration_script")
    change_reason = Column(String, default="Initial migration from YAML")
    is_active = Column(Boolean, default=True)

    # Only one active setting per category/key combination
    __table_args__ = (UniqueConstraint("category", "key", "is_active", name="_category_key_active_uc"),)


def load_yaml_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return None

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def flatten_config(config):
    """Convert nested YAML config to flat database records."""
    settings = []

    # Stream settings
    stream = config.get("stream_settings", {})
    settings.extend(
        [
            {"category": "stream", "key": "input_source", "value": stream.get("input_source")},
            {"category": "stream", "key": "fps", "value": stream.get("fps")},
            {"category": "stream", "key": "vehicle_classes", "value": stream.get("vehicle_classes")},
        ]
    )

    # Video processing
    video = config.get("video_processing", {})
    settings.extend(
        [
            {"category": "video_processing", "key": "scale", "value": video.get("scale")},
            {"category": "video_processing", "key": "crop_top", "value": video.get("crop_top")},
            {"category": "video_processing", "key": "crop_side", "value": video.get("crop_side")},
            {"category": "video_processing", "key": "frame_buffer_size", "value": video.get("frame_buffer_size")},
        ]
    )

    # Stop sign detection (zones and thresholds)
    stopsign = config.get("stopsign_detection", {})
    settings.extend(
        [
            # Zone coordinates
            {"category": "zones", "key": "stop_line", "value": stopsign.get("stop_line"), "coordinate_system": "raw"},
            {"category": "zones", "key": "stop_box_tolerance", "value": stopsign.get("stop_box_tolerance")},
            {
                "category": "zones",
                "key": "pre_stop_zone",
                "value": stopsign.get("pre_stop_zone"),
                "coordinate_system": "processing",
            },
            {
                "category": "zones",
                "key": "image_capture_zone",
                "value": stopsign.get("image_capture_zone"),
                "coordinate_system": "processing",
            },
            # Detection thresholds
            {
                "category": "detection",
                "key": "in_zone_frame_threshold",
                "value": stopsign.get("in_zone_frame_threshold"),
            },
            {
                "category": "detection",
                "key": "out_zone_frame_threshold",
                "value": stopsign.get("out_zone_frame_threshold"),
            },
            {"category": "detection", "key": "stop_speed_threshold", "value": stopsign.get("stop_speed_threshold")},
            {"category": "detection", "key": "max_movement_speed", "value": stopsign.get("max_movement_speed")},
            {"category": "detection", "key": "parked_frame_threshold", "value": stopsign.get("parked_frame_threshold")},
            {
                "category": "detection",
                "key": "unparked_frame_threshold",
                "value": stopsign.get("unparked_frame_threshold"),
            },
            {
                "category": "detection",
                "key": "unparked_speed_threshold",
                "value": stopsign.get("unparked_speed_threshold"),
            },
        ]
    )

    # Tracking
    tracking = config.get("tracking", {})
    settings.extend(
        [
            {"category": "tracking", "key": "use_kalman_filter", "value": tracking.get("use_kalman_filter")},
        ]
    )

    # Output
    output = config.get("output", {})
    settings.extend(
        [
            {"category": "output", "key": "save_video", "value": output.get("save_video")},
            {"category": "output", "key": "frame_skip", "value": output.get("frame_skip")},
            {"category": "output", "key": "jpeg_quality", "value": output.get("jpeg_quality")},
        ]
    )

    # Debugging/Visualization
    debug = config.get("debugging_visualization", {})
    settings.extend(
        [
            {"category": "visualization", "key": "draw_grid", "value": debug.get("draw_grid")},
            {"category": "visualization", "key": "grid_size", "value": debug.get("grid_size")},
        ]
    )

    # Filter out None values
    return [s for s in settings if s["value"] is not None]


def migrate_config(db_url, config_path="config.yaml", force_update=False):
    """Migrate YAML config to database."""
    logger.info("Starting config migration...")

    # Load YAML config
    yaml_config = load_yaml_config(config_path)
    if not yaml_config:
        logger.error("Failed to load YAML config")
        return False

    # Connect to database
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)  # Create table if it doesn't exist
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Convert YAML to flat settings
        settings = flatten_config(yaml_config)
        logger.info(f"Found {len(settings)} config settings to migrate")

        migrated_count = 0
        updated_count = 0

        for setting_data in settings:
            category = setting_data["category"]
            key = setting_data["key"]
            value = setting_data["value"]
            coord_system = setting_data.get("coordinate_system")

            # Check if setting already exists
            existing = session.query(ConfigSetting).filter_by(category=category, key=key, is_active=True).first()

            if existing:
                if force_update or existing.value != value:
                    # Deactivate old setting
                    existing.is_active = False

                    # Create new setting
                    new_setting = ConfigSetting(
                        category=category,
                        key=key,
                        value=value,
                        coordinate_system=coord_system,
                        updated_by="migration_script",
                        change_reason="Migration update" if force_update else "Value changed in YAML",
                    )
                    session.add(new_setting)
                    updated_count += 1
                    logger.info(f"Updated {category}.{key}")
                else:
                    logger.debug(f"Skipping {category}.{key} (unchanged)")
            else:
                # Create new setting
                new_setting = ConfigSetting(
                    category=category,
                    key=key,
                    value=value,
                    coordinate_system=coord_system,
                    updated_by="migration_script",
                    change_reason="Initial migration from YAML",
                )
                session.add(new_setting)
                migrated_count += 1
                logger.info(f"Migrated {category}.{key}")

        session.commit()
        logger.info(f"Migration complete: {migrated_count} new, {updated_count} updated")
        return True

    except Exception as e:
        session.rollback()
        logger.error(f"Migration failed: {e}")
        return False
    finally:
        session.close()


def main():
    """Main migration function."""
    # Get database URL from environment
    db_url = os.getenv("DB_URL")
    if not db_url:
        logger.error("DB_URL environment variable not set")
        sys.exit(1)

    # Get config path
    config_path = os.getenv("CONFIG_PATH", "config.yaml")

    # Check for force update flag
    force_update = "--force" in sys.argv

    if force_update:
        logger.warning("Force update enabled - will overwrite existing settings")

    # Run migration
    success = migrate_config(db_url, config_path, force_update)

    if success:
        logger.info("✅ Config migration completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Config migration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
