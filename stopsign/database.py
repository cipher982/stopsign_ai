import functools
import logging
import os
import time
from datetime import datetime
from typing import List

from sqlalchemy import JSON
from sqlalchemy import BigInteger
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

from stopsign.telemetry import get_tracer

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer("stopsign.database")
        with tracer.start_as_current_span(f"db_{func.__name__}") as span:
            span.set_attribute("db.operation", func.__name__)
            span.set_attribute("db.system", "postgresql")

            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            span.set_attribute("db.duration_seconds", elapsed)
            logger.debug(f"{func.__name__} completed in {elapsed:.2f} seconds")
            return result

    return wrapper


class CarStateHistory(Base):
    __tablename__ = "car_state_history"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    car_id = Column(Integer)
    last_seen = Column(DateTime)
    location = Column(JSON)
    speed = Column(Float)
    is_parked = Column(Boolean)
    in_stop_zone = Column(Boolean)
    time_in_zone = Column(Float)
    min_speed_in_zone = Column(Float)
    stop_duration = Column(Float)
    entry_time = Column(Float)
    exit_time = Column(Float)
    track = Column(JSON)


class VehiclePass(Base):
    __tablename__ = "vehicle_passes"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now())
    vehicle_id = Column(Integer)
    time_in_zone = Column(Float)
    stop_duration = Column(Float)
    min_speed = Column(Float)
    image_path = Column(String)


class ConfigSetting(Base):
    __tablename__ = "config_settings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    category = Column(String, nullable=False)  # "zones", "video_processing", "tracking", etc.
    key = Column(String, nullable=False)  # "stop_line", "scale", "fps", etc.
    value = Column(JSON, nullable=False)  # The actual config value
    coordinate_system = Column(String)  # "raw", "processing", null for non-zones
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    updated_by = Column(String, default="system")
    change_reason = Column(String)
    is_active = Column(Boolean, default=True)

    # Only one active setting per category/key combination
    __table_args__ = (UniqueConstraint("category", "key", "is_active", name="_category_key_active_uc"),)


class Database:
    def __init__(self, db_url: str):
        self.engine = create_engine(
            db_url,
            pool_size=20,  # Increased pool size
            max_overflow=0,  # No overflow
            pool_timeout=30,  # Timeout for getting a connection
            pool_recycle=300,  # Recycle connections after 5 minutes
            pool_pre_ping=True,  # Test connections before use
        )
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

        # Check if we're in read-only mode
        self.read_only_mode = os.getenv("READ_ONLY_MODE", "false").lower() == "true"
        if self.read_only_mode:
            logger.warning("🔒 DATABASE IN READ-ONLY MODE - All write operations will be blocked")

        logger.info(f"Database connection established at {db_url}")
        self.log_database_summary()

    @log_execution_time
    def log_database_summary(self):
        with self.Session() as session:
            total_passes = session.query(VehiclePass).count()
            summary = "Database summary:\n"
            summary += f"Total vehicle passes: {total_passes:,}\n"
            logger.info(summary)

    @log_execution_time
    def save_car_state(self, car):
        if self.read_only_mode:
            logger.debug("🚫 Blocked save_car_state (READ-ONLY MODE)")
            return

        with self.Session() as session:
            car_state = CarStateHistory(
                car_id=car.id,
                last_seen=datetime.fromtimestamp(car.state.last_update_time),
                location=car.state.location,
                speed=car.state.speed,
                is_parked=car.state.is_parked,
                in_stop_zone=car.state.in_stop_zone,
                time_in_zone=car.state.time_in_zone,
                min_speed_in_zone=car.state.min_speed_in_zone,
                stop_duration=car.state.stop_duration,
                entry_time=car.state.entry_time,
                exit_time=car.state.exit_time,
                track=car.state.track,
            )
            session.add(car_state)
            session.commit()

    @log_execution_time
    def add_vehicle_pass(
        self, vehicle_id: int, time_in_zone: float, stop_duration: float, min_speed: float, image_path: str
    ):
        if self.read_only_mode:
            logger.debug("🚫 Blocked add_vehicle_pass (READ-ONLY MODE)")
            return

        with self.Session() as session:
            vehicle_pass = VehiclePass(
                vehicle_id=vehicle_id,
                time_in_zone=time_in_zone,
                stop_duration=stop_duration,
                min_speed=min_speed,
                image_path=image_path,
            )
            session.add(vehicle_pass)
            session.commit()

    @log_execution_time
    def get_min_speed_percentile(self, min_speed: float, hours: int = 24) -> float:
        with self.Session() as session:
            total_count = (
                session.query(VehiclePass)
                .filter(VehiclePass.timestamp >= func.now() - text(f"INTERVAL '{hours} hours'"))
                .count()
            )

            count = (
                session.query(VehiclePass)
                .filter(
                    VehiclePass.min_speed <= min_speed,
                    VehiclePass.timestamp >= func.now() - text(f"INTERVAL '{hours} hours'"),
                )
                .count()
            )

            return (count * 100.0 / total_count) if total_count > 0 else 0

    @log_execution_time
    def get_time_in_zone_percentile(self, time_in_zone: float, hours: int = 24) -> float:
        with self.Session() as session:
            total_count = (
                session.query(VehiclePass)
                .filter(VehiclePass.timestamp >= func.now() - text(f"INTERVAL '{hours} hours'"))
                .count()
            )

            count = (
                session.query(VehiclePass)
                .filter(
                    VehiclePass.time_in_zone <= time_in_zone,
                    VehiclePass.timestamp >= func.now() - text(f"INTERVAL '{hours} hours'"),
                )
                .count()
            )

            return (count * 100.0 / total_count) if total_count > 0 else 0

    @log_execution_time
    def get_extreme_passes(self, field: str, order: str, limit: int, hours: int) -> List[VehiclePass]:
        valid_fields = ["min_speed", "time_in_zone", "stop_duration"]
        valid_orders = ["asc", "desc"]

        if field not in valid_fields:
            raise ValueError(f"Invalid field. Must be one of: {valid_fields}")
        if order.lower() not in valid_orders:
            raise ValueError(f"Invalid order. Must be one of: {valid_orders}")

        with self.Session() as session:
            result = (
                session.query(VehiclePass)
                .filter(VehiclePass.image_path.isnot(None))  # Not null
                .filter(VehiclePass.image_path != "")  # Not empty string
                .filter(VehiclePass.image_path.like("minio://%"))  # Proper minio prefix
                .filter(VehiclePass.timestamp >= func.now() - text(f"INTERVAL '{hours} hours'"))
                .order_by(text(f"{field} {order}"))
                .limit(limit)
                .all()
            )
            return result

    @log_execution_time
    def get_total_passes_last_24h(self):
        with self.Session() as session:
            return (
                session.query(VehiclePass)
                .filter(VehiclePass.timestamp >= func.now() - text("INTERVAL '24 hours'"))
                .count()
            )

    @log_execution_time
    def get_bulk_scores(self, passes: list[dict]) -> list[dict]:
        """Calculate scores for multiple passes based on 24h historical data"""
        with self.Session() as session:
            # Get all passes from last 24h for comparison
            historical = session.execute(
                text("""
                SELECT min_speed, time_in_zone 
                FROM vehicle_passes 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                """)
            ).fetchall()

            if not historical:
                return [
                    {"min_speed": p["min_speed"], "time_in_zone": p["time_in_zone"], "speed_score": 0, "time_score": 0}
                    for p in passes
                ]

            # Convert to sorted lists for percentile calcs
            hist_speeds = sorted(row[0] for row in historical)
            hist_times = sorted(row[1] for row in historical)
            total = len(historical)

            # Calculate scores for each pass
            results = []
            for p in passes:
                # Find position in sorted lists
                speed_pos = sum(1 for x in hist_speeds if x <= p["min_speed"])
                time_pos = sum(1 for x in hist_times if x <= p["time_in_zone"])

                # Calculate percentiles and convert to scores
                speed_score = round((1 - (speed_pos / total)) * 10)
                time_score = round((time_pos / total) * 10)

                results.append(
                    {
                        "min_speed": p["min_speed"],
                        "time_in_zone": p["time_in_zone"],
                        "speed_score": speed_score,
                        "time_score": time_score,
                    }
                )

            return results

    @log_execution_time
    def get_daily_statistics(self, date):
        # Assuming you have a DailyStatistics model
        # You would need to create this model and its corresponding table
        pass

    @log_execution_time
    def get_recent_vehicle_passes(self, limit=10):
        with self.Session() as session:
            result = (
                session.query(VehiclePass)
                .filter(VehiclePass.image_path.isnot(None))
                .filter(VehiclePass.image_path != "")
                .filter(VehiclePass.image_path.like("minio://%"))
                .order_by(VehiclePass.timestamp.desc())
                .limit(limit)
                .all()
            )
        return result

    def close(self):
        self.engine.dispose()

    @log_execution_time
    def get_config_setting(self, category: str, key: str):
        """Get a single active config setting."""
        with self.Session() as session:
            setting = session.query(ConfigSetting).filter_by(category=category, key=key, is_active=True).first()
            return setting.value if setting else None

    @log_execution_time
    def get_config_category(self, category: str) -> dict:
        """Get all active settings for a category."""
        with self.Session() as session:
            settings = session.query(ConfigSetting).filter_by(category=category, is_active=True).all()
            return {setting.key: setting.value for setting in settings}

    @log_execution_time
    def update_config_setting(
        self, category: str, key: str, value, coordinate_system=None, updated_by="system", change_reason="Config update"
    ):
        """Update a config setting, keeping history."""
        with self.Session() as session:
            # Deactivate old setting if it exists
            old_setting = session.query(ConfigSetting).filter_by(category=category, key=key, is_active=True).first()

            if old_setting:
                old_setting.is_active = False
                session.flush()  # Ensure constraint is satisfied before inserting new record

            # Create new setting
            new_setting = ConfigSetting(
                category=category,
                key=key,
                value=value,
                coordinate_system=coordinate_system,
                updated_by=updated_by,
                change_reason=change_reason,
            )
            session.add(new_setting)
            session.commit()
            logger.info(f"Updated config: {category}.{key} = {value}")

    @log_execution_time
    def get_config_history(self, category: str, key: str, limit: int = 10):
        """Get change history for a config setting."""
        with self.Session() as session:
            history = (
                session.query(ConfigSetting)
                .filter_by(category=category, key=key)
                .order_by(ConfigSetting.created_at.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "value": h.value,
                    "created_at": h.created_at,
                    "updated_by": h.updated_by,
                    "change_reason": h.change_reason,
                    "is_active": h.is_active,
                }
                for h in history
            ]
