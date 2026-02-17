import functools
import logging
import os
import time
from datetime import datetime
from typing import Dict
from typing import List

from sqlalchemy import JSON
from sqlalchemy import BigInteger
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import LargeBinary
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from sqlalchemy import create_engine
from sqlalchemy import or_
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB
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
    entry_time = Column(Float)
    exit_time = Column(Float)
    clip_path = Column(String)
    clip_status = Column(String)
    clip_error = Column(String)


class VehiclePassRaw(Base):
    __tablename__ = "vehicle_pass_raw"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    vehicle_pass_id = Column(BigInteger, ForeignKey("vehicle_passes.id"), unique=True, nullable=False)
    raw_payload = Column(JSONB, nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    raw_complete = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class ConfigSetting(Base):
    __tablename__ = "config_settings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    category = Column(String, nullable=False)  # "zones", "video_processing", "tracking", etc.
    key = Column(String, nullable=False)  # "stop_zone", "scale", "fps", etc.
    value = Column(JSON, nullable=False)  # The actual config value
    coordinate_system = Column(String)  # "raw", "processing", null for non-zones
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    updated_by = Column(String, default="system")
    change_reason = Column(String)
    is_active = Column(Boolean, default=True)

    # Only one active setting per category/key combination
    __table_args__ = (UniqueConstraint("category", "key", "is_active", name="_category_key_active_uc"),)


class VehicleEmbedding(Base):
    __tablename__ = "vehicle_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    vehicle_pass_id = Column(BigInteger, ForeignKey("vehicle_passes.id"), unique=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # 384 x float32 = 1,536 bytes
    model_name = Column(String(64), nullable=False)  # e.g. 'dinov2-small'
    created_at = Column(DateTime, default=func.now())


class VehicleAttribute(Base):
    __tablename__ = "vehicle_attributes"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    vehicle_pass_id = Column(BigInteger, ForeignKey("vehicle_passes.id"), unique=True, nullable=False)
    cluster_id = Column(Integer)
    vehicle_type = Column(String(64))  # sedan, suv, pickup, van, etc.
    color = Column(String(64))  # white, silver, black, etc.
    make_model = Column(String(128))  # Toyota Camry, Ford F-150, etc.
    confidence = Column(Float)  # LLM confidence 0.0-1.0
    is_representative = Column(Boolean, default=False)
    raw_llm_response = Column(JSON)  # Full structured response for debugging
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


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
        self._ensure_vehicle_pass_columns()
        self.log_database_summary()

    def _ensure_vehicle_pass_columns(self) -> None:
        """Add new columns to vehicle_passes if missing (lightweight migration)."""
        if self.read_only_mode:
            return

        desired = {
            "entry_time": "DOUBLE PRECISION",
            "exit_time": "DOUBLE PRECISION",
            "clip_path": "TEXT",
            "clip_status": "TEXT",
            "clip_error": "TEXT",
        }
        try:
            with self.Session() as session:
                result = session.execute(
                    text(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'vehicle_passes'
                        """
                    )
                )
                existing = {row[0] for row in result.fetchall()}

                missing = [name for name in desired.keys() if name not in existing]
                for name in missing:
                    session.execute(text(f"ALTER TABLE vehicle_passes ADD COLUMN {name} {desired[name]}"))
                if missing:
                    session.commit()
                    logger.info("Added missing columns to vehicle_passes: %s", ", ".join(missing))
        except Exception as e:
            logger.warning("Failed to ensure vehicle_passes columns: %s", e)

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
                is_parked=car.state.motion.is_parked,
                in_stop_zone=car.state.zone.in_zone,
                time_in_zone=car.state.zone.time_in_zone,
                min_speed_in_zone=car.state.zone.min_speed,
                stop_duration=car.state.zone.stop_duration,
                entry_time=car.state.zone.entry_time,
                exit_time=car.state.zone.exit_time,
                track=car.state.track,
            )
            session.add(car_state)
            session.commit()

    @log_execution_time
    def add_vehicle_pass(
        self,
        vehicle_id: int,
        time_in_zone: float,
        stop_duration: float,
        min_speed: float,
        image_path: str,
        entry_time: float | None = None,
        exit_time: float | None = None,
    ):
        if self.read_only_mode:
            logger.debug("🚫 Blocked add_vehicle_pass (READ-ONLY MODE)")
            return None

        with self.Session() as session:
            vehicle_pass = VehiclePass(
                vehicle_id=vehicle_id,
                time_in_zone=time_in_zone,
                stop_duration=stop_duration,
                min_speed=min_speed,
                image_path=image_path,
                entry_time=entry_time,
                exit_time=exit_time,
            )
            session.add(vehicle_pass)
            session.commit()
            return vehicle_pass.id

    @log_execution_time
    def save_vehicle_pass_raw(
        self,
        vehicle_pass_id: int,
        raw_payload: dict,
        sample_count: int,
        raw_complete: bool = True,
    ) -> bool:
        if self.read_only_mode:
            logger.debug("🚫 Blocked save_vehicle_pass_raw (READ-ONLY MODE)")
            return False

        with self.Session() as session:
            existing = (
                session.query(VehiclePassRaw).filter(VehiclePassRaw.vehicle_pass_id == vehicle_pass_id).one_or_none()
            )
            if existing:
                existing.raw_payload = raw_payload
                existing.sample_count = sample_count
                existing.raw_complete = raw_complete
            else:
                session.add(
                    VehiclePassRaw(
                        vehicle_pass_id=vehicle_pass_id,
                        raw_payload=raw_payload,
                        sample_count=sample_count,
                        raw_complete=raw_complete,
                    )
                )
            session.commit()
            return True

    @log_execution_time
    def get_pass_detail(self, pass_id: int):
        with self.Session() as session:
            result = (
                session.query(VehiclePass, VehiclePassRaw)
                .outerjoin(VehiclePassRaw, VehiclePassRaw.vehicle_pass_id == VehiclePass.id)
                .filter(VehiclePass.id == pass_id)
                .first()
            )
            if not result:
                return None
            vehicle_pass, raw = result
            # Expunge so objects survive session close
            session.expunge(vehicle_pass)
            if raw is not None:
                session.expunge(raw)
            return {"pass": vehicle_pass, "raw": raw}

    @log_execution_time
    def update_image_path(self, old_path: str, new_path: str) -> int:
        """Update image_path for a specific record. Returns rows updated."""
        if self.read_only_mode:
            logger.debug("🚫 Blocked update_image_path (READ-ONLY MODE)")
            return 0

        with self.Session() as session:
            rows = (
                session.query(VehiclePass).filter(VehiclePass.image_path == old_path).update({"image_path": new_path})
            )
            session.commit()
            return rows

    @log_execution_time
    def get_passes_needing_clips(self, limit: int = 10, min_exit_age_sec: float = 2.0):
        """Return passes that need clip work: new, retryable, or needing upload.

        Picks up:
        - clip_status IS NULL: never attempted
        - clip_status = 'retry': ffmpeg failed but retries remain
        - clip_status = 'local': built locally but MinIO upload pending
        """
        cutoff = time.time() - min_exit_age_sec
        with self.Session() as session:
            return (
                session.query(VehiclePass)
                .filter(VehiclePass.exit_time.isnot(None))
                .filter(VehiclePass.exit_time > 0)
                .filter(VehiclePass.exit_time <= cutoff)
                .filter(VehiclePass.clip_status.is_(None) | VehiclePass.clip_status.in_(["retry", "local"]))
                .order_by(VehiclePass.timestamp.desc())
                .limit(limit)
                .all()
            )

    @log_execution_time
    def get_clips_missing_locally(self, limit: int = 10):
        """Return passes with ready clips that may need local file restoration."""
        with self.Session() as session:
            return (
                session.query(VehiclePass)
                .filter(VehiclePass.clip_status == "ready")
                .filter(VehiclePass.clip_path.isnot(None))
                .order_by(VehiclePass.timestamp.desc())
                .limit(limit)
                .all()
            )

    @log_execution_time
    def update_clip_status(
        self,
        pass_id: int,
        status: str | None,
        clip_path: str | None = None,
        clip_error: str | None = None,
    ):
        if self.read_only_mode:
            logger.debug("🚫 Blocked update_clip_status (READ-ONLY MODE)")
            return

        with self.Session() as session:
            update_fields = {"clip_status": status}
            if clip_path is not None:
                update_fields["clip_path"] = clip_path
            if clip_error is not None:
                update_fields["clip_error"] = clip_error
            session.query(VehiclePass).filter(VehiclePass.id == pass_id).update(update_fields)
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
                .filter(
                    or_(
                        VehiclePass.image_path.like("minio://%"),  # Legacy minio prefix
                        VehiclePass.image_path.like("local://%"),  # Local prefix (pre-archive)
                        VehiclePass.image_path.like("bremen://%"),  # Bremen archive prefix
                    )
                )
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
                .filter(
                    or_(
                        VehiclePass.image_path.like("minio://%"),  # Legacy minio prefix
                        VehiclePass.image_path.like("local://%"),  # Local prefix (pre-archive)
                        VehiclePass.image_path.like("bremen://%"),  # Bremen archive prefix
                    )
                )
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

    # ==========================================
    # Full Config Backup/Restore
    # ==========================================

    def save_full_config(self, yaml_string: str) -> None:
        """Persist the full config YAML to the database for disaster recovery."""
        self.update_config_setting(
            category="system",
            key="full_config_yaml",
            value=yaml_string,
            updated_by="config_save",
            change_reason="Config file saved",
        )

    def load_full_config(self) -> str | None:
        """Load the most recent full config YAML from the database."""
        return self.get_config_setting(category="system", key="full_config_yaml")

    # ==========================================
    # Vehicle Attributes Methods
    # ==========================================

    @log_execution_time
    def get_vehicle_stats_summary(self) -> dict | None:
        """Get summary stats: total classified, distinct clusters, coverage %."""
        with self.Session() as session:
            result = session.execute(
                text("""
                SELECT
                    COUNT(*) AS total_classified,
                    COUNT(DISTINCT cluster_id) AS cluster_count,
                    ROUND(
                        COUNT(*)::numeric * 100.0
                        / GREATEST((SELECT COUNT(*) FROM vehicle_passes), 1),
                        1
                    ) AS coverage_pct
                FROM vehicle_attributes
                """)
            ).first()
            if result and result.total_classified > 0:
                return {
                    "total_classified": result.total_classified,
                    "cluster_count": result.cluster_count,
                    "coverage_pct": float(result.coverage_pct),
                }
            return None

    @log_execution_time
    def get_vehicle_type_distribution(self) -> list[dict]:
        """Get vehicle type counts ordered by frequency."""
        with self.Session() as session:
            rows = session.execute(
                text("""
                SELECT vehicle_type, COUNT(*) AS cnt
                FROM vehicle_attributes
                WHERE vehicle_type IS NOT NULL
                GROUP BY vehicle_type
                ORDER BY cnt DESC
                """)
            ).fetchall()
            return [{"vehicle_type": r.vehicle_type, "count": r.cnt} for r in rows]

    @log_execution_time
    def get_vehicle_color_distribution(self) -> list[dict]:
        """Get vehicle color counts ordered by frequency."""
        with self.Session() as session:
            rows = session.execute(
                text("""
                SELECT color, COUNT(*) AS cnt
                FROM vehicle_attributes
                WHERE color IS NOT NULL
                GROUP BY color
                ORDER BY cnt DESC
                """)
            ).fetchall()
            return [{"color": r.color, "count": r.cnt} for r in rows]

    @log_execution_time
    def get_top_make_models(self, limit: int = 15) -> list[dict]:
        """Get top make/models with a representative image path."""
        with self.Session() as session:
            rows = session.execute(
                text("""
                SELECT
                    va.make_model,
                    COUNT(*) AS cnt,
                    (
                        SELECT vp.image_path
                        FROM vehicle_attributes va2
                        JOIN vehicle_passes vp ON vp.id = va2.vehicle_pass_id
                        WHERE va2.make_model = va.make_model
                          AND vp.image_path IS NOT NULL
                          AND vp.image_path != ''
                        ORDER BY va2.confidence DESC NULLS LAST
                        LIMIT 1
                    ) AS image_path
                FROM vehicle_attributes va
                WHERE va.make_model IS NOT NULL
                GROUP BY va.make_model
                ORDER BY cnt DESC
                LIMIT :limit
                """),
                {"limit": limit},
            ).fetchall()
            return [{"make_model": r.make_model, "count": r.cnt, "image_path": r.image_path} for r in rows]

    @log_execution_time
    def get_cluster_gallery(self, limit: int = 30) -> list[dict]:
        """Get representative images for top clusters."""
        with self.Session() as session:
            rows = session.execute(
                text("""
                SELECT
                    va.cluster_id,
                    va.vehicle_type,
                    va.color,
                    va.make_model,
                    vp.image_path,
                    cs.cluster_size
                FROM vehicle_attributes va
                JOIN vehicle_passes vp ON vp.id = va.vehicle_pass_id
                LEFT JOIN (
                    SELECT cluster_id, COUNT(*) AS cluster_size
                    FROM vehicle_attributes
                    WHERE cluster_id IS NOT NULL
                    GROUP BY cluster_id
                ) cs ON cs.cluster_id = va.cluster_id
                WHERE va.is_representative = true
                  AND vp.image_path IS NOT NULL
                  AND vp.image_path != ''
                ORDER BY cs.cluster_size DESC NULLS LAST
                LIMIT :limit
                """),
                {"limit": limit},
            ).fetchall()
            return [
                {
                    "cluster_id": r.cluster_id,
                    "vehicle_type": r.vehicle_type,
                    "color": r.color,
                    "make_model": r.make_model,
                    "image_path": r.image_path,
                    "cluster_size": r.cluster_size,
                }
                for r in rows
            ]

    @log_execution_time
    def get_vehicle_attributes_for_passes(self, pass_ids: list[int]) -> Dict[int, dict]:
        """Bulk lookup vehicle attributes keyed by vehicle_pass_id."""
        if not pass_ids:
            return {}
        with self.Session() as session:
            rows = session.execute(
                text("""
                SELECT vehicle_pass_id, vehicle_type, color, make_model
                FROM vehicle_attributes
                WHERE vehicle_pass_id = ANY(:ids)
                """),
                {"ids": pass_ids},
            ).fetchall()
            return {
                r.vehicle_pass_id: {
                    "vehicle_type": r.vehicle_type,
                    "color": r.color,
                    "make_model": r.make_model,
                }
                for r in rows
            }

    # ==========================================
    # Real-time Insights Methods
    # ==========================================

    @log_execution_time
    def get_peak_hour_today(self):
        """Get today's peak hour with vehicle count in Chicago timezone."""
        with self.Session() as session:
            result = session.execute(
                text("""
                WITH hourly_series AS (
                    SELECT generate_series(0, 23) AS hour
                ),
                today_counts AS (
                    SELECT
                        EXTRACT(hour FROM timestamp AT TIME ZONE 'America/Chicago') AS hour,
                        COUNT(*) AS vehicle_count
                    FROM vehicle_passes
                    WHERE DATE(timestamp AT TIME ZONE 'America/Chicago') = CURRENT_DATE
                    GROUP BY EXTRACT(hour FROM timestamp AT TIME ZONE 'America/Chicago')
                )
                SELECT
                    hs.hour,
                    COALESCE(tc.vehicle_count, 0) AS count,
                    CASE
                        WHEN hs.hour = 0 THEN '12-1 AM'
                        WHEN hs.hour < 12 THEN hs.hour || '-' || (hs.hour + 1) || ' AM'
                        WHEN hs.hour = 12 THEN '12-1 PM'
                        ELSE (hs.hour - 12) || '-' || (hs.hour - 11) || ' PM'
                    END AS display_format
                FROM hourly_series hs
                LEFT JOIN today_counts tc ON hs.hour = tc.hour
                WHERE COALESCE(tc.vehicle_count, 0) > 0
                ORDER BY count DESC, hs.hour
                LIMIT 1
            """)
            ).first()

            if result and result.count > 0:
                return {"hour": result.hour, "count": result.count, "display": result.display_format}
            return None

    @log_execution_time
    def get_compliance_streak(self):
        """Get the longest compliance streak from recent passes using gaps-and-islands technique."""
        with self.Session() as session:
            result = session.execute(
                text("""
                WITH compliance_data AS (
                    SELECT
                        id,
                        timestamp,
                        time_in_zone >= 2.0 AS is_compliant,
                        ROW_NUMBER() OVER (ORDER BY timestamp) as rn
                    FROM vehicle_passes
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
                    ORDER BY timestamp
                ),
                groups AS (
                    SELECT
                        id,
                        timestamp,
                        is_compliant,
                        rn - ROW_NUMBER() OVER (PARTITION BY is_compliant ORDER BY timestamp) AS grp
                    FROM compliance_data
                ),
                streaks AS (
                    SELECT
                        is_compliant,
                        grp,
                        COUNT(*) AS streak_length,
                        MIN(timestamp) AS streak_start,
                        MAX(timestamp) AS streak_end
                    FROM groups
                    GROUP BY is_compliant, grp
                )
                SELECT
                    streak_length,
                    streak_start AT TIME ZONE 'America/Chicago' AS start_chicago,
                    streak_end AT TIME ZONE 'America/Chicago' AS end_chicago
                FROM streaks
                WHERE is_compliant = true
                ORDER BY streak_length DESC
                LIMIT 1
            """)
            ).first()

            if result:
                return {"length": result.streak_length, "start": result.start_chicago, "end": result.end_chicago}
            return None

    @log_execution_time
    def get_average_stop_time(self, hours=24):
        """Get average stop time for the last N hours."""
        with self.Session() as session:
            result = session.execute(
                text("""
                SELECT
                    AVG(time_in_zone) AS avg_time_in_zone,
                    AVG(stop_duration) AS avg_stop_duration,
                    COUNT(*) AS sample_size
                FROM vehicle_passes
                WHERE timestamp >= NOW() - INTERVAL ':hours hours'
            """),
                {"hours": hours},
            ).first()

            if result and result.sample_size > 0:
                return {
                    "avg_time_in_zone": round(result.avg_time_in_zone, 1),
                    "avg_stop_duration": round(result.avg_stop_duration or 0, 1),
                    "sample_size": result.sample_size,
                }
            return None

    @log_execution_time
    def get_fastest_vehicle_today(self):
        """Get fastest vehicle from today with timestamp and details."""
        with self.Session() as session:
            result = session.execute(
                text("""
                SELECT
                    min_speed,
                    timestamp AT TIME ZONE 'America/Chicago' AS chicago_time,
                    image_path
                FROM vehicle_passes
                WHERE DATE(timestamp AT TIME ZONE 'America/Chicago') = CURRENT_DATE
                ORDER BY min_speed DESC
                LIMIT 1
            """)
            ).first()

            if result:
                return {
                    "speed": round(result.min_speed, 1),
                    "time": result.chicago_time,
                    "image_path": result.image_path,
                }
            return None

    @log_execution_time
    def get_traffic_summary_today(self):
        """Get comprehensive traffic summary for today."""
        with self.Session() as session:
            result = session.execute(
                text("""
                SELECT
                    COUNT(*) AS total_vehicles,
                    AVG(time_in_zone) AS avg_stop_time,
                    SUM(CASE WHEN time_in_zone >= 2.0 THEN 1 ELSE 0 END) AS compliant_vehicles,
                    MAX(min_speed) AS fastest_speed,
                    COUNT(DISTINCT EXTRACT(hour FROM timestamp AT TIME ZONE 'America/Chicago'))
                        AS active_hours
                FROM vehicle_passes
                WHERE DATE(timestamp AT TIME ZONE 'America/Chicago') = CURRENT_DATE
            """)
            ).first()

            if result and result.total_vehicles > 0:
                # Calculate compliance rate in Python to avoid division by zero
                compliance_rate = (
                    (result.compliant_vehicles / result.total_vehicles * 100) if result.total_vehicles > 0 else 0
                )

                return {
                    "total_vehicles": result.total_vehicles,
                    "avg_stop_time": round(result.avg_stop_time, 1),
                    "compliance_rate": round(compliance_rate, 1),
                    "fastest_speed": round(result.fastest_speed, 1),
                    "active_hours": result.active_hours,
                }
            return None
