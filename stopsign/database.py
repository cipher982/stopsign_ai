import contextlib
import json
import logging
import os
import sqlite3
from datetime import datetime

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.conn = None
        self.ensure_connection()
        self.create_tables()
        logger.info(f"Database connection established at {self.db_file}")

    @contextlib.contextmanager
    def get_cursor(self):
        self.ensure_connection()
        cursor = self.conn.cursor()  # type: ignore
        try:
            yield cursor
            self.conn.commit()  # type: ignore
        except Exception as e:
            self.conn.rollback()  # type: ignore
            raise e
        finally:
            cursor.close()

    def ensure_connection(self):
        if not self.conn:
            directory = os.path.dirname(self.db_file)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.conn = sqlite3.connect(self.db_file)

    def create_tables(self):
        with self.get_cursor() as cursor:
            # Add new table for car state history
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS car_state_history (
                id INTEGER PRIMARY KEY,
                car_id INTEGER,
                last_seen DATETIME,
                location TEXT,
                speed REAL,
                is_parked BOOLEAN,
                in_stop_zone BOOLEAN,
                time_in_zone REAL,
                min_speed_in_zone REAL,
                stop_duration REAL,
                entry_time REAL,
                exit_time REAL,
                track TEXT
            )
            """)

            # Create vehicle_passes table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_passes (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                vehicle_id INTEGER,
                time_in_zone REAL,
                stop_duration REAL,
                min_speed REAL,
                image_path TEXT
            )
            """)

    def save_car_state(self, car):
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO car_state_history (
                    car_id, last_seen, location, speed, is_parked, in_stop_zone,
                    time_in_zone, min_speed_in_zone, stop_duration, entry_time, exit_time, track
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    car.id,
                    datetime.fromtimestamp(car.state.last_update_time),
                    json.dumps(car.state.location),
                    car.state.speed,
                    car.state.is_parked,
                    car.state.in_stop_zone,
                    car.state.time_in_zone,
                    car.state.min_speed_in_zone,
                    car.state.stop_duration,
                    car.state.entry_time,
                    car.state.exit_time,
                    json.dumps(car.state.track),
                ),
            )

    def add_vehicle_pass(
        self, vehicle_id: int, time_in_zone: float, stop_duration: float, min_speed: float, image_path: str
    ):
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO vehicle_passes 
                (timestamp, vehicle_id, time_in_zone, stop_duration, min_speed, image_path) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (datetime.now(), vehicle_id, time_in_zone, stop_duration, min_speed, image_path),
            )

    def get_min_speed_percentile(self, min_speed: float, hours: int = 24) -> float:
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) * 100.0 / (
                    SELECT COUNT(*) FROM vehicle_passes 
                    WHERE timestamp >= datetime('now', ? || ' hours')
                )
                FROM vehicle_passes
                WHERE min_speed <= ? AND timestamp >= datetime('now', ? || ' hours')
                """,
                (f"-{hours}", min_speed, f"-{hours}"),
            )
            return cursor.fetchone()[0]

    def get_time_in_zone_percentile(self, time_in_zone: float, hours: int = 24) -> float:
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) * 100.0 / (
                    SELECT COUNT(*) FROM vehicle_passes 
                    WHERE timestamp >= datetime('now', ? || ' hours')
                )
                FROM vehicle_passes WHERE time_in_zone <= ? AND timestamp >= datetime('now', ? || ' hours')
                """,
                (f"-{hours}", time_in_zone, f"-{hours}"),
            )
            return cursor.fetchone()[0]

    def get_total_passes_last_24h(self):
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) FROM vehicle_passes 
                WHERE timestamp >= datetime('now', '-24 hours')
                """
            )
            return cursor.fetchone()[0]

    def get_daily_statistics(self, date):
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM daily_statistics WHERE date = ?", (date,))
            return cursor.fetchone()

    def get_recent_vehicle_passes(self, limit=10):
        with self.get_cursor() as cursor:
            cursor.execute(
                """
            SELECT id, timestamp, vehicle_id, time_in_zone, stop_duration, min_speed, image_path
            FROM vehicle_passes
            ORDER BY timestamp DESC
            LIMIT ?
            """,
                (limit,),
            )
            return cursor.fetchall()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
