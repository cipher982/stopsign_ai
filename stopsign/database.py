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
                speed FLOAT,
                is_parked BOOLEAN,
                stop_zone_state TEXT,
                stop_score INTEGER,
                min_speed_in_zone FLOAT,
                time_at_zero FLOAT,
                entry_time FLOAT,
                exit_time FLOAT,
                track TEXT
            )
            """)

            # Create vehicle_passes table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_passes (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                vehicle_id INTEGER,
                stop_score INTEGER,
                stop_duration FLOAT,
                min_speed FLOAT,
                image_path TEXT
            )
            """)

            # Create daily_statistics table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_statistics (
                date DATE PRIMARY KEY,
                total_vehicles INTEGER,
                average_score FLOAT,
                complete_stops INTEGER,
                rolling_stops INTEGER,
                no_stops INTEGER,
                busiest_hour INTEGER
            )
            """)

    def save_car_state(self, car):
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO car_state_history (
                    car_id, last_seen, location, speed, is_parked, stop_zone_state,
                    stop_score, min_speed_in_zone, time_at_zero, entry_time, exit_time, track
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    car.id,
                    datetime.fromtimestamp(car.state.last_update_time),
                    json.dumps(car.state.location),
                    car.state.speed,
                    car.state.is_parked,
                    car.state.stop_zone_state,
                    car.state.stop_score,
                    car.state.min_speed_in_zone,
                    car.state.time_at_zero,
                    car.state.entry_time,
                    car.state.exit_time,
                    json.dumps(car.state.track),
                ),
            )

    def add_vehicle_pass(
        self, vehicle_id: int, stop_score: int, stop_duration: float, min_speed: float, image_path: str
    ):
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO vehicle_passes 
                (timestamp, vehicle_id, stop_score, stop_duration, min_speed, image_path) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (datetime.now(), vehicle_id, stop_score, stop_duration, min_speed, image_path),
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

    def get_total_passes_last_24h(self):
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) FROM vehicle_passes 
                WHERE timestamp >= datetime('now', '-24 hours')
                """
            )
            return cursor.fetchone()[0]

    def update_daily_statistics(self):
        with self.get_cursor() as cursor:
            today = datetime.now().date()

            # Get today's data
            cursor.execute(
                """
            SELECT COUNT(*), AVG(stop_score),
                SUM(CASE WHEN stop_score >= 8 THEN 1 ELSE 0 END),
                SUM(CASE WHEN stop_score >= 5 AND stop_score < 8 THEN 1 ELSE 0 END),
                SUM(CASE WHEN stop_score < 5 THEN 1 ELSE 0 END),
                strftime('%H', timestamp) as hour
            FROM vehicle_passes
            WHERE date(timestamp) = ?
            """,
                (today,),
            )

            total, avg_score, complete_stops, rolling_stops, no_stops, _ = cursor.fetchone()

            # Get busiest hour
            cursor.execute(
                """
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM vehicle_passes
            WHERE date(timestamp) = ?
            GROUP BY hour
            ORDER BY count DESC
            LIMIT 1
            """,
                (today,),
            )
            busiest_hour, _ = cursor.fetchone() or (None, 0)

            # Update or insert daily statistics
            cursor.execute(
                """
            INSERT OR REPLACE INTO daily_statistics
            (date, total_vehicles, average_score, complete_stops, rolling_stops, no_stops, busiest_hour)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (today, total, avg_score, complete_stops, rolling_stops, no_stops, busiest_hour),
            )

    def get_daily_statistics(self, date):
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM daily_statistics WHERE date = ?", (date,))
            return cursor.fetchone()

    def get_recent_vehicle_passes(self, limit=10):
        with self.get_cursor() as cursor:
            cursor.execute(
                """
            SELECT id, timestamp, vehicle_id, stop_score, stop_duration, min_speed, image_path
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
