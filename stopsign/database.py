import sqlite3
from datetime import datetime


class Database:
    def __init__(self, db_file="stop_sign_data.db"):
        self.db_file = db_file
        self.conn = None
        self.ensure_connection()
        self.create_tables()

    def ensure_connection(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_file)

    def create_tables(self):
        self.ensure_connection()
        if self.conn:
            cursor = self.conn.cursor()
        else:
            raise Exception("Database connection not established")

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

        self.conn.commit()

    def add_vehicle_pass(self, vehicle_id, stop_score, stop_duration, min_speed, image_path):
        self.ensure_connection()
        cursor = self.conn.cursor()
        cursor.execute(
            """
        INSERT INTO vehicle_passes (timestamp, vehicle_id, stop_score, stop_duration, min_speed, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (datetime.now(), vehicle_id, stop_score, stop_duration, min_speed, image_path),
        )
        self.conn.commit()

    def update_daily_statistics(self):
        self.ensure_connection()
        cursor = self.conn.cursor()
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

        self.conn.commit()

    def get_daily_statistics(self, date):
        self.ensure_connection()
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM daily_statistics WHERE date = ?", (date,))
        return cursor.fetchone()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
