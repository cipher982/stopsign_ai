"""Durable local outbox for completed vehicle passes.

A completed pass is accepted locally before any remote database delivery is attempted.
The outbox is SQLite-backed with full synchronous commits, and each payload remains
pending until the remote insert is acknowledged or an idempotency lookup proves that
the insert committed. A process restart therefore resumes delivery without losing a
pass, and capacity pressure never discards the oldest accepted record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SPOOL_DIR = os.getenv("PASS_SPOOL_DIR", "/app/data/pending-passes")
RETRY_INTERVAL_SECONDS = float(os.getenv("PASS_SPOOL_RETRY_SECONDS", "60"))
RETRY_BATCH = 20
MAX_SWEEP_ROWS = RETRY_BATCH * 5
RETRY_BACKOFF_BASE_SECONDS = 60.0
RETRY_BACKOFF_MAX_SECONDS = 3600.0

_lock = threading.Lock()


def _database_path() -> Path:
    return Path(SPOOL_DIR) / "passes.sqlite3"


def _retry_delay(attempts: int) -> float:
    """Move rejected rows out of the hot path without losing their evidence."""
    return min(
        RETRY_BACKOFF_MAX_SECONDS,
        RETRY_BACKOFF_BASE_SECONDS * (2 ** min(max(attempts, 0), 6)),
    )


def _ensure_schema(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(pending_passes)").fetchall()}
    if "next_attempt_at" not in columns:
        connection.execute("ALTER TABLE pending_passes ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0")
        connection.commit()


_retry_lock = threading.Lock()
_worker_started = False
_worker_wakeup = threading.Event()

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS pending_passes (
    pass_key TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at REAL NOT NULL,
    next_attempt_at REAL NOT NULL DEFAULT 0
)
"""

_CREATE_ARCHIVE_FLIPS_TABLE = """
CREATE TABLE IF NOT EXISTS pending_archive_flips (
    object_name TEXT PRIMARY KEY,
    enqueued_at REAL NOT NULL
)
"""


def _open_database(*, read_only: bool = False) -> sqlite3.Connection:
    path = _database_path()
    if read_only:
        if not path.is_file():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
        return connection
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(_CREATE_TABLE)
    connection.execute(_CREATE_ARCHIVE_FLIPS_TABLE)
    _ensure_schema(connection)
    connection.commit()
    return connection


def enqueue_archive_flip(object_name: str, enqueued_at: float | None = None) -> bool:
    """Persist an archived object's pending database path flip."""
    try:
        connection = _open_database()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO pending_archive_flips (object_name, enqueued_at) VALUES (?, ?) "
                    "ON CONFLICT(object_name) DO UPDATE SET enqueued_at = "
                    "MIN(pending_archive_flips.enqueued_at, excluded.enqueued_at)",
                    (object_name, enqueued_at if enqueued_at is not None else time.time()),
                )
        finally:
            connection.close()
        return True
    except (OSError, sqlite3.Error) as exc:
        logger.error("Could not persist archive flip for %s: %s", object_name, exc)
        return False


def pending_archive_flips() -> dict[str, float] | None:
    """Return durable archive flips, or None when the outbox is unreadable."""
    try:
        connection = _open_database(read_only=True)
        try:
            rows = connection.execute("SELECT object_name, enqueued_at FROM pending_archive_flips").fetchall()
        finally:
            connection.close()
        return {str(object_name): float(enqueued_at) for object_name, enqueued_at in rows}
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        logger.error("Could not read archive flip outbox: %s", exc)
        return None


def forget_archive_flip(object_name: str) -> bool:
    """Remove an archive flip after its database path and marker are durable."""
    try:
        connection = _open_database()
        try:
            with connection:
                connection.execute(
                    "DELETE FROM pending_archive_flips WHERE object_name = ?",
                    (object_name,),
                )
        finally:
            connection.close()
        return True
    except (OSError, sqlite3.Error) as exc:
        logger.error("Could not remove archive flip for %s: %s", object_name, exc)
        return False


def _encode_payload(kwargs: dict[str, Any]) -> str:
    return json.dumps(kwargs, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pass_key(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _write_fallback_json(pass_key: str, payload_json: str) -> Path:
    """Keep a durable file fallback when the SQLite outbox cannot be opened."""
    directory = Path(SPOOL_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"pass_{pass_key}.json"
    partial = target.with_suffix(".part")
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(payload_json)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(target)
    try:
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        logger.warning("Could not fsync pass outbox directory %s", directory)
    return target


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Unreadable legacy pass outbox file %s: %s", path.name, exc)
        return None
    if not isinstance(payload, dict):
        logger.error("Legacy pass outbox file %s does not contain an object", path.name)
        return None
    return payload


def _migrate_legacy_files(connection: sqlite3.Connection) -> None:
    """Adopt JSON records written by pre-SQLite versions without losing them."""
    directory = Path(SPOOL_DIR)
    if not directory.exists():
        return

    migrated: list[Path] = []
    with connection:
        for path in sorted(directory.glob("pass_*.json"), key=lambda item: item.stat().st_mtime):
            payload = _read_json_file(path)
            if payload is None:
                continue
            if "event_time" not in payload:
                legacy_time = payload.get("exit_time")
                if (
                    isinstance(legacy_time, bool)
                    or not isinstance(legacy_time, (int, float))
                    or not math.isfinite(legacy_time)
                ):
                    logger.error(
                        "Legacy pass outbox file %s has no trustworthy event time; leaving it for recovery",
                        path.name,
                    )
                    continue
                payload["event_time"] = float(legacy_time)
            try:
                payload_json = _encode_payload(payload)
            except (TypeError, ValueError) as exc:
                logger.error("Legacy pass outbox file %s is not serializable: %s", path.name, exc)
                continue
            now = time.time()
            connection.execute(
                "INSERT OR IGNORE INTO pending_passes "
                "(pass_key, payload_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (_pass_key(payload_json), payload_json, path.stat().st_mtime, now),
            )
            migrated.append(path)

    # Delete only after the SQLite transaction committed. A crash before this point
    # leaves the JSON file for the next boot; INSERT OR IGNORE makes that harmless.
    for path in migrated:
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Migrated pass outbox file %s but could not remove it: %s", path.name, exc)


def _pending_payloads() -> tuple[list[dict[str, Any]], bool]:
    """Return pending payloads and whether every durable source was readable."""
    payloads: list[dict[str, Any]] = []
    complete = True
    try:
        connection = _open_database(read_only=True)
        try:
            rows = connection.execute("SELECT payload_json FROM pending_passes").fetchall()
        finally:
            connection.close()
        for (payload_json,) in rows:
            try:
                payload = json.loads(payload_json)
            except (TypeError, ValueError):
                complete = False
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
            else:
                complete = False
    except (OSError, sqlite3.Error) as exc:
        logger.error("Could not read pass outbox database: %s", exc)
        complete = False

    directory = Path(SPOOL_DIR)
    if not directory.is_dir():
        return payloads, False
    try:
        legacy_paths = list(directory.glob("pass_*.json"))
    except OSError as exc:
        logger.error("Could not enumerate pass outbox files: %s", exc)
        return payloads, False
    for path in legacy_paths:
        payload = _read_json_file(path)
        if payload is not None:
            payloads.append(payload)
        else:
            complete = False
    return payloads, complete


def pending_pass_image_paths() -> set[str] | None:
    """Return image paths, or ``None`` when durable spool evidence is incomplete."""
    payloads, complete = _pending_payloads()
    if not complete:
        return None
    image_paths: set[str] = set()
    for payload in payloads:
        image_path = payload.get("image_path")
        if isinstance(image_path, str) and image_path.startswith("local://"):
            image_paths.add(image_path.removeprefix("local://"))
    return image_paths


def enqueue_pass(kwargs: dict[str, Any]) -> str:
    """Durably admit a completed pass before attempting remote delivery."""
    payload_json = _encode_payload(kwargs)
    pass_key = _pass_key(payload_json)
    now = time.time()
    try:
        connection = _open_database()
        try:
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO pending_passes "
                    "(pass_key, payload_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (pass_key, payload_json, now, now),
                )
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        # A second durable representation is preferable to dropping an accepted
        # pass when SQLite is locked/corrupt. It is adopted on the next sweep.
        logger.error("SQLite pass outbox unavailable; using durable file fallback: %s", exc)
        _write_fallback_json(pass_key, payload_json)

    _worker_wakeup.set()
    logger.info("Durably queued vehicle pass %s", kwargs.get("vehicle_id"))
    return pass_key


def _record_attempt_failure(connection: sqlite3.Connection, pass_key: str, error: str) -> None:
    now = time.time()
    row = connection.execute("SELECT attempts FROM pending_passes WHERE pass_key = ?", (pass_key,)).fetchone()
    attempts = int(row[0]) if row else 0
    next_attempt_at = now + _retry_delay(attempts)
    connection.execute(
        "UPDATE pending_passes SET attempts = attempts + 1, last_error = ?, "
        "updated_at = ?, next_attempt_at = ? WHERE pass_key = ?",
        (error[:2000], now, next_attempt_at, pass_key),
    )
    connection.commit()


def retry_pending_passes(db: Any) -> int:
    """Deliver pending passes; keep every row until acknowledgement or proof."""
    with _retry_lock:
        try:
            connection = _open_database()
        except (OSError, sqlite3.Error) as exc:
            logger.error("Pass outbox sweep cannot open SQLite database: %s", exc)
            return 0

        landed = 0
        processed = 0
        try:
            _migrate_legacy_files(connection)
            while processed < MAX_SWEEP_ROWS:
                rows = connection.execute(
                    "SELECT pass_key, payload_json FROM pending_passes "
                    "WHERE next_attempt_at <= ? "
                    "ORDER BY next_attempt_at, created_at, pass_key LIMIT ?",
                    (time.time(), min(RETRY_BATCH, MAX_SWEEP_ROWS - processed)),
                ).fetchall()
                if not rows:
                    break
                for pass_key, payload_json in rows:
                    processed += 1
                    try:
                        kwargs = json.loads(payload_json)
                        if not isinstance(kwargs, dict):
                            raise ValueError("payload is not an object")
                    except (TypeError, ValueError) as exc:
                        _record_attempt_failure(connection, pass_key, f"invalid payload: {exc}")
                        continue

                    try:
                        if db.has_vehicle_pass(kwargs.get("vehicle_id"), kwargs.get("exit_time")):
                            logger.info(
                                "Pending pass %s was already written; dropping the acknowledged copy",
                                pass_key,
                            )
                        else:
                            inserted_id = db.add_vehicle_pass(**kwargs)
                            if inserted_id is None:
                                raise RuntimeError("database insert was not acknowledged")
                            landed += 1
                            logger.info(
                                "Recovered pending pass %s for vehicle %s",
                                pass_key,
                                kwargs.get("vehicle_id"),
                            )
                        with connection:
                            connection.execute("DELETE FROM pending_passes WHERE pass_key = ?", (pass_key,))
                    except Exception as exc:  # noqa: BLE001 - the database is the retry boundary
                        logger.warning("Pending pass %s still cannot be written: %s", pass_key, exc)
                        _record_attempt_failure(connection, pass_key, str(exc))
                        continue
        finally:
            connection.close()
        return landed


def _outbox_worker_loop(db: Any) -> None:
    while True:
        try:
            retry_pending_passes(db)
        except Exception as exc:  # noqa: BLE001 - the worker outlives its failures
            logger.error("Pass outbox sweep failed: %s", exc)
        _worker_wakeup.wait(RETRY_INTERVAL_SECONDS)
        _worker_wakeup.clear()


def start_pass_outbox_worker(db: Any) -> None:
    """Start the replay worker once, migrating pre-rewrite JSON records first."""
    global _worker_started

    if db is None:
        return
    with _lock:
        if _worker_started:
            return
        try:
            connection = _open_database()
            try:
                _migrate_legacy_files(connection)
            finally:
                connection.close()
        except (OSError, sqlite3.Error) as exc:
            logger.error("Pass outbox migration failed at analyzer boot: %s", exc)
        _worker_started = True

    thread = threading.Thread(target=_outbox_worker_loop, args=(db,), daemon=True, name="pass-outbox")
    thread.start()
    logger.info("Pass outbox worker started (database %s, every %.0fs)", _database_path(), RETRY_INTERVAL_SECONDS)
