"""Durable spool for passes the database would not accept.

A pass is the product of the whole pipeline and it is written exactly once, at zone
exit. The insert retries in line for a few seconds, which covers a blip; past that the
pass was logged and dropped, so a database outage longer than the retry window - or a
restart mid-write - lost the vehicle permanently.

Failed writes are spooled to disk instead and replayed by a background worker, so the
row lands when the database comes back rather than never. The worker starts at analyzer
boot, which is also how a spool left behind by the previous process gets drained.

Replays are idempotent: a pass with the same vehicle id and zone exit time is looked up
before inserting, so a write that committed but never returned cannot duplicate.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from typing import Optional

logger = logging.getLogger(__name__)

SPOOL_DIR = os.getenv("PASS_SPOOL_DIR", "/app/data/pending-passes")
RETRY_INTERVAL_SECONDS = float(os.getenv("PASS_SPOOL_RETRY_SECONDS", "60"))
RETRY_BATCH = 20
SPOOL_MAX_FILES = 5000

_lock = threading.Lock()
_worker_started = False


def spool_failed_pass(db: Any, kwargs: dict[str, Any]) -> Optional[Path]:
    """Write a pass that could not be inserted to disk, for the worker to replay."""
    try:
        directory = Path(SPOOL_DIR)
        directory.mkdir(parents=True, exist_ok=True)

        existing = sorted(directory.glob("pass_*.json"), key=lambda p: p.stat().st_mtime)
        if len(existing) >= SPOOL_MAX_FILES:
            # Bounded like everything else here: the oldest goes before the newest is
            # refused, because a newer pass is likelier to still be in the zone.
            for stale in existing[: len(existing) - SPOOL_MAX_FILES + 1]:
                stale.unlink(missing_ok=True)
            logger.error("Pass spool full; discarded the oldest entries")

        target = directory / f"pass_{uuid.uuid4().hex}.json"
        partial = target.with_suffix(".part")
        # Write-then-rename: a half-written spool file is never replayed.
        partial.write_text(json.dumps(kwargs))
        partial.replace(target)
        logger.warning(
            "Spooled pass for vehicle %s (%s) after the database refused it",
            kwargs.get("vehicle_id"),
            target.name,
        )
        start_spool_worker(db)
        return target
    except Exception as exc:  # noqa: BLE001 - the caller is already on a failure path
        logger.error("Failed to spool pass for vehicle %s: %s", kwargs.get("vehicle_id"), exc)
        return None


def retry_spooled_passes(db: Any) -> int:
    """Insert every spooled pass the database will now accept. Returns how many landed."""
    directory = Path(SPOOL_DIR)
    if not directory.exists():
        return 0

    landed = 0
    pending = sorted(directory.glob("pass_*.json"), key=lambda p: p.stat().st_mtime)
    for path in pending[:RETRY_BATCH]:
        try:
            kwargs = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 - an unreadable spool file is dead weight
            logger.error("Discarding unreadable spool file %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
            continue

        try:
            if db.has_vehicle_pass(kwargs.get("vehicle_id"), kwargs.get("exit_time")):
                logger.info("Spooled pass %s was already written; dropping the copy", path.name)
                path.unlink(missing_ok=True)
                continue
            db.add_vehicle_pass(**kwargs)
        except Exception as exc:  # noqa: BLE001 - the database is the thing that is down
            logger.warning("Spooled pass %s still cannot be written: %s", path.name, exc)
            break  # still down: stop here and try again next sweep
        path.unlink(missing_ok=True)
        landed += 1
        logger.info("Recovered spooled pass for vehicle %s from %s", kwargs.get("vehicle_id"), path.name)
    return landed


def _spool_worker_loop(db: Any) -> None:
    while True:
        try:
            retry_spooled_passes(db)
        except Exception as exc:  # noqa: BLE001 - the worker outlives its failures
            logger.error("Pass spool sweep failed: %s", exc)
        time.sleep(RETRY_INTERVAL_SECONDS)


def start_spool_worker(db: Any) -> None:
    """Start the replay worker once; the first sweep runs immediately."""
    global _worker_started

    if db is None:
        return
    with _lock:
        if _worker_started:
            return
        _worker_started = True

    thread = threading.Thread(target=_spool_worker_loop, args=(db,), daemon=True, name="pass-spool")
    thread.start()
    logger.info("Pass spool worker started (dir %s, every %.0fs)", SPOOL_DIR, RETRY_INTERVAL_SECONDS)
