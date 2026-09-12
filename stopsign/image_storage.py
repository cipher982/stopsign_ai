import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import urllib3
from minio import Minio

from stopsign.database import Database
from stopsign.pass_spool import pending_pass_image_paths
from stopsign.settings import ARCHIVE_HEALTH_REDIS_KEY
from stopsign.settings import BREMEN_MINIO_ACCESS_KEY
from stopsign.settings import BREMEN_MINIO_BUCKET
from stopsign.settings import BREMEN_MINIO_ENDPOINT
from stopsign.settings import BREMEN_MINIO_SECRET_KEY
from stopsign.settings import LOCAL_IMAGE_DIR
from stopsign.settings import LOCAL_IMAGE_MAX_COUNT
from stopsign.settings import REDIS_URL

logger = logging.getLogger(__name__)

# Module-level upload queue and worker state
_upload_queue: queue.Queue = queue.Queue(maxsize=100)
_worker_started = False
_worker_lock = threading.Lock()
_prune_lock = threading.Lock()
_last_prune_monotonic = 0.0
PRUNE_INTERVAL_SECONDS = 60.0

# MinIO client timeout (seconds). The archive worker must never hang forever on a
# half-open connection; a timeout surfaces as a retryable error instead.
BREMEN_MINIO_TIMEOUT_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Archive health signal (shared with web_server via Redis; in-memory mirror for tests)
# ---------------------------------------------------------------------------
_health_lock = threading.Lock()
_health = {
    "local_saves": 0,
    "local_save_failures": 0,
    "upload_attempts": 0,
    "upload_successes": 0,
    "upload_failures": 0,
    "last_local_save_ts": None,
    "last_local_save_failure_ts": None,
    "last_upload_attempt_ts": None,
    "last_upload_success_ts": None,
    "last_upload_failure_ts": None,
}
_redis_client = None
_redis_attempted = False

# Per-file upload state so the pruner never deletes a file whose pass still needs it.
#   "pending"  -> queued / waiting to upload (DB path is local://, must stay on disk)
#   "failed"   -> upload or DB flip did not complete (DB path is local://, must stay)
#   "uploaded" -> object archived and the local copy is redundant, so prune-eligible:
#                 either the DB path flipped to bremen://, or no pass references the
#                 file at all
_upload_state: dict[str, str] = {}
_UPLOAD_STATE_MAX = 20000
_upload_state_lock = threading.Lock()

# Objects already archived in Bremen whose DB path flip has not landed yet. The pass
# row is written when the vehicle leaves the zone, which can be a minute after the
# image was captured, so the upload worker's inline retry window can expire first.
# Keep the durable outbox for a full day; a database outage can leave the pass spool
# waiting much longer than the ordinary zone-exit window.
_flip_pending: dict[str, float] = {}
_FLIP_PENDING_MAX = 5000
_FLIP_RETRY_MAX_AGE_SEC = 24 * 60 * 60
_FLIP_RETRY_BATCH = 50

# The sweep runs on the upload worker's idle loop, so it needs its own handle on the
# Database (recorded on every save) plus a rate limit.
FLIP_SWEEP_INTERVAL_SECONDS = 60.0
_last_flip_sweep_monotonic = 0.0
_flip_retry_db: Optional[Database] = None

# Local captures the worker gave up on (three failed attempts, a full queue, or a
# restart that lost the in-memory state) are re-queued from disk on the same idle
# loop. Bounded on purpose: a few files, oldest first, and never more often than the
# interval, so a long archive outage retries gently instead of hammering it.
REQUEUE_SWEEP_INTERVAL_SECONDS = 120.0
REQUEUE_BATCH = 5
REQUEUE_MIN_AGE_SECONDS = 60.0

# A marker is written only after the archive object exists and the database path
# is reconciled (or the capture is proven to have no future pass row). It turns
# the existing local image directory into a restart-safe outbox without another
# database or queue service.
ARCHIVE_MARKER_SUFFIX = ".uploaded"
_last_requeue_sweep_monotonic = 0.0


def _get_redis_client():
    """Lazily build a short-timeout Redis client (best-effort; never raises)."""
    global _redis_client, _redis_attempted
    if _redis_attempted:
        return _redis_client
    _redis_attempted = True
    try:
        import redis

        client = redis.from_url(REDIS_URL, socket_connect_timeout=0.3, socket_timeout=0.3)
        client.ping()
        _redis_client = client
    except Exception:
        _redis_client = None
    return _redis_client


_health_seeded = False


def _seed_health_from_redis() -> None:
    """Carry the archive counters and timestamps across a restart.

    They live in memory and the web reads them through Redis, so a restart reset them
    to zero - and the derived flags read "healthy" from zero. Every one of the 280
    analyzer restarts during the 2026-09-12 archive outage therefore announced a
    healthy archive for its first seconds, on a signal the pipeline-health check
    quotes. Seeding keeps the flag honest across restarts.
    """
    global _health_seeded

    if _health_seeded:
        return
    _health_seeded = True

    client = _get_redis_client()
    if client is None:
        return
    try:
        stored = json.loads(client.get(ARCHIVE_HEALTH_REDIS_KEY) or "{}")
    except Exception:
        return
    if not isinstance(stored, dict):
        return

    with _health_lock:
        for key in _health:
            value = stored.get(key)
            if isinstance(value, (int, float)):
                _health[key] = value


def _durable_pending_local_file_count() -> Optional[int]:
    """Count local captures without an archive acknowledgement marker."""
    image_dir = Path(LOCAL_IMAGE_DIR)
    try:
        if not image_dir.exists():
            return 0
        return sum(1 for path in image_dir.glob("*.jpg") if not _is_durably_archived(path.name))
    except OSError:
        return None


def _health_snapshot() -> dict:
    _seed_health_from_redis()
    with _health_lock:
        h = dict(_health)
    pending_local_files = _durable_pending_local_file_count()
    with _upload_state_lock:
        in_memory_pending = sum(1 for state in _upload_state.values() if state in ("pending", "failed"))
    if pending_local_files is None:
        h["pending_local_files"] = in_memory_pending
        h["archive_outbox_observed"] = False
    else:
        h["pending_local_files"] = max(pending_local_files, in_memory_pending)
        h["archive_outbox_observed"] = True
    h["upload_healthy"] = (
        pending_local_files is not None
        and (
            h["upload_failures"] == 0
            or (
                h["last_upload_success_ts"] is not None
                and h["last_upload_failure_ts"] is not None
                and h["last_upload_success_ts"] >= h["last_upload_failure_ts"]
            )
        )
        and h["pending_local_files"] == 0
    )
    h["local_save_healthy"] = h["local_save_failures"] == 0 or (
        h["last_local_save_ts"] is not None
        and h["last_local_save_failure_ts"] is not None
        and h["last_local_save_ts"] >= h["last_local_save_failure_ts"]
    )
    return h


def _write_health_to_redis() -> None:
    """Persist the health signal for the web_server to read. Best-effort, non-blocking."""
    try:
        client = _get_redis_client()
        if client is not None:
            client.set(ARCHIVE_HEALTH_REDIS_KEY, json.dumps(_health_snapshot()))
    except Exception:
        # The signal is best-effort; local in-memory counters still surface in logs/tests.
        pass


def get_archive_health() -> dict:
    """Return the current archive health signal (counters + timestamps + derived flags)."""
    return _health_snapshot()


def _record_local_save_success() -> None:
    with _health_lock:
        _health["local_saves"] += 1
        _health["last_local_save_ts"] = time.time()
    _write_health_to_redis()


def _record_local_save_failure() -> None:
    with _health_lock:
        _health["local_save_failures"] += 1
        _health["last_local_save_failure_ts"] = time.time()
    _write_health_to_redis()


def _record_upload_attempt() -> None:
    with _health_lock:
        _health["upload_attempts"] += 1
        _health["last_upload_attempt_ts"] = time.time()
    _write_health_to_redis()


def _record_upload_success() -> None:
    with _health_lock:
        _health["upload_successes"] += 1
        _health["last_upload_success_ts"] = time.time()
    _write_health_to_redis()


def _record_upload_failure() -> None:
    with _health_lock:
        _health["upload_failures"] += 1
        _health["last_upload_failure_ts"] = time.time()
    _write_health_to_redis()


def _mark_upload_state(object_name: str, state: str) -> None:
    with _upload_state_lock:
        if len(_upload_state) >= _UPLOAD_STATE_MAX:
            # Bound memory: evicting a state only makes the pruner more conservative
            # (unknown files are never pruned), so this is safe.
            _upload_state.pop(next(iter(_upload_state)))
        _upload_state[object_name] = state


def _get_upload_state(object_name: str) -> Optional[str]:
    with _upload_state_lock:
        state = _upload_state.get(object_name)
    if state is not None:
        return state
    return "uploaded" if _is_durably_archived(object_name) else None


def _archive_marker_path(object_name: str) -> Path:
    return Path(LOCAL_IMAGE_DIR) / f"{object_name}{ARCHIVE_MARKER_SUFFIX}"


def _is_durably_archived(object_name: str) -> bool:
    try:
        return _archive_marker_path(object_name).is_file()
    except OSError:
        return False


def _mark_durably_archived(object_name: str) -> bool:
    """Persist the archive acknowledgement beside the local outbox object."""
    local_path = Path(LOCAL_IMAGE_DIR) / object_name
    if not local_path.exists():
        # Tests and a completed prune may not have a local outbox file left to
        # mark. The archive/DB acknowledgement is still sufficient in that case.
        return True
    marker = _archive_marker_path(object_name)
    partial = marker.with_suffix(marker.suffix + ".part")
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(f"archived_at={time.time():.6f}\n")
        partial.replace(marker)
        return True
    except OSError as exc:
        logger.warning("Could not persist archive marker for %s: %s", object_name, exc)
        partial.unlink(missing_ok=True)
        return False


def _enqueue_flip_retry(object_name: str) -> None:
    """Queue an archived object whose DB path flip did not land inside the worker.

    Bounded: when full the oldest entry is dropped, which only means that object
    stays local. That is the pruner's conservative direction, never a deleted file.
    """
    with _upload_state_lock:
        if len(_flip_pending) >= _FLIP_PENDING_MAX:
            _flip_pending.pop(next(iter(_flip_pending)))
        _flip_pending.setdefault(object_name, time.time())


def _maybe_retry_pending_flips(now: Optional[float] = None) -> None:
    """Rate-limited sweep entry point, driven by the upload worker's idle loop."""
    global _last_flip_sweep_monotonic

    now = time.monotonic() if now is None else now
    if now - _last_flip_sweep_monotonic < FLIP_SWEEP_INTERVAL_SECONDS:
        return
    _last_flip_sweep_monotonic = now
    _retry_pending_flips(_flip_retry_db)


def _forget_pending_flip(object_name: str) -> None:
    with _upload_state_lock:
        _flip_pending.pop(object_name, None)


def _is_flip_pending(object_name: str) -> bool:
    """Is this object already archived, with only its database path left to flip?"""
    with _upload_state_lock:
        return object_name in _flip_pending


def _retry_pending_flips(db: Optional[Database]) -> None:
    """Retry DB path flips the upload worker gave up on.

    Bounded batch per sweep so a backlog cannot stall the worker, and nothing here
    runs on the capture path. A durable failed-pass spool reference keeps an
    uploaded object alive until the pass can be inserted after a long DB outage.
    """
    if db is None or not _flip_pending:
        return

    now = time.time()
    with _upload_state_lock:
        batch = sorted(_flip_pending.items(), key=lambda kv: kv[1])[:_FLIP_RETRY_BATCH]
    pending_pass_images = pending_pass_image_paths()

    for object_name, enqueued_at in batch:
        if _get_upload_state(object_name) == "uploaded":
            _forget_pending_flip(object_name)
            continue

        try:
            rows = db.update_image_path(f"local://{object_name}", f"bremen://{object_name}")
        except Exception as db_err:
            logger.warning("Delayed DB path flip failed for %s: %s", object_name, db_err)
            continue

        if rows:
            if _mark_durably_archived(object_name):
                _mark_upload_state(object_name, "uploaded")
                _forget_pending_flip(object_name)
                logger.info("Delayed DB path flip landed for %s: local:// -> bremen://", object_name)
            continue

        if now - enqueued_at > _FLIP_RETRY_MAX_AGE_SEC:
            if object_name in pending_pass_images:
                logger.info(
                    "Keeping %s: a durable failed-pass record still references the local image",
                    object_name,
                )
                continue
            # Still unreferenced after the retention window. The archive upload
            # succeeded (the only way an entry gets here), so the object is in
            # Bremen and this local copy serves no recorded pass.
            if not _mark_durably_archived(object_name):
                continue
            _mark_upload_state(object_name, "uploaded")
            _forget_pending_flip(object_name)
            logger.info(
                "No pass references %s after %.0fs; releasing the local copy (object is in Bremen)",
                object_name,
                now - enqueued_at,
            )


def start_upload_worker(db: Optional[Database] = None) -> None:
    """Start the archive worker and retain a DB handle for restart reconciliation."""
    global _flip_retry_db
    if db is not None:
        _flip_retry_db = db
    _start_upload_worker()


def _start_upload_worker():
    """Start the background upload worker thread if not already running."""
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            thread = threading.Thread(target=_bremen_upload_worker, daemon=True)
            thread.start()
            _worker_started = True
            logger.info("Bremen upload worker thread started")


def _flip_db_path_with_retry(db: Optional[Database], object_name: str) -> bool:
    """Flip the pass path local:// -> bremen:// once the pass row exists.

    The archive upload completes asynchronously, usually several seconds BEFORE the
    pass is recorded (the pass is persisted at zone exit). A single immediate
    update_image_path therefore finds 0 rows and silently no-ops, leaving the path
    stuck at local:// forever. Retry briefly so we catch the pass insert; a give-up
    is not final - the caller queues a delayed retry that the upload worker sweeps.
    """
    if db is None:
        return False
    old_path = f"local://{object_name}"
    new_path = f"bremen://{object_name}"
    for _ in range(20):
        try:
            rows = db.update_image_path(old_path, new_path)
        except Exception as db_err:
            logger.warning(f"Failed to update DB path for {object_name}: {db_err}")
            return False
        if rows:
            logger.info(f"Updated DB path for {object_name}: local:// -> bremen://")
            return True
        time.sleep(1.0)
    logger.warning(f"Gave up waiting to flip DB path for {object_name} (pass not recorded in time)")
    return False


def _process_upload_item(local_path: str, object_name: str, db: Optional[Database]) -> None:
    """Upload a single queued image to Bremen MinIO and flip its DB path.

    Extracted from the worker loop so tests can exercise one item deterministically.
    """
    # Skip if Bremen credentials not configured
    if not BREMEN_MINIO_SECRET_KEY:
        logger.debug(f"Bremen MinIO not configured, skipping archive of {object_name}")
        return

    _record_upload_attempt()
    # Retry up to 3 times with exponential backoff
    for attempt in range(3):
        try:
            client = Minio(
                BREMEN_MINIO_ENDPOINT,
                access_key=BREMEN_MINIO_ACCESS_KEY,
                secret_key=BREMEN_MINIO_SECRET_KEY,
                secure=False,  # Bremen is on local network
                http_client=urllib3.PoolManager(timeout=BREMEN_MINIO_TIMEOUT_SECONDS),
            )

            # Upload the file
            client.fput_object(
                BREMEN_MINIO_BUCKET,
                object_name,
                local_path,
                content_type="image/jpeg",
            )
            logger.debug(f"Archived {object_name} to Bremen MinIO")
            _record_upload_success()

            # Only release the local file for pruning once BOTH the object is
            # archived AND the DB path points at the archive. The marker makes
            # that acknowledgement survive an analyzer restart.
            if _flip_db_path_with_retry(db, object_name):
                if _mark_durably_archived(object_name):
                    _mark_upload_state(object_name, "uploaded")
                else:
                    _mark_upload_state(object_name, "failed")
                    _enqueue_flip_retry(object_name)
            else:
                _mark_upload_state(object_name, "failed")
                _enqueue_flip_retry(object_name)
            # The success write above happened before the state was final; persist
            # the accurate pending/uploaded counts.
            _write_health_to_redis()
            return
        except Exception as e:
            if attempt == 2:
                logger.error(f"Failed to archive {object_name} after 3 attempts: {e}")
                _mark_upload_state(object_name, "failed")
                _record_upload_failure()
            else:
                logger.warning(f"Bremen upload attempt {attempt + 1} failed for {object_name}: {e}")
                time.sleep(2**attempt)  # Exponential backoff: 1s, 2s


def _maybe_requeue_unarchived_uploads(now: Optional[float] = None) -> int:
    """Re-queue local captures that never reached the archive.

    The worker gives up after three attempts, a full queue skips the archive
    outright, and the per-file state that tracks both is in memory - so a capture that
    missed its window (an archive outage, an overflow, a restart mid-upload) would
    otherwise sit on disk forever while its pass kept pointing at ``local://``.
    Returns the number re-queued.

    Two kinds of file are left alone: anything already archived and flipped, and
    anything the flip sweep is still holding - that upload succeeded, and re-uploading
    it would be work for nothing.
    """
    global _last_requeue_sweep_monotonic

    now = time.monotonic() if now is None else now
    if now - _last_requeue_sweep_monotonic < REQUEUE_SWEEP_INTERVAL_SECONDS:
        return 0
    if not BREMEN_MINIO_SECRET_KEY:
        return 0

    image_dir = Path(LOCAL_IMAGE_DIR)
    if not image_dir.exists():
        return 0

    _last_requeue_sweep_monotonic = now
    wall = time.time()
    candidates: list[Path] = []
    for path in image_dir.glob("*.jpg"):
        try:
            # Skip whatever is still being written or is queued right now.
            if wall - path.stat().st_mtime < REQUEUE_MIN_AGE_SECONDS:
                continue
        except OSError:
            continue
        if _get_upload_state(path.name) != "uploaded" and not _is_flip_pending(path.name):
            candidates.append(path)

    candidates.sort(key=lambda p: p.stat().st_mtime)
    requeued = 0
    for path in candidates[:REQUEUE_BATCH]:
        try:
            _upload_queue.put_nowait((str(path), path.name, _flip_retry_db))
        except queue.Full:
            break
        requeued += 1

    if requeued:
        logger.info(f"Re-queued {requeued} local capture(s) that had not reached the archive")
    return requeued


def _bremen_upload_worker():
    """Background worker: uploads queued images, and sweeps pending DB path flips.

    The sweep rides this thread's idle path instead of owning a timer of its own, so
    a flip is still retried when no further image is ever saved - a quiet evening
    must not strand the last capture of the night.
    """
    while True:
        try:
            try:
                local_path, object_name, db = _upload_queue.get(timeout=FLIP_SWEEP_INTERVAL_SECONDS)
            except queue.Empty:
                _maybe_retry_pending_flips()
                _maybe_requeue_unarchived_uploads()
                continue
            try:
                _process_upload_item(local_path, object_name, db)
                _maybe_retry_pending_flips()
                # A steady stream can keep the queue non-empty forever. Sweep
                # after work as well as on timeout, or restart-recovered captures
                # starve until the camera goes quiet.
                _maybe_requeue_unarchived_uploads()
            finally:
                _upload_queue.task_done()
        except Exception as e:
            logger.error(f"Unexpected error in Bremen upload worker: {e}")


def _prune_old_images():
    """Remove oldest locally-captured images that have been safely archived.

    Only files whose object was uploaded AND whose DB path was flipped to bremen://
    are eligible. Files still pending or failed retain a local:// DB reference and
    must stay on disk so the pass continues to serve; never delete a file its pass
    still needs.
    """
    try:
        image_dir = Path(LOCAL_IMAGE_DIR)
        if not image_dir.exists():
            return

        # Get all jpg files with their modification times
        images = list(image_dir.glob("*.jpg"))

        if len(images) <= LOCAL_IMAGE_MAX_COUNT:
            return

        # Only known-uploaded files may be pruned.
        prunable = [p for p in images if _get_upload_state(p.name) == "uploaded"]
        if not prunable:
            logger.debug("No archived images eligible for pruning; keeping pending/failed local files")
            return

        # Sort by modification time (oldest first)
        prunable.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest uploaded files to get back under limit
        to_remove = len(images) - LOCAL_IMAGE_MAX_COUNT
        removed = 0
        for img_path in prunable[:to_remove]:
            try:
                img_path.unlink()
                _archive_marker_path(img_path.name).unlink(missing_ok=True)
                removed += 1
                logger.debug(f"Pruned archived image: {img_path.name}")
            except Exception as e:
                logger.warning(f"Failed to prune {img_path.name}: {e}")

        logger.info(f"Pruned {removed} archived images to maintain {LOCAL_IMAGE_MAX_COUNT} limit")
    except Exception as e:
        logger.error(f"Error pruning old images: {e}")


def _run_prune_worker() -> None:
    try:
        _prune_old_images()
    finally:
        _prune_lock.release()


def _start_prune_worker() -> None:
    thread = threading.Thread(target=_run_prune_worker, daemon=True)
    thread.start()


def _maybe_prune_old_images(now: Optional[float] = None) -> None:
    """Rate-limit pruning and keep directory scans out of the capture path."""
    global _last_prune_monotonic

    now = time.monotonic() if now is None else now
    if now - _last_prune_monotonic < PRUNE_INTERVAL_SECONDS:
        return

    if not _prune_lock.acquire(blocking=False):
        return

    _last_prune_monotonic = now
    _start_prune_worker()


# Broad capture margin around the detection bbox. The saved crop is the archival
# artifact: recall over precision, since tight display crops can be re-derived
# later but missing pixels are gone forever. YOLO boxes on moving cars regularly
# trail or under-cover the vehicle by more than the old 10%.
CROP_PADDING_FACTOR = 0.4


def compute_crop_rect(
    bbox: Tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    padding_factor: float = CROP_PADDING_FACTOR,
) -> Tuple[int, int, int, int]:
    """Padded, frame-clamped crop rect (XYXY ints) for a detection bbox."""
    bx1, by1, bx2, by2 = bbox
    w = bx2 - bx1
    h = by2 - by1
    padding_x = int(w * padding_factor)
    padding_y = int(h * padding_factor)
    x1, y1 = int(bx1 - padding_x), int(by1 - padding_y)
    x2, y2 = int(bx2 + padding_x), int(by2 + padding_y)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width, x2), min(frame_height, y2)
    return x1, y1, x2, y2


def save_vehicle_image(
    frame: np.ndarray,
    timestamp: float,
    bbox: Tuple[float, float, float, float],
    db: Optional[Database] = None,
) -> str:
    """Save vehicle image locally and queue upload to Bremen MinIO for archival.

    Returns the local:// path when the LOCAL save succeeds, regardless of whether the
    later async Bremen upload succeeds or fails. Returns "" only when the LOCAL save
    itself fails (the capture failed and must not masquerade as success).
    """
    # Generate a random UUID for the filename
    file_id = uuid.uuid4().hex
    filename = f"vehicle_{file_id}_{int(timestamp)}.jpg"

    # Crop the image - bbox is XYXY format (x1, y1, x2, y2)
    x1, y1, x2, y2 = compute_crop_rect(bbox, frame.shape[1], frame.shape[0])

    cropped_image = frame[y1:y2, x1:x2]

    # Ensure local image directory exists
    image_dir = Path(LOCAL_IMAGE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)

    local_path = image_dir / filename

    # Save to local filesystem. Closing the file makes the image visible to the
    # upload worker without blocking the analyzer on a disk fsync.
    try:
        _, img_encoded = cv2.imencode(".jpg", cropped_image)
        with open(local_path, "wb") as f:
            f.write(img_encoded.tobytes())

        logger.debug(f"Saved vehicle image locally: {filename}")
        _record_local_save_success()

        # Protect the file from pruning until it is archived and the DB path flips.
        _mark_upload_state(filename, "pending")

        # Retain the database handle before starting the worker. A restart can
        # have an idle outbox sweep before the next vehicle capture.
        global _flip_retry_db
        if db is not None:
            _flip_retry_db = db
        _start_upload_worker()

        # Queue upload to Bremen MinIO (non-blocking)
        try:
            _upload_queue.put_nowait((str(local_path), filename, db))
        except queue.Full:
            logger.warning(f"Upload queue full, skipping archive of {filename}")

        _maybe_prune_old_images()

        return f"local://{filename}"

    except Exception as e:
        logger.error(f"Failed to save vehicle image locally: {str(e)}", exc_info=True)
        _record_local_save_failure()
        return ""
