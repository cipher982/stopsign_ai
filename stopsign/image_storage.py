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
#   "uploaded" -> object archived AND DB path flipped to bremen:// (local copy redundant)
_upload_state: dict[str, str] = {}
_UPLOAD_STATE_MAX = 20000
_upload_state_lock = threading.Lock()


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


def _health_snapshot() -> dict:
    with _health_lock:
        h = dict(_health)
    h["upload_healthy"] = h["upload_failures"] == 0 or (
        h["last_upload_success_ts"] is not None and h["last_upload_success_ts"] >= h["last_upload_failure_ts"]
    )
    h["local_save_healthy"] = h["local_save_failures"] == 0 or (
        h["last_local_save_ts"] is not None and h["last_local_save_ts"] >= h["last_local_save_failure_ts"]
    )
    with _upload_state_lock:
        h["pending_local_files"] = sum(1 for s in _upload_state.values() if s in ("pending", "failed"))
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
        return _upload_state.get(object_name)


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
    stuck at local:// forever. Retry briefly so we catch the pass insert.
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
            # archived AND the DB path points at the archive (bremen://).
            if _flip_db_path_with_retry(db, object_name):
                _mark_upload_state(object_name, "uploaded")
            else:
                _mark_upload_state(object_name, "failed")
            return
        except Exception as e:
            if attempt == 2:
                logger.error(f"Failed to archive {object_name} after 3 attempts: {e}")
                _mark_upload_state(object_name, "failed")
                _record_upload_failure()
            else:
                logger.warning(f"Bremen upload attempt {attempt + 1} failed for {object_name}: {e}")
                time.sleep(2**attempt)  # Exponential backoff: 1s, 2s


def _bremen_upload_worker():
    """Background worker that uploads images to Bremen MinIO with retry logic."""
    while True:
        try:
            local_path, object_name, db = _upload_queue.get()
            try:
                _process_upload_item(local_path, object_name, db)
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
    bx1, by1, bx2, by2 = bbox
    w = bx2 - bx1
    h = by2 - by1
    padding_factor = 0.1
    padding_x = int(w * padding_factor)
    padding_y = int(h * padding_factor)
    x1, y1 = int(bx1 - padding_x), int(by1 - padding_y)
    x2, y2 = int(bx2 + padding_x), int(by2 + padding_y)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

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

        # Start upload worker if not already running
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
