import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
from minio import Minio

from stopsign.database import Database
from stopsign.settings import BREMEN_MINIO_ACCESS_KEY
from stopsign.settings import BREMEN_MINIO_BUCKET
from stopsign.settings import BREMEN_MINIO_ENDPOINT
from stopsign.settings import BREMEN_MINIO_SECRET_KEY
from stopsign.settings import LOCAL_IMAGE_DIR
from stopsign.settings import LOCAL_IMAGE_MAX_COUNT

logger = logging.getLogger(__name__)

# Module-level upload queue and worker state
_upload_queue: queue.Queue = queue.Queue(maxsize=100)
_worker_started = False
_worker_lock = threading.Lock()


def _start_upload_worker():
    """Start the background upload worker thread if not already running."""
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            thread = threading.Thread(target=_bremen_upload_worker, daemon=True)
            thread.start()
            _worker_started = True
            logger.info("Bremen upload worker thread started")


def _bremen_upload_worker():
    """Background worker that uploads images to Bremen MinIO with retry logic."""
    while True:
        try:
            local_path, object_name, db = _upload_queue.get()

            # Skip if Bremen credentials not configured
            if not BREMEN_MINIO_SECRET_KEY:
                logger.debug(f"Bremen MinIO not configured, skipping archive of {object_name}")
                _upload_queue.task_done()
                continue

            # Retry up to 3 times with exponential backoff
            for attempt in range(3):
                try:
                    client = Minio(
                        BREMEN_MINIO_ENDPOINT,
                        access_key=BREMEN_MINIO_ACCESS_KEY,
                        secret_key=BREMEN_MINIO_SECRET_KEY,
                        secure=False,  # Bremen is on local network
                    )

                    # Upload the file
                    client.fput_object(
                        BREMEN_MINIO_BUCKET,
                        object_name,
                        local_path,
                        content_type="image/jpeg",
                    )
                    logger.debug(f"Archived {object_name} to Bremen MinIO")

                    # Flip DB path from local:// to bremen://
                    if db is not None:
                        try:
                            rows = db.update_image_path(
                                f"local://{object_name}",
                                f"bremen://{object_name}",
                            )
                            if rows:
                                logger.info(f"Updated DB path for {object_name}: local:// -> bremen://")
                        except Exception as db_err:
                            logger.warning(f"Failed to update DB path for {object_name}: {db_err}")

                    break
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Failed to archive {object_name} after 3 attempts: {e}")
                    else:
                        logger.warning(f"Bremen upload attempt {attempt + 1} failed for {object_name}: {e}")
                        time.sleep(2**attempt)  # Exponential backoff: 1s, 2s

            _upload_queue.task_done()
        except Exception as e:
            logger.error(f"Unexpected error in Bremen upload worker: {e}")


def _prune_old_images():
    """Remove oldest images when count exceeds LOCAL_IMAGE_MAX_COUNT."""
    try:
        image_dir = Path(LOCAL_IMAGE_DIR)
        if not image_dir.exists():
            return

        # Get all jpg files with their modification times
        images = list(image_dir.glob("*.jpg"))

        if len(images) <= LOCAL_IMAGE_MAX_COUNT:
            return

        # Sort by modification time (oldest first)
        images.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest images to get back under limit
        to_remove = len(images) - LOCAL_IMAGE_MAX_COUNT
        for img_path in images[:to_remove]:
            try:
                img_path.unlink()
                logger.debug(f"Pruned old image: {img_path.name}")
            except Exception as e:
                logger.warning(f"Failed to prune {img_path.name}: {e}")

        logger.info(f"Pruned {to_remove} old images to maintain {LOCAL_IMAGE_MAX_COUNT} limit")
    except Exception as e:
        logger.error(f"Error pruning old images: {e}")


def save_vehicle_image(
    frame: np.ndarray,
    timestamp: float,
    bbox: Tuple[float, float, float, float],
    db: Optional[Database] = None,
) -> str:
    """Save vehicle image locally and queue upload to Bremen MinIO for archival.

    Returns local:// path for fast serving from cube.
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

    # Save to local filesystem (synchronous, ensure file is fully written)
    try:
        _, img_encoded = cv2.imencode(".jpg", cropped_image)
        with open(local_path, "wb") as f:
            f.write(img_encoded.tobytes())
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        logger.debug(f"Saved vehicle image locally: {filename}")

        # Start upload worker if not already running
        _start_upload_worker()

        # Queue upload to Bremen MinIO (non-blocking)
        try:
            _upload_queue.put_nowait((str(local_path), filename, db))
        except queue.Full:
            logger.warning(f"Upload queue full, skipping archive of {filename}")

        # Prune old images to maintain disk space
        _prune_old_images()

        return f"local://{filename}"

    except Exception as e:
        logger.error(f"Failed to save vehicle image locally: {str(e)}", exc_info=True)
        return ""
