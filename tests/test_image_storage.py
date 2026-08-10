import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from stopsign import image_storage


@pytest.fixture(autouse=True)
def isolate_image_storage_state(monkeypatch):
    monkeypatch.setattr(image_storage, "_prune_lock", threading.Lock())
    monkeypatch.setattr(image_storage, "_last_prune_monotonic", 0.0)
    monkeypatch.setattr(image_storage, "_upload_state", {})
    monkeypatch.setattr(image_storage, "_health", dict(image_storage._health))
    monkeypatch.setattr(image_storage, "_redis_attempted", True)
    monkeypatch.setattr(image_storage, "_redis_client", None)


def test_save_vehicle_image_writes_local_file_without_inline_prune(monkeypatch, tmp_path):
    upload_queue = queue.Queue()
    prune_calls = []

    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    monkeypatch.setattr(image_storage, "_upload_queue", upload_queue)
    monkeypatch.setattr(image_storage, "_start_upload_worker", lambda: None)
    monkeypatch.setattr(image_storage, "_maybe_prune_old_images", lambda: prune_calls.append(True))

    frame = np.full((100, 120, 3), 127, dtype=np.uint8)
    image_path = image_storage.save_vehicle_image(
        frame=frame,
        timestamp=1234.5,
        bbox=(20.0, 20.0, 80.0, 80.0),
        db=None,
    )

    assert image_path.startswith("local://vehicle_")
    filename = image_path.removeprefix("local://")
    assert (tmp_path / filename).exists()
    assert upload_queue.qsize() == 1
    assert prune_calls == [True]

    # A successful local save must be recorded as successful and marked pending,
    # even though the async upload has not run yet.
    health = image_storage.get_archive_health()
    assert health["local_saves"] == 1
    assert health["local_save_failures"] == 0
    assert health["last_local_save_ts"] is not None
    assert image_storage._get_upload_state(filename) == "pending"


def test_save_vehicle_image_returns_empty_and_records_failure_when_local_save_fails(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    monkeypatch.setattr("cv2.imencode", boom)
    monkeypatch.setattr(image_storage, "_start_upload_worker", lambda: None)

    frame = np.full((100, 120, 3), 127, dtype=np.uint8)
    image_path = image_storage.save_vehicle_image(
        frame=frame,
        timestamp=1234.5,
        bbox=(20.0, 20.0, 80.0, 80.0),
        db=None,
    )

    assert image_path == ""
    health = image_storage.get_archive_health()
    assert health["local_save_failures"] == 1
    assert health["last_local_save_failure_ts"] is not None
    assert health["local_save_healthy"] is False


def test_maybe_prune_old_images_rate_limits_background_work(monkeypatch):
    starts = []

    def fake_start_prune_worker():
        starts.append(True)
        image_storage._prune_lock.release()

    monkeypatch.setattr(image_storage, "_last_prune_monotonic", 0.0)
    monkeypatch.setattr(image_storage, "_start_prune_worker", fake_start_prune_worker)

    image_storage._maybe_prune_old_images(now=100.0)
    image_storage._maybe_prune_old_images(now=110.0)
    image_storage._maybe_prune_old_images(now=161.0)

    assert starts == [True, True]


def test_prune_worker_releases_lock_on_error(monkeypatch):
    def raise_from_prune():
        raise RuntimeError("prune failed")

    monkeypatch.setattr(image_storage, "_prune_old_images", raise_from_prune)

    assert image_storage._prune_lock.acquire(blocking=False)
    with pytest.raises(RuntimeError):
        image_storage._run_prune_worker()

    assert not image_storage._prune_lock.locked()


def test_prune_old_images_only_removes_safely_archived_files(monkeypatch, tmp_path):
    """Only files that are uploaded AND DB-flipped (state 'uploaded') may be pruned.

    Pending/failed/unknown files still back a local:// DB path and must stay on disk.
    """
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_MAX_COUNT", 3)

    # 5 files: two uploaded (prunable), one pending, one failed, one unknown.
    for idx in range(5):
        path = tmp_path / f"vehicle_{idx}.jpg"
        path.write_bytes(b"jpg")
        os.utime(path, (idx, idx))

    image_storage._mark_upload_state("vehicle_0.jpg", "uploaded")
    image_storage._mark_upload_state("vehicle_1.jpg", "uploaded")
    image_storage._mark_upload_state("vehicle_2.jpg", "pending")
    image_storage._mark_upload_state("vehicle_3.jpg", "failed")
    # vehicle_4.jpg left unknown (e.g. from before this process started)

    image_storage._prune_old_images()

    remaining = sorted(path.name for path in Path(tmp_path).glob("*.jpg"))
    # Both uploaded files are pruned (5 - 3 = 2 pruned); pending/failed/unknown stay.
    assert remaining == ["vehicle_2.jpg", "vehicle_3.jpg", "vehicle_4.jpg"]


def test_prune_preserves_everything_when_nothing_is_uploaded(monkeypatch, tmp_path):
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_MAX_COUNT", 2)

    for idx in range(4):
        path = tmp_path / f"vehicle_{idx}.jpg"
        path.write_bytes(b"jpg")
        os.utime(path, (idx, idx))
    image_storage._mark_upload_state("vehicle_0.jpg", "pending")
    image_storage._mark_upload_state("vehicle_1.jpg", "failed")

    image_storage._prune_old_images()

    remaining = sorted(path.name for path in Path(tmp_path).glob("*.jpg"))
    assert remaining == ["vehicle_0.jpg", "vehicle_1.jpg", "vehicle_2.jpg", "vehicle_3.jpg"]


def test_upload_worker_flips_db_path_with_retry(monkeypatch):
    """The flip must wait for the pass row (inserted after the async upload) and retry."""
    from unittest.mock import MagicMock

    db = MagicMock()
    db.update_image_path.side_effect = [0, 0, 1]  # pass row appears on 3rd attempt

    monkeypatch.setattr(time, "sleep", lambda _: None)

    minio_client = MagicMock()
    minio_client.fput_object = MagicMock()
    minio_class = MagicMock(return_value=minio_client)
    monkeypatch.setattr(image_storage, "Minio", minio_class)

    monkeypatch.setattr(image_storage, "BREMEN_MINIO_SECRET_KEY", "secret")
    monkeypatch.setattr(image_storage, "BREMEN_MINIO_ENDPOINT", "100.98.103.56:9000")
    monkeypatch.setattr(image_storage, "BREMEN_MINIO_ACCESS_KEY", "root")
    monkeypatch.setattr(image_storage, "BREMEN_MINIO_BUCKET", "vehicle-images")

    q = queue.Queue()
    q.put(("/tmp/x_123.jpg", "x_123.jpg", db))
    monkeypatch.setattr(image_storage, "_upload_queue", q)

    image_storage._process_upload_item("/tmp/x_123.jpg", "x_123.jpg", db)

    assert image_storage._get_upload_state("x_123.jpg") == "uploaded"
    # Flip retried until rows>0
    assert db.update_image_path.call_count == 3
    health = image_storage.get_archive_health()
    assert health["upload_successes"] == 1
    assert health["upload_failures"] == 0
