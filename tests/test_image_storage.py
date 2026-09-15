import json
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
    monkeypatch.setattr(image_storage, "_flip_pending", {})
    monkeypatch.setattr(image_storage, "_flip_retry_db", None)
    monkeypatch.setattr(image_storage, "_last_flip_sweep_monotonic", 0.0)
    monkeypatch.setattr(image_storage, "_last_requeue_sweep_monotonic", 0.0)
    monkeypatch.setattr(image_storage, "_health", dict(image_storage._health))
    monkeypatch.setattr(image_storage, "_health_seeded", True)
    monkeypatch.setattr(image_storage, "_redis_attempted", True)
    monkeypatch.setattr(image_storage, "_redis_client", None)
    monkeypatch.setattr(image_storage, "enqueue_archive_flip", lambda *_args: True)
    monkeypatch.setattr(image_storage, "forget_archive_flip", lambda *_args: True)
    monkeypatch.setattr(image_storage, "pending_archive_flips", lambda: {})


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


def test_archive_outage_captures_are_retried_from_disk(monkeypatch, tmp_path):
    """A capture the worker gave up on is re-queued, so an archive outage or a queue
    overflow cannot strand it on local:// forever."""
    upload_queue: queue.Queue = queue.Queue()
    monkeypatch.setattr(image_storage, "_upload_queue", upload_queue)
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    monkeypatch.setattr(image_storage, "BREMEN_MINIO_SECRET_KEY", "secret")

    old = time.time() - 600
    for name, state in (("vehicle_failed.jpg", "failed"), ("vehicle_never_queued.jpg", None)):
        path = tmp_path / name
        path.write_bytes(b"jpg")
        os.utime(path, (old, old))
        if state:
            image_storage._mark_upload_state(name, state)
    fresh_path = tmp_path / "vehicle_fresh.jpg"
    fresh_path.write_bytes(b"jpg")
    image_storage._mark_upload_state("vehicle_fresh.jpg", "pending")
    done_path = tmp_path / "vehicle_uploaded.jpg"
    done_path.write_bytes(b"jpg")
    os.utime(done_path, (old, old))
    image_storage._mark_upload_state("vehicle_uploaded.jpg", "uploaded")

    requeued = image_storage._maybe_requeue_unarchived_uploads(now=1000.0)

    queued = {upload_queue.get_nowait()[1] for _ in range(upload_queue.qsize())}
    assert requeued == 2
    assert queued == {"vehicle_failed.jpg", "vehicle_never_queued.jpg"}


def test_a_capture_waiting_on_its_row_is_not_uploaded_twice(monkeypatch, tmp_path):
    """The object is already in the archive; only the database path flip is outstanding."""
    upload_queue: queue.Queue = queue.Queue()
    monkeypatch.setattr(image_storage, "_upload_queue", upload_queue)
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    monkeypatch.setattr(image_storage, "BREMEN_MINIO_SECRET_KEY", "secret")

    old = time.time() - 600
    path = tmp_path / "vehicle_awaiting_flip.jpg"
    path.write_bytes(b"jpg")
    os.utime(path, (old, old))
    image_storage._mark_upload_state("vehicle_awaiting_flip.jpg", "failed")
    with image_storage._upload_state_lock:
        image_storage._flip_pending["vehicle_awaiting_flip.jpg"] = old

    assert image_storage._maybe_requeue_unarchived_uploads(now=1000.0) == 0
    assert upload_queue.empty()


def test_archive_health_survives_a_restart(monkeypatch):
    """A restart must not announce a healthy archive that was failing a second ago."""
    monkeypatch.setattr(image_storage, "_health_seeded", False)
    monkeypatch.setattr(image_storage, "_health", dict(image_storage._health))

    stored = {
        "upload_failures": 3,
        "upload_successes": 0,
        "last_upload_failure_ts": 2000.0,
        "last_upload_success_ts": 1000.0,
    }

    class _Client:
        def get(self, _key):
            return json.dumps(stored)

    monkeypatch.setattr(image_storage, "ARCHIVE_HEALTH_REDIS_KEY", "k")
    monkeypatch.setattr(image_storage, "_get_redis_client", lambda: _Client())

    health = image_storage._health_snapshot()

    assert health["upload_failures"] == 3
    assert health["upload_healthy"] is False


def test_archive_health_counts_disk_outbox_after_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    pending = tmp_path / "vehicle_pending.jpg"
    pending.write_bytes(b"jpg")
    os.utime(pending, (100.0, 100.0))
    monkeypatch.setattr(image_storage.time, "time", lambda: 200.0)

    health = image_storage._health_snapshot()

    assert health["pending_local_files"] == 1
    assert health["oldest_pending_local_age_seconds"] == 100.0
    assert health["oldest_pending_local_ts"] == 100.0
    assert health["archive_health_observed_at"] == 200.0
    assert health["archive_outbox_observed"] is True
    assert health["upload_transport_healthy"] is True
    assert health["archive_reconciliation_healthy"] is False
    assert health["upload_healthy"] is True


def test_archive_health_reports_unknown_when_local_outbox_is_missing(monkeypatch, tmp_path):
    missing = tmp_path / "not-created"
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(missing))

    health = image_storage._health_snapshot()

    assert health["pending_local_files"] is None
    assert health["archive_outbox_observed"] is False
    assert health["upload_transport_healthy"] is False
    assert health["archive_reconciliation_healthy"] is False


def test_archive_health_does_not_inflate_disk_backlog_with_stale_worker_state(monkeypatch, tmp_path):
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    (tmp_path / "vehicle_on_disk.jpg").write_bytes(b"jpg")
    image_storage._mark_upload_state("vehicle_on_disk.jpg", "failed")
    image_storage._mark_upload_state("vehicle_stale_memory.jpg", "pending")

    health = image_storage._health_snapshot()

    assert health["pending_local_files"] == 1
    assert health["worker_pending_files"] == 2


def test_archive_health_ignores_failed_memory_state_after_durable_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    image = tmp_path / "vehicle_archived.jpg"
    image.write_bytes(b"jpg")
    image_storage._mark_durably_archived(image.name)
    image_storage._mark_upload_state(image.name, "failed")

    health = image_storage._health_snapshot()

    assert health["pending_local_files"] == 0
    assert health["worker_pending_files"] == 0
    assert image_storage._get_upload_state(image.name) == "uploaded"


def test_upload_worker_flips_db_path_with_retry(monkeypatch):
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


def _archive_once(monkeypatch, db, object_name="x_123.jpg"):
    """Run one upload-queue item against a mocked MinIO, with `db` supplying the flip."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(time, "sleep", lambda _: None)
    minio_client = MagicMock()
    minio_client.fput_object = MagicMock()
    monkeypatch.setattr(image_storage, "Minio", MagicMock(return_value=minio_client))
    monkeypatch.setattr(image_storage, "BREMEN_MINIO_SECRET_KEY", "secret")
    monkeypatch.setattr(image_storage, "BREMEN_MINIO_ENDPOINT", "100.98.103.56:9000")
    monkeypatch.setattr(image_storage, "BREMEN_MINIO_ACCESS_KEY", "root")
    monkeypatch.setattr(image_storage, "BREMEN_MINIO_BUCKET", "vehicle-images")

    image_storage._process_upload_item(f"/tmp/{object_name}", object_name, db)


def test_archive_copy_verification_checks_content_not_only_size(tmp_path):
    from types import SimpleNamespace

    matching = tmp_path / "matching.jpg"
    matching.write_bytes(b"same bytes")
    different = tmp_path / "different.jpg"
    different.write_bytes(b"other data")

    class _Response:
        def __init__(self, payload):
            self.payload = payload
            self.offset = 0

        def read(self, size):
            chunk = self.payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

        def close(self):
            pass

        def release_conn(self):
            pass

    class _Client:
        def stat_object(self, _bucket, name):
            return SimpleNamespace(size=len(b"same bytes"), etag=name)

        def get_object(self, _bucket, _name):
            return _Response(b"same bytes")

    client = _Client()
    assert image_storage._archive_copy_matches_local(client, matching) is True
    assert image_storage._archive_copy_matches_local(client, different) is False


def test_startup_reconciliation_flips_archived_files_and_requeues_missing(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    archived = tmp_path / "archived.jpg"
    archived.write_bytes(b"archived")
    missing = tmp_path / "missing.jpg"
    missing.write_bytes(b"missing")
    upload_queue = queue.Queue()
    db = MagicMock()
    db.update_image_path.return_value = 1

    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    monkeypatch.setattr(image_storage, "BREMEN_MINIO_SECRET_KEY", "secret")
    monkeypatch.setattr(image_storage, "_archive_client", lambda: object())
    monkeypatch.setattr(
        image_storage,
        "_archive_copy_matches_local",
        lambda _client, path: path.name == archived.name,
    )
    monkeypatch.setattr(image_storage, "_upload_queue", upload_queue)

    image_storage._reconcile_local_archive_on_startup(db)

    queued = upload_queue.get_nowait()
    assert queued[:2] == (str(missing), missing.name)
    assert db.update_image_path.call_args.args == (
        "local://archived.jpg",
        "bremen://archived.jpg",
    )
    assert image_storage._get_upload_state(archived.name) == "uploaded"
    assert (tmp_path / "archived.jpg.uploaded").exists()


def test_upload_worker_queues_a_late_flip_instead_of_leaking_the_file(monkeypatch):
    """A pass row that lands after the inline window must not strand the local file.

    Zone exit can persist the pass a minute after the capture; the inline window is
    seconds. An abandoned flip means a local:// path the pruner can never remove.
    """
    from unittest.mock import MagicMock

    db = MagicMock()
    db.update_image_path.return_value = 0  # row never appears inside the window

    _archive_once(monkeypatch, db)

    assert image_storage._get_upload_state("x_123.jpg") == "failed"
    assert "x_123.jpg" in image_storage._flip_pending


def test_delayed_flip_retry_marks_uploaded_once_the_pass_row_lands():
    from unittest.mock import MagicMock

    db = MagicMock()
    db.update_image_path.side_effect = [0, 1]
    image_storage._mark_upload_state("x_123.jpg", "failed")
    image_storage._enqueue_flip_retry("x_123.jpg")

    image_storage._retry_pending_flips(db)  # pass row not written yet
    assert image_storage._get_upload_state("x_123.jpg") == "failed"

    image_storage._retry_pending_flips(db)  # row landed -> file becomes prunable
    assert image_storage._get_upload_state("x_123.jpg") == "uploaded"
    assert "x_123.jpg" not in image_storage._flip_pending


def test_delayed_flip_retry_restores_persisted_queue_after_restart(monkeypatch):
    from unittest.mock import MagicMock

    db = MagicMock()
    db.update_image_path.return_value = 1
    monkeypatch.setattr(
        image_storage,
        "pending_archive_flips",
        lambda: {"x_after_restart.jpg": time.time()},
    )

    image_storage._retry_pending_flips(db)

    assert image_storage._get_upload_state("x_after_restart.jpg") == "uploaded"
    assert "x_after_restart.jpg" not in image_storage._flip_pending


def test_expired_flip_with_no_referencing_pass_releases_the_local_copy(monkeypatch):
    """An archived capture with no pass reference eventually becomes prune-eligible."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(image_storage, "pending_pass_image_paths", lambda: set())
    db = MagicMock()
    db.update_image_path.return_value = 0
    image_storage._mark_upload_state("x_orphan.jpg", "failed")
    image_storage._flip_pending["x_orphan.jpg"] = time.time() - image_storage._FLIP_RETRY_MAX_AGE_SEC - 1

    image_storage._retry_pending_flips(db)

    assert "x_orphan.jpg" not in image_storage._flip_pending
    assert image_storage._get_upload_state("x_orphan.jpg") == "uploaded"


def test_expired_flip_waits_for_durable_failed_pass(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from stopsign import pass_spool

    spool_dir = tmp_path / "pending-passes"
    spool_dir.mkdir()
    (spool_dir / "pass_1.json").write_text(json.dumps({"image_path": "local://x_pending.jpg", "vehicle_id": 1}))
    monkeypatch.setattr(pass_spool, "SPOOL_DIR", str(spool_dir))

    db = MagicMock()
    db.update_image_path.return_value = 0
    image_storage._mark_upload_state("x_pending.jpg", "failed")
    image_storage._flip_pending["x_pending.jpg"] = time.time() - image_storage._FLIP_RETRY_MAX_AGE_SEC - 1

    image_storage._retry_pending_flips(db)

    assert "x_pending.jpg" in image_storage._flip_pending
    assert image_storage._get_upload_state("x_pending.jpg") == "failed"


def test_expired_flip_keeps_local_copy_when_pass_spool_is_unreadable(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr(image_storage, "pending_pass_image_paths", lambda: None)
    db = MagicMock()
    db.update_image_path.return_value = 0
    image_storage._mark_upload_state("x_unknown.jpg", "failed")
    image_storage._flip_pending["x_unknown.jpg"] = time.time() - image_storage._FLIP_RETRY_MAX_AGE_SEC - 1

    image_storage._retry_pending_flips(db)

    assert "x_unknown.jpg" in image_storage._flip_pending
    assert image_storage._get_upload_state("x_unknown.jpg") == "failed"


def test_start_upload_worker_retains_database_for_idle_reconciliation(monkeypatch):
    calls = []
    db = object()
    monkeypatch.setattr(image_storage, "_start_upload_worker", lambda: calls.append(True))

    image_storage.start_upload_worker(db)

    assert image_storage._flip_retry_db is db
    assert calls == [True]


def test_save_records_db_for_the_background_flip_sweep(monkeypatch, tmp_path):
    """Wiring: a save must leave the sweep able to reach the database while idle."""
    calls = []

    class _DB:
        def update_image_path(self, old, new):
            calls.append((old, new))
            return 1

    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    monkeypatch.setattr(image_storage, "_upload_queue", queue.Queue())
    monkeypatch.setattr(image_storage, "_start_upload_worker", lambda: None)
    monkeypatch.setattr(image_storage, "_maybe_prune_old_images", lambda: None)
    monkeypatch.setattr(image_storage, "_flip_pending", {"vehicle_orphan.jpg": time.time()})

    image_storage.save_vehicle_image(
        frame=np.full((100, 120, 3), 127, dtype=np.uint8),
        timestamp=1234.5,
        bbox=(20.0, 20.0, 80.0, 80.0),
        db=_DB(),
    )
    image_storage._maybe_retry_pending_flips(now=10_000.0)

    assert calls == [("local://vehicle_orphan.jpg", "bremen://vehicle_orphan.jpg")]


def test_flip_sweep_is_rate_limited(monkeypatch):
    calls = []
    db = object()
    monkeypatch.setattr(image_storage, "_flip_retry_db", db)
    monkeypatch.setattr(image_storage, "_retry_pending_flips", lambda passed: calls.append(passed))

    image_storage._maybe_retry_pending_flips(now=100.0)
    image_storage._maybe_retry_pending_flips(now=110.0)
    image_storage._maybe_retry_pending_flips(now=161.0)

    assert calls == [db, db]


def test_upload_worker_sweeps_pending_flips_on_its_idle_path(monkeypatch):
    """The sweep has to run with an empty queue: a quiet evening adds no captures."""
    sweeps = []

    class _Stop(BaseException):
        pass

    def _fake_sweep(now=None):
        sweeps.append(now)
        if len(sweeps) >= 2:
            raise _Stop

    monkeypatch.setattr(image_storage, "_maybe_retry_pending_flips", _fake_sweep)
    monkeypatch.setattr(threading, "excepthook", lambda _args: None)
    monkeypatch.setattr(image_storage, "_upload_queue", queue.Queue())
    monkeypatch.setattr(image_storage, "FLIP_SWEEP_INTERVAL_SECONDS", 0.05)

    worker = threading.Thread(target=image_storage._bremen_upload_worker, daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert len(sweeps) >= 2


def test_upload_worker_sweeps_outbox_while_queue_has_work(monkeypatch):
    monkeypatch.setattr(threading, "excepthook", lambda _args: None)
    requeues = []

    class _Stop(BaseException):
        pass

    monkeypatch.setattr(image_storage, "_maybe_retry_pending_flips", lambda: None)
    monkeypatch.setattr(image_storage, "_process_upload_item", lambda *_args: None)

    def _fake_requeue(now=None):
        requeues.append(now)
        raise _Stop

    monkeypatch.setattr(image_storage, "_maybe_requeue_unarchived_uploads", _fake_requeue)
    work = queue.Queue()
    work.put(("/tmp/capture.jpg", "capture.jpg", None))
    monkeypatch.setattr(image_storage, "_upload_queue", work)

    worker = threading.Thread(target=image_storage._bremen_upload_worker, daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert requeues
