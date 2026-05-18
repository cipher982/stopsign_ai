import os
import queue
import threading
from pathlib import Path

import numpy as np
import pytest

from stopsign import image_storage


@pytest.fixture(autouse=True)
def isolate_image_storage_state(monkeypatch):
    monkeypatch.setattr(image_storage, "_prune_lock", threading.Lock())
    monkeypatch.setattr(image_storage, "_last_prune_monotonic", 0.0)


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


def test_prune_old_images_removes_oldest_files(monkeypatch, tmp_path):
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_DIR", str(tmp_path))
    monkeypatch.setattr(image_storage, "LOCAL_IMAGE_MAX_COUNT", 3)

    for idx in range(5):
        path = tmp_path / f"vehicle_{idx}.jpg"
        path.write_bytes(b"jpg")
        os.utime(path, (idx, idx))

    image_storage._prune_old_images()

    remaining = sorted(path.name for path in Path(tmp_path).glob("*.jpg"))
    assert remaining == ["vehicle_2.jpg", "vehicle_3.jpg", "vehicle_4.jpg"]
