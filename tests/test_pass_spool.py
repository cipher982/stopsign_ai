"""A pass the database will not accept must survive the outage that refused it.

The insert retries in line for a few seconds; past that the vehicle used to be logged
and dropped. These cover the spool that replaces the drop: written once, replayed until
it lands, never duplicated, and bounded.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from stopsign import pass_spool
from stopsign.pass_spool import retry_spooled_passes
from stopsign.pass_spool import spool_failed_pass


class FakeDatabase:
    def __init__(self, *, failing: bool = False, known: bool = False):
        self.failing = failing
        self.known = known
        self.inserted: list[dict] = []

    def has_vehicle_pass(self, vehicle_id, exit_time):
        return self.known

    def add_vehicle_pass(self, **kwargs):
        if self.failing:
            raise RuntimeError("clifford is unreachable")
        self.inserted.append(kwargs)
        return len(self.inserted)


@pytest.fixture(autouse=True)
def isolate_spool(monkeypatch, tmp_path):
    monkeypatch.setattr(pass_spool, "SPOOL_DIR", str(tmp_path))
    monkeypatch.setattr(pass_spool, "_worker_started", True)  # no background thread in tests
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: _NoopThread())
    yield


class _NoopThread:
    def start(self):
        pass


def _pass_kwargs(vehicle_id=7, exit_time=1000.5):
    return {
        "vehicle_id": vehicle_id,
        "time_in_zone": 2.5,
        "stop_duration": 1.0,
        "min_speed": 0.0,
        "image_path": "local://vehicle_x.jpg",
        "entry_time": 998.0,
        "exit_time": exit_time,
        "entry_speed": 12.0,
        "decel_score": None,
        "track_quality": 0.9,
        "stop_pos_x": 800.0,
        "stop_pos_y": 700.0,
        "stream_queue_depth_exit": 3,
        "stream_lag_est_sec": 0.2,
        "raw_payload": {"samples": [[1.0, 2.0]], "raw_complete": True},
        "sample_count": 1,
        "raw_complete": True,
    }


def test_failed_pass_is_written_to_disk():
    db = FakeDatabase(failing=True)

    written = spool_failed_pass(db, _pass_kwargs())

    assert written is not None and written.exists()
    assert json.loads(written.read_text())["vehicle_id"] == 7
    assert not list(Path(pass_spool.SPOOL_DIR).glob("*.part")), "no half-written spool files"


def test_spooled_pass_is_replayed_when_the_database_returns():
    db = FakeDatabase(failing=True)
    spool_failed_pass(db, _pass_kwargs())

    assert retry_spooled_passes(db) == 0, "still down: nothing lands, nothing is lost"

    db.failing = False
    assert retry_spooled_passes(db) == 1
    assert db.inserted[0]["vehicle_id"] == 7
    assert not list(Path(pass_spool.SPOOL_DIR).glob("pass_*.json")), "the spool drains"


def test_replay_does_not_duplicate_a_pass_that_committed_but_never_returned():
    db = FakeDatabase(known=True)
    spool_failed_pass(db, _pass_kwargs())

    assert retry_spooled_passes(db) == 0
    assert db.inserted == []
    assert not list(Path(pass_spool.SPOOL_DIR).glob("pass_*.json"))


def test_the_spool_is_bounded(monkeypatch):
    monkeypatch.setattr(pass_spool, "SPOOL_MAX_FILES", 3)
    db = FakeDatabase(failing=True)

    written = [spool_failed_pass(db, _pass_kwargs(vehicle_id=i)) for i in range(5)]

    remaining = list(Path(pass_spool.SPOOL_DIR).glob("pass_*.json"))
    assert len(remaining) == 3
    assert written[-1] is not None and written[-1].exists(), "the newest pass is kept"
    assert written[0] is not None and not written[0].exists(), "the oldest is the one dropped"


def test_unreadable_spool_files_are_discarded():
    db = FakeDatabase()
    junk = Path(pass_spool.SPOOL_DIR) / "pass_deadbeef.json"
    junk.write_text("{not json")

    assert retry_spooled_passes(db) == 0
    assert not junk.exists()


def test_a_newer_pass_is_replayed_before_an_older_one():
    db = FakeDatabase(failing=True)
    spool_failed_pass(db, _pass_kwargs(vehicle_id=1))
    time.sleep(0.01)
    spool_failed_pass(db, _pass_kwargs(vehicle_id=2))

    db.failing = False
    retry_spooled_passes(db)

    assert [p["vehicle_id"] for p in db.inserted] == [1, 2]
