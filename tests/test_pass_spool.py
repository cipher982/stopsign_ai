"""Completed passes remain durable until remote delivery is acknowledged."""

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from stopsign import pass_spool
from stopsign.pass_spool import enqueue_pass
from stopsign.pass_spool import retry_pending_passes


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
def isolate_outbox(monkeypatch, tmp_path):
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
        "event_time": exit_time,
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


def _pending_count() -> int:
    with sqlite3.connect(Path(pass_spool.SPOOL_DIR) / "passes.sqlite3") as connection:
        return connection.execute("SELECT COUNT(*) FROM pending_passes").fetchone()[0]


def test_completed_pass_is_durable_in_sqlite_outbox():
    key = enqueue_pass(_pass_kwargs())

    assert len(key) == 64
    assert (Path(pass_spool.SPOOL_DIR) / "passes.sqlite3").exists()
    assert _pending_count() == 1
    assert pass_spool.pending_pass_image_paths() == {"vehicle_x.jpg"}


def test_pending_pass_is_replayed_when_database_returns():
    db = FakeDatabase(failing=True)
    enqueue_pass(_pass_kwargs())

    assert retry_pending_passes(db) == 0, "still down: durable row remains"
    assert _pending_count() == 1

    db.failing = False
    assert retry_pending_passes(db) == 1
    assert db.inserted[0]["vehicle_id"] == 7
    assert _pending_count() == 0


def test_replay_does_not_duplicate_a_pass_that_committed_before_crash():
    db = FakeDatabase(known=True)
    enqueue_pass(_pass_kwargs())

    assert retry_pending_passes(db) == 0
    assert db.inserted == []
    assert _pending_count() == 0


def test_outbox_never_discards_accepted_passes_when_many_are_pending():
    db = FakeDatabase(failing=True)
    for vehicle_id in range(5):
        enqueue_pass(_pass_kwargs(vehicle_id=vehicle_id))

    assert _pending_count() == 5
    db.failing = False
    assert retry_pending_passes(db) == 5
    assert [payload["vehicle_id"] for payload in db.inserted] == list(range(5))
    assert _pending_count() == 0


def test_unreadable_legacy_file_is_preserved_for_operator_recovery():
    db = FakeDatabase()
    junk = Path(pass_spool.SPOOL_DIR) / "pass_deadbeef.json"
    junk.write_text("{not json")

    assert retry_pending_passes(db) == 0
    assert junk.exists()


def test_pending_image_paths_are_unknown_when_legacy_evidence_is_malformed():
    junk = Path(pass_spool.SPOOL_DIR) / "pass_deadbeef.json"
    junk.write_text("{not json")

    assert pass_spool.pending_pass_image_paths() is None


def test_pending_image_paths_are_unknown_when_sqlite_is_missing():
    assert pass_spool.pending_pass_image_paths() is None


def test_pending_image_paths_are_unknown_when_sqlite_cannot_be_read(monkeypatch):
    def fail_open(*, read_only=False):
        raise sqlite3.DatabaseError("database is corrupt")

    monkeypatch.setattr(pass_spool, "_open_database", fail_open)

    assert pass_spool.pending_pass_image_paths() is None


def test_valid_legacy_file_is_migrated_before_replay():
    db = FakeDatabase()
    legacy = Path(pass_spool.SPOOL_DIR) / "pass_deadbeef.json"
    legacy.write_text(json.dumps(_pass_kwargs()))

    assert retry_pending_passes(db) == 1
    assert db.inserted[0]["vehicle_id"] == 7
    assert not legacy.exists()
    assert _pending_count() == 0


def test_older_pending_pass_is_replayed_before_newer_one():
    db = FakeDatabase(failing=True)
    enqueue_pass(_pass_kwargs(vehicle_id=1))
    time.sleep(0.01)
    enqueue_pass(_pass_kwargs(vehicle_id=2))

    db.failing = False
    retry_pending_passes(db)

    assert [payload["vehicle_id"] for payload in db.inserted] == [1, 2]


def test_one_rejected_pass_does_not_block_later_durable_passes():
    class SelectiveDatabase(FakeDatabase):
        def add_vehicle_pass(self, **kwargs):
            if kwargs["vehicle_id"] == 1:
                raise RuntimeError("one malformed row is rejected")
            return super().add_vehicle_pass(**kwargs)

    db = SelectiveDatabase()
    enqueue_pass(_pass_kwargs(vehicle_id=1))
    enqueue_pass(_pass_kwargs(vehicle_id=2))

    assert retry_pending_passes(db) == 1
    assert [payload["vehicle_id"] for payload in db.inserted] == [2]
    assert _pending_count() == 1


def test_legacy_pass_migration_preserves_exit_time_as_event_time():
    db = FakeDatabase()
    payload = _pass_kwargs()
    payload.pop("event_time")
    legacy = Path(pass_spool.SPOOL_DIR) / "pass_deadbeef.json"
    legacy.write_text(json.dumps(payload))

    assert retry_pending_passes(db) == 1
    assert db.inserted[0]["event_time"] == payload["exit_time"]
