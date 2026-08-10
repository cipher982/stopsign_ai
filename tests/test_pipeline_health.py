"""Tests for the pipeline-health surface and captured-pass stats semantics.

Covers the deliberately-fuzzy boundary that made the archive stall invisible:
what counts as a "captured" pass (local:// counts), how live-stats aggregates a
mix of path states, and the /api/archive-health + /api/pipeline-health shapes
the Sauron alerting jobs depend on.
"""

import asyncio
import json
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import app first so create_app() fully initializes before the route modules load;
# routes.api imports stopsign.web.app (for templates), which re-imports routes.api.
import stopsign.web.app  # noqa: E402,F401
from stopsign.database import Database
from stopsign.database import VehiclePass
from stopsign.web.routes.api import get_live_stats  # noqa: E402
from stopsign.web.routes.health import archive_health  # noqa: E402
from stopsign.web.routes.health import pipeline_health  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sqlite_database() -> Database:
    """Build a Database instance over an in-memory SQLite vehicle_passes table.

    Skipping __init__ avoids the production Postgres connection; only the
    vehicle_passes table is created so the JSONB raw table never enters the mix.
    """
    engine = create_engine("sqlite:///:memory:")
    VehiclePass.__table__.create(engine)
    db = Database.__new__(Database)
    db.Session = sessionmaker(bind=engine)
    return db


_pass_id_counter = 0


def _insert_pass(db: Database, ts: datetime, image_path: str, time_in_zone: float = 3.0) -> None:
    global _pass_id_counter
    _pass_id_counter += 1
    with db.Session() as session:
        session.add(
            VehiclePass(
                id=_pass_id_counter,
                timestamp=ts,
                vehicle_id=1,
                time_in_zone=time_in_zone,
                min_speed=5.0,
                image_path=image_path,
            )
        )
        session.commit()


class _FakeRedis:
    def __init__(self, data: dict):
        self._data = data

    def get(self, key):
        return self._data.get(key)


# ---------------------------------------------------------------------------
# C1: get_recent_vehicle_passes — empty no-capture vs captured (local://, bremen://)
# ---------------------------------------------------------------------------


def test_recent_passes_excludes_no_capture_and_keeps_local_and_bremen():
    db = _sqlite_database()
    now = datetime.now()
    _insert_pass(db, now, "")  # no image: pass recorded without a capture
    _insert_pass(db, now, "local://vehicle_aaa.jpg")  # captured, awaiting async flip
    _insert_pass(db, now, "bremen://vehicle_bbb.jpg")  # captured + archived
    _insert_pass(db, now, None)  # legacy no-capture record

    passes = db.get_recent_vehicle_passes(limit=10)

    assert len(passes) == 2
    assert sorted(p.image_path for p in passes) == [
        "bremen://vehicle_bbb.jpg",
        "local://vehicle_aaa.jpg",
    ]


def test_recent_passes_orders_newest_first_and_respects_limit():
    db = _sqlite_database()
    base = datetime.now()
    _insert_pass(db, base - timedelta(minutes=5), "local://vehicle_old.jpg")
    _insert_pass(db, base, "bremen://vehicle_new.jpg")

    passes = db.get_recent_vehicle_passes(limit=1)

    assert len(passes) == 1
    assert passes[0].image_path == "bremen://vehicle_new.jpg"


# ---------------------------------------------------------------------------
# C2: live-stats aggregation across mixed image_path states
# ---------------------------------------------------------------------------


def test_live_stats_aggregates_mixed_image_path_states(monkeypatch):
    """local:// and bremen:// passes both count as captured; compliance counts
    compliant passes (time_in_zone >= 2s) regardless of which storage prefix the
    image currently lives under."""
    now = datetime.now()
    passes = [
        SimpleNamespace(timestamp=now, time_in_zone=3.2, min_speed=4.0),  # bremen (flipped)
        SimpleNamespace(timestamp=now, time_in_zone=2.1, min_speed=6.0),  # local:// (pending flip)
        SimpleNamespace(timestamp=now, time_in_zone=0.8, min_speed=12.0),  # local:// non-compliant
        SimpleNamespace(timestamp=now, time_in_zone=4.0, min_speed=3.0),  # bremen compliant
    ]
    fake_db = MagicMock()
    fake_db.get_recent_vehicle_passes.return_value = passes
    fake_db.get_total_passes_last_24h.return_value = 4

    monkeypatch.setattr("stopsign.web.routes.api._ensure_db", lambda request: fake_db)
    monkeypatch.setattr("stopsign.web.routes.api.get_real_insights", lambda db, recent: "Monitoring active")

    request = SimpleNamespace()
    response = asyncio.run(get_live_stats(request))

    body = response.body.decode()
    assert 'data-stat="compliance">75%</span>' in body  # 3 of 4 compliant
    assert 'data-stat="vehicles">4</span>' in body
    assert 'data-stat="lastDetection">0m ago</span>' in body
    assert 'data-stat="violations">1</span>' in body


# ---------------------------------------------------------------------------
# B1/C2: /api/archive-health response shape
# ---------------------------------------------------------------------------


def test_archive_health_response_shape(monkeypatch):
    signature = {
        "local_saves": 12,
        "local_save_failures": 0,
        "upload_attempts": 12,
        "upload_successes": 11,
        "upload_failures": 1,
        "last_local_save_ts": 1700000000.0,
        "last_local_save_failure_ts": None,
        "last_upload_attempt_ts": 1700000001.0,
        "last_upload_success_ts": 1700000002.0,
        "last_upload_failure_ts": 1700000003.0,
        "upload_healthy": True,
        "local_save_healthy": True,
        "pending_local_files": 3,
    }
    fake = _FakeRedis({"stopsign.archive.health": json.dumps(signature)})
    monkeypatch.setattr("stopsign.web.routes.health.redis_lib.from_url", lambda url, **kw: fake)

    response = asyncio.run(archive_health())
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["available"] is True
    for key in (
        "local_saves",
        "local_save_failures",
        "upload_attempts",
        "upload_successes",
        "upload_failures",
        "last_local_save_ts",
        "last_upload_attempt_ts",
        "last_upload_success_ts",
        "last_upload_failure_ts",
        "upload_healthy",
        "local_save_healthy",
        "pending_local_files",
    ):
        assert key in payload, f"missing key {key}"
    assert payload["upload_successes"] == 11


def test_archive_health_shape_when_no_signal_yet(monkeypatch):
    fake = _FakeRedis({})
    monkeypatch.setattr("stopsign.web.routes.health.redis_lib.from_url", lambda url, **kw: fake)

    response = asyncio.run(archive_health())
    payload = json.loads(response.body)

    assert payload["available"] is False
    assert "detail" in payload


# ---------------------------------------------------------------------------
# B1: /api/pipeline-health response shape (the Sauron pipeline-health job's contract)
# ---------------------------------------------------------------------------


def test_pipeline_health_response_shape(monkeypatch):
    now = __import__("time").time()
    fake = _FakeRedis(
        {
            "stopsign.archive.health": json.dumps({"upload_healthy": True, "local_save_healthy": True}),
            "stopsign.analyzer.last_frame_at": str(now - 4),
            "stopsign.analyzer.boot_ts": str(now - 3600),
            "stopsign.ffmpeg.health": json.dumps({"fps": 15.0, "new_fps": 14.8, "dup_pct": 1.3, "ts": now}),
        }
    )
    monkeypatch.setattr("stopsign.web.routes.health.redis_lib.from_url", lambda url, **kw: fake)

    response = asyncio.run(pipeline_health())
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["available"] is True
    assert payload["archive"]["upload_healthy"] is True
    assert payload["analyzer"]["available"] is True
    assert payload["analyzer"]["frame_age_seconds"] == 4.0
    assert isinstance(payload["analyzer"]["uptime_seconds"], float)
    assert payload["ffmpeg"]["dup_pct"] == 1.3
    assert "hls" in payload


def test_pipeline_health_reports_stall_reason(monkeypatch):
    now = __import__("time").time()
    fake = _FakeRedis(
        {
            "stopsign.analyzer.last_frame_at": str(now - 600),
            "stopsign.analyzer.stall": json.dumps(
                {
                    "triggered_at": now - 120,
                    "lag_seconds": 600.0,
                    "threshold_seconds": 120.0,
                    "reason": "no frames processed for 600.0s (threshold 120.0s)",
                }
            ),
        }
    )
    monkeypatch.setattr("stopsign.web.routes.health.redis_lib.from_url", lambda url, **kw: fake)

    response = asyncio.run(pipeline_health())
    payload = json.loads(response.body)

    assert payload["analyzer"]["frame_age_seconds"] == 600.0
    assert payload["analyzer"]["last_stall"]["lag_seconds"] == 600.0
