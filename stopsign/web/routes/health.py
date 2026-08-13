"""Health check routes."""

import json
import logging
import os
import time

import redis as redis_lib
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from sqlalchemy import text

from stopsign.database import Database
from stopsign.hls_health import parse_hls_playlist
from stopsign.settings import ANALYZER_BOOT_TS_KEY
from stopsign.settings import ANALYZER_LAST_FRAME_AT_KEY
from stopsign.settings import ANALYZER_STALL_KEY
from stopsign.settings import ARCHIVE_HEALTH_REDIS_KEY
from stopsign.settings import DB_URL
from stopsign.settings import FFMPEG_HEALTH_KEY
from stopsign.settings import GRACE_STARTUP_SEC
from stopsign.settings import REDIS_URL
from stopsign.web.app import STREAM_FS_PATH
from stopsign.web.app import WEB_START_TIME

logger = logging.getLogger(__name__)

router = APIRouter()

_HLS_PARSE_WARN_LAST_TS = 0.0


def _parse_hls_playlist(path: str) -> dict:
    global _HLS_PARSE_WARN_LAST_TS
    try:
        info = parse_hls_playlist(path)
    except Exception as e:
        now = time.time()
        if now - _HLS_PARSE_WARN_LAST_TS > 60:
            logger.warning(f"HLS playlist parse failed: {e}")
            _HLS_PARSE_WARN_LAST_TS = now
        else:
            logger.debug(f"HLS playlist parse failed: {e}")
        info = {
            "exists": os.path.exists(path),
            "playlist_mtime": os.path.getmtime(path) if os.path.exists(path) else None,
            "age_seconds": None,
            "segments_count": 0,
            "threshold_sec": 60.0,
        }
        try:
            stream_dir = os.path.dirname(path)
            if os.path.isdir(stream_dir):
                ts_count = len([f for f in os.listdir(stream_dir) if f.endswith(".ts")])
                info["segments_count"] = ts_count
        except Exception:
            pass
    return info


class DBHealthTracker:
    def __init__(self):
        self.last_failure_time = None
        self.failure_count = 0
        self.max_failure_duration = 300

    def record_failure(self):
        current_time = time.time()
        if self.last_failure_time is None:
            self.last_failure_time = current_time
        self.failure_count += 1

    def record_success(self):
        self.last_failure_time = None
        self.failure_count = 0

    def is_failure_persistent(self) -> bool:
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) > self.max_failure_duration


db_health_tracker = DBHealthTracker()


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.api_route("/readyz", methods=["GET", "HEAD"])
async def readyz():
    resp = JSONResponse({"status": "ready"})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@router.get("/api/archive-health")
async def archive_health():
    """Surface the analyzer's archive health signal (written to Redis on each event).

    The video_analyzer writes counters/timestamps for local saves and Bremen uploads;
    this lets the next hardening pass alert when archival is silently degrading.
    """
    try:
        r = redis_lib.from_url(REDIS_URL, socket_connect_timeout=0.3, socket_timeout=0.3)
        raw = r.get(ARCHIVE_HEALTH_REDIS_KEY)
        if not raw:
            return JSONResponse(
                {"available": False, "detail": "No archive health signal yet (analyzer has not recorded one)"}
            )
        payload = json.loads(raw)
        payload["available"] = True
        return JSONResponse(payload)
    except Exception as e:
        logger.warning(f"archive_health read failed: {e}")
        return JSONResponse({"available": False, "error": str(e)})


@router.get("/api/label-health")
async def label_health(request: Request):
    """Surface vehicle-label freshness (COUNT + MAX(updated_at) in vehicle_labels).

    The daily cube cron (scripts/label_increment.sh) labels newly captured passes.
    A silent failure there (rotated key, dead cron, broken uv env) freezes labels
    exactly like the Feb 2026 cluster freeze, while the detection pipeline keeps
    looking healthy. Sauron's stopsign-label-freshness job polls this endpoint.
    """
    try:
        db = getattr(request.app.state, "db", None)
        if db is None:
            db = Database(db_url=DB_URL)
            request.app.state.db = db
        with db.Session() as session:
            row = session.execute(text("SELECT COUNT(*) AS total, MAX(updated_at) AS last FROM vehicle_labels")).first()
        if row is None or row.last is None:
            return JSONResponse({"available": False, "detail": "No labels recorded yet"})
        return JSONResponse(
            {
                "available": True,
                "total_labeled": row.total,
                "last_labeled_at": row.last.isoformat(),
                "last_labeled_epoch": row.last.timestamp(),
            }
        )
    except Exception as e:
        logger.warning(f"label_health read failed: {e}")
        return JSONResponse({"available": False, "error": str(e)})


@router.get("/api/pipeline-health")
async def pipeline_health():
    """Aggregated capture->analyzer->archive chain health for alerting.

    Exposes, in one read-only endpoint: archive upload health (same signal as
    /api/archive-health), analyzer frame age / uptime / last stall reason, the
    ffmpeg new-vs-dup ratio, and HLS playlist freshness. The Sauron
    stopsign-pipeline-health job polls this so a silent freeze anywhere in the
    chain pages instead of going unnoticed for days.
    """
    try:
        r = redis_lib.from_url(REDIS_URL, socket_connect_timeout=0.3, socket_timeout=0.3)
    except Exception as e:
        return JSONResponse({"available": False, "error": f"redis client error: {e}"})

    now = time.time()
    payload: dict = {"generated_at": now, "available": True}

    # Archive upload health (mirrors /api/archive-health).
    try:
        raw = r.get(ARCHIVE_HEALTH_REDIS_KEY)
        if raw:
            archive = json.loads(raw)
            archive["available"] = True
        else:
            archive = {"available": False, "detail": "No archive health signal yet (analyzer has not recorded one)"}
    except Exception as e:
        archive = {"available": False, "error": str(e)}
    payload["archive"] = archive

    # Analyzer: last-frame time (age grows after death by design), boot time, last stall.
    analyzer: dict = {}
    try:
        last_frame_raw = r.get(ANALYZER_LAST_FRAME_AT_KEY)
        boot_raw = r.get(ANALYZER_BOOT_TS_KEY)
        stall_raw = r.get(ANALYZER_STALL_KEY)
        analyzer["available"] = bool(last_frame_raw or boot_raw)
        if last_frame_raw:
            last_frame_at = float(last_frame_raw)
            analyzer["last_frame_at"] = last_frame_at
            analyzer["frame_age_seconds"] = round(now - last_frame_at, 1)
        if boot_raw:
            boot_ts = float(boot_raw)
            analyzer["started_at"] = boot_ts
            analyzer["uptime_seconds"] = round(now - boot_ts, 1)
        if stall_raw:
            try:
                analyzer["last_stall"] = json.loads(stall_raw)
            except Exception:
                analyzer["last_stall"] = None
    except Exception as e:
        analyzer = {"available": False, "error": str(e)}
    payload["analyzer"] = analyzer

    # FFmpeg: dup-ratio snapshot written every 5s. dup_pct -> ~100 means ffmpeg is
    # repeating the last frame because the analyzer stopped producing new ones.
    try:
        ff_raw = r.get(FFMPEG_HEALTH_KEY)
        if ff_raw:
            ff = json.loads(ff_raw)
            ff["available"] = True
            ff_ts = ff.get("ts")
            ff["snapshot_age_seconds"] = round(now - float(ff_ts), 1) if ff_ts else None
        else:
            ff = {"available": False, "detail": "No ffmpeg health snapshot yet (service starting?)"}
    except Exception as e:
        ff = {"available": False, "error": str(e)}
    payload["ffmpeg"] = ff

    # HLS playlist freshness (same source as /health/stream; one-stop for alerting).
    try:
        hls = _parse_hls_playlist(STREAM_FS_PATH)
        age = hls.get("age_seconds")
        payload["hls"] = {
            "fresh": bool(hls.get("exists")) and age is not None and age <= hls.get("threshold_sec", 60.0),
            "age_seconds": age,
            "segments_count": hls.get("segments_count", 0),
        }
    except Exception as e:
        payload["hls"] = {"available": False, "error": str(e)}

    return JSONResponse(payload)


@router.get("/health/stream")
async def health_stream(request: Request):
    tracer = request.app.state.tracer
    with tracer.start_as_current_span("health_stream") as span:
        info = _parse_hls_playlist(STREAM_FS_PATH)
        age = info.get("age_seconds")
        exists = bool(info.get("exists"))
        threshold = info.get("threshold_sec", 60.0)
        warming_up = (time.time() - WEB_START_TIME) <= GRACE_STARTUP_SEC
        fresh = (exists and age is not None and age <= threshold) or warming_up

        span.set_attribute("hls.exists", exists)
        if age is not None:
            span.set_attribute("hls.age_seconds", float(age))
        span.set_attribute("hls.segments_count", info.get("segments_count", 0))
        span.set_attribute("hls.threshold_sec", threshold)
        span.set_attribute("hls.fresh", fresh)

        status = 200 if fresh else 503
        payload = {
            "fresh": bool(fresh),
            "exists": bool(exists),
            "age_seconds": age,
            "threshold_sec": threshold,
            "segments_count": info.get("segments_count", 0),
        }
        resp = HTMLResponse(status_code=status, content=json.dumps(payload))
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Content-Type"] = "application/json"
        return resp


@router.get("/health")
async def health(request: Request):
    tracer = request.app.state.tracer
    with tracer.start_as_current_span("health_check") as span:
        try:
            if not hasattr(request.app.state, "db"):
                request.app.state.db = Database(db_url=DB_URL)

            db_start = time.time()
            with request.app.state.db.Session() as session:
                session.execute(text("SELECT 1 /* health check */"), execution_options={"timeout": 5}).scalar()
            db_duration = time.time() - db_start

            db_health_tracker.record_success()
            span.set_attribute("health.database_ok", True)
            span.set_attribute("health.database_duration_seconds", db_duration)
            span.set_attribute("health.status", "healthy")

            hls_healthy = os.path.exists(STREAM_FS_PATH)
            span.set_attribute("health.hls_stream_ok", hls_healthy)

            stream_dir = os.path.dirname(STREAM_FS_PATH)
            if os.path.exists(stream_dir):
                files = [f for f in os.listdir(stream_dir) if f.endswith(".ts")]
                span.set_attribute("health.hls_segments_count", len(files))

            resp = HTMLResponse(status_code=200, content="Healthy: Database connection verified")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            db_health_tracker.record_failure()
            span.set_attribute("health.database_ok", False)
            span.set_attribute("health.error", str(e))

            if db_health_tracker.is_failure_persistent():
                span.set_attribute("health.status", "unhealthy")
                span.set_attribute("health.persistent_failure", True)
                resp = HTMLResponse(
                    status_code=503,
                    content=f"Unhealthy: Database connection issues for over 5 minutes - {str(e)}",
                )
                resp.headers["Cache-Control"] = "no-store"
                return resp
            else:
                span.set_attribute("health.status", "degraded")
                span.set_attribute("health.persistent_failure", False)
                resp = HTMLResponse(
                    status_code=200, content="Healthy: Tolerating temporary database connectivity issue"
                )
                resp.headers["Cache-Control"] = "no-store"
                return resp
