"""Health check routes."""

import json
import logging
import os
import time

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse
from sqlalchemy import text

from stopsign.database import Database
from stopsign.hls_health import parse_hls_playlist
from stopsign.settings import DB_URL
from stopsign.settings import GRACE_STARTUP_SEC
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
