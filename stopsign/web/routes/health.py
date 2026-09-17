"""Health check routes."""

import asyncio
import json
import logging
import math
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
from stopsign.settings import ANALYZER_LAST_INFERENCE_AT_KEY
from stopsign.settings import ANALYZER_STALL_KEY
from stopsign.settings import ARCHIVE_HEALTH_REDIS_KEY
from stopsign.settings import DB_URL
from stopsign.settings import FFMPEG_HEALTH_KEY
from stopsign.settings import REDIS_URL
from stopsign.web.app import STREAM_FS_PATH

logger = logging.getLogger(__name__)
RELEASE_GENERATION = os.getenv("RELEASE_GENERATION")
if not RELEASE_GENERATION or RELEASE_GENERATION == "unknown":
    RELEASE_GENERATION = os.getenv("SOURCE_COMMIT", "unknown")
PROJECT_IDENTITY = os.getenv("PROJECT_IDENTITY", "stopsign")
RTSP_HEALTH_KEY = os.getenv("RTSP_HEALTH_KEY", "stopsign.rtsp.health")
ANALYZER_HEALTH_KEY = os.getenv("ANALYZER_HEALTH_KEY", "stopsign.analyzer.health")

router = APIRouter()

_HLS_PARSE_WARN_LAST_TS = 0.0
TIMESTAMP_FUTURE_TOLERANCE_SEC = float(os.getenv("STOPSIGN_TIMESTAMP_FUTURE_TOLERANCE_SEC", "2.0"))


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


def _refresh_archive_health(payload: dict) -> dict:
    """Recompute age-bearing archive fields from persisted timestamps on read."""
    now = time.time()
    for timestamp_key, age_key in (
        ("oldest_pending_local_ts", "oldest_pending_local_age_seconds"),
        ("oldest_pending_archive_ts", "oldest_pending_archive_age_seconds"),
        ("oldest_pending_reconciliation_ts", "oldest_pending_reconciliation_age_seconds"),
    ):
        timestamp = payload.get(timestamp_key)
        if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
            payload[age_key] = now - timestamp
    observed_at = payload.get("archive_health_observed_at")
    if isinstance(observed_at, (int, float)) and not isinstance(observed_at, bool):
        # Preserve negative ages: a future-dated observation is invalid evidence,
        # not a fresh observation.
        payload["archive_health_age_seconds"] = now - observed_at
    return payload


ARCHIVE_HEALTH_MAX_AGE_SEC = float(os.getenv("ARCHIVE_HEALTH_MAX_AGE_SEC", "900"))


def _classify_archive_health(payload: dict) -> tuple[str, str]:
    """Classify archive evidence without inventing health from a present key."""
    failures = []
    if payload.get("local_save_healthy") is False:
        failures.append("local capture persistence is unhealthy")
    if payload.get("upload_healthy") is False:
        failures.append("archive upload transport is unhealthy")
    if failures:
        return "failed", "; ".join(failures)
    if payload.get("archive_outbox_observed") is False:
        return "deferred", "archive durable outbox could not be observed"

    observed_age = payload.get("archive_health_age_seconds")
    if not isinstance(observed_age, (int, float)) or isinstance(observed_age, bool):
        return "deferred", "archive health signal has no valid observation age"
    if observed_age < 0:
        return "deferred", "archive health observation is future-dated"
    if observed_age > ARCHIVE_HEALTH_MAX_AGE_SEC:
        return "deferred", f"archive health signal is {observed_age:.1f}s old"

    if payload.get("archive_reconciliation_healthy") is False:
        pending_archive = payload.get("pending_archive_files")
        pending_reconciliation = payload.get("pending_reconciliation_files")
        if (
            payload.get("archive_outbox_observed") is True
            and pending_archive == 0
            and isinstance(pending_reconciliation, int)
            and pending_reconciliation > 0
        ):
            return (
                "reconciling",
                f"{pending_reconciliation} proven archive object(s) are awaiting database path reconciliation",
            )
        return "degraded", "archive reconciliation has unverified captures"
    if payload.get("local_save_healthy") is not True or payload.get("upload_healthy") is not True:
        return "deferred", "archive health signal lacks explicit healthy transport flags"
    return "healthy", "local capture persistence and archive transport are healthy"


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


def _tracker_for(request: Request) -> DBHealthTracker:
    """Keep DB failure state scoped to the live app instance."""
    tracker = getattr(request.app.state, "db_health_tracker", None)
    if tracker is None:
        tracker = db_health_tracker
        request.app.state.db_health_tracker = tracker
    return tracker


def _check_database(db: Database, query: str) -> None:
    """Run one bounded-by-caller synchronous DB probe in its own thread."""
    with db.Session() as session:
        session.execute(text(query)).scalar()


@router.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "outcome": "healthy",
        "liveness": "healthy",
        "release_generation": RELEASE_GENERATION,
        "project_identity": PROJECT_IDENTITY,
    }


@router.api_route("/readyz", methods=["GET", "HEAD"])
async def readyz(request: Request):
    """Web-serving readiness, deliberately independent of video freshness."""
    payload = {
        "schema_version": 2,
        "release_generation": RELEASE_GENERATION,
        "project_identity": PROJECT_IDENTITY,
        "ready": False,
        "status": "unavailable",
        "reason": "database readiness has not been checked",
        "database": {"status": "unavailable"},
    }
    tracker = _tracker_for(request)
    try:
        db = getattr(request.app.state, "db", None)
        if db is None:
            db = Database(db_url=DB_URL)
            request.app.state.db = db
        await asyncio.wait_for(
            asyncio.to_thread(_check_database, db, "SELECT 1 /* ready check */"),
            timeout=5.0,
        )
        tracker.record_success()
        payload["ready"] = True
        payload["status"] = "healthy"
        payload["reason"] = "web server and database are available"
        payload["database"] = {"status": "healthy"}
        status_code = 200
    except Exception as exc:
        tracker.record_failure()
        payload["reason"] = f"database unavailable: {exc}"
        payload["database"] = {"status": "unavailable", "reason": str(exc)}
        status_code = 503
    resp = JSONResponse(payload, status_code=status_code)
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
        payload = _refresh_archive_health(json.loads(raw))
        payload["available"] = True
        payload["status"], payload["reason"] = _classify_archive_health(payload)
    except Exception as e:
        logger.warning(f"archive_health read failed: {e}")
        return JSONResponse({"available": False, "error": str(e)})
    return JSONResponse(payload)


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


def _read_stage_health(client, key: str, stage: str, now: float) -> dict:
    """Read a stage heartbeat and downgrade stale claims instead of trusting labels."""
    try:
        raw = client.get(key)
    except Exception as exc:
        return {
            "stage": stage,
            "available": False,
            "status": "unavailable",
            "reason": str(exc),
            "_heartbeat_present": False,
            "_redis_error": True,
        }
    if not raw:
        return {
            "stage": stage,
            "available": False,
            "status": "unavailable",
            "reason": "no stage heartbeat",
            "_heartbeat_present": False,
        }
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return {
            "stage": stage,
            "available": False,
            "status": "unavailable",
            "reason": f"invalid heartbeat: {exc}",
            "_heartbeat_present": True,
        }
    if not isinstance(payload, dict):
        return {
            "stage": stage,
            "available": False,
            "status": "unavailable",
            "reason": "heartbeat is not an object",
            "_heartbeat_present": True,
        }
    payload = dict(payload)
    payload["stage"] = stage
    payload["available"] = True
    payload["_heartbeat_present"] = True
    observed = payload.get("updated_at", payload.get("ts"))
    observed_age = None
    if isinstance(observed, (int, float)) and not isinstance(observed, bool) and math.isfinite(float(observed)):
        observed_age = now - float(observed)
        payload["heartbeat_age_seconds"] = round(observed_age, 1)
        # The Redis key has a TTL, but a mocked/older Redis reader can still
        # expose a repeated value. Never call an old heartbeat healthy merely
        # because its producer labelled it that way.
        if observed_age < -TIMESTAMP_FUTURE_TOLERANCE_SEC:
            payload["status"] = "failed"
            payload["reason"] = "stage heartbeat timestamp is in the future"
        elif observed_age > float(os.getenv("STAGE_HEARTBEAT_STALE_SEC", "300")):
            payload["status"] = "failed"
            payload["reason"] = "stage heartbeat is stale"
    status = payload.get("status")
    if status is None:
        # The first FFmpeg health writer emitted fps/duplication fields and a
        # timestamp but no structured stage status. Treat a current snapshot
        # as compatibility evidence, never as a timeless healthy claim.
        status = (
            "healthy" if stage == "ffmpeg" and observed_age is not None and 0 <= observed_age <= 300 else "deferred"
        )
        payload["status"] = status

    def timestamp_age(key: str) -> float | None:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        return now - float(value)

    stale_reason = None
    if stage == "rtsp":
        reference_age = timestamp_age("last_publish_ts")
        threshold = float(payload.get("push_stale_threshold_seconds", 10.0))
        if reference_age is None:
            stale_reason = "Redis publish timestamp is missing or invalid"
        elif reference_age < -TIMESTAMP_FUTURE_TOLERANCE_SEC:
            stale_reason = "Redis publish timestamp is in the future"
        elif reference_age > threshold:
            stale_reason = "Redis publish heartbeat is stale"
    elif stage == "analyzer":
        reference_age = timestamp_age("last_output_capture_ts")
        if reference_age is None:
            reference_age = payload.get("capture_age_seconds")
            if (
                isinstance(reference_age, bool)
                or not isinstance(reference_age, (int, float))
                or not math.isfinite(float(reference_age))
            ):
                reference_age = None
        threshold = float(os.getenv("ANALYZER_CATCHUP_SEC", "15"))
        if reference_age is None:
            stale_reason = "processed capture timestamp is missing or invalid"
        elif float(reference_age) < -TIMESTAMP_FUTURE_TOLERANCE_SEC:
            stale_reason = "processed capture timestamp is in the future"
        elif float(reference_age) > threshold:
            stale_reason = "processed capture evidence is stale"
    elif stage == "ffmpeg":
        reference_age = timestamp_age("last_encoded_capture_ts")
        if reference_age is None:
            reference_age = payload.get("encoded_capture_age_seconds")
            if (
                isinstance(reference_age, bool)
                or not isinstance(reference_age, (int, float))
                or not math.isfinite(float(reference_age))
            ):
                reference_age = None
        threshold = float(os.getenv("FRAME_STALL_SEC", "120"))
        if reference_age is None:
            stale_reason = "encoded capture timestamp is missing or invalid"
        elif float(reference_age) < -TIMESTAMP_FUTURE_TOLERANCE_SEC:
            stale_reason = "encoded capture timestamp is in the future"
        elif float(reference_age) > threshold:
            stale_reason = "encoded capture evidence is stale"
        if payload.get("hls_fresh") is False:
            stale_reason = stale_reason or "HLS playlist is stale"
    if stale_reason and (status == "healthy" or "future" in stale_reason):
        payload["status"] = "failed"
        payload["reason"] = stale_reason
    payload.setdefault("reason", stale_reason or f"{stage} reported {payload.get('status')}")
    if observed_age is not None and stage == "ffmpeg":
        payload.setdefault("snapshot_age_seconds", round(max(0.0, observed_age), 1))
    return payload


def _legacy_float(raw, now: float) -> tuple[float | None, float | None]:
    """Parse a legacy timestamp and return (timestamp, age)."""
    if raw in (None, b"", ""):
        return None, None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None, None
    if not math.isfinite(value):
        return None, None
    age = now - value
    if -TIMESTAMP_FUTURE_TOLERANCE_SEC <= age < 0:
        age = 0.0
    return value, round(age, 1)


def _apply_legacy_analyzer_evidence(analyzer: dict, legacy: dict, now: float) -> bool:
    """Attach legacy analyzer keys and synthesize a truthful stage result.

    The timestamp keys predate stage heartbeats and are still written by the
    analyzer.  They are useful compatibility evidence, but only their age can
    establish freshness; a persisted stall or timestamp is never silently
    treated as healthy.
    """
    last_frame_at, frame_age = _legacy_float(legacy.get("last_frame"), now)
    last_inference_at, inference_age = _legacy_float(legacy.get("last_inference"), now)
    started_at, _ = _legacy_float(legacy.get("boot"), now)
    if started_at is not None:
        analyzer["uptime_seconds"] = round(max(0.0, now - started_at), 1)
    stall = legacy.get("stall")
    analyzer["inference_available"] = last_inference_at is not None
    if last_inference_at is None and analyzer.get("_heartbeat_present"):
        heartbeat_inference_age = analyzer.get("inference_age_seconds")
        analyzer["inference_available"] = (
            isinstance(heartbeat_inference_age, (int, float))
            and heartbeat_inference_age >= 0
            and heartbeat_inference_age <= float(os.getenv("ANALYZER_CATCHUP_SEC", "15"))
        )
    if last_frame_at is not None:
        analyzer["last_frame_at"] = last_frame_at
        analyzer["frame_age_seconds"] = frame_age
    if last_inference_at is not None:
        analyzer["last_inference_at"] = last_inference_at
        analyzer["inference_age_seconds"] = inference_age
    if (frame_age is not None and frame_age < -TIMESTAMP_FUTURE_TOLERANCE_SEC) or (
        inference_age is not None and inference_age < -TIMESTAMP_FUTURE_TOLERANCE_SEC
    ):
        analyzer["status"] = "failed"
        analyzer["reason"] = "legacy analyzer timestamp is in the future"
    if stall not in (None, b"", ""):
        try:
            analyzer["last_stall"] = json.loads(stall)
        except (TypeError, ValueError):
            analyzer.setdefault("reason", "legacy analyzer stall evidence is invalid")

    has_evidence = any(value not in (None, b"", "") for value in legacy.values())
    if analyzer.get("_heartbeat_present") or not has_evidence:
        return has_evidence

    analyzer["available"] = True
    threshold = float(os.getenv("ANALYZER_CATCHUP_SEC", "15"))
    last_stall = analyzer.get("last_stall")
    stall_at = last_stall.get("triggered_at") if isinstance(last_stall, dict) else None
    unrecovered_stall = isinstance(stall_at, (int, float)) and (
        last_frame_at is None or last_frame_at <= float(stall_at)
    )
    if last_frame_at is None:
        analyzer["status"] = "deferred"
        analyzer["reason"] = "legacy analyzer frame evidence is unavailable"
    elif frame_age is None or frame_age < -TIMESTAMP_FUTURE_TOLERANCE_SEC or frame_age > threshold:
        analyzer["status"] = "failed"
        analyzer["reason"] = "legacy analyzer frame evidence is stale"
    elif unrecovered_stall:
        analyzer["status"] = "failed"
        analyzer["reason"] = (
            last_stall.get("reason", "legacy analyzer stall recorded")
            if isinstance(last_stall, dict)
            else "legacy analyzer stall recorded"
        )
    elif (
        last_inference_at is None
        or inference_age is None
        or inference_age < -TIMESTAMP_FUTURE_TOLERANCE_SEC
        or inference_age > threshold
    ):
        analyzer["status"] = "degraded"
        analyzer["reason"] = "legacy analyzer inference evidence is stale or unavailable"
    else:
        analyzer["status"] = "healthy"
        analyzer["reason"] = "legacy analyzer frame and inference evidence are current"
    return has_evidence


def _mark_missing_legacy_stage(stage: dict, stage_name: str, has_legacy_evidence: bool) -> None:
    """Keep a Redis-connected legacy response structured without false health."""
    if has_legacy_evidence and not stage.get("_heartbeat_present") and not stage.get("_redis_error"):
        stage["status"] = "deferred"
        stage["reason"] = f"{stage_name} heartbeat unavailable; legacy evidence retained"


@router.get("/api/pipeline-health")
async def pipeline_health():
    """Canonical capture -> analyzer -> FFmpeg -> HLS aggregate."""
    now = time.time()
    try:
        r = redis_lib.from_url(REDIS_URL, socket_connect_timeout=0.3, socket_timeout=0.3)
        # Older Redis fakes/clients used by compatibility consumers do not
        # expose ping; the stage reads below still prove connectivity there.
        ping = getattr(r, "ping", None)
        if ping is not None:
            ping()
    except Exception as exc:
        unavailable = {
            "available": False,
            "status": "unavailable",
            "reason": f"Redis health read unavailable: {exc}",
            "generated_at": now,
            "release_generation": RELEASE_GENERATION,
            "project_identity": PROJECT_IDENTITY,
        }
        return JSONResponse(unavailable)

    rtsp = _read_stage_health(r, RTSP_HEALTH_KEY, "rtsp", now)
    analyzer = _read_stage_health(r, ANALYZER_HEALTH_KEY, "analyzer", now)
    ffmpeg = _read_stage_health(r, FFMPEG_HEALTH_KEY, "ffmpeg", now)

    # Preserve the older analyzer evidence fields for existing pollers. Read
    # these independently: one expired/malformed key must not hide the others.
    legacy = {}
    for name, key in (
        ("last_frame", ANALYZER_LAST_FRAME_AT_KEY),
        ("last_inference", ANALYZER_LAST_INFERENCE_AT_KEY),
        ("boot", ANALYZER_BOOT_TS_KEY),
        ("stall", ANALYZER_STALL_KEY),
    ):
        try:
            legacy[name] = r.get(key)
        except Exception:
            legacy[name] = None
    has_legacy_evidence = _apply_legacy_analyzer_evidence(analyzer, legacy, now)
    _mark_missing_legacy_stage(rtsp, "rtsp", has_legacy_evidence)
    _mark_missing_legacy_stage(ffmpeg, "ffmpeg", has_legacy_evidence)
    try:
        raw_archive = r.get(ARCHIVE_HEALTH_REDIS_KEY)
        if raw_archive:
            archive = _refresh_archive_health(json.loads(raw_archive))
            archive["available"] = True
            archive["status"], archive["reason"] = _classify_archive_health(archive)
        else:
            archive = {
                "available": False,
                "status": "deferred",
                "reason": "no archive health signal yet",
            }
    except Exception as exc:
        archive = {"available": False, "status": "unavailable", "reason": str(exc)}

    info = _parse_hls_playlist(STREAM_FS_PATH)
    hls_age = info.get("age_seconds")
    hls_fresh = bool(info.get("exists")) and hls_age is not None and hls_age <= info.get("threshold_sec", 60.0)
    encoded_fresh = bool(ffmpeg.get("hls_fresh")) and (
        ffmpeg.get("encoded_capture_age_seconds") is not None
        and ffmpeg.get("encoded_capture_age_seconds") <= float(os.getenv("FRAME_STALL_SEC", "120"))
    )
    hls_status = "healthy" if hls_fresh and encoded_fresh else ("deferred" if not ffmpeg.get("available") else "failed")
    hls_reason = (
        "HLS playlist and encoded capture evidence are current"
        if hls_status == "healthy"
        else "HLS requires fresh playlist and FFmpeg encoded capture evidence"
    )
    hls = {
        "available": True,
        "status": hls_status,
        "reason": hls_reason,
        "fresh": hls_fresh and encoded_fresh,
        "playlist_fresh": hls_fresh,
        "encoded_capture_fresh": encoded_fresh,
        "age_seconds": hls_age,
        "segments_count": info.get("segments_count", 0),
    }

    for stage in (rtsp, analyzer, ffmpeg):
        stage.pop("_heartbeat_present", None)
        stage.pop("_redis_error", None)

    stages = {"rtsp": rtsp, "analyzer": analyzer, "ffmpeg": ffmpeg, "hls": hls, "archive": archive}
    # Archive evidence is exposed above but does not gate live video readiness;
    # the dedicated archive alert owns that failure domain.
    statuses = [stage.get("status", "unavailable") for name, stage in stages.items() if name != "archive"]
    if "unavailable" in statuses:
        overall_status = "unavailable"
    elif "failed" in statuses:
        overall_status = "failed"
    elif "degraded" in statuses:
        overall_status = "degraded"
    elif "deferred" in statuses:
        overall_status = "deferred"
    else:
        overall_status = "healthy"
    return JSONResponse(
        {
            "schema_version": 2,
            "generated_at": now,
            "available": True,
            "ready": overall_status == "healthy",
            "status": overall_status,
            "reason": "one or more pipeline stages are not healthy"
            if overall_status != "healthy"
            else "all pipeline stages report current evidence",
            "release_generation": RELEASE_GENERATION,
            "project_identity": PROJECT_IDENTITY,
            "stages": stages,
            # Legacy top-level keys remain available to existing alert jobs.
            "rtsp": rtsp,
            "archive": archive,
            "analyzer": analyzer,
            "ffmpeg": ffmpeg,
            "hls": hls,
        }
    )


@router.get("/health/stream")
async def health_stream(request: Request):
    tracer = request.app.state.tracer
    with tracer.start_as_current_span("health_stream") as span:
        now = time.time()
        info = _parse_hls_playlist(STREAM_FS_PATH)
        age = info.get("age_seconds")
        exists = bool(info.get("exists"))
        threshold = info.get("threshold_sec", 60.0)
        playlist_fresh = exists and age is not None and age <= threshold
        ffmpeg = {"available": False, "status": "unavailable", "reason": "FFmpeg heartbeat unavailable"}
        try:
            client = redis_lib.from_url(REDIS_URL, socket_connect_timeout=0.3, socket_timeout=0.3)
            ffmpeg = _read_stage_health(client, FFMPEG_HEALTH_KEY, "ffmpeg", now)
        except Exception as exc:
            ffmpeg["reason"] = str(exc)
        encoded_age = ffmpeg.get("encoded_capture_age_seconds")
        encoded_fresh = (
            ffmpeg.get("available", False)
            and ffmpeg.get("hls_fresh", False)
            and isinstance(encoded_age, (int, float))
            and float(encoded_age) <= float(os.getenv("FRAME_STALL_SEC", "120"))
        )
        fresh = bool(playlist_fresh and encoded_fresh)
        if fresh:
            stage_status = "healthy"
            reason = "HLS playlist and FFmpeg encoded capture evidence are current"
        elif not ffmpeg.get("available"):
            stage_status = "unavailable"
            reason = "FFmpeg encoded capture evidence is unavailable"
        elif not playlist_fresh:
            stage_status = "failed"
            reason = "HLS playlist is missing or stale"
        else:
            stage_status = "failed"
            reason = "FFmpeg encoded capture evidence is stale or missing"

        span.set_attribute("hls.exists", exists)
        if age is not None:
            span.set_attribute("hls.age_seconds", float(age))
        span.set_attribute("hls.segments_count", info.get("segments_count", 0))
        span.set_attribute("hls.threshold_sec", threshold)
        span.set_attribute("hls.fresh", fresh)

        payload = {
            "schema_version": 2,
            "status": stage_status,
            "reason": reason,
            "fresh": fresh,
            "exists": exists,
            "age_seconds": age,
            "threshold_sec": threshold,
            "playlist_fresh": playlist_fresh,
            "encoded_capture_fresh": encoded_fresh,
            "encoded_capture_age_seconds": encoded_age,
            "segments_count": info.get("segments_count", 0),
            "ffmpeg": ffmpeg,
            "release_generation": RELEASE_GENERATION,
            "project_identity": PROJECT_IDENTITY,
        }
        resp = HTMLResponse(status_code=200 if fresh else 503, content=json.dumps(payload))
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Content-Type"] = "application/json"
        return resp


@router.get("/health")
async def health(request: Request):
    tracer = request.app.state.tracer
    tracker = _tracker_for(request)
    with tracer.start_as_current_span("health_check") as span:
        try:
            if not hasattr(request.app.state, "db"):
                request.app.state.db = Database(db_url=DB_URL)

            db_start = time.time()
            await asyncio.wait_for(
                asyncio.to_thread(_check_database, request.app.state.db, "SELECT 1 /* health check */"),
                timeout=5.0,
            )
            db_duration = time.time() - db_start

            tracker.record_success()
            span.set_attribute("health.database_ok", True)
            span.set_attribute("health.database_duration_seconds", db_duration)
            span.set_attribute("health.status", "healthy")

            hls_healthy = os.path.exists(STREAM_FS_PATH)
            span.set_attribute("health.hls_stream_ok", hls_healthy)

            stream_dir = os.path.dirname(STREAM_FS_PATH)
            if os.path.exists(stream_dir):
                files = [f for f in os.listdir(stream_dir) if f.endswith(".ts")]
                span.set_attribute("health.hls_segments_count", len(files))

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            tracker.record_failure()
            span.set_attribute("health.database_ok", False)
            span.set_attribute("health.error", str(e))

            if tracker.is_failure_persistent():
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
