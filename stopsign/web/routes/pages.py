# ruff: noqa: E501
"""Page routes — full HTML pages."""

import logging
import os
import time

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse

from stopsign.database import Database
from stopsign.settings import DB_URL
from stopsign.web.app import templates
from stopsign.web.services.images import resolve_image_url
from stopsign.web.services.scoring import get_speed_color
from stopsign.web.services.scoring import get_time_color
from stopsign.web.services.seo import PAGE_METADATA
from stopsign.web.services.seo import build_json_ld

logger = logging.getLogger(__name__)

router = APIRouter()

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
BASE_URL = "https://crestwoodstopsign.com"


def _get_db(request: Request):
    if not hasattr(request.app.state, "db"):
        request.app.state.db = Database(db_url=DB_URL)
    return request.app.state.db


@router.get("/")
async def home(request: Request):
    meta = PAGE_METADATA["home"]

    # Server-render initial stats so they're visible immediately
    stats = {
        "compliance_rate": "--",
        "violation_count": "--",
        "vehicle_count": "--",
        "last_detection": "--",
    }
    try:
        db = _get_db(request)
        total_passes_24h = db.get_total_passes_last_24h()
        recent_passes = db.get_recent_vehicle_passes(limit=100)

        if recent_passes:
            compliant_count = sum(1 for p in recent_passes if p.time_in_zone >= 2.0)
            compliance_rate = round((compliant_count / len(recent_passes)) * 100)
        else:
            compliance_rate = 0

        last_detection = "N/A"
        if recent_passes:
            last_time = recent_passes[0].timestamp
            minutes_ago = int((time.time() - last_time.timestamp()) / 60)
            if minutes_ago < 60:
                last_detection = f"{minutes_ago}m ago"
            else:
                hours_ago = int(minutes_ago / 60)
                last_detection = f"{hours_ago}h ago"

        violation_count = total_passes_24h - int(total_passes_24h * compliance_rate / 100)

        stats = {
            "compliance_rate": f"{compliance_rate}%",
            "violation_count": str(violation_count),
            "vehicle_count": str(total_passes_24h),
            "last_detection": last_detection,
        }
    except Exception as e:
        logger.warning(f"Failed to pre-fetch homepage stats: {e}")

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "active_page": "home",
            "meta_title": "Stop Sign Nanny",
            "meta_description": meta["description"],
            "canonical_url": meta["url"],
            "meta_image": meta["image"],
            "json_ld": build_json_ld(BASE_URL, meta, "home"),
            "stats": stats,
        },
    )


@router.get("/passes/{pass_id}")
async def pass_detail(request: Request, pass_id: int):
    try:
        db = _get_db(request)
        detail = db.get_pass_detail(pass_id)
        if not detail:
            return HTMLResponse("<h1>Pass not found</h1>", status_code=404)

        vehicle_pass = detail["pass"]
        raw = detail["raw"]

        vehicle_attrs = db.get_vehicle_attributes_for_passes([pass_id])
        attrs = vehicle_attrs.get(pass_id, {})

        image_url = resolve_image_url(vehicle_pass.image_path)

        clip_url = None
        if vehicle_pass.clip_path and vehicle_pass.clip_status in ("ready", "local"):
            clip_url = f"/clips/{vehicle_pass.clip_path}"

        # Format timestamp
        from zoneinfo import ZoneInfo

        time_display = ""
        if vehicle_pass.timestamp:
            utc_time = vehicle_pass.timestamp.replace(tzinfo=ZoneInfo("UTC"))
            chicago_time = utc_time.astimezone(ZoneInfo("America/Chicago"))
            time_display = chicago_time.strftime("%b %d, %Y at %-I:%M %p")

        # Badge text
        badge_parts = []
        type_color = " ".join(filter(None, [attrs.get("color"), attrs.get("vehicle_type")]))
        if type_color:
            badge_parts.append(type_color.title())
        if attrs.get("make_model"):
            badge_parts.append(attrs["make_model"])
        badge_text = " | ".join(badge_parts)

        # Raw payload summary
        raw_summary = None
        if raw and raw.raw_payload:
            payload = raw.raw_payload
            raw_summary = {
                "version": payload.get("version"),
                "sample_count": raw.sample_count,
                "coordinate_space": payload.get("coordinate_space"),
                "dimensions": payload.get("dimensions"),
                "has_config": bool(payload.get("config_snapshot")),
                "has_model": bool(payload.get("model_snapshot")),
                "model_name": (payload.get("model_snapshot") or {}).get("model_name"),
            }

        return templates.TemplateResponse(
            "pass_detail.html",
            {
                "request": request,
                "active_page": "",
                "meta_title": f"Pass #{pass_id} - Stop Sign Nanny",
                "pass_id": pass_id,
                "image_url": image_url,
                "clip_url": clip_url,
                "time_display": time_display,
                "min_speed": vehicle_pass.min_speed,
                "time_in_zone": vehicle_pass.time_in_zone,
                "stop_duration": vehicle_pass.stop_duration,
                "speed_color": get_speed_color(vehicle_pass.min_speed),
                "time_color": get_time_color(vehicle_pass.time_in_zone),
                "badge_text": badge_text,
                "entry_time": vehicle_pass.entry_time,
                "exit_time": vehicle_pass.exit_time,
                "raw_summary": raw_summary,
            },
        )
    except Exception as e:
        logger.error(f"Error loading pass detail {pass_id}: {e}")
        return HTMLResponse(f"<h1>Error loading pass #{pass_id}</h1>", status_code=500)


@router.get("/records")
async def records(request: Request):
    meta = PAGE_METADATA["records"]
    return templates.TemplateResponse(
        "records.html",
        {
            "request": request,
            "active_page": "records",
            "meta_title": "Records - Stop Sign Nanny",
            "meta_description": meta["description"],
            "canonical_url": meta["url"],
            "meta_image": meta["image"],
            "json_ld": build_json_ld(BASE_URL, meta, "records"),
        },
    )


@router.get("/vehicles")
async def vehicles(request: Request):
    meta = PAGE_METADATA["vehicles"]
    return templates.TemplateResponse(
        "vehicles.html",
        {
            "request": request,
            "active_page": "vehicles",
            "meta_title": "Vehicles - Stop Sign Nanny",
            "meta_description": meta["description"],
            "canonical_url": meta["url"],
            "meta_image": meta["image"],
            "json_ld": build_json_ld(BASE_URL, meta, "vehicles"),
        },
    )


@router.get("/about")
async def about(request: Request):
    meta = PAGE_METADATA["about"]
    hero_metrics = [
        {"value": "15 FPS", "label": "real-time inference budget from capture to stream"},
        {"value": "SSFM timestamps", "label": "capture-time metadata preserved across every queue"},
        {"value": "4 dedicated services", "label": "ingest, analyze, render, and serve the experience"},
    ]

    pipeline_nodes = [
        {
            "name": "RTSP Camera",
            "badge": "Capture",
            "detail": "Network camera or sample MP4 feed streaming into the site via RTSP.",
        },
        {
            "name": "rtsp_to_redis",
            "badge": "Frame ingestion",
            "detail": "Encodes frames as JPEG, wraps them in the SSFM header, and LPUSHes into Redis with FIFO semantics.",
        },
        {
            "name": "Redis - RAW",
            "badge": "Buffer",
            "detail": "Deterministic queueing keeps capture order intact while smoothing network jitter.",
        },
        {
            "name": "video_analyzer",
            "badge": "Detection & scoring",
            "detail": "YOLO inference, Kalman-smoothed tracking, and stop-zone scoring feed Postgres + MinIO evidence.",
        },
        {
            "name": "Redis - PROCESSED",
            "badge": "Frame bus",
            "detail": "Annotated frames with timestamps stay ready for streaming without blocking the analyzer.",
        },
        {
            "name": "ffmpeg_service",
            "badge": "Streaming",
            "detail": "FFmpeg (NVENC or libx264) assembles HLS playlists, guarded by watchdog and readiness probes.",
        },
        {
            "name": "web_server",
            "badge": "Experience layer",
            "detail": "FastAPI + Jinja2 + htmx deliver the live player, dashboards, and developer tooling.",
        },
        {
            "name": "Operators",
            "badge": "Interface",
            "detail": "Browsers consume HLS, review recent passes, and adjust zones without redeploying.",
        },
    ]

    support_nodes = [
        {
            "name": "PostgreSQL",
            "badge": "Structured history",
            "detail": "Stores vehicle pass records, compliance scoring, and trend queries for insights.",
        },
        {
            "name": "MinIO",
            "badge": "Evidence store",
            "detail": "Holds annotated JPEG clips and exposes them through signed URLs in the UI.",
        },
        {
            "name": "Grafana + Prometheus",
            "badge": "Observability",
            "detail": "Dashboards visualize FPS, inference latency, queue depth, and HLS freshness.",
        },
    ]

    service_cards = [
        {
            "title": "rtsp_to_redis",
            "subtitle": "Frame ingestion & SSFM packaging",
            "items": [
                "LPUSHes JPEG frames with SSFM headers so capture timestamps survive downstream hops.",
                "Bounded queues (FRAME_BUFFER_SIZE) smooth out bursty networks without going stale.",
                "Exports Prometheus counters/timers plus runtime status mixins for health probes.",
            ],
        },
        {
            "title": "video_analyzer",
            "subtitle": "Computer vision core",
            "items": [
                "Runs YOLO via ONNX Runtime with GPU acceleration (configured via YOLO_MODEL_NAME/YOLO_DEVICE).",
                "CarTracker + Kalman filter blend trajectories for reliable stop detection.",
                "Persists scores to Postgres, ships annotated evidence to MinIO, and surfaces live insights.",
            ],
        },
        {
            "title": "ffmpeg_service",
            "subtitle": "HLS edge",
            "items": [
                "Consumes processed frames from Redis and renders annotated video at 15 FPS.",
                "Configurable FFmpeg encoders (NVENC, libx264) with presets tuned for low latency.",
                "Watchdog + /ready + /health endpoints restart the stream if freshness drifts.",
            ],
        },
        {
            "title": "web_server",
            "subtitle": "Experience + APIs",
            "items": [
                "FastAPI + Jinja2 pages powered by htmx for live updates without heavy JS.",
                "Interactive records view, live HLS.js player, and /debug zone editor for calibration.",
                "Caches insights, proxies media from MinIO, and exposes /health/stream for monitors.",
            ],
        },
    ]

    observability_items = [
        "Prometheus exporters on every service feed Grafana boards shipped in `static/`.",
        "Health surface: `/healthz` for liveness, `/ready` for freshness, `/health/stream` for external probes.",
        "ServiceStatus mixins report queue depth, Redis/DB connectivity, and error counters for triage.",
        "Insights cache highlights live trends (peak hour, average stop time, fastest vehicle).",
    ]

    resilience_items = [
        "Analyzer catch-up trims Redis backlogs when frames age past ANALYZER_CATCHUP_SEC.",
        "FFmpeg watchdog exits when HLS segments age beyond playlist thresholds so orchestrators restart cleanly.",
        "Single-source config (`config/config.yaml`) hot-reloads across services and persists via volumes.",
        "Debug UI + CLI tools (`tools/set_stop_zone.py`) let operators retune stop zones without downtime.",
    ]

    developer_items = [
        "`docker/local/docker-compose.yml` spins up the full stack with Redis, Postgres, and MinIO dependencies.",
        "`Makefile` automates setup (`make setup`), streaming (`make stream-local`), and linting.",
        "`sample_data/` video lets you replay the pipeline offline; `uv` manages Python deps reproducibly.",
        "Documentation lives under `docs/` covering architecture, health modeling, and deployment strategy.",
    ]

    return templates.TemplateResponse(
        "about.html",
        {
            "request": request,
            "active_page": "about",
            "meta_title": "About Stop Sign Nanny",
            "meta_description": meta["description"],
            "canonical_url": meta["url"],
            "meta_image": meta["image"],
            "json_ld": build_json_ld(BASE_URL, meta, "about"),
            "hero_metrics": hero_metrics,
            "pipeline_nodes": pipeline_nodes,
            "support_nodes": support_nodes,
            "service_cards": service_cards,
            "observability_items": observability_items,
            "resilience_items": resilience_items,
            "developer_items": developer_items,
        },
    )


@router.get("/debug")
async def debug_page(request: Request):
    meta = PAGE_METADATA["debug"]
    return templates.TemplateResponse(
        "debug.html",
        {
            "request": request,
            "active_page": "debug",
            "meta_title": "Stop Sign Debug",
            "meta_description": meta["description"],
            "canonical_url": meta["url"],
            "meta_image": meta["image"],
        },
    )


@router.get("/debug-perf")
async def debug_perf_page(request: Request):
    return templates.TemplateResponse(
        "debug_perf.html",
        {
            "request": request,
            "active_page": "",
            "meta_title": "Performance Debug",
        },
    )


@router.get("/statistics")
async def statistics(request: Request):
    return templates.TemplateResponse(
        "statistics.html",
        {
            "request": request,
            "active_page": "",
            "meta_title": "Statistics - Stop Sign Nanny",
            "grafana_url": GRAFANA_URL,
        },
    )
