# ruff: noqa: E501
"""Page routes — full HTML pages."""

import os

from fastapi import APIRouter
from fastapi import Request

from stopsign.web.app import templates
from stopsign.web.services.seo import PAGE_METADATA
from stopsign.web.services.seo import build_json_ld

router = APIRouter()

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
BASE_URL = "https://crestwoodstopsign.com"


@router.get("/")
async def home(request: Request):
    meta = PAGE_METADATA["home"]
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
        },
    )


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
