# 🚦 StopSign AI

Real-time computer-vision pipeline that watches a stop sign, detects vehicles, and measures whether they actually stop.  
Runs end-to-end from an RTSP camera to a web dashboard with nothing more than Docker and Python.

🌐 **Live demo:** <https://crestwoodstopsign.com>   [![Status](https://img.shields.io/uptimerobot/status/m797914657-f517a98377b6b7a2e883d57a)](https://stats.uptimerobot.com/m797914657-f517a98377b6b7a2e883d57a)

![Afternoon screenshot](static/screenshot_afternoon.png)

---

## Tech-Stack Badges

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/) 
[![Ruff](https://img.shields.io/badge/style-Ruff-000000.svg)](https://github.com/astral-sh/ruff) 
[![UV](https://img.shields.io/badge/deps-UV-4A4A4A.svg)](https://github.com/astral-sh/uv) 
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics) 
[![FastHTML](https://img.shields.io/badge/frontend-FastHTML-orange.svg)](https://github.com/davidrose/fasthtml) 
[![Redis](https://img.shields.io/badge/cache-Redis-red.svg)](https://redis.io/) 
[![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL-blue.svg)](https://www.postgresql.org/) 
[![MinIO](https://img.shields.io/badge/storage-MinIO-00C6FF.svg)](https://min.io/) 

---

## Table of Contents

1. Quick Start (local development)
2. Project Architecture
3. Configuration
4. Monitoring & Metrics
5. Production Deployment
6. Directory Layout
7. Contributing & Development

---

## 1. Quick Start

The entire stack (camera simulator, AI pipeline, databases, web UI) can be launched locally with **one command**:

```bash
# 1️⃣ Install prerequisites (Docker + Make)
# 2️⃣ Run the setup script once – creates volumes, a sample video, and .env
make setup

# 3️⃣ Spin everything up
make dev

# 4️⃣ Open the UI ➜  http://localhost:8000
```

Need to stop or rebuild?

```bash
make dev-down     # stop containers
make dev-logs     # follow logs
make dev-build    # rebuild images
make dev-clean    # wipe everything (volumes, images, containers)
```

👉 For detailed developer docs see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## 2. Project Architecture

Service | Purpose | Code | Docker image (local)
---|---|---|---
RTSP → Redis | Grabs frames from an RTSP feed (or sample .mp4) and publishes JPEGs to Redis | `rtsp_to_redis/rtsp_to_redis.py` | `Dockerfile.rtsp.local`
Video Analyzer | YOLOv8 inference + object tracking + stop-sign logic. Stores metadata in Postgres and images in MinIO. Publishes annotated frames. | `stopsign/video_analyzer.py` | `Dockerfile.processor.local`
FFmpeg Service | Converts annotated frames → HLS stream (m3u8 + .ts) | `stopsign/ffmpeg_service.py` | `Dockerfile.ffmpeg.local`
Web Server | Simple FastAPI + FastHTML UI that shows the live stream & recent violations | `stopsign/web_server.py` | `Dockerfile.web.local`
Infrastructure | Redis, Postgres, MinIO (+ console) | Official upstream images | –

All of the above are declared in `docker/local/docker-compose.yml` and wired together with environment variables in `docker/local/.env` (created by `make setup`).

---

## 3. Configuration

Key settings are controlled via **environment variables** so that the exact same containers work in development and production.

Local (`docker/local/.env`):

```env
ENV=local
RTSP_URL=file:///app/sample_data/sample.mp4  # uses sample video
YOLO_MODEL_NAME=yolov8n.pt                   # light-weight CPU model
REDIS_URL=redis://redis:6379/0
DB_URL=postgresql://postgres:password@postgres:***REMOVED***/stopsign
MINIO_ENDPOINT=minio:9000
# … see template for all options
```

Production: supply the same variables via your orchestrator (Docker Swarm, Kubernetes, Fly.io, etc.).  GPU models (`yolov8x.pt`) & NVIDIA runtimes are fully supported.

Some advanced vision parameters (stop-line coordinates, buffer sizes, etc.) live in `config.yaml`.

---

## 4. Monitoring & Metrics

Every custom service exposes a Prometheus `/metrics` endpoint.  Mount a Prometheus/Grafana stack (or use the included Grafana data-source) to get:

* FPS, processing latency, dropped frames
* YOLO inference time, device utilisation (CPU/GPU)
* Redis/DB query timings
* FFmpeg encoder throughput

Grafana dashboards are provided in `static/`.

### Robust Stream Health Monitoring

Silent failures in HLS segment generation can be hard to catch with simple HTTP liveness checks. This repo now includes freshness-aware health endpoints and restart policies:

- `ffmpeg_service` health: `http://localhost:8080/health` returns 200 only when the HLS manifest is being updated; returns 503 if stale. Liveness is still available at `/healthz`.
- `web_server` stream health: `http://localhost:8000/health/stream` returns 200 when fresh and 503 when stale. Use this for external monitoring/alerting without coupling to DB availability.

How it determines freshness (no extra config):

- The services parse the HLS playlist (`stream.m3u8`) and compute a dynamic threshold from the manifest itself (target duration and the window of segments, via `#EXTINF`/`#EXT-X-PROGRAM-DATE-TIME`).
- A stream is considered healthy if the last segment timestamp is newer than ~3× the playlist window, with a safe floor of ~60 seconds. This adapts automatically to your HLS settings and avoids tuning env vars.

Defaults and resilience:

- `restart: always` added to core services for automatic recovery.
- Optional auto-restart watchdog is available in the encoder service code (disabled by default); it can be enabled by changing a constant if you want the container to self-restart after persistent stalls.

Examples

- Encoder readiness: `curl -i http://localhost:8080/health`
- Encoder liveness:  `curl -i http://localhost:8080/healthz`
- Web stream health: `curl -i http://localhost:8000/health/stream`

Notes

- Health endpoints set `Cache-Control: no-store` to avoid caching by proxies.
- Docker `restart: always` restarts containers only when the process exits. If you want automatic recovery on stalls, pair it with the encoder watchdog (which exits on persistent staleness).
- `/health` on the encoder is readiness-like; use `/healthz` for pure liveness probes.

---

## 5. Production Deployment

The legacy production setup is preserved in `docker/production/`.  Images are CUDA-enabled, use external managed databases, and **do not rely on .env files** – instead configure via environment variables / secrets.

Minimal example:

```bash
cd docker/production
docker compose --profile all up -d  # or your preferred orchestrator
```

Ensure the following external services are reachable:

* Redis 7+
* PostgreSQL 14+
* S3-compatible object storage (MinIO, AWS S3, etc.)

---

## 6. Directory Layout (top-level)

```
.
├── docker/             # Dockerfiles & compose files (local & production)
├── stopsign/           # Application source code (Python)
├── models/             # Pre-downloaded YOLO models
├── volumes/            # Bind-mounted data for local development
├── static/             # UI assets, screenshots, Grafana dashboards
├── sample_data/        # Sample video used in local mode
├── DEVELOPMENT.md      # Deep-dive developer guide
└── README.md           # You are here 💁
