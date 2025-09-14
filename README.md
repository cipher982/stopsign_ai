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
4. Frame Format & Timestamp Accuracy
5. Monitoring & Metrics
6. Production Deployment
7. Directory Layout
8. Contributing & Development

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
RTSP → Redis | Grabs frames from an RTSP feed (or sample .mp4) and publishes JPEGs to Redis with SSFM frame headers containing capture timestamps | `rtsp_to_redis/rtsp_to_redis.py` | `Dockerfile.rtsp.local`
Video Analyzer | YOLOv8 inference + object tracking + stop-sign logic. Uses capture timestamps for accurate timing. Stores metadata in Postgres and images in MinIO. Publishes annotated frames. | `stopsign/video_analyzer.py` | `Dockerfile.processor.local`
FFmpeg Service | Converts annotated frames → HLS stream (m3u8 + .ts) with Redis resilience and auto-recovery watchdog | `stopsign/ffmpeg_service.py` | `Dockerfile.ffmpeg.local`
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
DB_URL=postgresql://postgres:password@postgres:5432/stopsign
MINIO_ENDPOINT=minio:9000
# … see template for all options
```

Production: supply the same variables via your orchestrator (Docker Swarm, Kubernetes, Fly.io, etc.).  GPU models (`yolov8x.pt`) & NVIDIA runtimes are fully supported.

Some advanced vision parameters (stop-line coordinates, buffer sizes, etc.) live in `config.yaml`.

---

## 4. Frame Format & Timestamp Accuracy

### SSFM Wire Format

The pipeline uses a custom SSFM (StopSign Frame Message) format to ensure timestamp accuracy throughout the video processing chain:

**Frame Structure:**
- Bytes 0-3: `b'SSFM'` (magic header)
- Byte 4: Version (currently `1`)
- Bytes 5-8: Big-endian uint32 JSON metadata length
- Bytes 9+: JSON metadata + JPEG frame data

**JSON Metadata:**
```json
{
  "ts": 1694621234.567,  // Capture timestamp (epoch float)
  "w": 1920,             // Frame width
  "h": 1080,             // Frame height
  "src": "rtsp"          // Source identifier
}
```

**Benefits:**
- **Accurate timestamps**: Video overlay shows actual frame capture time, not processing time
- **Pipeline visibility**: Metadata includes `latency_sec` showing capture-to-processing delay
- **Backward compatibility**: Falls back gracefully for frames without SSFM headers

### Timestamp Sources

- **Capture timestamp**: Set at RTSP ingestion (`cap.read()` time) and preserved throughout pipeline
- **Processing timestamp**: Available in metadata for latency calculation
- **Video overlay**: Now displays capture timestamp in America/Chicago timezone for accuracy

---

## 5. Monitoring & Metrics

Every custom service exposes a Prometheus `/metrics` endpoint.  Mount a Prometheus/Grafana stack (or use the included Grafana data-source) to get:

* FPS, processing latency, dropped frames
* YOLO inference time, device utilisation (CPU/GPU)
* Redis/DB query timings
* FFmpeg encoder throughput

Grafana dashboards are provided in `static/`.

### Robust Stream Health Monitoring

Silent failures in HLS segment generation can be hard to catch with simple HTTP liveness checks. This repo includes comprehensive health endpoints and auto-recovery:

**Health Endpoints:**
- `ffmpeg_service` readiness: `http://localhost:8080/ready` - Composite health (HLS fresh + Redis connected + recent frames)
- `ffmpeg_service` stream health: `http://localhost:8080/health` - HLS freshness only (legacy)
- `ffmpeg_service` liveness: `http://localhost:8080/healthz` - Simple process health
- `web_server` stream health: `http://localhost:8000/health/stream` - Stream availability for external monitoring

**Auto-Recovery:** FFmpeg service includes a configurable watchdog that automatically restarts the container when HLS generation stalls, eliminating the need for manual intervention during network hiccups.

How it determines freshness (no extra config):

- The services parse the HLS playlist (`stream.m3u8`) and compute a dynamic threshold from the manifest itself (target duration and the window of segments, via `#EXTINF`/`#EXT-X-PROGRAM-DATE-TIME`).
- A stream is considered healthy if the last segment timestamp is newer than ~3× the playlist window, with a safe floor of ~60 seconds. This adapts automatically to your HLS settings and avoids tuning env vars.

Defaults and resilience:

- `restart: always` added to core services for automatic recovery
- **Redis resilience**: Exponential backoff reconnection logic handles network interruptions gracefully
- **Auto-restart watchdog**: Configurable via `PIPELINE_WATCHDOG_SEC` environment variable (e.g., 180 for 3-minute timeout)
- **FIFO frame processing**: Proper queue semantics ensure frames are processed in correct order

Examples

- Encoder composite health: `curl -i http://localhost:8080/ready`
- Encoder stream freshness: `curl -i http://localhost:8080/health`
- Encoder liveness: `curl -i http://localhost:8080/healthz`
- Web stream health: `curl -i http://localhost:8000/health/stream`

Notes

- Health endpoints set `Cache-Control: no-store` to avoid caching by proxies
- **Watchdog configuration**: Set `PIPELINE_WATCHDOG_SEC=180` to enable 3-minute auto-restart on HLS staleness
- **Redis configuration**: Optional `REDIS_MAX_BACKOFF_SEC=30` and `FRAME_STALL_SEC=120` for fine-tuning
- Use `/ready` for comprehensive readiness checks, `/healthz` for simple liveness, `/health` for stream-specific monitoring
- All services include exponential backoff Redis reconnection to handle network instability

---

## 6. Production Deployment

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

## 7. Directory Layout (top-level)

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
