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
[![YOLO11](https://img.shields.io/badge/AI-YOLO11-green.svg)](https://github.com/ultralytics/ultralytics) 
[![FastHTML](https://img.shields.io/badge/frontend-FastHTML-orange.svg)](https://github.com/answerdotai/fasthtml) 
[![Redis](https://img.shields.io/badge/cache-Redis-red.svg)](https://redis.io/) 
[![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL-blue.svg)](https://www.postgresql.org/) 
[![MinIO](https://img.shields.io/badge/storage-MinIO-00C6FF.svg)](https://min.io/) 

---

## Table of Contents

1. What’s Interesting
2. Architecture
3. Configuration
4. Frame Format & Timestamp Accuracy
5. Health Model & Metrics
6. Resilience Knobs
7. Production Deployment
8. Directory Layout

---

## 1. What’s Interesting

- Capture‑time correctness end‑to‑end via SSFM headers (actual camera capture time used for all logic/overlays).
- Deterministic FIFO across both legs of the pipeline (producer `LPUSH` + consumer `BRPOP`).
- Clear health separation: liveness (`/healthz`) vs readiness (`/ready`) vs HLS freshness (`/health`, `/health/stream`).
- Dynamic HLS freshness threshold derived from playlist window (no brittle hard‑coded timers).
- Analyzer catch‑up mode that trims backlog after stalls to minimize time‑to‑fresh.
- Observability triad: capture_age, process_age, hls_age to pinpoint where staleness originates.

For setup and development details, see the docs/ directory (e.g., `docs/architecture/first-principles-streaming.md`).

---

## 2. Project Architecture

### Architecture at a Glance

```mermaid
graph LR
  C[RTSP Camera] -->|RTSP| I[rtsp_to_redis]
  I -->|SSFM JPEG LPUSH RAW| R[(Redis)]
  R -->|BRPOP RAW| A[video_analyzer]
  A -->|LPUSH PROCESSED| R
  R -->|BRPOP PROCESSED| F[ffmpeg_service]
  F -->|HLS .m3u8/.ts| W[web_server]
  W -->|HTTP| B[(Browser)]

  subgraph Health & Telemetry
    F -.->|/healthz /ready /health| M[Monitoring]
    W -.->|/health/stream| M
  end
```

Service | Purpose | Code | Docker image (local)
---|---|---|---
RTSP → Redis | Grabs frames from an RTSP feed (or sample .mp4) and publishes JPEGs to Redis with SSFM frame headers containing capture timestamps | `rtsp_to_redis/rtsp_to_redis.py` | `Dockerfile.rtsp.local`
Video Analyzer | YOLO11 inference + object tracking + stop-sign logic. Uses capture timestamps for accurate timing. Stores metadata in Postgres and images in MinIO. Publishes self-describing annotated BGR frames. | `stopsign/video_analyzer.py` | `Dockerfile.processor.local`
FFmpeg Service | Converts annotated frames → HLS stream (m3u8 + .ts) with Redis resilience and auto-recovery watchdog | `stopsign/ffmpeg_service.py` | `Dockerfile.ffmpeg.local`
Web Server | Simple FastAPI + FastHTML UI that shows the live stream & recent violations | `stopsign/web_server.py` | `Dockerfile.web.local`
Infrastructure | Redis, Postgres, MinIO (+ console) | Official upstream images | –

All of the above are declared in `docker/local/docker-compose.yml` and wired together with environment variables in `docker/local/.env` (created by `make setup`).

### Queue Semantics & Backlog Policy

- Raw → Analyzer: `LPUSH` (producer) + `BRPOP` (consumer) on `RAW_FRAME_KEY` = FIFO; buffer bounded via `LTRIM`.
- Analyzer → FFmpeg: `LPUSH` + `BRPOP` on `PROCESSED_FRAME_KEY` = FIFO. Processed frames use the `SSF2` envelope so dimensions travel with the frame; FFmpeg still accepts legacy raw BGR bytes during migration.
- Catch‑up policy: if a raw frame’s capture timestamp is older than `ANALYZER_CATCHUP_SEC`, trim RAW to last `ANALYZER_CATCHUP_KEEP_N` frames to jump back to near‑live.

---

## 3. Configuration

Key settings are controlled via **environment variables** so that the exact same containers work in development and production.

Local (`docker/local/.env`):

```env
ENV=local
RTSP_URL=file:///app/sample_data/sample.mp4  # uses sample video
YOLO_MODEL_NAME=yolo11n.pt                   # light-weight CPU model
REDIS_URL=redis://redis:6379/0
DB_URL=postgresql://postgres:password@postgres:5432/stopsign
MINIO_ENDPOINT=minio:9000
# … see template for all options
```

Production: supply the same variables via your orchestrator (Docker Swarm, Kubernetes, Fly.io, etc.).  GPU models (`yolo11x.pt`) & NVIDIA runtimes are fully supported.

### Configuration Management

**Config File:** `config/config.yaml` (NOT in git - created from `config.example.yaml`)

**Key Features:**
- **Persistent across deployments:** Config stored in Docker volume, not in image
- **Single source of truth:** One config file shared by all services
- **Atomic updates:** Changes are written atomically with automatic versioning
- **Instant reload:** Analyzer detects changes every frame (< 100ms typical)
- **Fail-fast:** Services exit immediately if config is missing or invalid

**Initial Setup:**
```bash
# Local development
cp config/config.example.yaml config/config.yaml

# Production (after first deploy)
docker exec <container> cp /app/config.example.yaml /app/config/config.yaml
```

**CLI Tools:**
```bash
# View current configuration
./tools/print_config.py

# Update stop zone coordinates (four corners)
./tools/set_stop_zone.py \
  --x1 300 --y1 225 \
  --x2 380 --y2 225 \
  --x3 380 --y3 295 \
  --x4 300 --y4 295
```

**Web UI:** Navigate to `/debug` to visually adjust stop zones with instant feedback.

**Production Persistence:** Use Coolify persistent volumes or Docker named volumes to maintain config across deployments. See `docs/config-deployment.md` for details.

---

## 4. Frame Format & Timestamp Accuracy

### Frame Wire Formats

The pipeline uses compact frame headers so timestamps and dimensions travel with frame bytes:

**Legacy raw input structure:**
- Bytes 0-3: `b'SSFM'` (magic header for RTSP JPEG frames)
- Byte 4: Version (currently `1`)
- Bytes 5-8: Big-endian uint32 JSON metadata length
- Bytes 9+: JSON metadata + JPEG frame data

**Processed-frame structure:**
- Bytes 0-3: `b'SSF2'` (self-describing frame envelope)
- Byte 4: Version (currently `1`)
- Bytes 5-8: Big-endian uint32 JSON metadata length
- Bytes 9+: JSON metadata + BGR24 frame bytes

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

## 5. Health Model & Metrics

Every custom service exposes a Prometheus `/metrics` endpoint. Mount a Prometheus/Grafana stack (or use the included Grafana data-source) to get:

* FPS, processing latency, dropped frames
* YOLO inference time, device utilisation (CPU/GPU)
* Redis/DB query timings
* FFmpeg encoder throughput

Grafana dashboards are provided in `static/`.

### Health endpoints and semantics

- **rtsp_to_redis** — `/healthz` (liveness) and `/ready` (Redis push freshness + visual freeze detection).
- **video_analyzer** — `/healthz` (liveness) and `/ready` (frame gap ≤ `ANALYZER_STALL_SEC`).
- **ffmpeg_service** — `/healthz` (liveness) and `/ready` (fresh HLS + Redis + recent frame).
- **web_server** — `/healthz` (process up), `/readyz` (public uptime probe, supports GET/HEAD), and `/health/stream` (HLS freshness for external monitors).
- Legacy `/health` on ffmpeg_service remains for backwards compatibility.

Docker healthchecks now target the liveness endpoints (`/healthz`) so short upstream hiccups don’t flip container health; alerting systems should watch `/ready` and treat `/health/stream` as an external freshness signal.

### Netdata alert tuning

Netdata ships with aggressive defaults that page on any short-lived HTTP failure. With the new readiness split:

1. Monitor `video_analyzer` `/ready` and `ffmpeg_service` `/ready` with a **warning** threshold at ~2 minutes and a **critical** alert only after ≥10 minutes of failure. This filters transient Wi-Fi drops while still paging on real outages.
2. Optionally keep `web_server` `/health/stream` as an informational alarm; set `delay: up`/`delay: down` to a few minutes so the recovered notification isn’t spammy.
3. Use the new OTEL metrics `frame_queue_depth`, `frame_pipeline_lag_seconds`, and `redis_empty_polls_total` to build dashboards that show where the stall originated (camera ingest, analyzer, or encoder) before escalating.

These thresholds ensure operators see sustained issues without the “unhealthy/recovered” churn that previously flooded email.

### Freshness and the “three ages”

- `capture_age` = now − SSFM.capture_ts (ingest health)
- `process_age` = now − last frame processed in ffmpeg_service (pipeline health)
- `hls_age` = now − last `#EXT-X-PROGRAM-DATE-TIME` in playlist (edge/player health)

Freshness threshold is derived from the playlist window (~3× window, floored at 60s) from `#EXTINF` and PDT.

### Robust Stream Health Monitoring

Silent failures in HLS segment generation can be hard to catch with simple HTTP liveness checks. This repo includes comprehensive health endpoints and auto-recovery:

**Health Endpoints:**
- `rtsp_to_redis` readiness: `http://localhost:8080/ready` – Redis push freshness + frozen-frame guardrail
- `rtsp_to_redis` liveness: `http://localhost:8080/healthz`
- `video_analyzer` readiness: `http://localhost:${ANALYZER_HEALTH_PORT:-8081}/ready` – frame pipeline (“can I serve fresh frames?”)
- `video_analyzer` liveness: `http://localhost:${ANALYZER_HEALTH_PORT:-8081}/healthz`
- `ffmpeg_service` readiness: `http://localhost:8080/ready` – HLS + Redis + recent frames
- `ffmpeg_service` liveness: `http://localhost:8080/healthz`
- `web_server` liveness: `http://localhost:8000/healthz`
- `web_server` public uptime probe: `http://localhost:8000/readyz`
- `web_server` stream freshness: `http://localhost:8000/health/stream`

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

## 6. Resilience Knobs

- `RTSP_FREEZE_DETECT_SEC` (default 120): mark ingest as frozen if no visual motion over this window.
- `RTSP_FREEZE_MAD_THRESHOLD` (default 0.015): frame-diff sensitivity for freeze detection.
- `RTSP_FREEZE_RECONNECT_SEC` (default 180): force RTSP reconnect on sustained freeze.
- `RTSP_FREEZE_REMEDIATION_CMD` (default empty): optional command hook for camera reboot automation.
- `RTSP_FREEZE_REMEDIATION_SEC` (default 420): freeze age before running remediation command.
- `ANALYZER_CATCHUP_SEC` (default 15): skip/trim stale raw frames older than this age to jump back to live.
- `ANALYZER_CATCHUP_KEEP_N` (default 30): how many newest raw frames to retain when trimming.
- `FRAME_STALL_SEC` (default 120): readiness requires frames newer than this.
- `PIPELINE_WATCHDOG_SEC` (disabled by default): restart ffmpeg_service on sustained staleness.
- `REDIS_MAX_BACKOFF_SEC` (default 30) and `REDIS_INITIAL_BACKOFF_SEC` (default 0.5): reconnect strategy.

---

## 7. Production Deployment

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

## 8. Directory Layout (top-level)

```
.
├── docker/             # Dockerfiles & compose files (local & production)
├── stopsign/           # Application source code (Python)
├── models/             # Pre-downloaded YOLO models
├── volumes/            # Bind-mounted data for local development
├── static/             # UI assets, screenshots, Grafana dashboards
├── sample_data/        # Sample video used in local mode
├── docs/               # Design/architecture notes
└── README.md           # You are here 💁
