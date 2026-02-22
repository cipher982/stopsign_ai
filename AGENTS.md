# Stop Sign AI - Deployment & Architecture

## Deployment Overview
This service is deployed to **cube** via Coolify and runs in multiple Docker containers:
1. **rtsp_to_redis** - Standalone service that captures RTSP stream and pushes frames to Redis
2. **Main compose stack** (docker/production/docker-compose.yml):
   - `video_analyzer` - AI detection and tracking (GPU-accelerated)
   - `ffmpeg_service` - HLS stream generation for web viewing
   - `web_server` - FastAPI web interface and API

## Data Persistence Architecture

### PostgreSQL Database (clifford, NOT cube)
- **Location**: `5.161.97.53:5432/stopsign`
- **Container**: Find with `ssh clifford "docker ps | grep postgres"` (Coolify hash changes on deploy)
- **Tables**:
  - `vehicle_passes` - Core metrics (timestamp, speed, stop duration, time_in_zone, image_path)
  - `car_state_history` - Full tracking history per vehicle
  - `config_settings` - Dynamic configuration with version history
- **Current Data**: ~41k passes as of Dec 2025 (query `SELECT COUNT(*) FROM vehicle_passes` for current)
- **Environment Var**: `DB_URL` in cube containers points to clifford's PostgreSQL

### MinIO S3 Storage (clifford, NOT cube)
- **Endpoint**: `minio-nwcs0c4g0w8gcgow0gscgckg.5.161.97.53.sslip.io`
- **Public URL**: `https://api.files.drose.io`
- **Container**: Find with `ssh clifford "docker ps | grep minio"` (Coolify hash changes on deploy)
- **Bucket**: `vehicle-images`
- **Storage**: ~1.5 GiB, 119k+ images as of Dec 2025 (use `mc ls` for current)
- **Format**: Cropped vehicle JPEGs named `vehicle_{uuid}_{timestamp}.jpg`
- **Image Path in DB**: Stored as `minio://vehicle-images/vehicle_*.jpg`

### Data Flow
1. **RTSP Camera** → rtsp_to_redis container → **Redis** (raw frames on cube)
2. **video_analyzer** reads frames from Redis → YOLO detection → tracking logic
3. When vehicle exits stop zone:
   - Cropped vehicle image → **MinIO** (stopsign/tracking.py:547-593)
   - Metrics record → **PostgreSQL** (stopsign/database.py:153-169)
   - Both on **clifford** via network calls from cube
4. **web_server** queries PostgreSQL + serves images via MinIO public URL

## HLS Stream Architecture

### Internal Pipeline (cube)
```
Camera (WiFi) → RTSP → rtsp_to_redis → Redis → ffmpeg_service → HLS files → web_server
```

- **rtsp_to_redis**: Captures RTSP stream, pushes raw frames to Redis
- **ffmpeg_service**: Reads from Redis, generates HLS segments (10 segments × 2 seconds each)
- **web_server**: Serves HLS playlist at `/stream/stream.m3u8`

### External Access (as of Jan 2026)

| Path | URL | Latency |
|------|-----|---------|
| **Direct** (recommended) | `http://stream.crestwoodstopsign.com:8443/stream/stream.m3u8` | ~70ms |
| Via Cloudflare Tunnel | `https://crestwoodstopsign.com/stream/stream.m3u8` | 4-14s |

**Direct access setup:**
- **DNS**: `stream.crestwoodstopsign.com` → home IP (Cloudflare DNS-only, grey cloud, NOT proxied)
- **Router**: AT&T port forward 8443 → cube:8002 (TCP)
- **Why**: Cloudflare Tunnel relays all traffic through their network, adding unacceptable latency for video streaming

**Hybrid architecture:**
- Main website (`crestwoodstopsign.com`) → Cloudflare Tunnel (security, no exposed ports)
- Video stream → Direct port forward (performance, ~70ms vs 4-14s)

**Used by:** Sauron's `crestwood-camera` job for AI vision monitoring every 15 minutes.

## Key Implementation Details

### Image Capture
- Images captured at "capture line" crossing (configurable via debug UI)
- 10% padding added around vehicle bounding box
- Uploaded synchronously to MinIO during vehicle pass processing
- Path stored in database as `image_path` field for 1:1 mapping

### Configuration Management
- Config file: `/app/config/config.yaml` (mounted volume in containers)
- Dynamic updates via web UI stored in `config_settings` table with full history
- Stop zones, lines, thresholds all configurable and versioned

### Redis Usage
- Ephemeral frame buffer only (no persistence)
- Keys: `raw_frames` (RTSP input), `processed_frames` (annotated output)
- Used for inter-container communication on cube

## Troubleshooting Data Issues

**PostgreSQL is host-level on clifford, NOT in a container.** The postgres containers visible via `docker ps` on clifford are for traccar and umami — not stopsign. Do not use them.

```bash
# Get credentials from the running web_server container on cube
ssh cube 'docker exec $(docker ps --format "{{.Names}}" | grep web_server) sh -c "echo \$DB_URL"'
# Returns: postgresql://postgres:<password>@clifford.coin-castor.ts.net:5432/stopsign

# Query directly from local machine (Tailscale required)
PGPASSWORD="<from above>" psql -h clifford.coin-castor.ts.net -p 5432 -U postgres -d stopsign
```

- MinIO access: Use mc client or web console at MinIO endpoint
- All credentials in environment variables (see docker/production/.env on cube's Coolify deployment)

## Speed Tracking Implementation

### Speed Pipeline (tracking.py)
Speed goes through three layers before being stored:
1. **Median over last 6 raw YOLO positions** → kills single-frame bbox outliers
2. **EMA α=0.5** → `raw_speed` (lighter smoothing, used for stop detection)
3. **EMA α=0.3** → `speed` (heavier lag, used for display/parked logic)

`min_speed` stored in DB = **5th percentile of all `raw_speed` samples** collected while the car was in the stop zone.

### Kalman Filter Limitation
The Kalman filter smooths `state.location` (used for display/visualization) but **speed is computed from `state.track` which stores raw YOLO positions** (tracking.py:134). The filter has no effect on speed measurement. Smoothing is handled by the median + EMA pipeline above.

Potential fix: append Kalman-smoothed position to `state.track` instead of raw. Not yet implemented.

### Key Thresholds (config/config.yaml)
- `stop_speed_threshold: 20` — raw_speed ≤ 20 px/s counts toward `stop_duration`
- `max_movement_speed: 20` — same threshold for "stationary" state
- `unparked_speed_threshold: 30` — speed to trigger unparked transition

### Data Quality — What to Filter
When doing any analysis on `vehicle_passes`:
- **Exclude `time_in_zone >= 30`** (191 records as of Feb 2026): parked/street-parked cars where the 60s zone timeout didn't fire. These have `min_speed ≈ 0` and `time_in_zone` of hours — tracking artifacts, not real passes.
- **Anomaly window `2026-02-19 18:00` – `2026-02-21 14:00`**: Stop zone was misconfigured (placed in intersection). Feb 20 has zero records; surrounding hours have elevated speed medians. Only 39 affected records.

### Calibrated Stop Thresholds
From parked-car noise floor analysis (Feb 2026, ~53k passes — see `docs/analysis/2026-02-21-stop-calibration.md`):
- **Noise floor ceiling: ~6 px/s** (parked car p95 < 2.3 px/s, max 6.1 px/s)
- **`min_speed < 10 px/s`** = hardware-calibrated "stopped" signal (sits clearly above noise, below the bimodal dip at 10–15 px/s)
- **Full stop definition**: `min_speed < 10 AND time_in_zone >= 3s` ≈ 10.8% of clean traffic
- Do NOT use speed alone for scoring — `time_in_zone` is the primary ranking signal; `min_speed` is a binary gate

## Google Ads Campaign

**Site**: https://crestwoodstopsign.com

### Campaign Details (Dec 2025)
- **Campaign ID**: `23374137482`
- **Ad Group ID**: `193403091587`
- **Name**: Stop Sign Nanny - DIY Hackers
- **Budget**: $1/day (~$30/month)
- **Strategy**: Search only, exact match, manual CPC
- **Landing Page**: /about

### Keywords (19 exact match, 3 audience segments)
- **DIY/Maker**: raspberry pi traffic camera, raspberry pi yolo, diy ai camera, yolo live demo, computer vision live stream, traffic camera diy project, ai traffic camera project, home traffic camera ai, diy object detection
- **Frustrated Neighbor**: cars running stop signs, stop sign violations neighborhood, do cars stop at stop signs, traffic safety residential
- **Data/Voyeur**: traffic pattern analysis, vehicle counting camera, intersection traffic data, live traffic camera, real time street camera, watch traffic live

### Common Commands
```bash
cd ~/git/google-ads-cli

# Check campaign performance
uv run ads report summary --period week
uv run ads keywords list --campaign-id 23374137482

# Pause/enable campaign
uv run ads campaigns pause 23374137482
uv run ads campaigns enable 23374137482

# Add more keywords
uv run ads keywords add 193403091587 "new keyword here" --match-type exact
```

### Notes
- No conversion tracking (awareness campaign only)
- To add gtag.js later: follow HDRPop pattern in ~/git/hdr (GOOGLE_ADS_ID env var + gtag snippet)

## SEO & Structured Data

**Implemented Dec 2025:**
- **Meta tags**: Description, Open Graph, Twitter Cards in `stopsign/components.py` (`page_head_component`)
- **JSON-LD**: WebSite, Person, WebApplication, VideoObject, Dataset schemas (same file)
- **robots.txt**: `/static/robots.txt` served at `/robots.txt`
- **sitemap.xml**: `/static/sitemap.xml` served at `/sitemap.xml`
- **Google Search Console**: Sitemap submitted, 3 pages discovered

Routes for SEO files defined in `stopsign/web_server.py` (search for `robots.txt`).