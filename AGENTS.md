# Stop Sign AI - Deployment & Architecture

## Deployment Overview
Production: two Docker tracks on **cube**:
1. **rtsp_to_redis** Compose at `/home/drose/manual-apps/stopsign_ai/rtsp_to_redis` — captures RTSP stream, pushes frames to Redis. Capture runs `restart: always` (both stacks do).
2. **Main stack** — estate manual-app `stopsign` (manifest `~/git/me/domains/mytech/infrastructure/manual-apps/stopsign/app.json`): checkout `/home/drose/manual-apps/stopsign/repo`, compose `docker/production/docker-compose.yml`, compose project `is844go80k088kcgo88s8cs8` (preserves volumes):
   - `video_analyzer` - GPU AI detection/tracking
   - `ffmpeg_service` - HLS segment generation for web
   - `web_server` - FastAPI web interface + API

Deploy: commit+push, then `~/git/me/domains/mytech/bin/manual-app deploy stopsign --repo-dir ~/git/stopsign_ai` (refuses unpushed/dirty HEAD; verifies remote-local + public `/healthz`). YOLO weights live in `models/` (gitignored); keep them on cube, don't commit.

## Frontend Design System
"Field Instrument" (Aug 2026, replaced the win98 theme): dark asphalt ground, verdict semantics green/amber/red (`--ok/--warn/--bad`) as the ONLY data colors, magenta `--accent` as identity-only, Overpass + Overpass Mono type, verdict stamp chips (FULL STOP / ROLLING STOP / NO STOP) site-wide. Tokens in `static/base.css`; page CSS in each template's style block. Rules: evidence images always `object-fit: contain` on `--well` (cover re-crops the tight capture crops); thumbnails served by `/vehicle-thumb/{obj}?v=<variant>` — bump `THUMBNAIL_VARIANT` (stopsign/web/services/images.py) on any rendering change because Cloudflare caches immutably by URL.

## Data Persistence

### PostgreSQL (shared container on clifford, NOT cube)
- **Location**: `clifford.coin-castor.ts.net:5432/stopsign` (Tailscale TCP)
- **Container**: `kgcos0o4cw4ok0ss0g08wswo` — one retained PostgreSQL 16 daemon shared with Collector; the empty legacy `rag` DB was retired 2026-07-29
- **Tables**: `vehicle_passes` (timestamp, speed, stop duration, time_in_zone, image_path), `vehicle_pass_raw` (per-pass raw evidence incl `capture` bbox/crop geometry since Aug 2026), `vehicle_labels` (per-pass gpt-5.6-luna closed-vocab labels, Aug 2026, 67.7k rows ≈ 99% of imaged passes), `vehicle_attributes` (legacy cluster pipeline, frozen Feb 2026 — still feeds /vehicles), `config_settings` (versioned config, full history)
- **Data**: ~73k passes as of Aug 2026 (`SELECT COUNT(*) FROM vehicle_passes`)
- **Labeling**: `scripts/label_passes.py` — resumable, idempotent, 50-way concurrent; ~$0.0004/pass; rerun anytime to label new passes (needs DB_URL, OPENAI_API_KEY, BREMEN_MINIO_* env)
- **Env**: Infisical `DB_URL` injects restricted owner/login `stopsign_app` into analyzer + web. The old shared-superuser URL is a bounded rollback artifact through 2026-08-05, not an app credential.

### MinIO S3 (clifford, NOT cube)
- **Endpoint**: `minio-nwcs0c4g0w8gcgow0gscgckg.5.161.97.53.sslip.io`; public `https://api.files.drose.io`
- **App**: estate `object-storage` on clifford (`/home/drose/manual-apps/object-storage`, compose project `object-storage`, stable container `minio`)
- **Bucket**: `vehicle-images` — ~1.5 GiB / 119k+ images as of Dec 2025 (`mc ls` for current)
- **Format**: cropped JPEGs `vehicle_{uuid}_{timestamp}.jpg`; DB `image_path` = `minio://vehicle-images/vehicle_*.jpg`

### Data Flow
1. RTSP camera → rtsp_to_redis → **Redis** (raw frames, cube)
2. `video_analyzer`: Redis frames → YOLO → tracking logic
3. Stop-zone exit: crop → **MinIO** (`stopsign/tracking.py:547-593`); metrics → **PostgreSQL** (`stopsign/database.py:153-169`) — both network calls from cube to clifford
4. `web_server` queries PostgreSQL, serves images via MinIO public URL

## HLS Stream

```
Camera (WiFi) → RTSP → rtsp_to_redis → Redis → ffmpeg_service → HLS → web_server
```
- `ffmpeg_service`: 10 segments × 2s; `web_server` serves `/stream/stream.m3u8`

External access (Jan 2026):

| Path | URL | Latency |
|------|-----|---------|
| **Direct** (recommended) | `http://stream.crestwoodstopsign.com:8443/stream/stream.m3u8` | ~70ms |
| Cloudflare Tunnel | `https://crestwoodstopsign.com/stream/stream.m3u8` | 4-14s |

Direct setup: DNS `stream.crestwoodstopsign.com` → home IP (Cloudflare grey cloud, NOT proxied); AT&T router port-forwards 8443 → cube:8002 (TCP). Tunnel relay adds unacceptable video latency.
Hybrid: main site behind Cloudflare Tunnel (no exposed ports); stream direct (70ms vs 4-14s).
Used by Sauron's `crestwood-camera` job (AI vision, every 15 min).

## Implementation Details
- **Capture**: images at "capture line" crossing (configurable via debug UI), 40% bbox padding since Aug 2026 (broad archival crop; detection bbox + crop rect recorded in `vehicle_pass_raw.capture` for later tight re-cropping), synchronous local save + async Bremen upload, 1:1 `image_path` mapping
- **Config**: `/app/config/config.yaml` (volume-mounted); dynamic web-UI updates → `config_settings` (versioned stop zones, lines, thresholds)
- **Redis**: ephemeral frame buffer only (no persistence); `raw_frames` (in), `processed_frames` (out); inter-container comms on cube

## Troubleshooting Data
PostgreSQL is in the shared clifford container, not a native host service. `DB_URL` is injected, not visible via `docker exec env`.
```bash
# Get creds from Infisical (Tailscale required for query)
python3 ~/git/me/scripts/infisical-get.py DB_URL --project-id 9c373776-768f-454b-a7b3-d1cc40deb475 --env prod
# → postgresql://stopsign_app:<password>@clifford.coin-castor.ts.net:5432/stopsign
PGPASSWORD="<above>" psql -h clifford.coin-castor.ts.net -p 5432 -U stopsign_app -d stopsign
```
- MinIO: `mc` client or web console at endpoint
- Creds are server-side env vars: RTSP at `/home/drose/manual-apps/stopsign_ai/rtsp_to_redis/.env`; main stack at `/home/drose/manual-apps/stopsign_ai/docker/production/.env`

## Speed Tracking

Pipeline (tracking.py), layers before storage:
1. **Median of last 6 raw YOLO positions** → kills single-frame bbox outliers
2. **EMA α=0.5** → `raw_speed` (used for stop detection)
3. **EMA α=0.3** → `speed` (heavier lag, display/parked logic)

`min_speed` stored in DB = **5th percentile of all `raw_speed`** samples while car in stop zone.

### Kalman Limitation
Kalman smooths `state.location` (display) but speed = `state.track` (raw YOLO positions, tracking.py:134) — filter has no effect on speed; smoothing is the median+EMA pipeline above. Possible fix: append Kalman-smoothed position to `state.track`. Not implemented.

### Thresholds (config/config.yaml)
- `stop_speed_threshold: 20` — raw_speed ≤ 20 px/s counts toward `stop_duration`
- `max_movement_speed: 20` — same threshold for "stationary"
- `unparked_speed_threshold: 30` — triggers unparked transition

### Data Quality — What to Filter
- **Exclude `time_in_zone >= 30`** (191 records, Feb 2026): parked cars where the 60s zone timeout didn't fire; `min_speed ≈ 0`, hours-long zones — tracking artifacts
- **Anomaly window `2026-02-19 18:00` – `2026-02-21 14:00`**: stop zone misconfigured (in intersection); Feb 20 zero records; 39 affected records
- **Calibrated stops** (parked-car noise floor, Feb 2026, ~53k passes — `docs/analysis/2026-02-21-stop-calibration.md`): noise floor ~6 px/s (parked p95 < 2.3, max 6.1); `min_speed < 10 px/s` = hardware-calibrated stop (above noise, below bimodal 10-15 dip); full stop = `min_speed < 10 AND time_in_zone >= 3s` ≈ 10.8% of clean traffic
- Score on `time_in_zone` (primary); `min_speed` is a binary gate — don't use speed alone

## Google Ads

Campaign **Stop Sign Nanny - DIY Hackers**: ID `23374137482`, ad group `193403091587`. Search only, exact match, manual CPC, $1/day (~$30/mo), landing `/about`. No conversion tracking (awareness only).

Keywords (19 exact, 3 audience segments):
- **DIY/Maker**: raspberry pi traffic camera, raspberry pi yolo, diy ai camera, yolo live demo, computer vision live stream, traffic camera diy project, ai traffic camera project, home traffic camera ai, diy object detection
- **Frustrated Neighbor**: cars running stop signs, stop sign violations neighborhood, do cars stop at stop signs, traffic safety residential
- **Data/Voyeur**: traffic pattern analysis, vehicle counting camera, intersection traffic data, live traffic camera, real time street camera, watch traffic live

```bash
cd ~/git/google-ads-cli
uv run ads report summary --period week
uv run ads keywords list --campaign-id 23374137482
uv run ads campaigns pause 23374137482
uv run ads campaigns enable 23374137482
uv run ads keywords add 193403091587 "new keyword here" --match-type exact
```
Later gtag.js: follow HDRPop pattern in `~/git/hdr` (GOOGLE_ADS_ID env var + gtag snippet).

## SEO
Dec 2025: meta (description, Open Graph, Twitter Cards) + JSON-LD (WebSite, Person, WebApplication, VideoObject, Dataset) in `stopsign/components.py` (`page_head_component`); `/static/robots.txt` → `/robots.txt`; `/static/sitemap.xml` → `/sitemap.xml`; Google Search Console sitemap submitted, 3 pages. Routes in `stopsign/web_server.py` (search `robots.txt`).