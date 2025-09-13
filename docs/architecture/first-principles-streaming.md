# StopSign AI Streaming Pipeline — First‑Principles Redesign

Status: proposal (kept on file for future adoption)

Owner: davidrose • Last updated: 2025‑09‑12


## Context & Goals

Product goal: a real‑time website that shows a stop‑sign camera feed with overlaid YOLO detections and tracks, plus persistent analytics/metrics in a database and good operational observability.

What exists today (high level):
- RTSP camera → `rtsp_to_redis` (encodes JPEG) → Redis list (`RAW_FRAME_KEY`).
- Video Analyzer (YOLO) → processes frames from Redis → writes JPEG to Redis list (`PROCESSED_FRAME_KEY`) + metadata to DB.
- FFmpeg service → consumes processed JPEGs from Redis → encodes HLS to disk → web server serves HLS; separate health endpoints.
- Web server → UI + REST endpoints; some health checks for HLS freshness and DB; OTEL telemetry overall.

This was a first production attempt outside core ML. It works, but complexity + fragile edges (networking, Redis blocking, separate compose stacks) produce intermittent alerts and occasional stalls.


## Current Architecture (as implemented)

Compose/topology (local and production variants):
- `docker/local/docker-compose.yml` runs `redis`, `postgres`, `minio`, `rtsp_to_redis`, `video_analyzer`, `ffmpeg_service`, `web_server` on one bridge network.
- `docker/production/docker-compose.yml` splits services across multiple networks (e.g., `coolify`, `monitoring`).

Primary data path:
1) Camera (RTSP) → `rtsp_to_redis/rtsp_to_redis.py` encodes frames to JPEG and `LPUSH`es into `RAW_FRAME_KEY` (Redis list) with `LTRIM` for a bounded buffer.
2) `stopsign/video_analyzer.py` `BLPOP`s from `RAW_FRAME_KEY`, runs YOLO + visualization, then `LPUSH`es a processed JPEG into `PROCESSED_FRAME_KEY` (again bounded) and writes per‑frame metadata to Redis/DB.
3) `stopsign/ffmpeg_service.py` `BLPOP`s from `PROCESSED_FRAME_KEY`, decodes JPEG → raw BGR → pipes raw frames to `ffmpeg` for HLS packaging to `/app/data/stream/*.m3u8/*.ts`.
4) Web server serves the HLS output; health checks validate HLS freshness and DB; OTEL instrumentation emits traces/metrics.

Observability:
- OTEL traces/metrics in `stopsign/telemetry.py`, plus ad‑hoc health endpoints in services.
- HLS freshness parsed by `stopsign/hls_health.py` (program date‑time, segment window, mtime).


## Pain Points and Bottlenecks

Failure modes observed in ops reports and code review:
- HLS freshness alerts: The FFmpeg service health probes flag HLS staleness (e.g., last `.ts` older than threshold), even while the lightweight HTTP health server is “up”. Root cause is usually the consumer loop stalling (Redis hiccup, decode edge case, ffmpeg stdin closed, etc.).
- Redis as frame transport: Using Redis lists as a frame bus couples several moving parts:
  - Blocking `BLPOP`/`BRPOP` loops can hang on network partitions; reconnection logic historically spotty.
  - FIFO/LIFO semantics must be carefully matched (producer LPUSH + consumer BRPOP is FIFO; mixed BLPOP/LPOP/LPUSH can surprise).
  - No visibility into lag/backpressure beyond `LLEN`; no per‑consumer state when scaling out.
- Many processes, many edges: RTSP capture, analyzer, FFmpeg encoder, and web server all in separate containers with different health semantics and cross‑network routing (in production). This multiplies “unknowns” during transient network or resource pressure.
- Server‑side compositing: Burning overlays server‑side (or assuming server will always output final composed frames) increases the critical path and tightens coupling; any hiccup in analyzer → no video on site.
- Alert flapping: HLS-only readiness conflates “video freshness” with “service liveness”; brief stalls or grace periods trigger alerts; little signal about which stage broke (RTSP, Analyzer, FFmpeg).

Where CPU/GPU and IO time goes:
- Double JPEG transcodes (RTSP→JPEG in producer; JPEG→BGR→HLS in FFmpeg) and repeated decode/encode steps add latency/CPU and create more points of failure.
- Disk IO for HLS segments is fine for scale=1 but not ideal for sub‑second latency or strict freshness SLOs.


## First‑Principles Redesign

Design principles:
- Offload media transport and packaging to a battle‑tested media hub.
- Keep Python focused on AI + metadata; do not move pixel buffers through Redis.
- Decouple “video playback” from “AI overlays” by rendering overlays in the browser.
- Express clear liveness vs readiness and stage‑specific stall detectors to avoid noisy alerts.

Proposed high‑level architecture:
1) Media Hub (MediaMTX)
   - Ingest RTSP from the camera.
   - Output WebRTC for sub‑second playback; HLS as a fallback.
   - Optional: accept a second input for a server‑burned overlay stream if ever needed.

2) Analyzer (single Python service)
   - Ingest directly from MediaMTX over RTSP using PyAV (or GStreamer) — no Redis for frames.
   - Run YOLO; produce structured metadata: timestamp, boxes, tracks, events.
   - Persist analytics to Postgres.
   - Publish real‑time metadata over WebSocket to the web app (or Redis Pub/Sub if multi‑consumer fanout is needed).

3) Web App
   - Play the MediaMTX stream (WebRTC preferred; HLS fallback).
   - Open a WebSocket to receive metadata; draw overlays client‑side on a `<canvas>` layered over the `<video>` element.
   - Maintain a small time offset (auto‑calibrated slider) to align metadata timestamps to video PTS/PDT.

Immediate simplifications:
- Remove the FFmpeg service and the RTSP→Redis frame shuttle entirely.
- Fewer containers and networks; fewer blocking queues; less disk churn.
- If the analyzer stalls, the video keeps playing; if the video stutters, metadata still updates — decoupled UX.


## Detailed Component Responsibilities

MediaMTX (media plane)
- RTSP ingest, auto‑reconnect, and rate control.
- WebRTC (preferred) and HLS outputs for browser clients.
- Optional DVR/recording if needed later.

Analyzer (compute plane)
- Decode frames with timestamps (frame.pts × time_base → UTC).
- YOLO inference, tracking, eventing (stop zone, dwell, violations).
- Persist to Postgres (tables: `detections`, `tracks`, `events`, `session_metrics`).
- Publish metadata over WebSocket (JSON schema suggested below).
- Expose `/healthz` (liveness: thread/process running) and `/ready` (recent frames < T and DB reachable) with clear timestamps.

Web App (UI plane)
- Subscribe to WebRTC/HLS stream from MediaMTX.
- Maintain `time_offset_ms` between server video time and client wallclock; align overlay paint loop via `requestAnimationFrame`.
- Show analytics overlays, debug toggles, and basic controls (offset slider, FPS stats).


## Message Schema (Analyzer → Web via WS)

```json
{
  "ts_utc": 1726179150.123,        // float seconds since epoch (frame capture time)
  "frame": 123456,                 // monotonically increasing frame counter (optional)
  "tracks": [
    { "id": 42, "bbox": [x1,y1,x2,y2], "speed": 12.3, "direction": 1.57, "is_parked": false }
  ],
  "zones": { "stop": [[x,y], [x,y], [x,y], [x,y]] },
  "raw_dimensions": {"w": 1920, "h": 1080},
  "scaled_dimensions": {"w": 1280, "h": 720}
}
```


## Migration Plan (Incremental)

Step 1 — Stand up MediaMTX
- Add MediaMTX container to compose; point camera to it; validate browser playback via WebRTC and HLS.

Step 2 — Switch web playback
- Update the web app to consume the MediaMTX stream (no overlays yet). Keep the existing HLS until WebRTC is verified.

Step 3 — Refactor analyzer input/output
- Replace `BLPOP RAW_FRAME_KEY` with PyAV RTSP reader from MediaMTX.
- Remove Redis frame I/O from analyzer; keep DB writes.
- Add WebSocket server and start publishing metadata.

Step 4 — Client overlays
- Implement `<video>` + `<canvas>` with WS metadata overlay; add a small UI to tune `time_offset_ms` and show FPS/lag.

Step 5 — Decommission old services
- Remove `rtsp_to_redis` and `ffmpeg_service` containers.
- Keep DB and telemetry unchanged; keep HLS fallback via MediaMTX.


## Operational Model

Health checks
- Liveness (`/healthz`): process/thread is alive — do not include external dependencies.
- Readiness (`/ready`): AND of recent frames (< T), DB reachable, internal error budget not exceeded.
- MediaMTX: rely on its native health and auto‑reconnect (export metrics if available).

Alerting (reduce flapping)
- Page only when readiness fails for > 5 minutes or restarts exceed X in 30 minutes.
- Warn on short staleness or brief reconnects; no pages.

Watchdogs
- Analyzer stall watchdog: if no frames decoded for `N` seconds → exit(1) to let orchestrator restart.
- WebSocket backpressure checks: monitor queue depth; drop oldest messages past a threshold to preserve recency.

Observability
- Continue with OpenTelemetry: emit spans/metrics for `decode`, `infer`, `db_write`, `ws_broadcast`.
- Emit gauges/attributes for: `last_decode_ts`, `last_infer_ts`, `ws_queue_depth`, `video_pts_wallclock_delta`.
- Dashboard “staleness chain”: camera ingest → decode → infer → WS emit → browser render; show deltas to isolate the breaking link quickly.


## Tooling Alternatives & Trade‑offs

Baseline (recommended): MediaMTX + PyAV + WebRTC/HLS + WS overlays
- Pros: minimal moving parts, robust media transport, lowest code; browser can scale overlays; decouples failures.
- Cons: two moving systems (media hub and analyzer); requires client overlay rendering (JS/Canvas).

Single‑process GStreamer pipeline
- `rtspsrc ! decode ! appsink -> (YOLO) -> appsrc ! encoder ! webrtcbin/hlssink`
- Pros: one process, lowest latency; excellent for single‑box edge deployments (systemd supervised).
- Cons: steeper learning curve, more glue in Python; still need a simple signaling server for WebRTC.

NVIDIA DeepStream (GPU servers)
- Pros: production‑grade decode/infer/OSD/encode with HW acceleration; great at multi‑camera scale.
- Cons: heavier setup; vendor‑specific; UI customization often more involved.

Redis Streams (if you must keep Redis in the data path)
- Use `XADD MAXLEN ~ N` + consumer groups (XREADGROUP) instead of lists.
- Pros: durable offsets, lag visibility, backpressure.
- Cons: still more moving parts than a media hub; higher latency than direct RTSP decode.


## Example Compose Snippet (Conceptual)

```yaml
services:
  mediamtx:
    image: bluenviron/mediamtx:latest
    ports: ["8554:8554", "8889:8889", "8123:8123"] # RTSP, WebRTC, API (example)
    volumes:
      - ./mediamtx.yml:/mediamtx.yml:ro

  analyzer:
    build: ./analyzer
    environment:
      RTSP_URL: rtsp://mediamtx:8554/camera
      DB_URL: ${DB_URL}
    depends_on: [mediamtx]

  web:
    build: ./web
    ports: ["8000:8000"]
    environment:
      STREAM_URL: webrtc://mediamtx:8889/camera # or HLS fallback
      WS_URL: ws://analyzer:9000/metadata
    depends_on: [analyzer, mediamtx]
```


## Database Sketch

Tables (simplified):
- `detections(id, ts_utc, track_id, bbox, speed, direction, is_parked, frame_id)`
- `tracks(id, first_ts, last_ts, summary)`
- `events(id, ts_utc, kind, track_id, payload_json)`  // e.g., stop‑zone entry/exit
- `session_metrics(ts_utc, fps_infer, latency_ms_p50, latency_ms_p95, gpu_util, cpu_util)`


## Security & Networking

- Single Docker network for all services to avoid DNS/cross‑network surprises.
- WebRTC DTLS/SRTP handled by MediaMTX; terminate TLS at the reverse proxy for the web app and WS endpoints.
- Secrets via env or a secret manager; avoid baking into images.
- Rate‑limit and auth for WS connections if externalized.


## Risks & Open Questions

- Browser overlay fidelity vs server‑rendered overlays: client GPUs vary; most are fine at 1080p/60 with simple canvas.
- Multi‑camera scale: MediaMTX handles N inputs; analyzer instances can scale horizontally; WS fanout becomes relevant.
- Historical video with overlays: client‑side is easiest for live; for archive playback, server‑burned VOD may be preferable.


## Decision & Next Steps

This document captures a simpler architecture that eliminates the custom FFmpeg and Redis frame shuttle, favoring a media hub and client overlays. We’ll keep it as a staged migration option. If we choose to proceed:
1) Pilot MediaMTX alongside the current stack; validate browser playback.
2) Build analyzer PyAV input + WS metadata prototype.
3) Implement client overlays and tune time offset.
4) Plan cutover; decommission old services.

Appendix A: Keeping Today’s Stack Stable in the Interim
- We’ve added resiliency to `stopsign/ffmpeg_service.py`: Redis backoff, FIFO BRPOP, `/ready` endpoint with composite checks, optional watchdog, and status integration.
- If we stick with Redis for a while longer, consider migrating lists → Streams and moving all consumers to `XREADGROUP` with lag metrics and dead‑letter queues.

