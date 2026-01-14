# StopSign AI — Decoupled Video Playback + AI Overlays (MediaMTX + WebRTC/LL‑HLS + WS Metadata)

Status: **Phase 3 complete** (analyzer can read from MediaMTX RTSP)

Owner: davidrose • Last updated: 2026-01-13

## Implementation Progress

| Phase | Status | Notes |
|-------|--------|-------|
| 0 | ✅ Complete | Baseline latency documented (~20s HLS, ~33s Redis backlog) |
| 1 | ✅ Complete | MediaMTX + video_publisher in docker-compose, LL-HLS working |
| 2 | ✅ Complete | VIDEO_SOURCE config, multi-source video-player.js |
| 3 | ✅ Complete | FrameSource abstraction, FRAME_SOURCE config, analyzer uses RTSPFrameSource |
| 4 | 🔲 Pending | WebSocket metadata feed + browser canvas overlays |
| 5 | 🔲 Pending | Decommission rtsp_to_redis, ffmpeg_service |

### Files changed (Phases 1-3)
- `docker/local/docker-compose.yml` - Added mediamtx, video_publisher services; env vars for video/frame sources
- `docker/local/.env.example` - Documented VIDEO_SOURCE, FRAME_SOURCE, MEDIAMTX_* settings
- `stopsign/settings/__init__.py` - Added VIDEO_SOURCE, MEDIAMTX_HLS_URL, MEDIAMTX_WEBRTC_URL, FRAME_SOURCE, ANALYZER_RTSP_URL
- `static/js/video-player.js` - Rewritten for multi-source (legacy_hls, mediamtx_hls, mediamtx_webrtc)
- `stopsign/components.py` - Injects video config JSON for frontend
- `stopsign/frame_source.py` - **New**: FrameSource ABC, RedisFrameSource, RTSPFrameSource
- `stopsign/video_analyzer.py` - Uses frame_source abstraction
- `stopsign/web_server.py` - Health checks now conditional on VIDEO_SOURCE

### Known limitations (to address before production)

1. **WebRTC implementation is basic**: The current `mediamtx_webrtc` mode uses a minimal WHEP client without trickle ICE or session teardown. This works on friendly networks but may be brittle with NAT/firewalls. For production, consider using MediaMTX's version-matched `reader.js` or implementing full trickle ICE.

2. **Timestamp semantics differ**: `FRAME_SOURCE=rtsp` uses read-time timestamps (not camera capture time). See "Timestamp semantics" section below for details and implications.

3. **Legacy pipeline still required for output**: The analyzer still writes processed JPEGs to Redis, and `ffmpeg_service` still produces `/stream/stream.m3u8`. Full decouple (browser canvas overlays) is Phase 4.


## Why this doc exists

The current pipeline works, but it tightly couples “video playback” to “AI processing” and moves full frames through Redis and multiple JPEG transcodes. That creates avoidable latency (~20s today) and a lot of moving parts that can stall.

This document proposes a **first‑principles simplification**:

1. Put a stable, purpose‑built **media plane** in charge of RTSP ingest and browser playback.
2. Make the Python “AI plane” do only detection/tracking/eventing + persistence.
3. Render boxes/zones in the browser using a WebSocket metadata feed.

If the analyzer is down or redeploying, the site continues streaming raw video (“AI off is fine”).


## Goals (explicit)

- Keep a **single stable ingest** component that is rarely redeployed (historically the RTSP camera was flaky).
- Reduce end‑to‑end latency from “~20 seconds behind” to **as low as practical**.
- Keep **inference + tracking on the server** (no model in the browser).
- Allow **video playback to continue** even if AI is restarting or temporarily unhealthy.
- Preserve the existing persistence model:
  - Postgres `vehicle_passes`, `car_state_history`
  - MinIO `vehicle-images` objects referenced as `minio://bucket/object`
- Avoid big‑bang changes; migrate in phases with a clear rollback path.

## Non‑goals

- Multi-camera support.
- Rewriting tracking logic or the database schema (unless explicitly needed).
- Pixel‑perfect frame‑accurate overlay alignment on day 1 (we’ll design for it, but ship an MVP first).

## Current deployment context (as of today)

This matters because “WebRTC low-latency” can be blocked by NAT/ports, and because the existing pipeline spans multiple hosts.

- The app is deployed to **cube** via Coolify.
- Redis is on **cube** (frame buffer today).
- Postgres and MinIO are on **clifford**; cube services reach them over the network (`DB_URL`, `MINIO_*`).
- Production is currently split across multiple compose stacks / networks:
  - `rtsp_to_redis/docker-compose.yml` (stable ingest service)
  - `docker/production/docker-compose.yml` (analyzer + ffmpeg + web)

The migration plan below keeps this spirit: **MediaMTX becomes the new “stable ingest service”** that you almost never touch, while the analyzer + UI can be redeployed freely.


## 0→1 review of the current pipeline (what exists today)

### Services (current)

| Service | Responsibility | Transport |
|---|---|---|
| `rtsp_to_redis` | OpenCV reads RTSP (or `file://`), stamps capture time, JPEG‑encodes, pushes frames into Redis | RTSP → Redis list |
| `video_analyzer` | YOLO + tracking + stop-sign logic; burns overlays into pixels; uploads vehicle crops to MinIO; writes passes to Postgres; writes per-frame metadata to Redis | Redis list → AI → Redis list (+ DB/MinIO) |
| `ffmpeg_service` | Reads processed JPEGs from Redis, decodes to raw BGR, pipes to `ffmpeg` to generate HLS segments | Redis list → `ffmpeg` → HLS on disk |
| `web_server` | Serves HLS + UI; reads Postgres + MinIO; debug UI writes config.yaml and pokes Redis flags | HTTP + HLS |
| Redis / Postgres / MinIO | Infra | — |

### Frame format

Raw frames (on `RAW_FRAME_KEY`) use a custom SSFM header that carries capture timestamp:

- `rtsp_to_redis` packs: `b"SSFM" + version + json_len + json + jpeg`
- `video_analyzer` requires SSFM + `ts` (it drops frames without `ts`)

This is a real requirement: **stop timing should be based on camera capture time**, not “whatever time the analyzer happened to run”.

### Queue semantics

- RAW: `LPUSH RAW_FRAME_KEY` and analyzer uses `BRPOP` ⇒ FIFO (oldest first).
- PROCESSED: `LPUSH PROCESSED_FRAME_KEY` and encoder uses `RPOP` ⇒ FIFO (oldest first).
- Both queues are bounded via `LTRIM`.

### Configuration / debug control plane

- The single source of truth is `config/config.yaml` (mounted into containers).
- Debug UI updates the YAML atomically and then sets Redis key `config_updated=1` for “instant reload”.
- Debug zones are toggled via Redis key `debug_zones_enabled`.

### Latency: why you’re ~20s behind right now

The latency isn’t “mysterious”; it’s mostly baked in:

1. HLS window size in `stopsign/ffmpeg_service.py`:
   - `-hls_time 2` and `-hls_list_size 10` ⇒ ~20 seconds of segments in the playlist.
2. Player tolerance in `static/js/video-player.js`:
   - `liveMaxLatencyDurationCount: 10` ⇒ allows being ~10 segments behind (~20s at 2s segments).
3. Redis buffers are huge for “near-live”:
   - `FRAME_BUFFER_SIZE=500` at 15 FPS ⇒ ~33 seconds of possible backlog (per queue).
4. The encoder “stability fix” duplicates frames to maintain 15 FPS output when input is sparse.
   - That improves “continuous video” but can mask upstream stalls and doesn’t itself reduce latency.

### Coupling (what’s hard today)

Video playback depends on the entire chain:

RTSP → Redis RAW → Analyzer → Redis PROCESSED → ffmpeg → HLS → Web UI

If any stage stalls, the website video is stale. That’s why the current system needs lots of health checks, watchdogs, and reconnect logic.

### Hidden landmines worth noting

- `web_server` reads “latest frame metadata” with `redis.keys("frame_metadata:*")` which is not a safe pattern at scale (and ordering isn’t guaranteed).
- `rtsp_to_redis` logs env vars verbatim; RTSP URLs often embed credentials.
- `docker/local/docker-compose.prod-view.yml` contains what appear to be production credentials. Treat that file as a security incident and remove/rotate as needed before making anything more public.


## Target architecture (first principles)

Split the system into 3 planes:

1. **Media plane (stable):** MediaMTX ingests RTSP and serves browser playback (WebRTC for lowest latency; LL‑HLS as a fallback).
2. **AI plane (redeployable):** Analyzer reads frames from MediaMTX, runs YOLO/tracking, persists events (Postgres/MinIO), publishes overlay metadata.
3. **UI plane (redeployable):** Web server serves HTML/JS; browser draws boxes/zones from metadata.

### What this buys you

- The camera connection logic lives in a purpose‑built media router instead of your app.
- Redis is no longer a frame bus.
- `ffmpeg_service` disappears (MediaMTX does protocol conversion and low‑latency HLS; WebRTC for best latency).
- Analyzer crashes no longer kill the live video.

### New data path (conceptual)

```mermaid
graph LR
  Cam[RTSP Camera] -->|RTSP| M[MediaMTX]
  M -->|WebRTC or LL-HLS| Browser[Browser Video Playback]
  M -->|RTSP pull| A[Analyzer: YOLO + Tracking]
  A -->|WS metadata| Browser
  A -->|vehicle_passes| PG[(Postgres)]
  A -->|vehicle crops| S3[(MinIO)]
  UI[Web Server] -->|HTML/JS| Browser
```

Key behavior change: **If the analyzer is down, the video still plays.** Overlays disappear temporarily, which is acceptable.


## Major decision points (don’t assume — verify)

### 1) Camera codec / browser support

Your current pipeline always outputs browser-friendly H264 via `ffmpeg_service`.

MediaMTX does **not** transcode; it remuxes/proxies.

So before changing playback, verify what the camera actually emits:

- If the camera can output **H264**, great: you can usually play via WebRTC/LL‑HLS without extra transcode.
- If it outputs **H265**:
  - Safari can play H265 in WebRTC; Chrome often cannot (without flags/GPU).
  - You may need to change camera settings or add a dedicated **transcode-to-MediaMTX** sidecar (still simpler than the current pipeline).

Concrete “no guessing” checks:

- From any machine with ffmpeg installed:

  ```sh
  ffprobe -hide_banner -rtsp_transport tcp -i "rtsp://USER:PASS@CAMERA_HOST:554/whatever"
  ```

  Look for `Video: h264` vs `Video: hevc` (H265).

- From MediaMTX logs:
  - it prints the incoming track codecs when a path starts.

If you must transcode for browser support, prefer a stable sidecar that publishes H264 into MediaMTX:

```sh
ffmpeg -rtsp_transport tcp -i "rtsp://USER:PASS@CAMERA_HOST:554/whatever" \
  -an \
  -c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency -g 30 \
  -f rtsp "rtsp://MEDIAMTX_HOST:8554/camera"
```

This keeps the “stable media plane” concept: camera flakiness + codec mismatch are handled outside your app.

### 2) Networking for WebRTC

WebRTC gives the best latency, but it needs UDP connectivity (and sometimes STUN/TURN).

MediaMTX docs emphasize:

- Configure `webrtcAdditionalHosts` (LAN + public hostnames/IPs).
- Expose UDP port `8189/udp` (and optionally TCP fallback if UDP is blocked).

If WebRTC is hard in your deployment environment, **LL‑HLS is a very good intermediate step** (typically ~0.5–3s latency when tuned).

### 3) Timestamp alignment (nice to have vs required)

For a first ship:

- Show “latest overlays” (not frame-perfect) and keep latency low by dropping old metadata.

For a more precise overlay:

- Preserve absolute timestamps (`useAbsoluteTimestamp`) and align overlays to WebRTC playout time via `RTCRtpReceiver.getStats()` and `estimatedPlayoutTimestamp` (where supported).

This is doable, but it’s not the first dependency to unblock the architecture.


## Proposed migration plan (phased, reversible)

This is designed so you can stop after any phase with a working system.

### Phase 0 — Baseline + guardrails (do before changing anything)

1) **Measure current real latency**

- From the outside: compare “wall clock” to what you see on the overlay timestamp (if shown).
- From metrics: the analyzer already exports `frame_capture_lag_seconds`.

Write down:

- median latency
- worst-case latency
- how often the pipeline stalls

Concrete ways to measure today (no guessing):

- HLS playback delay: fetch the playlist and compare `#EXT-X-PROGRAM-DATE-TIME` to wall clock.
- Redis backlog: `LLEN raw_frames` / `LLEN processed_frames` and convert to seconds (`len / fps`).
- Analyzer metric: `frame_capture_lag_seconds` (capture → analyzer).

2) **Decide where MediaMTX will live**

Your original motivation was perfect: a stable ingest service that deployments don’t disturb.

Recommendation:

- Run MediaMTX as its own **separate Coolify app / container** (or even systemd service) with minimal change frequency.
- Redeploy `web_server` / `video_analyzer` freely without touching MediaMTX.

3) **Security cleanup**

- Treat RTSP URLs as secrets.
- Remove/rotate any credentials present in repo files (notably `docker/local/docker-compose.prod-view.yml`).

Exit criteria for Phase 0:

- You can state “current latency is X–Y seconds” with evidence.
- You have a planned “where MediaMTX runs” decision (separate from the redeploy loop).


### Phase 1 — Stand up MediaMTX (no StopSign code changes yet)

1) **Pick a MediaMTX version and pin it**

Don’t use `:latest` in production. Pick a tag and pin it (for example `bluenviron/mediamtx:1` to stay on 1.x, or a specific release tag).

2) **Configure MediaMTX (config file vs env vars)**

You have two good configuration options:

- **Option A (recommended for secrets): env vars** in Coolify / docker compose.
  - This keeps RTSP credentials out of a file.
- **Option B: a `mediamtx.yml` file** mounted into the container.
  - Fine if the file is treated as a secret and stored in a private volume.

Option A example (env vars):

```sh
# pull camera into path "camera"
MTX_PATHS_CAMERA_SOURCE="rtsp://USER:PASS@CAMERA_HOST:554/whatever"

# if the camera network is flaky, force RTSP over TCP
MTX_PATHS_CAMERA_RTSPTRANSPORT=tcp

# for WebRTC, advertise addresses reachable by clients
MTX_WEBRTCADDITIONALHOSTS="192.168.x.x,my-public-dns.example.org"
```

Option B example (`mediamtx.yml`, adjust as needed; verify against your environment):

```yaml
logLevel: info

# optional but useful
metrics: yes
metricsAddress: :9998

# If you want absolute timestamps (only if your camera actually provides them):
pathDefaults:
  useAbsoluteTimestamp: false

# WebRTC needs to know what address clients should use to reach the server.
# Fill these after you decide deployment topology (LAN + public).
webrtcAdditionalHosts: []

paths:
  camera:
    # Pull from your camera RTSP server (this is the “single stable ingest”)
    source: rtsp://USER:PASS@CAMERA_HOST:554/whatever

    # If the camera/wi-fi is flaky, TCP is often more reliable than UDP.
    # rtspTransport: tcp
```

3) **Expose the right ports (docker / firewall / proxy)**

MediaMTX defaults (verify in docs):

- RTSP: `8554/tcp`
- HLS: `8888/tcp`
- WebRTC web/WHEP: `8889/tcp`
- WebRTC media: `8189/udp` (and optionally `8189/tcp` if you enable TCP fallback)

If you’re deploying behind NAT / containers, the WebRTC doc path is:

- forward/allow `8189/udp`
- set `webrtcAdditionalHosts`
- if UDP is blocked, enable TCP fallback (`webrtcLocalTCPAddress: :8189`) and forward `8189/tcp`

4) **Validate MediaMTX in isolation**

Checklist:

- Open WebRTC page: `http://<host>:8889/camera`
- Open HLS page: `http://<host>:8888/camera`
- Open RTSP in VLC: `rtsp://<host>:8554/camera`
- Scrape metrics (if enabled): `curl http://<host>:9998/metrics`

Do not proceed until you can reliably read the stream with at least one protocol.

Exit criteria for Phase 1:

- MediaMTX reliably pulls from the camera (survives transient camera flakiness).
- You can read the stream with *at least one* browser-friendly protocol (LL‑HLS or WebRTC) from the same network you care about (LAN vs public).

#### Local compose sketch (recommended)

If you want to run this in `docker/local/docker-compose.yml`, the shape usually looks like:

```yaml
services:
  mediamtx:
    image: bluenviron/mediamtx:1
    environment:
      # don’t put secrets in git; set in docker/local/.env or your orchestrator
      MTX_PATHS_CAMERA_SOURCE: ${RTSP_URL}
      MTX_PATHS_CAMERA_RTSPTRANSPORT: tcp
      MTX_WEBRTCADDITIONALHOSTS: ${WEBRTC_ADDITIONAL_HOSTS}
      MTX_METRICS: "yes"
      MTX_METRICSADDRESS: ":9998"
    ports:
      - "8554:8554"
      - "8888:8888"
      - "8889:8889"
      - "8189:8189/udp"
```

Notes:

- This uses env vars instead of committing `mediamtx.yml`.
- `WEBRTC_ADDITIONAL_HOSTS` should include `localhost`/LAN IPs for local testing.


### Phase 2 — Switch the website to play the MediaMTX stream (video decoupling)

The objective is: **the website should show live video even if the analyzer is stopped**.

Implementation strategy:

- Keep the existing `/stream/stream.m3u8` path working as rollback.
- Add a new “video source” mode and switch gradually.

1) **Add a `VIDEO_SOURCE` knob**

Conceptually:

- `VIDEO_SOURCE=legacy_hls` (current behavior)
- `VIDEO_SOURCE=mediamtx_hls`
- `VIDEO_SOURCE=mediamtx_webrtc`

2) **Start with LL‑HLS (lowest risk)**

Update the frontend stream URL from `/stream/stream.m3u8` to:

- `http(s)://<mediamtx-host>:8888/camera/index.m3u8`

Notes:

- LL‑HLS can be ~0.5–3 seconds latency when tuned (and when keyframes are frequent enough).
- If your public site is HTTPS, avoid mixed-content issues by either:
  - serving MediaMTX behind the same origin (reverse proxy), or
  - enabling TLS on MediaMTX for HLS (also required for Safari iOS low-latency HLS).

3) **Then add WebRTC (best latency)**

The goal is direct access to the `<video>` element so you can overlay a `<canvas>`.

MediaMTX provides a JS helper (`reader.js`) to load WHEP into a `<video>` tag.

Approach:

- Download the `reader.js` that matches your MediaMTX version tag (so the protocol stays compatible).
- Serve it from your `static/` assets.
- Initialize it with WHEP URL: `http(s)://<mediamtx-host>:8889/camera/whep`

This yields the lowest practical latency if your network supports UDP.

Concrete “reader.js” workflow (recommended for overlays):

1) Download the version-matched reader:
   - From the MediaMTX repo at the same tag you deploy (see their “Embed streams in a website” doc).
2) Serve it from `static/js/reader.js`.
3) Replace the HLS.js initialization with something like:
   - Create `new MediaMTXWebRTCReader({ url: "https://<host>:8889/camera/whep", ... })`
   - Set `video.srcObject = evt.streams[0]` in `onTrack`.

(Don’t use the iframe embed if you want canvas overlays; you need the raw `<video>` element.)

4) **UI sanity checks**

- With the analyzer stopped:
  - video should still play
  - the site should not error-loop
  - health checks should not mark the site “down” just because overlays are missing

Exit criteria for Phase 2:

- You can stop `video_analyzer` and still have a working live video page.
- Your site health model is updated so “AI down” != “site down”.

Rollback for Phase 2:

- Flip `VIDEO_SOURCE` back to `legacy_hls` and you’re back to the old pipeline.


### Phase 3 — Refactor analyzer input: read from MediaMTX (drop Redis frames)

Now that the site video is decoupled, you can delete the biggest complexity: frames-in-Redis.

1) **Replace `Redis BRPOP RAW_FRAME_KEY` with RTSP decode**

New input should be:

- Local: `rtsp://mediamtx:8554/camera`
- Prod: `rtsp://<mediamtx-host>:8554/camera` (or internal DNS name)

2) **Pick a decoder library**

Options:

- **PyAV** (recommended): better access to frame timing/PTS; more control.
- OpenCV: simpler but timing metadata is usually weak.

If you add PyAV, follow your library integration protocol (check version, introspect API, live test).

3) **Keep the AI logic the same**

Goal is strictly I/O refactor:

- input: RTSP frames
- output: overlay metadata (WS) + DB/MinIO writes

4) **Stop producing processed frames**

Once overlays are in the browser, you no longer need:

- `PROCESSED_FRAME_KEY`
- `ffmpeg_service`
- JPEG encode/decode between services

The analyzer can still keep an “annotated frame” internally for debugging snapshots if needed, but it should not be on the hot path.

Exit criteria for Phase 3:

- Analyzer no longer depends on Redis for frames.
- DB + MinIO writes still work (vehicle pass rows keep appearing; images still upload).

Rollback for Phase 3:

- Keep the legacy analyzer code path behind a flag (`FRAME_SOURCE=redis` vs `FRAME_SOURCE=rtsp`) until the new reader is stable.


### Phase 4 — Add a WebSocket metadata feed + browser canvas overlays

This is the “AI overlays” channel.

#### 4.1 Metadata message schema (proposed)

Keep it small, stable, and easy to version:

```json
{
  "v": 1,
  "ts_utc": 1736790000.123,
  "raw": {"w": 1920, "h": 1080},
  "tracks": [
    {
      "id": 42,
      "bbox": [x1, y1, x2, y2],
      "speed": 12.3,
      "in_stop_zone": true,
      "violating_stop": false
    }
  ],
  "zones": {
    "stop_zone": [[x,y],[x,y],[x,y],[x,y]],
    "pre_stop_line": [[x,y],[x,y]],
    "capture_line": [[x,y],[x,y]]
  }
}
```

Important: **publish in raw coordinates**. Then overlay mapping is just scaling raw → display.

If the analyzer runs crop/scale internally, it should convert back to raw coordinates before publishing.

#### 4.2 Where to host the WebSocket

Two reasonable options:

- **Option A (simplest):** WebSocket endpoint in `web_server`, analyzer POSTs/pushes into it or connects as a client and publishes messages.
- **Option B (keeps analyzer dumb):** analyzer publishes to Redis pub/sub; `web_server` subscribes and fans out to browsers.

Given the goal is simplification, Option A can eliminate Redis entirely once you’ve removed frames-from-Redis.

#### 4.3 Browser overlay implementation

- Render `<video id="videoPlayer">` with MediaMTX source.
- Add `<canvas id="overlay">` absolutely positioned over the video.
- On every animation frame:
  - clear canvas
  - draw boxes/labels/zones based on latest metadata

Overlay drawing should be tolerant:

- if metadata is stale, fade it or clear it
- drop old messages; always keep the latest

Exit criteria for Phase 4:

- Overlays appear in the browser.
- When the analyzer restarts, video keeps playing and overlays recover without page refresh.


### Phase 5 — Decommission legacy services

Once Phase 2–4 are stable:

- Remove `rtsp_to_redis`
- Remove `ffmpeg_service`
- Remove Redis frame keys and any Redis dependency that only existed for video/metadata
- Simplify health checks:
  - Web server health should not depend on an on-disk HLS playlist anymore
  - Analyzer readiness should depend on “frames recently decoded + DB reachable”
  - MediaMTX should expose its own health/metrics

Exit criteria for Phase 5:

- The only remaining “always-on” video component is MediaMTX.
- No frames traverse Redis.
- There is no ffmpeg packaging code in your app stack (unless you intentionally keep a transcode sidecar).


## Latency tuning: practical playbook

### Biggest wins (in order)

1) **Stop using legacy HLS windowing** (your current pipeline is “designed” for 20s).
2) Prefer **WebRTC** if you can open UDP and configure ICE correctly.
3) If WebRTC is hard, use **LL‑HLS** via MediaMTX and tune:
   - `hlsPartDuration` (example: `500ms`)
   - `hlsSegmentDuration` (shorter segments help, but keyframes matter)
4) Ensure camera keyframe interval is reasonable (for HLS variants).

### WebRTC networking checklist (MediaMTX)

From MediaMTX docs:

- set `webrtcAdditionalHosts` to include addresses clients can use to reach the server (LAN + public DNS/IP).
- forward/allow `8189/udp` to the MediaMTX container/host.
- if UDP is blocked, enable TCP listener (`webrtcLocalTCPAddress: :8189`) and forward TCP 8189.
- if NAT traversal is hard, configure STUN:

```yaml
webrtcICEServers2:
  - url: stun:stun.l.google.com:19302
```

If that still fails, add TURN (client-only or full relay).


## Timestamp semantics: RTSP vs Redis/SSFM

**Important behavioral difference when using `FRAME_SOURCE=rtsp`:**

The legacy Redis/SSFM path embeds the **camera capture timestamp** in each frame's metadata. This is the time `rtsp_to_redis` received the frame from the camera. Stop-duration and speed calculations use this timestamp to measure real-world timing.

The new `RTSPFrameSource` uses `time.time()` at **read time** - when the analyzer dequeues the frame from OpenCV's buffer. This is subtly different:

| Source | Timestamp represents | Accuracy |
|--------|---------------------|----------|
| Redis/SSFM | Camera capture time | True timing (rtsp_to_redis stamps it) |
| RTSP direct | Analyzer read time | Affected by buffering, CPU load |

**Practical implications:**

1. **Stop duration**: If the analyzer processes frames faster than real-time (draining a buffer), timestamps will be closer together → stop duration appears shorter. The rate-limiting fix (sleeping to target FPS) mitigates this but doesn't eliminate it.

2. **Speed calculation**: Same issue - if frames drain quickly, `dt` between frames shrinks, making calculated speeds spike.

3. **`frame_capture_lag_seconds` metric**: With RTSP, this metric shows "time since analyzer read the frame" rather than true pipeline latency.

**Recommendations:**

- For **Phase 3 testing**, the current implementation is acceptable - the rate limiter keeps frame timing reasonably stable.
- For **production accuracy**, consider:
  - Using PyAV instead of OpenCV to access frame PTS (presentation timestamps)
  - Anchoring PTS to wall-clock time at stream start
  - Or accepting the slight timing variance as a tradeoff for lower latency

**Future enhancement (Phase 4+):** If overlay timestamp alignment becomes critical, implement PTS-based timing using PyAV or similar. This is not a Phase 3 blocker.


## Timestamp alignment (advanced, optional)

You have three levels of alignment:

1) **MVP:** draw latest boxes, drop stale metadata. Low latency, minimal complexity.
2) **Manual offset:** add a slider in the UI that offsets metadata time relative to video.
3) **Absolute timestamp alignment:** preserve source timestamps and align to playout time.

MediaMTX can preserve absolute timestamps if configured (`useAbsoluteTimestamp`) and the camera actually provides them.

In WebRTC, browsers may expose an `estimatedPlayoutTimestamp` stat which can be used to align media time with absolute time.

This is the right direction if you ever need multi-stream sync or perfect frame alignment, but it is not required to get the architectural simplification.


## Testing plan (what to test, and how)

### 1) Local development without a real camera

Use a sample video and publish it into MediaMTX with FFmpeg:

- loop sample file forever (`-stream_loop -1`)
- publish RTSP to MediaMTX `rtsp://mediamtx:8554/camera`

Then verify:

- Web playback works (HLS and/or WebRTC)
- analyzer can read RTSP from MediaMTX
- metadata overlays render

### 2) Integration tests (container-level)

Add checks that assert “decoupling actually happened”:

- With analyzer stopped:
  - MediaMTX still serves video
  - web_server still serves UI
- With MediaMTX stopped:
  - UI shows clear “video unavailable” but site is up
- With Postgres unreachable:
  - analyzer readiness fails (but video still plays)

### 3) Unit tests (fast, deterministic)

Focus on the pieces that are easy to regress:

- coordinate transform raw ↔ display
- metadata schema validation (including version field)
- “drop stale messages” logic (ensure overlay uses the newest data)

### 4) Latency measurement

Add a tiny “latency HUD” in the overlay:

- show `now - metadata.ts_utc` (age of metadata)
- show “ws connected / last message time”

If you implement absolute timestamp alignment, also display:

- estimated playout timestamp (if available)
- computed video→metadata offset

### 5) Soak testing

Run for 24–72h and track:

- MediaMTX reconnect behavior to camera
- analyzer memory growth
- overlay WS stability (reconnects, message rate)


## CORS and production deployment considerations

When deploying MediaMTX separately from the web app, you'll encounter cross-origin issues:

### The problem

- Web app runs on `:8000` (or your domain)
- MediaMTX HLS runs on `:8888`
- MediaMTX WebRTC runs on `:8889`

Browsers will block:
- HLS.js fetching manifests/segments from different origin
- WebRTC WHEP POST requests to different origin
- Mixed content (HTTPS site loading HTTP streams)

### Solutions (pick one)

**Option A: Reverse proxy (recommended for production)**

Route MediaMTX through your main domain:
- `/media/hls/*` → `mediamtx:8888/*`
- `/media/whep/*` → `mediamtx:8889/*`

Then set:
```
MEDIAMTX_HLS_URL=/media/hls/camera/index.m3u8
MEDIAMTX_WEBRTC_URL=/media/whep/camera/whep

# Backend health checks (must be reachable from inside the web_server container/network)
MEDIAMTX_HLS_INTERNAL_URL=http://<mediamtx-host>:8888/camera/index.m3u8
MEDIAMTX_WEBRTC_INTERNAL_URL=http://<mediamtx-host>:8889/camera/whep
```

**Option B: Enable CORS on MediaMTX**

Add to MediaMTX config (env vars or mediamtx.yml):
```yaml
# Allow cross-origin requests (less secure than reverse proxy)
hlsAllowOrigin: "*"
webrtcAllowOrigin: "*"
```

Or via environment variables:
```
MTX_HLSALLOWORIGIN=*
MTX_WEBRTCALLOWORIGIN=*
```

Note: Wildcard origins are less secure; prefer reverse proxy for production.

**Option C: Same-origin deployment (local dev)**

For local development, both services are on localhost with different ports.
Modern browsers allow this for `localhost`, but HTTPS sites will still block HTTP streams.


## Rollout strategy (prod)

1) Deploy MediaMTX as a separate "stable ingest" app/service.
2) Switch the website video source to MediaMTX (HLS first, WebRTC second).
3) Migrate analyzer input/output.
4) Enable overlays in UI (behind a feature flag).
5) Remove the legacy pipeline.

Keep a rollback switch at each step:

- UI can go back to legacy `/stream/stream.m3u8`.
- Analyzer can keep writing DB/MinIO independent of overlays.


## Implementation checklist (repo changes, concrete)

This is the “what files change” map. It’s intentionally specific so you can execute it linearly.

### Media plane (new)

- Add (local): `docker/local/docker-compose.yml` service `mediamtx` (or a `docker-compose.mediamtx.yml` overlay).
- Add (prod): a separate Coolify app (recommended) that runs MediaMTX with pinned version.
- Add env vars (prod, in Coolify):
  - `MTX_PATHS_CAMERA_SOURCE` (RTSP URL; treat as secret)
  - `MTX_PATHS_CAMERA_RTSPTRANSPORT=tcp` (if needed)
  - `MTX_WEBRTCADDITIONALHOSTS=...`
  - Optional: `MTX_METRICS=yes`, `MTX_METRICSADDRESS=:9998`
- Ensure networking/ports:
  - `8554/tcp`, `8888/tcp`, `8889/tcp`, `8189/udp` reachable where needed

### UI plane (web_server)

- Update `static/js/video-player.js` to support `VIDEO_SOURCE` modes:
  - `legacy_hls` (current `/stream/stream.m3u8`)
  - `mediamtx_hls` (MediaMTX `/index.m3u8`)
  - `mediamtx_webrtc` (MediaMTX WHEP + `reader.js`)
- Add `static/js/overlay.js` (canvas renderer) and include it on pages that show video.
- Update `stopsign/web_server.py`:
  - add a WebSocket endpoint for overlay metadata (e.g. `/ws/overlay`)
  - update `/health/stream` (stop checking local HLS file; instead check MediaMTX reachability or simply remove “stream freshness” as a hard dependency)

### AI plane (video_analyzer)

- Add a new “frame source” abstraction:
  - `RedisFrameSource` (existing SSFM → numpy)
  - `RTSPFrameSource` (MediaMTX RTSP → numpy)
- Update `stopsign/video_analyzer.py`:
  - swap input to `RTSPFrameSource` when enabled
  - stop writing processed frames to Redis
  - publish metadata messages (schema above) to the web_server WS endpoint (or Redis pub/sub if you keep Redis)
- Keep DB + MinIO behavior unchanged (only refactor I/O).

### Delete path (after stabilized)

- Remove `rtsp_to_redis/` from prod deployment.
- Remove `stopsign/ffmpeg_service.py` from prod deployment.
- Remove Redis from prod only if it no longer has a job (config flags + pubsub can move elsewhere).


## Appendix: immediate “low effort” improvements if you must keep the current architecture (optional)

If you want a quick improvement before migrating:

- Reduce Redis buffers from 500 to something like 30–60 frames.
- Reduce HLS playlist window (list size).
- Add a true “drop-to-live” policy in both analyzer and encoder (drain queues, keep newest).

These can reduce latency, but they don’t address the root complexity: frames moving through Redis + ffmpeg.
