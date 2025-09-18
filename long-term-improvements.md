# Longer-Term Pipeline Improvements

## 1. Media Hub & Transport Redesign
- Replace the Redis + ffmpeg packaging path with a dedicated media server (MediaMTX or GStreamer) to ingest RTSP and serve WebRTC/HLS directly.
- Python services subscribe only for analytics, decoupling video freshness from AI throughput.

## 2. Analyzer Input/Output Refactor
- Use PyAV or GStreamer to read frames directly from the media hub/camera instead of Redis lists.
- Drop Redis for raw frame transport; keep it (or Postgres) for metadata fan-out only.
- Removes double JPEG encode/decode, reduces latency, simplifies backpressure.

## 3. Client-Side Overlays
- Deliver unmodified video to the browser and push detection metadata via WebSocket.
- Render boxes/labels on a `<canvas>` in the client for better UX and tolerance of analyzer hiccups.

## 4. Stage-Specific Watchdogs
- Extend the analyzer stall watchdog pattern to RTSP ingest and encoder stages (or their replacements).
- Each component exits/restarts when it exceeds a "no fresh frames" threshold, minimizing manual intervention.

## 5. Observability Enhancements
- Build dashboards around new metrics (`frame_queue_depth`, `frame_pipeline_lag_seconds`, `redis_empty_polls_total`).
- Add tracing spans/timers for decode → infer → encode to pinpoint first-failure stages quickly.

## 6. Alerting & Policy Cleanup
- After the readiness split stabilizes, tune alerts (Netdata/Grafana) to fire on sustained readiness failures and restart storms, not transient blips.
- Align paging thresholds with business tolerance (e.g., warn at 2 minutes, critical at 10+ minutes).

Reference: `docs/architecture/first-principles-streaming.md` for a deeper proposal.
