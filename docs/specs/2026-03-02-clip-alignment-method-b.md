# Replay Clip Alignment (Method B)

Date: March 2, 2026  
Status: Proposed  
Owner: stopsign_ai

## 1) Problem Statement

Replay clips are frequently offset from the vehicle pass event. Example from production on **February 27, 2026**:

- Pass `#54908`: DB `exit_time` is `2026-02-27 17:08:13` CST.
- The generated clip `clip_54908_1772233693.mp4` shows on-frame overlay ending around `17:08:03`.
- Observed mismatch: approximately **9-10 seconds early**.

Root cause from first principles:

1. Pass timestamps (`entry_time`, `exit_time`) are on the **capture timeline**.
2. Clip extraction currently uses HLS `.ts` file `mtime` on the **segment write timeline**.
3. With FIFO consumption and queue backlog, segment write timeline lags capture timeline.
4. Clip builder assumes these clocks match and extracts the wrong window.

## 2) Constraints / Guardrails

We explicitly do **not** want a default behavior change to the live stream pipeline.

- Keep ffmpeg default mode as FIFO (`BRPOP` oldest-first).
- Do not drop frames by default to "fix clips".
- Clip alignment must be solved in clip-building logic and pass metadata.

## 3) Research Findings

### 3.1 Live sampling (production, March 2, 2026)

30-second sample from `cube`:

- `processed_queue_depth`: min 232, p50 247, p95 247, max 247
- `raw_queue_depth`: min 0, p50 0, p95 9.6, max 20
- naive lag signal (`latest_segment_end - latest_capture_timestamp`):
  - min -1.55s, p50 -0.55s, p95 -0.40s, max +0.50s

Interpretation:

1. FIFO processed queue can remain deeply backlogged.
2. `latest_capture_timestamp` tracks producer head, not the frame currently encoded.
3. Therefore `latest_segment_end - latest_capture_timestamp` is not a valid estimator for clip lag.

### 3.2 Why current "dynamic lag from latest metadata" is incorrect

If ffmpeg consumes oldest frames and analyzer publishes newest metadata:

- Segment content corresponds to an old frame in queue.
- Metadata corresponds to newest frame.
- Their difference mostly reflects producer-head recency, not queue delay.

This explains why that signal stays near 0s while clips are still 10s+ off.

## 4) Candidate Approaches

### A. Make ffmpeg consume latest frames (drop stale backlog)

Mechanics:

- Switch to newest-first pop + queue trimming.

Effects:

- Lower live latency.
- Clip alignment improves indirectly (backlog reduced).
- But stale frames are dropped; stream no longer a faithful FIFO replay source.

Risk:

- Medium/high behavior change to live pipeline.

Conclusion:

- Keep as optional mode only, not default.

### B. Keep FIFO; compensate clip windows from FIFO lag (recommended)

Mechanics:

1. Capture per-pass queue/lag metadata at pass exit.
2. Shift clip window by estimated lag in clip builder.
3. Keep gap-aware segment-to-concat offset mapping.

Effects:

- No default stream semantics change.
- No frame-drop policy change.
- Fix is localized to replay clip construction.

Risk:

- Lower operational risk.
- Accuracy depends on lag estimate quality.

## 5) Method B Recommended Design

### 5.1 Data Model additions

Add columns to `vehicle_passes`:

- `stream_queue_depth_exit` `INT NULL`
- `stream_lag_est_sec` `DOUBLE PRECISION NULL`

Rationale:

- FIFO lag is primarily queue delay: `lag_sec ~= queue_depth / consumer_fps`.
- This estimate is per-pass and time-local, unlike "current live lag" at clip build time.

### 5.2 Lag capture at pass exit

At pass finalization (in stop detector path), record:

- `stream_queue_depth_exit` from analyzer’s processed queue depth (same frame cycle).
- `stream_lag_est_sec = stream_queue_depth_exit / fps` (fps from config, default 15).

Formula:

```
lag_est_sec = queue_depth_exit / FRAME_RATE
clip_shift_sec = lag_est_sec + CLIP_TIME_OFFSET_SEC
```

Where `CLIP_TIME_OFFSET_SEC` is a small calibration knob (default 0).

### 5.3 Clip builder windowing

For each pass:

1. Base window:
   - `entry_time - PREPAD`
   - `exit_time + POSTPAD`
2. Apply shift:
   - `aligned_entry = entry_time + clip_shift_sec`
   - `aligned_exit = exit_time + clip_shift_sec`
3. Segment selection uses aligned window.
4. Offset conversion is **gap-aware** (already required): map wall clock into concat timeline by cumulative segment durations.
5. If selected segment coverage is partial, do not build; retry later.

### 5.4 Fallback behavior

For old passes (missing new fields):

- Fallback order:
  1. `CLIP_TIME_OFFSET_SEC` (manual)
  2. No shift
- Do not use "latest metadata lag" as fallback in FIFO mode.

### 5.5 Bounds / safety

Clamp computed lag:

- `CLIP_LAG_MIN_SEC` default `0`
- `CLIP_LAG_MAX_SEC` default `45`

Reason:

- Negative lag is invalid for FIFO queue delay estimate.
- Max clamp prevents pathological offsets from bad samples.

## 6) Non-Goals

- Re-architecting ingest/analyzer/ffmpeg queues.
- Changing default live-latency strategy.
- Perfect forensic reconstruction for old passes without historical lag data.

## 7) Validation Plan

### 7.1 Unit tests

- Queue-depth-to-lag conversion and clamps.
- Gap-aware timeline offset conversion.
- Window alignment with parked and max-duration caps.

### 7.2 Integration tests

- Synthetic segment timeline with known lag and gaps.
- Verify clip builder targets expected event window.

### 7.3 Production acceptance criteria

For a sample of fresh passes on the same day:

- Median alignment error <= 2.0s
- P95 alignment error <= 4.0s
- No increase in clip build failure rate (`retry`, `failed`, `no_segments`)

Alignment error definition:

- Absolute delta between expected exit overlay time and clip tail overlay time.

## 8) Rollout Plan

### Phase 0: Instrument only

- Add columns and populate for new passes.
- No clip shift enabled yet.
- Observe `stream_queue_depth_exit`, `stream_lag_est_sec` distributions for 24h.

### Phase 1: Enable clip shift for new passes

- Enable `stream_lag_est_sec`-based alignment in clip builder.
- Keep manual calibration env available.

### Phase 2: Optional rebuild tool

- Add script to rebuild recent clips where metadata exists and status is `ready/local`.

## 9) Operational Notes

- Existing clip misalignment from February 27, 2026 remains in old artifacts unless rebuilt.
- This method intentionally prioritizes correctness with current FIFO semantics over live latency tuning.

## 10) Open Questions

1. Should `stream_queue_depth_exit` be sampled pre- or post-enqueue for best estimate?
2. Do we want to include a small fixed encoder/segment delay term in default calibration (e.g., +1s)?
3. Should pass detail UI expose `exit_time` directly (in addition to DB `timestamp`) for debugging alignment?

