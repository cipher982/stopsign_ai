# Frozen Stream Guardrail Spec (2026-02-27, revised 2026-09-11)

## Goal

Detect camera streams that are technically connected but visually frozen, then
recover automatically before downstream "no recent passes" alerts fire.

## Requirements

1. Detect visual freeze in `rtsp_to_redis` (source-of-truth ingress stage).
2. Detect a live camera that is delivering almost no frames (lossy link), which
   is invisible to freeze detection: the picture changes, it just rarely arrives.
3. Fail readiness during either condition (without killing liveness).
4. Auto-remediate in escalating stages:
   - Stage 1: force RTSP reconnect, on sustained freeze *or* sustained low input rate.
   - Stage 2: exit the process when the rate has stayed below the floor past the
     reconnect attempts, so `restart: always` gives capture a fresh process and a
     fresh RTSP session.
5. Avoid alert spam by emitting one open/resolve incident pair per freeze event.

## Detection Strategy

Two independent signals, because they fail in different ways:

- **Content freeze** — mean absolute difference (MAD) between downsampled
  grayscale frames. Compute `mad = mean(abs(curr_gray - prev_gray))`; track
  `last_motion_ts` whenever `mad > RTSP_FREEZE_MAD_THRESHOLD`; declare frozen
  when `now - last_motion_ts >= RTSP_FREEZE_DETECT_SEC`. Matches FFmpeg
  `freezedetect` semantics (noise threshold + duration persistence).
- **Frame starvation** — the input FPS the loop already computes, scored over
  `RTSP_RATE_WINDOW_SEC` rather than per second. A lossy link delivers frames in
  bursts, so a one-second sample reads 1.7 FPS inside a gap and 15 FPS inside a
  burst for a stream that is fine; the window is what makes the signal mean
  something. Declare starved when the windowed rate stays below
  `RTSP_MIN_INPUT_FPS`, which is the mode a WiFi link in packet loss produces
  (0.15-4 FPS of live content, against 9.5-15 FPS healthy on this camera).

A frozen stream is not starved and a starved stream is not frozen; a guard that
only measures one of them reports healthy through the other.

## Runtime Behavior

### Health endpoints

- `/healthz`: liveness (always 200 if process is alive).
- `/ready`: readiness (503 when ingest stalled, visually frozen, or starved past
  `READY_LOW_INPUT_FPS_SEC` of grace).
- `/health`: backward-compatible alias to `/ready`.

The compose healthcheck probes `/ready`.

### Stage 1 remediation

When `freeze_age >= RTSP_FREEZE_RECONNECT_SEC`, or the input rate has been below
`RTSP_MIN_INPUT_FPS` for `RTSP_LOW_FPS_RECONNECT_SEC`, force RTSP reconnect by
reinitializing capture. Both triggers share one cooldown
(`RTSP_FREEZE_RECONNECT_COOLDOWN_SEC`).

### Stage 2 remediation

When the input rate has stayed below the floor for `RTSP_LOW_FPS_EXIT_SEC`, log
and `os._exit(1)`. `restart: always` re-creates the container, re-opening the
RTSP session and clearing any wedged decoder state. This is the same shape as
the video analyzer's stall watchdog, and it is skipped when Redis is unreachable
at that moment — decided by a live ping, because the cached connectivity flag is
written by the publisher and is therefore exactly as stale as the outage.

A stream that yields no frames at all never reaches this stage, and does not need
to: the read times out, the loop re-opens capture, and it keeps re-opening.

Stage 1 is deliberately *not* gated on Redis: a reconnect is an action on the
camera, and re-opening the RTSP session is correct whether or not Redis is up.

A sustained *visual freeze* deliberately has no stage 2: it is retried with
stage-1 reconnects and left to the chain-level alert. A frozen camera still
delivers frames at full rate, so the rate guard cannot see it, and the reconnect
is the remedy that has actually cleared one (2026-09-09: resolved 62 s after the
forced reconnect).

> The 2026-02-27 version of this spec defined stage 2 as an operator-supplied
> `RTSP_FREEZE_REMEDIATION_CMD` shell hook. It shipped disabled, was never
> enabled, and on 2026-09-11 it was the reason a 90-minute ingest collapse took
> no remedial action at all. The hook is deleted in favour of the self-restart.

## Tunables

- `RTSP_FREEZE_DETECT_SEC` (default `120`)
- `RTSP_FREEZE_MAD_THRESHOLD` (default `0.015`)
- `RTSP_FREEZE_SAMPLE_WIDTH` (default `160`)
- `RTSP_FREEZE_SAMPLE_HEIGHT` (default `90`)
- `RTSP_FREEZE_RECONNECT_SEC` (default `180`)
- `RTSP_FREEZE_RECONNECT_COOLDOWN_SEC` (default `60`)
- `RTSP_MIN_INPUT_FPS` (default `6`, `0` disables the rate guard)
- `RTSP_RATE_WINDOW_SEC` (default `10`)
- `RTSP_LOW_FPS_RECONNECT_SEC` (default `120`)
- `RTSP_LOW_FPS_EXIT_SEC` (default `900`, `0` disables the restart)
- `OPENCV_FFMPEG_CAPTURE_OPTIONS` (default `rtsp_transport;tcp`)

## Rollout Plan

1. Deploy code (stage 2 disabled with `RTSP_LOW_FPS_EXIT_SEC=0` if a pure
   observation window is wanted).
2. Verify `/ready` transitions to 503 under synthetic freeze and synthetic
   starvation.
3. Enable `RTSP_LOW_FPS_EXIT_SEC` once the restart is confirmed to recover.
4. Route external alerting at the chain level (`/api/pipeline-health`), with
   dedupe windows, rather than at this container's probe.
