# Frozen Stream Guardrail Spec (2026-02-27)

## Goal

Detect camera streams that are technically connected but visually frozen, then
recover automatically before downstream "no recent passes" alerts fire.

## Requirements

1. Detect visual freeze in `rtsp_to_redis` (source-of-truth ingress stage).
2. Fail readiness during sustained freeze (without killing liveness).
3. Auto-remediate in two stages:
   - Stage 1: force RTSP reconnect.
   - Stage 2: run optional operator-defined remediation command (camera reboot hook).
4. Avoid alert spam by emitting one open/resolve incident pair per freeze event.

## Detection Strategy

Use mean absolute difference (MAD) between downsampled grayscale frames:

- Compute `mad = mean(abs(curr_gray - prev_gray))`.
- Track `last_motion_ts` whenever `mad > RTSP_FREEZE_MAD_THRESHOLD`.
- Declare frozen when `now - last_motion_ts >= RTSP_FREEZE_DETECT_SEC`.

This approach is robust to noise and matches FFmpeg `freezedetect` semantics
(`noise` threshold + `duration` persistence).

## Runtime Behavior

### Health endpoints

- `/healthz`: liveness (always 200 if process is alive).
- `/ready`: readiness (503 when ingest stalled or visually frozen).
- `/health`: backward-compatible alias to `/ready`.

### Stage 1 remediation

When `freeze_age >= RTSP_FREEZE_RECONNECT_SEC`, force RTSP reconnect by
reinitializing capture. Cooldown controlled by
`RTSP_FREEZE_RECONNECT_COOLDOWN_SEC`.

### Stage 2 remediation

When `freeze_age >= RTSP_FREEZE_REMEDIATION_SEC`, run
`RTSP_FREEZE_REMEDIATION_CMD` once per incident (and not more often than
`RTSP_FREEZE_REMEDIATION_COOLDOWN_SEC`).

## Tunables

- `RTSP_FREEZE_DETECT_SEC` (default `120`)
- `RTSP_FREEZE_MAD_THRESHOLD` (default `0.25`)
- `RTSP_FREEZE_SAMPLE_WIDTH` (default `160`)
- `RTSP_FREEZE_SAMPLE_HEIGHT` (default `90`)
- `RTSP_FREEZE_RECONNECT_SEC` (default `180`)
- `RTSP_FREEZE_RECONNECT_COOLDOWN_SEC` (default `60`)
- `RTSP_FREEZE_REMEDIATION_SEC` (default `420`)
- `RTSP_FREEZE_REMEDIATION_CMD` (default empty, disabled)
- `RTSP_FREEZE_REMEDIATION_COOLDOWN_SEC` (default `1800`)
- `RTSP_FREEZE_REMEDIATION_TIMEOUT_SEC` (default `45`)

## Rollout Plan

1. Deploy code with Stage 2 command unset (safe mode).
2. Verify `/ready` transitions to 503 under synthetic freeze.
3. Enable `RTSP_FREEZE_REMEDIATION_CMD` once credentialed reboot command is validated.
4. Route external alerting to `/ready` (not `/healthz`) and apply alert dedupe windows.
