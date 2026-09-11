# Ingest Starvation — 2026-09-11 Post-Mortem

## What the alert said

`stopsign-pipeline-health` (Sauron, 15-minute cron) paged at 22:22 UTC:

> ffmpeg dup ratio 93% (threshold 90%) — output frozen, 25 repeated frame(s) in
> last snapshot | hls_fresh=True hls_age=4.6s archive_upload_healthy=True

The first investigation concluded "the RTSP camera feed at Crestwood froze —
camera-side" and recommended setting `RTSP_FREEZE_REMEDIATION_CMD`.

## What the evidence says

Containers, logs and the Redis signals were pulled from cube for the 26 h around
the event.

**It was not a freeze, and not one 15-minute window.** The ingress freeze detector
(`rtsp_to_redis`, MAD between consecutive downsampled frames against
`RTSP_FREEZE_MAD_THRESHOLD`) never fired: it logged `frozen=0` at every one-minute
sample from 21:52 to 22:32, and `freeze_age` peaked at 27.5 s against a 120 s
threshold. In 32 days of container life it has fired exactly once, on 2026-09-09
20:44 UTC, where `freeze_age` reached 120 s with `mad=0.0000` — and the stage-1
forced reconnect resolved that real freeze 62 s later.

**What actually happened is frame starvation.** The camera kept producing *live,
changing* frames, delivered at 0.15–4 FPS in bursts:

```
2026-09-11 22:10:20  FPS=0.15  freeze_age=0.0s  mad=0.2619   <- camera, not picture
2026-09-11 22:17:22  FPS=3.93  freeze_age=0.0s  mad=0.0242
2026-09-11 22:20:26  FPS=0.65  freeze_age=0.0s  mad=0.0174
```

Thirteen starvation episodes in 26 h, twelve of them between 20:46 and 22:34 UTC
(2–8 min each), plus one at 01:12. Over that window ffmpeg logged 181 five-second
samples at ≥90 % starved, 286 at ≥50 %, and 3775 at 10–49 % (~5.2 h of the 26 h
running below the 15 FPS target).

**The transport is the cause.** The rtsp log carries 957 h264 decode errors in
26 h — `error while decoding MB`, `left block unavailable for requested intra4x4
mode`, `reference picture missing during reorder`, `Missing reference picture`.
That is packet loss on the camera's WiFi link (`192.168.1.151`), and it is
chronic: 313 of 1560 sampled minutes had decode errors, framing every starvation
episode. Capture runs UDP (`OPENCV_FFMPEG_CAPTURE_OPTIONS` unset, so FFmpeg's RTSP
default), so lost packets are never retransmitted.

## Why nothing caught it

1. **Freeze detection answers the wrong question.** `frozen=0` was correct — the
   picture was moving. "Is the picture moving?" and "are frames arriving?" are
   different questions, and only the first was instrumented.
2. **Nothing acted on ingest health.** `/ready` on the rtsp container already
   computed a composite readiness, but the container's compose healthcheck probed
   `/healthz` (liveness only) and no poller read `/ready`. Plain `docker compose`
   does not restart a container for being *unhealthy* either, so even a correct
   probe would have produced no remediation.
3. **Stage 2 was a no-op by construction.** `RTSP_FREEZE_REMEDIATION_CMD` shipped
   empty in Feb 2026 ("safe mode", `docs/frozen-stream-guardrail-spec.md`), was
   never configured, and the 420 s escalation therefore logged and did nothing.
4. **`dup_pct` was read as a freeze signal.** The name comes from counting output
   slots where BRPOP returned nothing: it means *no fresh frame reached the
   encoder*, which a starved-but-live camera produces identically. The alert
   wording ("output frozen", "ffmpeg dup ratio") pointed the investigation at the
   picture and the analyzer rather than at the camera link.
5. **History was already there.** `docs/tasks/homepage-performance-experiments.md`
   recorded the same signature on 2026-04-18 — input-rate collapses to 0.19 FPS,
   `No frame available in Redis`, "transient ingest / pipeline instability on the
   camera-to-analyzer path" — recovered without code changes, and never acted on.

## What changed

**Ingress rate guard** (`rtsp_to_redis/rtsp_to_redis.py`) — arrival rate is tracked
alongside content motion, recomputed from frame arrival timestamps over
`RTSP_RATE_WINDOW_SEC` on every frame. Below `RTSP_MIN_INPUT_FPS` for
`RTSP_LOW_FPS_RECONNECT_SEC` forces an RTSP reconnect (shared cooldown with the
freeze trigger); below it for `RTSP_LOW_FPS_EXIT_SEC` the process exits so
`restart: always` re-opens capture with a fresh session. The exit is skipped when
Redis is unreachable *at that moment* — checked with a live ping, because the
cached status flag cannot be trusted while nothing is publishing. Reconnects are
deliberately not gated on Redis: a reconnect is about the camera. A stream that
yields no frames at all never reaches either stage, and does not need to: the
read times out, the loop re-opens capture, and it keeps re-opening — the same
remediation without the restart. This is the analyzer's existing watchdog idiom,
and the remediation the shell hook never performed. The dead
`RTSP_FREEZE_REMEDIATION_*` hook is deleted.

**Readiness now means something** — `/ready` fails on sustained low input rate,
and the compose healthcheck probes `/ready` instead of `/healthz`.

**RTSP over TCP** — `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` in the
deployed compose, so the lossy WiFi link retransmits instead of feeding the
decoder corrupt slices. Verified against this camera before rolling out.

**Honest naming** — `dup_pct` keeps its published key, but the ffmpeg log line,
the `stopsign/web/routes/health.py` comment, and the Sauron job's docstring,
failure text and diagnosis steps now say "starved by upstream underrun" and point
at `analyzer.frame_age_seconds` vs the rtsp `/ready` to tell an analyzer stall from
an ingest stall from a real freeze.

**Archive path flips retry** (`stopsign/image_storage.py`) — unrelated bug found
while reviewing the chain. `_flip_db_path_with_retry` gives up after 20 s and the
give-up was final: the pass kept `local://` forever, so its file could never be
pruned — 7109 stranded files, 8394 on disk against a 500 cap, 15750 `local://`
pass rows in the database. The inline retry was waiting for the wrong thing.
Querying the five give-ups from the incident window afterwards, **none has a pass
row at all** (0/5): they are the capture-line images of cars that never completed
a pass, so no retry could ever have succeeded. Failed flips are now swept from the
upload worker's idle loop (50 per 60 s sweep, 15 min horizon), and at the end of
that window a file no pass references is released for pruning — the row is written
at zone exit, seconds after the capture, so a row still missing after 15 minutes
is not late, it is never coming. Two caveats: the sweep is worker-driven, so it
does not run while the worker is blocked on an upload, and a database error
delays expiry rather than forcing it. Both are the conservative direction —
files are retained, never deleted on a doubt. Releasing stays safe even if that
judgement is wrong: `resolve_image_url` falls back to `/vehicle-image/<name>`,
which streams the object from Bremen.

## What is still open

- **The camera link is a permanent constraint, not a defect.** The camera is on
  WiFi through the building's exterior walls and stays there — no cable will be
  run for this project. Packet loss, and the occasional corrupt slice or starved
  minute that come with it, is accepted cost of the deployment. Everything above
  exists to *bound* that damage rather than to remove it: TCP removes corruption,
  the rate guard converts an unbounded starvation into a reconnect or a restart,
  and the chain-level alert is the backstop for whatever still gets through.
- **The existing `local://` backlog** — 15750 pass rows (of which ~7300 point at a
  file that is already gone) and ~8400 files on disk — is only handled going
  forward: the release path applies to objects this process watched fail. Clearing
  the backlog needs a one-off sweep that stats each object in Bremen and then
  either flips or releases it. That is a production data change and was left for
  an explicit decision. Nothing is user-visible either way, because
  `resolve_image_url` falls back to the archive.
- **`dup_pct >= 90`** remains the alert threshold, so 10–49 % starvation (5.2 h of
  the 26 h studied) still has no witness. That is a product question — how choppy
  may the public stream be before it is worth an email — not a bug.
- **`PIPELINE_WATCHDOG_SEC`** still defaults to 0 and is unset in the ffmpeg
  compose service, so ffmpeg cannot self-restart on HLS staleness.
