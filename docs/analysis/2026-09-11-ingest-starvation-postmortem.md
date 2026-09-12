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

## Where this stands

- **The camera link is a permanent constraint, not a defect.** The camera is on
  WiFi through the building's exterior walls and stays there — no cable will be
  run for this project. Packet loss, and the occasional corrupt slice or starved
  minute that come with it, is accepted cost of the deployment. Everything above
  exists to *bound* that damage rather than to remove it: TCP removes corruption,
  the rate guard converts an unbounded starvation into a reconnect or a restart,
  and the chain-level alert is the backstop for whatever still gets through.
- **The `local://` backlog is reconciled** — `scripts/reconcile_local_images.py`
  (dry run by default) flipped 15009 rows the archive already held, recovered 225
  more by re-uploading the local copy, left 516 pre-July rows that are gone from
  both sides, and reclaimed 8405 files. `bremen://` rows went 4935 → 20169 and the
  image directory 8407 files → 0. Nothing user-visible moved: a row still on
  `local://` serves through `/vehicle-image/`, and the thumbnail builder falls back
  to the archive when the local file is absent. The 516 unrecoverable rows keep
  their `local://` path and 404 their image; making them render the placeholder
  instead would rewrite the record of what was captured, so that stays a decision.
- **`dup_pct >= 90`** remains the alert threshold, so 10–49 % starvation (5.2 h of
  the 26 h studied) still has no witness. That is a product question — how choppy
  may the public stream be before it is worth an email — not a bug.
- **`PIPELINE_WATCHDOG_SEC`** is set to 180 in the ffmpeg compose service, so a
  wedged encoder now restarts itself. With the ≤180 s freshness threshold in
  `stopsign/hls_health.py` and a 10 s poll, a stalled encoder exits and is
  re-created roughly six minutes after its last playlist write.

## What is left

- **516 pass rows** (0.7 % of 77991) whose image is gone from both the local disk
  and the archive, all created before July. Their `local://` path 404s. Rendering
  the placeholder instead means nulling the path — the same choice as above.
- **`dup_pct >= 90`** remains the alert threshold, so 10–49 % starvation (5.2 h of
  the 26 h studied) still has no witness. That is a product question — how choppy
  may the public stream be before it is worth an email — not a bug.
- **Passes with no image are fixed.** The rate is not 8.6% overall - it is ~35% of
  passes every day, and 75-85% after dark, since 2026-05-18 (6,672 rows in total).
  That date is when `9be2204` and `47cbb28` started *recording* passes from tracks
  the detector only picks up past the capture line, instead of discarding them;
  the pass counts rose ~50% that week and the extra passes are exactly the ones
  with no picture. They are real, distinct vehicles, not duplicates (only 4.8%
  have an imaged pass within ±10s, against a 3.1% control). The defect was
  structural: a capture needs `passed_pre_stop` *then* a capture-line crossing, and
  74.5% of these tracks never cross that line while tracked - they are first seen a
  median of 270 px past it, at full size (bbox 233x95 px, so the picture was always
  there to take). The capture gate now also fires on the first usable view of a
  track acquired past the line, provided the vehicle is still upstream of the zone
  centre and moving along the approach. Replayed over 600 stored no-image
  trajectories: 96.2% would be photographed, a median of 2 frames after
  acquisition; over 600 imaged ones the line shot is unchanged for 97.5%. The
  remaining ~4% are first seen at or behind the zone centre (0.2% already inside
  the zone), where the only "picture" available is an exit-angle crop.
  Live after deploy: the first two passes recorded were both late-track captures
  (capture x=935 and 869 against a line at 1269) and both serve as images
  (416x173, 433x183 - the same size a line capture produces), against a 31%
  no-image rate in the six hours before. The labeling cron picks these up with no
  change, since it selects unlabeled passes that have an `image_path`.

## 2026-09-12 — the outage that was not the camera (and the reviews)

Ten hours of nothing. The analyzer processed its last frame at 05:42Z and the last
pass was written at 04:37Z; from then until 15:20Z the pipeline recorded nothing,
while the website, the HLS stream and the database all kept serving happily. The
camera was never the problem — it answered a ping from cube in 3 ms the whole time.

**What actually happened.** `cube` reaches `192.168.1.151` (the camera) over
`tailscale0`, because the netmap carried a subnet route for `192.168.1.0/24`
advertised by `RICHMCBNAS` — the NAS — and Tailscale's policy routing (table 52,
rule 5270) precedes the main table. Cube is itself on that subnet
(`192.168.1.66/24`, directly attached), so the route was never needed: it made a
remote node the path to cube's own LAN. `RICHMCBNAS` left the tailnet at
05:22:16Z, and every packet cube sent to the camera went into a tunnel to an
offline node. The ingest guard did what it could — `rtsp_to_redis` retried,
went unhealthy, never reconnected — but no retry can fix a route.

**Fix, applied and verified:** `tailscale set --accept-routes=false` on cube. The
camera is reachable again on the local interface (3.3 ms, `dev enp7s0`), the
ingest container reconnected at full 15 fps, and `new_fps 15.0 / dup_pct 0.0` with
a fresh analyzer frame age confirmed the whole chain. The only IPv4 subnet route
in the netmap was that one, so nothing else lost a path. Rollback is
`tailscale set --accept-routes=true`.

**Same node, second consequence.** `RICHMCBNAS` is also
`BREMEN_MINIO_ENDPOINT` (`100.98.103.56:9000`) — the archive every capture is
uploaded to. Since 05:22Z every upload has failed with a 10 s connect timeout, so
captures accumulate on local disk and their passes keep a `'/Users/davidrose/.omp/agent/sessions/-git-stopsign_ai/2026-09-11T23-20-51-227Z_01a092c6-191b-767e-98c2-17657a314be2/local'` path. The
site serves them anyway now, but the archive itself is unreachable until that node
returns to the tailnet; that is the one item here that needs hands, not code.

**Reviews (hatch `codex astra` + `cursor grok`, both returning BLOCKED) landed
four fixes:**

- The late-track capture is latched once per tracked vehicle, and constrained to
  the approach corridor — the band the stop zone and pre-stop line are drawn
  across. Without the latch, a vehicle that left the zone through a side edge
  could be photographed twice; without the corridor, a track crossing well clear
  of the roadway was photographed as traffic. Measured over 600 stored late-track
  trajectories: real vehicles sit within 1.8 roadway half-widths, the off-road
  counterexample at ~3.5.
- `/vehicle-image` now reads the local copy first, like the thumbnail route always
  did, so an image is served while its upload is pending or the archive is
  unreachable. Verified against an object that exists only locally: 200, 317x177
  JPEG (previously 404), and `..%2F..` now answers 400.
- Unarchived local captures are re-queued from disk on the upload worker's idle
  path — the worker gave up after three attempts and kept its state in memory, so
  a capture that missed its window was stranded permanently. This is what the
  2026-09-11 pile was made of.
- The reconciliation script's documented apply procedure was impossible (`docker
  exec` into a container it had just stopped) and its final line called every
  remaining row "gone from both", including captures still inside the settle
  window. Both corrected.

**Checked and not reproducing:** the analyzer skips YOLO on frames older than
100 ms, which review read as "starved ingest disables tracking". Measured on the
live queue: frame age p50 32 ms, max 105 ms over 25 samples, 1 over the gate — the
raw queue is drained on arrival, so the gate is a backpressure valve, not a
detection blocker. It is worth a counter before trusting that on a bad night.

**Still open:** analyzer restarts lose the in-flight upload queue and the pass
insert gives up after three attempts (durable state, not a rescan, is the real
fix); `dup_pct` 10-49% still has no witness; a pass has no reason column, so a
row with no image cannot say why. The ffmpeg startup path still clears the HLS
directory, which 404s the public window for the length of a restart.
