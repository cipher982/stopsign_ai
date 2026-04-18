# Homepage Performance Experiments — Task Log

## Goal
- Improve `https://crestwoodstopsign.com/` startup and live-feed responsiveness in production.
- Preserve the current user experience unless a regression is explicitly approved.
- Run one change at a time, measure it the same way each time, and record the result here.

## Guardrails
- Do not remove major homepage capabilities just to make the page lighter.
- Do not reduce information density as an optimization tactic without explicit approval.
- Prefer transport, caching, render-path, and byte-efficiency wins over content cuts.
- Treat the public production path as the source of truth. Direct-origin checks are diagnostic only.

## Decision Log
### Decision: Production public path is the primary benchmark
**Context:** The current slowness is materially worse on the public hostname than on direct cube origin.
**Choice:** Judge every experiment by public production measurements first; use direct-origin measurements to localize the bottleneck.
**Rationale:** The user cares about the public site, not the box in isolation.
**Revisit if:** We find a change that cannot be safely tested in prod incrementally.

### Decision: Change one thing at a time
**Context:** Several plausible optimizations overlap, and bundled changes make attribution weak.
**Choice:** Apply, deploy, and measure one experiment at a time.
**Rationale:** Keeps causality clean and rollback cheap.
**Revisit if:** Two changes are inseparable in practice.

### Decision: Optimize without degrading UX
**Context:** Early ideas that removed or delayed visible functionality lower load by lowering product quality.
**Choice:** Favor no-regression optimizations first.
**Rationale:** The target is a faster site, not a thinner site.
**Revisit if:** The user explicitly wants a tradeoff.

## Consistent Profiling Setup
### Browser cold-load pass
- Tool: `browser-use` with a fresh Chromium session
- Method: open homepage with a cache-busting query param, wait 8s, read `window.performance`
- Capture:
  - navigation TTFB
  - DCL / load duration
  - top slow resources
  - image/XHR/script/link counts

### URL matrix
- Tool: `curl`
- Compare the same path list on:
  - public: `https://crestwoodstopsign.com`
  - direct: `http://100.125.140.78:8002`
- Capture:
  - `time_starttransfer`
  - `time_total`
  - response size
  - response headers when relevant

### Origin diagnostics
- In-container timings against `http://127.0.0.1:8000`
- Cube direct timings against `http://127.0.0.1:8002`
- `/debug-performance` for route-internal DB/Redis/HLS timing
- `docker stats`, `nvidia-smi`, `cloudflared-local` logs for system context

### Experimental cadence
1. Capture the baseline with the protocol above
2. Apply one code or infra change
3. Deploy to production if the hypothesis depends on the public path
4. Re-run the same protocol
5. Record outcome here
6. Keep or revert based on measured result

## Current Production Baseline
Date captured: `2026-04-18`

### App-side diagnostics
- Inside `web_server`, homepage-related routes are fast:
  - `/` about `0.23s`
  - `/api/recent-vehicle-passes` about `0.19s`
  - `/api/live-stats` about `0.20s`
  - `/stream/stream.m3u8` about `0.001s`
- `/debug-performance` on prod origin:
  - DB recent passes: about `99-108ms`
  - DB 24h stats: about `89-93ms`
  - Redis: about `1-2ms`

### Public vs direct-origin measurements
Representative samples:

| Path | Public | Direct cube | Read |
|---|---:|---:|---|
| `/` | `1.27-1.66s` | `0.73s` | Public path adds material latency |
| `/static/base.css` | `1.13s` | `0.35s` | Cached, still slower publicly |
| `/static/js/home.js` | `2.09s` | `0.35s` | Public variance is high |
| `/static/js/video-player.js` | `0.77s` | `0.35s` | Less problematic |
| `/load-video` | `1.22s` | `0.35s` | Extra public roundtrip |
| `/api/recent-vehicle-passes` | `4.83s` TTFB / `9.46s` total | `0.54s` / `0.71s` | Worst startup request |
| `/api/live-stats` | `1.97s` | `0.57s` | Public dynamic slow |
| `/stream/stream.m3u8` | `1.05s` / `1.49s` | `0.34s` / `0.52s` | Media path slower publicly |
| sample thumbnails | `0.84-1.46s` each | `0.52-0.55s` each | Individually okay, collectively expensive |

### Browser cold-load runs
- Navigation TTFB: `732ms` to `1213ms`
- Load / DCL: `3.2s` to `5.4s`
- Slowest resources were usually:
  - `vehicle-images/*`
  - `/stream/stream.m3u8`
  - `/api/recent-vehicle-passes`
- Initial request fan-out observed in the browser:
  - `4` XHR/fetch requests
  - `8` image requests fired during initial load
  - `30` recent-pass cards / `30` recent-pass images present in the DOM

### Cache / delivery observations
- Static CSS/JS are Cloudflare cache hits
- Thumbnail JPEGs are Cloudflare cache hits
- Homepage partials are dynamic
- HLS manifest is `no-store`
- `ASSET_VERSION` currently falls back to process start time, so restarts unnecessarily churn asset URLs

### Control-plane observations
- `crestwoodstopsign.com` is currently `tunnel_public` to cube
- `cloudflared-local` is running an older version (`2025.1.0`) and shows reconnect churn
- `stream.crestwoodstopsign.com` does not currently resolve and is not present in hostname inventory

## Theories / Candidate Experiments

| ID | Change | Theory | Expected outcome | Status |
|---|---|---|---|---|
| E0 | Codify measurement harness | A repeatable harness will reduce false positives and make every later result comparable | Stable before/after measurements for every prod experiment | Complete |
| E1 | Inline video shell in `/` and remove `/load-video` fetch | The extra HTMX request buys no UX benefit and costs one public roundtrip | Lower startup latency with identical visible video panel | Complete |
| E2 | Server-render recent passes in the initial homepage HTML | The worst startup request is the recents partial; moving the same UI into the first response should avoid the slow extra public roundtrip | Faster time-to-useful-content with same cards and images | Complete |
| E3 | Add short edge caching for homepage HTML fragments | The app is fast at origin; public dynamic latency is the real tax | Big reduction in public-path variability for `/`, `/api/recent-vehicle-passes`, `/api/live-stats`, `/load-video` | Reverted |
| E4 | Use a stable build hash for asset versioning | Restart-based asset busting causes unnecessary cold static fetches after deploys | Better cache retention across deploys and container restarts | Planned |
| E5 | Load analytics after page load / idle | Umami should not compete with startup-critical scripts | Lower DCL variance with no visible UX change | Complete |
| E6 | Restore a dedicated direct HTTPS media hostname for HLS | Same-origin tunnel delivery is the wrong transport for low-latency media | Faster live-feed startup and lower HLS manifest latency without changing the panel | Complete (Reverted) |
| E7 | Generate true thumbnail variants for recent-pass cards | The cards render tiny images but download larger originals | Lower image tail latency with unchanged card visuals at current size | Planned |
| E8 | Tune request headers / cache policy for media and thumbnails only if measurements justify it | Existing thumbnail cache hits may already be good enough; optimize only if remaining image tail is still dominant | Possible incremental win, but only after E1-E7 are tested | Planned |
| E9 | Self-host HTMX | External `unpkg` stalls can block startup despite HTMX being a tiny, stable dependency | Lower bad-tail startup variance with no UX change | Complete |
| E10 | Self-host and pin `hls.js` | `@latest` from a third-party CDN adds both startup variance and version drift | More consistent player startup with no UX change | Complete |

## Experiment Log

### E0 — Measurement Harness
- Status: complete
- Planned work:
  - Codify the URL matrix
  - Codify the browser cold-load capture
  - Store outputs in a repeatable form for before/after comparisons
- Predicted result:
  - Less noise, easier regression detection
- Actual result:
  - Added `scripts/profile_homepage_perf.py`
  - Smoke test command:
    - `python3 scripts/profile_homepage_perf.py --runs 1 --image-count 2 --output /tmp/homepage-perf-baseline.json`
  - The harness now captures:
    - public vs direct URL matrix
    - browser cold-load pass
    - startup-window metrics separate from ongoing HLS churn
  - First successful smoke-test output:
    - nav TTFB: `426ms`
    - load: `738ms`
    - startup window: `30` images, `5` XHR, `6` scripts, `2` links
    - startup hot spots included:
      - `/api/recent-vehicle-passes` at about `478ms`
      - `/stream/stream.m3u8` at about `322ms`
      - first `.ts` segment at about `622ms`
  - Important note:
    - single-run public/direct comparisons can invert depending on cache warmth and stream timing, so later experiments should use multiple runs and compare medians rather than one-off numbers

### E1 — Inline Video Shell
- Status: complete
- Predicted result:
  - Remove one extra startup request
  - No visual or behavioral regression
- Actual result:
  - Change deployed on `2026-04-18`
  - The homepage now server-renders the same `partials/video.html` shell directly in `/` instead of fetching it through `hx-get="/load-video"`
  - Public HTML confirmed the new inline `<video id="videoPlayer">` markup after deploy
  - Browser profiler result vs the E0 baseline:
    - baseline startup XHR count: `5`
    - E1 startup XHR count: `4`, `4`, `5`
    - `/load-video` disappeared from the startup request path on all E1 runs
    - nav TTFB stayed in the same range: baseline `426ms`, E1 `413-480ms`
    - load time did not show a consistent improvement: baseline `738ms`, E1 `641-1134ms`
  - Interpretation:
    - this is a real cleanup and a real request reduction
    - it is not a major performance win because the remaining startup cost is still dominated by `/api/recent-vehicle-passes`, HLS segment fetches, and analytics variability
  - Keep/revert:
    - keep
    - it preserves the exact UX, simplifies the page, and removes a pointless roundtrip even though the measured page-level win is small

### E2 — Server-Render Recent Passes
- Status: complete
- Predicted result:
  - Eliminate the slowest startup XHR from the critical path
  - Same visible card list, earlier
- Actual result:
  - Change deployed on `2026-04-18`
  - The homepage now renders the same 30 recent-pass cards directly in `/` instead of waiting on the initial `/api/recent-vehicle-passes` HTMX request
  - Public HTML no longer contains `hx-get="/api/recent-vehicle-passes"` and the cards are present in the first response
  - Browser profiler result vs E1:
    - `/api/recent-vehicle-passes` disappeared from the startup request path on all runs
    - median nav TTFB moved from `423ms` to `506ms`
    - median DCL improved from `1005ms` to `805ms`
    - median load moved from `1034ms` to `1114ms`
    - first recent-pass image requests started earlier: median about `1126ms` in E1 to about `754ms` in E2
  - URL matrix / HTML size result:
    - homepage HTML grew from about `17 KB` to about `51 KB`
    - public `/` TTFB in the matrix moved from about `0.453s` to about `0.566s`
  - Interpretation:
    - this removes the slow recents XHR cleanly and makes the panel content available earlier
    - it does not reduce total work; it shifts the card markup and image discovery into the initial document
    - the trade is earlier visible recents vs a heavier initial HTML response and slightly worse shell TTFB
  - Keep/revert:
    - keep for now
    - it improves perceived completeness of the homepage without removing any functionality, but it is not the standalone fix for overall startup latency

### E3 — Short Edge Caching for Homepage Fragments
- Status: complete
- Predicted result:
  - Large improvement on public-path variability
  - No visible product loss if TTL stays short
- Actual result:
  - Change deployed on `2026-04-18` as a targeted homepage test:
    - `/` returned `Cache-Control: public, max-age=0, s-maxage=10, stale-while-revalidate=60`
    - `CDN-Cache-Control` and `Cloudflare-CDN-Cache-Control` were also set
  - Repeated header checks on the public hostname still showed:
    - `cf-cache-status: DYNAMIC`
    - no `Age` header
    - no transition from MISS to HIT across repeated requests
  - Browser profiler result during the experiment:
    - no consistent improvement over E5
    - public-path variance remained high, including a bad-tail run where the external HTMX script stalled for several seconds
  - Interpretation:
    - app-side cache headers alone do not enable edge HTML caching on the current Cloudflare path
    - to make homepage HTML caching real, this needs Cloudflare-side cache rules or a different delivery setup, not just response headers from FastAPI
  - Keep/revert:
    - reverted
    - the headers were not delivering the intended edge behavior, so they should not stay in prod as a misleading no-op

### E4 — Stable Asset Versioning
- Status: complete
- Predicted result:
  - Fewer cold static fetches after deploys
  - Little change on a single already-cold run, larger change across deploy churn
- Actual result:
  - Change deployed on `2026-04-18`
  - Static asset URLs now use a content hash instead of process start time:
    - before: timestamp-like values such as `base.css?v=1776530771`
    - after: stable content hash `base.css?v=2a721aeab080`
  - Verification deploy sequence:
    - deploy 1 switched production HTML to the hash-based versioned asset URLs
    - deploy 2 was a no-content redeploy of the same code
    - the asset version stayed exactly the same across both deploys:
      - `/static/base.css?v=2a721aeab080`
      - `/static/js/video-player.js?v=2a721aeab080`
      - `/static/js/home.js?v=2a721aeab080`
  - Cache verification on the new hashed asset URL:
    - first direct CSS request: `cf-cache-status: MISS`
    - immediate second request: `cf-cache-status: HIT`
  - Browser profiler result after the change:
    - 2-run median nav TTFB: about `512ms`
    - 2-run median DCL: about `797ms`
    - 2-run median load: about `806ms`
  - Interpretation:
    - this is mostly a cache-correctness and deploy-churn fix, not a dramatic first-load speed win
    - the important result is that restarts and no-op redeploys no longer force brand-new asset URLs
    - that should preserve browser and CDN cache usefulness across routine deploys instead of making every user refetch CSS/JS after each restart
  - Keep/revert:
    - keep

### E5 — Post-Load Analytics
- Status: complete
- Predicted result:
  - Lower startup contention and fewer bad-tail cases
  - No visible regression
- Actual result:
  - Change deployed on `2026-04-18`
  - The Umami script is no longer a static head `<script ... defer>` tag; the page now injects it after `load` via `requestIdleCallback` with a short timeout fallback
  - Browser profiler result vs E2:
    - median nav TTFB moved from `506ms` to `527ms`
    - median DCL moved from `805ms` to `861ms`
    - median load improved from `1114ms` to `1040ms`
    - startup XHR count moved from `4,4,4` to `3,4,4`
  - Important observation:
    - the analytics script still showed up inside the startup window on `1/3` runs because the page finished `load` quickly and the idle callback fired immediately after
    - the analytics send request still appeared in the startup window on all runs
  - Interpretation:
    - this is a small cleanup, not a primary lever
    - moving analytics out of the head reduces some overlap with the core app path, but the current implementation still allows analytics traffic to land during the first couple of seconds
  - Keep/revert:
    - keep
    - harmless and directionally better, but not enough on its own; if we want analytics fully out of startup we need to delay it more aggressively than “right after load”

### E6 — Dedicated Direct HTTPS HLS Hostname
- Status: complete, reverted in production
- Predicted result:
  - Meaningful live-feed startup improvement
  - Cleaner separation of dashboard traffic vs media traffic
- Actual result:
  - The first direct-to-cube version of this idea was fully investigated and rejected:
    - the historical router docs were stale
    - the old `8443` forwarding path actually targets internal `8002`
    - even after correcting the stale router target and opening the host firewall, public packets still never reached cube
    - that path is not a viable browser-facing solution
  - The safer version was then implemented on `2026-04-18`:
    - added `stream.crestwoodstopsign.com` on `clifford`
    - installed a manual Caddy route that proxies `/stream/*` and `/health/stream` to cube over Tailscale
    - terminated public HTTPS on `clifford`
    - constrained CORS to `https://crestwoodstopsign.com`
    - tightened cube `8002/tcp` so only `clifford`'s Tailscale IP (`100.118.94.100`) can reach it
    - added app support for an external `PUBLIC_STREAM_URL`
    - deployed production with the homepage player pointed at `https://stream.crestwoodstopsign.com/stream/stream.m3u8`
  - The production measurements did not support keeping it:
    - page-shell metrics stayed roughly flat vs the pre-E6 baseline:
      - pre-E6 median nav TTFB about `512ms`
      - E6 median nav TTFB about `515ms`
      - pre-E6 median DCL about `797ms`
      - E6 median DCL about `815ms`
      - pre-E6 median load about `806ms`
      - E6 median load about `819ms`
    - the HLS startup path got materially worse in the browser:
      - pre-E6 startup manifest median about `195ms`
      - E6 startup manifest median about `521ms`
      - pre-E6 startup segment median about `804ms`
      - E6 startup segment median about `4198ms`
    - direct like-for-like public fetches made the regression unambiguous:
      - tunnel manifest: about `240-255ms`
      - clifford manifest: about `529-731ms`
      - tunnel first segment: about `0.57-0.89s`
      - clifford first segment: about `3.42s`, `8.22s`, and `17.16s`
  - First-principles interpretation:
    - the clifford media proxy removes Cloudflare Tunnel from the path, but it also hairpins every large HLS segment through `cube -> clifford -> browser`
    - from this measurement location, that doubled path is worse than letting Cloudflare carry the stream from cube to a nearby edge POP
    - the result is that the architecture is cleaner, but the actual delivered video bytes arrive slower
  - Revert / post-check:
    - production was reverted to same-origin `/stream/stream.m3u8`
    - the homepage HTML is back to `data-stream-url="/stream/stream.m3u8"`
    - short post-revert sanity profiling showed the HLS startup path back near the earlier baseline:
      - revert startup manifest median about `195ms`
      - revert startup segment median about `765ms`
  - Keep/revisit:
    - keep the generic app support for `PUBLIC_STREAM_URL`; it is a harmless capability hook for future experiments
    - keep cube `8002/tcp` restricted to `clifford` only; this closes the earlier broad exposure
    - do not use the clifford proxy path for production streaming in its current form
    - if live-feed performance work resumes, the next path should be a transport change that does not relay each HLS segment through a second public server

### E7 — Thumbnail Variants
- Status: complete
- Predicted result:
  - Lower image tail without losing the recent-pass panel
  - Smaller payoff than fixing the public-path partials, but still real
- Actual result:
  - Change deployed on `2026-04-18`
  - Recent-pass cards now request a dedicated `/vehicle-thumb/{object}` URL while keeping the existing full-size image URL for detail views and card links
  - The thumbnail route generates a `112x80` JPEG, caches it locally on cube, and returns immutable public cache headers
  - Public cache verification on a warmed thumbnail showed:
    - `cf-cache-status: HIT`
    - `age: 85`
    - `cache-control: public, max-age=31536000, immutable`
  - Sample byte comparison on five live cards:
    - full images: median about `24.5 KB`
    - thumbnails: median about `3.6 KB`
    - median byte reduction: about `85.5%`
  - Browser profiler result vs E10:
    - median nav TTFB improved from `544ms` to `511ms`
    - median DCL improved from `920ms` to `759ms`
    - median load improved from `924ms` to `762ms`
  - Cold/warm behavior matters:
    - first browser run after deploy still showed thumbnail requests around `592-596ms` each during startup
    - by runs 2 and 3, warmed thumbnail requests in the startup window were down to about `62-65ms`
  - Interpretation:
    - this is a real no-regression win
    - the cards keep the same count, layout, and click-through behavior, but the browser and edge move far fewer bytes
    - the remaining biggest startup costs are still HLS segment fetches and the analytics send request, not the card images
  - Keep/revert:
    - keep

### E8 — Request Header / Cache Tuning Follow-Up
- Status: not started
- Predicted result:
  - Possible incremental win once media and image paths are isolated more clearly
- Actual result:
  - TBD

### E9 — Self-Host HTMX
- Status: complete
- Predicted result:
  - Remove one third-party startup dependency from the critical path
  - Reduce bad-tail cases caused by `unpkg.com`
- Actual result:
  - Change deployed on `2026-04-18`
  - The site now serves the exact existing HTMX version from `/static/vendor/htmx-1.9.4.min.js` instead of loading from `https://unpkg.com/htmx.org@1.9.4`
  - Public HTML confirmed the local script path, and Cloudflare treated it as a normal static asset (`cf-cache-status: MISS` on first fetch, then cacheable)
  - Browser profiler result vs E5:
    - median nav TTFB moved from `527ms` to `523ms`
    - median DCL improved from `861ms` to `765ms`
    - median load improved from `1040ms` to `1028ms`
    - `unpkg.com/htmx.org` disappeared from the startup request path on all runs
  - Interpretation:
    - this is a real no-regression improvement
    - the gain is modest on median timings, but it removes one observed bad-tail source from the critical path and makes startup more reproducible
  - Keep/revert:
    - keep

### E10 — Self-Host and Pin `hls.js`
- Status: complete
- Predicted result:
  - Remove `jsdelivr` variance and eliminate `@latest` version drift
  - Make player startup more reproducible across deploys
- Actual result:
  - Change deployed on `2026-04-18`
  - The site now serves the exact currently-live `hls.js` version from `/static/vendor/hls-1.6.16.min.js` instead of `https://cdn.jsdelivr.net/npm/hls.js@latest`
  - Public HTML confirmed the local pinned script path
  - Browser profiler result vs E9:
    - median nav TTFB moved from `523ms` to `544ms`
    - median DCL moved from `765ms` to `920ms`
    - median load improved from `1028ms` to `924ms`
    - `jsdelivr` disappeared from the startup request path on all runs
    - the local HLS script appeared in the startup path on all runs, as expected
  - Interpretation:
    - this is a determinism / dependency-control win more than a clear median-speed win
    - it removes third-party CDN dependency and eliminates `@latest` drift, but the measured startup improvement is mixed rather than dramatic
  - Keep/revert:
    - keep

## Notes
- The first two suggested “optimizations” from the earlier audit were rejected because they degraded the product. This experiment plan intentionally avoids that class of change.
- The strongest current evidence is that the public delivery path is the dominant problem, not the Python app itself.
- The profiling harness now resolves the current versioned local CSS/JS asset URLs from the homepage HTML before running the public/direct URL matrix. That avoids stale query-string cache keys skewing later comparisons.

## Operational Note — Detection Gap on 2026-04-18
- Trigger:
  - During the performance work, the homepage showed `lastDetection` at about `58-59m ago`, which looked suspicious for a clear Saturday around noon.
- What was verified:
  - This was a real data gap, not just stale frontend HTML.
  - Direct DB query showed the most recent pass timestamps in Central Time were:
    - `2026-04-18 11:21`
    - then nothing until `2026-04-18 12:20:43`
    - then another pass at `2026-04-18 12:23:03`
  - `vehicle_passes` counts confirmed the same gap even when checking all rows, not just rows used by the recent-pass UI.
- Was it caused by the latest homepage deploy?
  - No evidence for that.
  - The current app containers restarted at about `2026-04-18 11:46:07` Central, which is about 25 minutes after the `11:21` detection gap began.
  - So the gap started before the latest restart / deploy.
- What the live pipeline showed during the investigation:
  - `rtsp_to_redis` stayed connected to both Redis and RTSP, but it showed intermittent input-rate collapses during the same general window:
    - about `1.30 FPS` at `11:17`
    - about `1.92 FPS` at `11:37`
    - about `0.19 FPS` at `11:47`
    - then back to about `14.98 FPS`
  - After the `11:46` restart, `video_analyzer` logged repeated `No frame available in Redis` warnings from about `11:46` through `12:00`, then more sporadically around `12:10-12:11` and once again at `12:19`.
  - `ffmpeg_service` was not dead during that time. It kept serving HLS, but logs showed short periods of degraded freshness / backlog, including:
    - new-frame rate dips
    - elevated duplicate / snap counts
    - FIFO backlog trimming events
- Current state at the end of the investigation:
  - The system recovered without any code changes.
  - Live checks from inside the containers showed:
    - analyzer `/ready`: `200`, `frame_lag_seconds` about `0.03s`
    - ffmpeg `/ready`: `200`, `hls_ok=true`, `redis_ok=true`, `recent_frame_ok=true`
    - web `/health`: `200`
    - web `/health/stream`: `fresh=true`
  - Redis state at recovery:
    - `raw_frame_buffer`: `0`
    - `processed_frame_buffer`: `9`
    - latest frame metadata showed fresh capture timestamps and active tracked cars
  - Public homepage stats also recovered during the check, moving from about `59m ago` to about `2m ago`
- Interpretation:
  - There was a real detection lull visible in prod.
  - The timing does not support “the homepage perf changes caused detection to stop.”
  - The best evidence points to transient ingest / pipeline instability on the camera-to-analyzer path, not a web rendering change.
  - The exact size of the “real outage” versus “just an unusually quiet traffic window” remains mixed:
    - the `11:21-12:20` Saturday window had only `1` pass today
    - prior four Saturdays in that same window were `5`, `1`, `8`, `8`
  - So today was low and suspicious, but not statistically impossible even before considering the observed ingest jitter.
