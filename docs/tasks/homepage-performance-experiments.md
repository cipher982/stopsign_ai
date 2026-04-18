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
| E0 | Codify measurement harness | A repeatable harness will reduce false positives and make every later result comparable | Stable before/after measurements for every prod experiment | Planned |
| E1 | Inline video shell in `/` and remove `/load-video` fetch | The extra HTMX request buys no UX benefit and costs one public roundtrip | Lower startup latency with identical visible video panel | Planned |
| E2 | Server-render recent passes in the initial homepage HTML | The worst startup request is the recents partial; moving the same UI into the first response should avoid the slow extra public roundtrip | Faster time-to-useful-content with same cards and images | Planned |
| E3 | Add short edge caching for homepage HTML fragments | The app is fast at origin; public dynamic latency is the real tax | Big reduction in public-path variability for `/`, `/api/recent-vehicle-passes`, `/api/live-stats`, `/load-video` | Planned |
| E4 | Use a stable build hash for asset versioning | Restart-based asset busting causes unnecessary cold static fetches after deploys | Better cache retention across deploys and container restarts | Planned |
| E5 | Load analytics after page load / idle | Umami should not compete with startup-critical scripts | Lower DCL variance with no visible UX change | Planned |
| E6 | Restore a dedicated direct HTTPS media hostname for HLS | Same-origin tunnel delivery is the wrong transport for low-latency media | Faster live-feed startup and lower HLS manifest latency without changing the panel | Planned |
| E7 | Generate true thumbnail variants for recent-pass cards | The cards render tiny images but download larger originals | Lower image tail latency with unchanged card visuals at current size | Planned |
| E8 | Tune request headers / cache policy for media and thumbnails only if measurements justify it | Existing thumbnail cache hits may already be good enough; optimize only if remaining image tail is still dominant | Possible incremental win, but only after E1-E7 are tested | Planned |

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
- Status: not started
- Predicted result:
  - Large improvement on public-path variability
  - No visible product loss if TTL stays short
- Actual result:
  - TBD

### E4 — Stable Asset Versioning
- Status: not started
- Predicted result:
  - Fewer cold static fetches after deploys
  - Little change on a single already-cold run, larger change across deploy churn
- Actual result:
  - TBD

### E5 — Post-Load Analytics
- Status: not started
- Predicted result:
  - Lower startup contention and fewer bad-tail cases
  - No visible regression
- Actual result:
  - TBD

### E6 — Dedicated Direct HTTPS HLS Hostname
- Status: not started
- Predicted result:
  - Meaningful live-feed startup improvement
  - Cleaner separation of dashboard traffic vs media traffic
- Actual result:
  - TBD

### E7 — Thumbnail Variants
- Status: not started
- Predicted result:
  - Lower image tail without losing the recent-pass panel
  - Smaller payoff than fixing the public-path partials, but still real
- Actual result:
  - TBD

## Notes
- The first two suggested “optimizations” from the earlier audit were rejected because they degraded the product. This experiment plan intentionally avoids that class of change.
- The strongest current evidence is that the public delivery path is the dominant problem, not the Python app itself.
- The profiling harness now resolves the current versioned local CSS/JS asset URLs from the homepage HTML before running the public/direct URL matrix. That avoids stale query-string cache keys skewing later comparisons.
