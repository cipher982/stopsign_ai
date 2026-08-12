# Crestwood Stop Sign: Dataset, Vision, and the "Label Everything" Pivot

Status: EXECUTED 2026-08-11 — `scripts/label_passes.py` labeled 67,742 of 68,276
image-bearing passes (99.2%, $25.16, 0 failures) into `vehicle_labels`.
Remaining: point /vehicles at the new table; schedule incremental labeling.
The rest of this doc is the original decision-support text.
All numbers below were measured live on 2026-08-10 (SQL counts and real OpenAI API usage), not guessed.

## 1. Context

`crestwoodstopsign.com` ("Stop Sign Nanny") is a public, personal-project website
watching a single 4-way stop sign at Crestwood and [street] in David's neighborhood.
A fixed RTSP camera feeds a YOLO tracker that records every vehicle pass: entry/exit
time, time-in-zone, min speed, stop compliance, a JPEG crop of the vehicle, and a
short replay clip. A Postgres DB holds ~72,760 passes going back to Oct 2024.
The site has a Windows-98 desktop aesthetic, is a one-developer project, generates
no revenue, and costs almost nothing to run. Cloudflare in front, cube (RTX 3090)
behind.

What makes this dataset unusual: it is a **complete, continuous, multi-year record
of one physical intersection**, with per-vehicle images and clips, stop-compliance
labels, and timestamps. That is closer to a civic-sensor dataset than a demo.

## 2. The dataset (measured)

- 72,760 vehicle passes, Oct 2024 -> now. ~210/day.
- 68,088 passes have a saved JPEG crop (29 KB avg). 4,695 have no image (a
  historical capture bug; future passes always retain images after a July 2026
  archive fix).
- Live compliance ~93% (passes stopping >= 2s in the stop zone); ~3 violations/day.
- 52,471 passes have `vehicle_attributes` (type/color/make/confidence) written by
  an offline pipeline; **last written 2026-02-06. Classification is frozen 6
  months**; ~24k passes since then have zero classification.
- The 52,471 attributes came from 1,215 clusters: DINOv2 embeddings + UMAP/HDBSCAN,
  then ONE OpenAI vision call per cluster representative, label propagated to all
  cluster members.
- **40% of attributes (20,927) are cluster noise: never labeled, invisible on the
  site.**
- Measured per-pass LLM cost (gpt-5.6-luna, real API usage, 20-sample): ~178
  prompt tokens + ~226 completion tokens = ~$0.000307/pass.
  - Label ALL 68,088 image-bearing passes: ~$21 one-time.
  - Incremental: ~$0.065/day, ~$24/year.
  - Runtime: ~3.4 s/call; with 20-50 concurrent workers, 1-3 hours. Resumable.
- Model: `gpt-5.6-luna` (OpenAI direct). $0.20/1M input, $1.20/1M output.

## 3. Current product

Public pages: Live (HLS stream + live stats + compliance), Records (recent passes
with images), Vehicles (classification explorer: summary, type/color breakdowns,
top make/models with photos, cluster gallery), About. The Vehicles page is the
dataset's showcase and currently renders a 6-month-stale snapshot with 40% of its
data invisible.

## 4. Pipeline history (why it's shaped this way)

Original design: `download` -> `embed` (DINOv2) -> `cluster` (UMAP+HDBSCAN) ->
`label` (one OpenAI vision call per cluster representative). David's stated reason
for clustering: **"if you ask to describe a car N times, freeform text is so
high-dimensional it's hard to aggregate/analytics on"** — i.e. clustering was the
aggregation mechanism because freeform descriptions were unaggregatable.

That pipeline froze Feb 2026 at the embed stage (never resumed; 15,617 passes
pending). The label stage has no backlog (all 1,215 reps labeled).

## 5. The pivot: label EVERYTHING (decided)

Decision: every image-bearing pass gets its own gpt-5.6-luna vision call, and the
output is **closed-vocabulary structured fields, not freeform prose**. This
dissolves the original aggregation problem at the source: clustering is demoted
from "labeling substrate" to "optional garnish for a regulars view". Per-pass
labels join directly with compliance, hour, weather, clips.

Proposed schema (per pass): vehicle_type enum, color enum, make enum (24 common +
other/unknown), model (canonical, validated), roof_rack/commercial/damage bools,
passengers int, headlights_on bool, per-field confidence, plus a freeform
`description` column that is human-only (never aggregated). Prompt ships the enum
lists; JSON validated; one retry on failure. Idempotent upsert by pass_id.
~$21 to label full history, then ~$0.065/day.

What this buys: no noise hole (all 68k image passes labeled), no cluster
inheritance error (a bad representative image no longer mislabels hundreds of
vehicles), real analytics (SQL over 68k rows), consistent schema across the whole
history, confidence thresholding.

## 6. The question for you

Given this dataset (2 years of one intersection: every vehicle labeled with
type/color/make/model/flags + stop compliance + images + clips + timestamps),
modern AI (vision labeling at fractions of a cent per image, LLM narrative
generation, retrieval over the corpus), and the constraint set (public site,
personal project, Windows-98 aesthetic, one developer, near-zero budget, batch
pipeline, data refreshes daily):

**What is the WOW-UX? What should we actually build?**

Deliver:
1. A one-paragraph vision statement for what the site becomes.
2. A shortlist of 3-5 wow features, each with: name, one-line pitch, why it's wow,
   what dataset/AI capability it needs, rough effort (S/M/L), and the main risk.
3. The ONE feature you'd build first, with a 3-step build path.
4. Anything you notice about the dataset that the owner may be underusing.
5. Which ideas are gimmicks and should be rejected, and why.

Prior brainstorm (owner's own list, for reference, may be stolen or rejected):
honesty meter for stale classification; cluster spotlight ("neighborhood
regulars"); compliance by make ("Volvo XC60s stop 4.2s avg"); hour-of-day
time-lapse of the traffic mix; seasonal trend; vehicle "mugshots" wall; a
"guess the stop" game; weather x compliance correlation; worst-offender
leaderboard.

Constraints on answers: be concrete and specific to THIS dataset, not generic
"add charts" advice. Assume the $21 labeling run happens and analytics are
possible over every vehicle. Prefer ideas that compound (one feature feeds the
next). No revenue/monetization talk.
