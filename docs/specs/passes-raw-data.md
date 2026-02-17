# Passes + Raw Tracking Persistence (SDP-1)

## Executive Summary
We will introduce a durable “Passes” data model that stores both fast summary records and full raw tracking payloads for every vehicle pass. The raw payload will capture the *entire tracked lifespan* of a pass (from first detection to pass completion), plus a frozen config/model snapshot for reproducible analysis. We will then build a `/passes/{id}` detail page that displays the clip, capture image, and key facts, and link to it from existing pass lists.

## Requirements
- Persist all raw tracking/source data for each pass (entire tracked lifespan).
- Persist config + model metadata with each pass for future analysis.
- Pass detail page lives under `/passes/{id}` and is the new destination for clip links.
- Keep list views fast (do not load raw JSON payloads for lists).
- Commit frequently with SDP-1 format.

## Decision Log
### Decision: Separate raw table (`vehicle_pass_raw`) from summary table
**Context:** Need to store large raw payloads without slowing list queries.
**Choice:** Add `vehicle_pass_raw` with a JSONB payload linked by `vehicle_pass_id`.
**Rationale:** Keeps `vehicle_passes` lightweight while allowing full-fidelity storage.
**Revisit if:** We move to a columnar/analytics store.

### Decision: Use compact sample schema arrays
**Context:** Raw samples can grow large.
**Choice:** Store samples as arrays + `sample_schema` instead of dicts.
**Rationale:** Smaller payload size and faster serialization.
**Revisit if:** Readability becomes more important than size.

### Decision: Coordinate space = processed
**Context:** Tracking operates in processed (cropped/scaled) coordinates.
**Choice:** Store samples in processed coordinates and persist crop/scale in snapshot.
**Rationale:** Avoids recomputation; raw conversion is possible later.
**Revisit if:** We need raw-space overlays frequently.

### Decision: Pass lifespan ends at pass completion
**Context:** Tracking resets after pass completion.
**Choice:** Capture samples from first detection through pass completion; mark raw payload complete at exit.
**Rationale:** Aligns with current pass definition and avoids mixing multiple passes.
**Revisit if:** We decide to bind passes to full track-to-removal lifecycle.

### Decision: Snapshot config + model metadata in raw payload
**Context:** Configs and model settings drift over time.
**Choice:** Store config + model snapshot per pass.
**Rationale:** Enables reproducible analysis even after config changes.
**Revisit if:** We implement a separate versioned config registry.

### Decision: Raw payload snapshots use live config + YOLO metadata at pass completion
**Context:** Need consistent, low-latency capture of config/model state without extra DB lookups.
**Choice:** Persist a deep-copied config snapshot plus YOLO model name/path/device, thresholds, and providers from the running analyzer.
**Rationale:** Captures the exact runtime settings with minimal overhead.
**Revisit if:** We add a centralized model registry or config/version service.

## Data Model
### Existing: `vehicle_passes` (summary)
Keep as-is. Optional future additions (not required for initial work): `camera_id`, `analysis_version`.

### New: `vehicle_pass_raw`
- `id` BIGSERIAL
- `vehicle_pass_id` BIGINT UNIQUE FK → `vehicle_passes.id`
- `raw_payload` JSONB
- `sample_count` INT
- `raw_complete` BOOL
- `created_at`, `updated_at`

## Raw Payload Schema (v1)
```json
{
  "version": 1,
  "coordinate_space": "processed",
  "sample_schema": ["t","x","y","x1","y1","x2","y2","raw_speed","speed"],
  "samples": [
    [1708200123.12, 512.3, 284.9, 490.1, 260.2, 540.9, 320.7, 23.4, 21.2]
  ],
  "summary": {
    "entry_time": 1708200124.01,
    "exit_time": 1708200129.88,
    "time_in_zone": 5.87,
    "stop_duration": 2.14,
    "min_speed": 8.7,
    "stop_position": [503.2, 291.7],
    "image_path": "local://vehicle_....jpg",
    "clip_path": "clip_123_1708200129.mp4"
  },
  "dimensions": {
    "raw": {"width": 2560, "height": 1440},
    "cropped": {"width": 2304, "height": 1296},
    "processed": {"width": 1152, "height": 648}
  },
  "config_snapshot": { ... },
  "model_snapshot": { ... },
  "raw_complete": true
}
```

## Capture Flow
1. Add `CarState.samples` to collect sample tuples on every update.
2. When a pass completes, insert `vehicle_passes` and immediately store `vehicle_pass_raw` payload.
3. Mark `raw_complete=true` and `sample_count=len(samples)`.

## API / Pages
- `/passes/{id}`: detail page (clip, capture image, facts, classification, config snapshot).
- Existing list partials link to `/passes/{id}` instead of direct clip URLs.

## Phases
### Phase 0: Spec (current)
**Acceptance:** Spec + task doc committed.

### Phase 1: Raw Capture + Data Model
**Acceptance:**
- New `vehicle_pass_raw` table exists (SQLAlchemy model).
- Raw payload stored for each pass.
- Config + model snapshot included.
- Pass creation returns `id` for linking.

### Phase 2: Pass Detail Page + Linking
**Acceptance:**
- `/passes/{id}` renders a detail page with clip + capture + facts.
- Pass list items link to `/passes/{id}`.
- No nested anchors in templates.

## Test Commands
- `uv run pytest` (optional; run only if already passing locally)
