# Tracking Pipeline Rewrite Spec

**Date:** 2026-02-16
**Status:** Final — reviewed by Codex, ready for implementation
**Scope:** `stopsign/tracking.py`, `stopsign/kalman_filter.py`, new modules
**Review:** See `docs/tracking-rewrite-review.md` for Codex critique

## Problem Statement

The tracking pipeline in `tracking.py` (~775 lines) has grown organically and accumulated several design issues:

1. **Three competing speed calculations** — Kalman velocity (computed but discarded), `speed` (10-frame median + EMA), `raw_speed` (6-frame median + EMA). Two are exposed as state, the best one is thrown away.
2. **Monolithic file** — Car state machine, stop zone detection, image capture, and upload infrastructure all in one file.
3. **CarState is a flat bag of 25+ fields** — Implicit coupling between fields, manual resets of 12+ fields, easy to miss one.
4. **Fragile execution ordering** — `car.update()` must run before `stop_detector.update_car_stop_status()` because they share mutable state. The `prev_update_time` fix is a workaround for this.
5. **Frame-count debouncing on zone entry/exit** — `in_zone_frame_threshold` and `out_zone_frame_threshold` have the same FPS-dependency we already fixed for parked detection.
6. **Direction calculation** — `_update_direction()` computes a least-squares regression every frame; result is only used in debug overlay labels, not decision logic.

## Goals

- Single source of truth for vehicle speed
- Clear module boundaries with single responsibilities
- Grouped state that resets atomically
- Time-based zone debouncing (FPS-independent)
- Remove fragile ordering dependency between update() and stop detection
- No behavior regressions — validation script confirms metrics stay stable

## Non-Goals

- Changing the YOLO detection or model
- Changing the database schema
- Changing the web UI
- Changing the HLS/clip pipeline

---

## Execution Order (revised per review)

Low-risk structural changes first. Speed model last behind a flag.

| Order | Phase | Risk | LOC Impact |
|-------|-------|------|------------|
| 1 | Extract image module | Minimal | Net zero (move), -130 lines from tracking.py |
| 2 | Explicit timestamp passing | Low | ~-5 lines, removes ordering footgun |
| 3 | Make direction lazy | Low | ~-20 lines from hot path |
| 4 | Group CarState into sub-states | Medium | ~+20 lines, many accessor changes |
| 5 | Time-based zone debouncing | Medium | ~+15 lines, needs config + tuning |
| 6 | Unify speed model (Kalman) | High | -30 lines, behind flag, needs validation |

---

## Phase 1: Extract Image Saving / Upload Module

### Current State

`tracking.py` contains:
- `save_vehicle_image()` — crop, save locally, queue Bremen upload (~60 lines)
- `_bremen_upload_worker()` — background thread with retry (~50 lines)
- `_prune_old_images()` — disk space management (~25 lines)
- Module-level `_upload_queue`, `_worker_started`, `_worker_lock`

None of this is related to tracking.

### Target State

New module: `stopsign/image_storage.py`
- Contains `save_vehicle_image()`, upload worker, pruning
- Exports a single function: `save_vehicle_image(frame, timestamp, bbox, db) -> str`
- No changes to the interface — just moves the code

### Implementation

1. Create `stopsign/image_storage.py`
2. Move `save_vehicle_image`, `_bremen_upload_worker`, `_prune_old_images`, `_start_upload_worker`, and the module-level queue/lock/flag
3. Move related imports (`uuid`, `queue`, `threading`, `Path`, `Minio`, Bremen settings) to new module
4. Update import in `tracking.py`: `from stopsign.image_storage import save_vehicle_image`
5. Remove moved code from `tracking.py`
6. Clean up unused imports in `tracking.py` (`uuid`, `queue`, `threading`, `Path`, `Minio`, etc.)

### Risk

Minimal — pure code move with no behavior change.

---

## Phase 2: Explicit Timestamp Passing

### Current State

```python
# video_analyzer.py line 564
self.car_tracker.update_cars(boxes, ts_for_logic, processed_frame)  # sets last_update_time
# video_analyzer.py line 589
self.stop_detector.update_car_stop_status(car, ts_for_logic, processed_frame)  # reads prev_update_time
```

Stop detector reads `car.state.prev_update_time` because `car.update()` already overwrote `last_update_time`. If call order changes, stop_duration breaks silently again.

### Target State

`car.update()` returns `prev_timestamp` so the caller can pass it through:

```python
# Car.update() returns prev_timestamp
def update(self, location, timestamp, bbox) -> float:
    prev = self.state.last_update_time
    self._update_location(location, timestamp)
    # ...
    return prev

# video_analyzer.py
prev_ts = car.update(location, timestamp, bbox)
# later...
self.stop_detector.update_car_stop_status(car, timestamp, prev_ts, frame)
```

### Implementation

1. `Car.update()` captures and returns `prev_timestamp` before overwriting
2. Remove `prev_update_time` from `CarState`
3. `update_car_stop_status()` accepts `prev_timestamp` parameter
4. `_update_stop_duration()` uses the explicit parameter
5. Update `video_analyzer.py` to thread the value through
6. Update `CarTracker.update_cars()` to store prev_timestamps for the stop detection loop

### Risk

Low — straightforward parameter threading. The interface change touches video_analyzer.py but is mechanical.

---

## Phase 3: Make Direction Lazy

### Current State

`_update_direction()` runs a least-squares regression every frame (~25 lines). The result (`state.direction`) is only used in two debug overlay labels in `video_analyzer.py` (lines 904, 919).

### Target State

Remove `_update_direction()` from the per-frame `update()` call. Provide a `get_direction()` method that computes on demand. Debug overlay calls `car.get_direction()` instead of reading `car.state.direction`.

### Implementation

1. Remove `self._update_direction()` call from `Car.update()`
2. Convert `_update_direction` to a public `get_direction() -> float` method
3. Remove `direction` field from `CarState`
4. Update `video_analyzer.py` lines ~904 and ~919 to call `car.get_direction()`

### Risk

Low — computation moves from every frame to only when debug overlay renders. No decision logic affected.

---

## Phase 4: Group CarState into Sub-States

### Current State

25+ fields in one flat dataclass. `_reset_car_state` manually zeros 12+ fields.

### Target State

```python
@dataclass
class MotionState:
    speed: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    is_parked: bool = True
    stationary_since: float = 0.0
    moving_since: float = 0.0

@dataclass
class ZoneState:
    in_zone: bool = False
    passed_pre_stop: bool = False
    entry_time: float = 0.0
    exit_time: float = 0.0
    time_in_zone: float = 0.0
    consecutive_in_frames: int = 0
    consecutive_out_frames: int = 0
    speed_samples: List[float] = field(default_factory=list)
    min_speed: float = float("inf")
    stop_duration: float = 0.0
    stop_position: Point = (0.0, 0.0)

@dataclass
class CaptureState:
    image_captured: bool = False
    image_path: str = ""

@dataclass
class CarState:
    location: Point = (0.0, 0.0)
    bbox: Tuple = (0.0, 0.0, 0.0, 0.0)
    raw_speed: float = 0.0          # kept until Phase 6
    speed: float = 0.0              # kept at top level (used everywhere)
    prev_speed: float = 0.0
    track: List = field(default_factory=list)
    last_update_time: float = 0.0
    motion: MotionState = field(default_factory=MotionState)
    zone: ZoneState = field(default_factory=ZoneState)
    capture: CaptureState = field(default_factory=CaptureState)
```

Reset: `car.state.zone = ZoneState()` — one line, can't miss a field.

### Implementation

1. Define sub-dataclasses above `CarState`
2. Move fields into sub-states
3. Update all accessors (grep + mechanical replacement):
   - `car.state.in_stop_zone` → `car.state.zone.in_zone`
   - `car.state.is_parked` → `car.state.motion.is_parked`
   - `car.state.image_captured` → `car.state.capture.image_captured`
   - etc.
4. Replace `_reset_car_state` body with sub-state reconstruction
5. Update tests and `video_analyzer.py` accessors
6. Note: `track` is persisted to DB via `Database.save_car_state` — structure must not change

### Risk

Medium — many accessor changes across files. No behavior change but needs careful grep. Run full test suite after.

---

## Phase 5: Time-Based Zone Debouncing

### Current State

```python
if car.state.consecutive_in_zone_frames >= self.in_zone_frame_threshold:
    # enter zone
elif car.state.consecutive_out_zone_frames >= self.out_zone_frame_threshold:
    # exit zone
```

Frame-count thresholds. At 15 FPS, `in_zone_frame_threshold=3` = 200ms. At 8 FPS, same = 375ms.

### Target State

Track `first_seen_in_zone` and `first_seen_out_zone` timestamps. Compare elapsed time against configurable thresholds.

### Implementation

1. Add `in_zone_time_threshold` and `out_zone_time_threshold` to Config (default 0.2s each, matching current ~3 frames at 15 FPS)
2. Add `first_seen_in_zone: float = 0.0` and `first_seen_out_zone: float = 0.0` to ZoneState
3. Replace frame-count comparisons with time comparisons
4. Guard against large `dt` gaps (frame drops, analyzer catch-up) — clamp elapsed time or require minimum consecutive observations
5. Keep frame counters for debugging/logging but remove from decision logic

### Risk

Medium — needs tuning. Large `dt` gaps from frame drops could cause instant zone entry/exit. Mitigate with a minimum observation count (e.g., at least 2 observations within the time window).

---

## Phase 6: Unify Speed Model (Kalman velocity) — FLAGGED

**Per review:** This phase ships behind `tracking.use_kalman_velocity` config flag (default: false). Current `raw_speed` + `speed` pipeline remains the default until validated.

### Current State

Three speed computations: Kalman velocity (discarded), `speed` (10-frame median + EMA), `raw_speed` (6-frame median + EMA).

### Target State (when flag enabled)

```
YOLO bbox center → Kalman filter → kf.x[:2] = position, kf.x[2:4] = velocity
                                  → state.speed = norm(kf.x[2:4])
```

### Known Risks (from review)

1. **Kalman tuning** — Current `process_noise=0.1`, `measurement_noise=1.0` is tuned for smooth position, not responsive velocity. Q becomes very small at actual dt (~0.067s), making velocity slow to react to stops. Needs retuning.
2. **Warm-up** — Filter starts with `v=0` and large covariance. Early frames will have incorrect velocity. Gate on minimum track length (e.g., 10 updates) before using Kalman velocity for stop logic.
3. **Threshold recalibration** — `stop_speed_threshold`, `max_movement_speed`, `unparked_speed_threshold` are calibrated on `raw_speed`. Kalman velocity will have different scale/noise characteristics. Must recalibrate.
4. **Interpolation change** — `get_interpolated_bbox` uses `state.velocity`. Kalman velocity will change visual smoothness.

### Implementation

1. Add `use_kalman_velocity: bool` to config (default: false)
2. When enabled:
   - Read `kf.x[2:4]` after Kalman update
   - Set `state.speed = norm(velocity)`, `state.velocity = (vx, vy)`
   - Skip `_update_speed()` entirely
3. When disabled: current behavior unchanged
4. Retune Kalman parameters (increase `process_noise` for faster velocity response)
5. Add warm-up guard: use `raw_speed` for first N frames until Kalman converges

### Validation (mandatory before enabling by default)

1. Run validation script before/after
2. Compare: stop_duration counts, stop_duration distribution, min_speed distribution, compliance rate
3. Deploy behind flag, enable on cube, monitor 48h
4. Only make default after confirming stop counts don't regress

---

## Validation Strategy

1. `scripts/validate_tracking.py snapshot` before starting any phase
2. After each phase: `uv run pytest tests/` — all must pass
3. After Phases 1-5: deploy, run `validate_tracking.py compare` after 24h
4. Phase 6 (speed model): deploy behind flag, validate stop counts + stop durations specifically
5. `validate_tracking.py health` after all phases to confirm no regressions
