**Tracking Rewrite Review**

**Summary**
- Phase 1 (Kalman-only speed) is not safe with current tuning. Expect laggy deceleration and/or noisy velocity, which directly impacts stop duration and minimum speed metrics.
- The spec misses a few real dependencies: `raw_speed` is used in stop detection and DB metrics, `direction` is used in debug overlays, and `track` is saved to the DB.
- Phase ordering should put explicit timestamps earlier and treat Phase 1 as an opt-in experiment behind a flag.

**Feasibility And Hidden Dependencies**
- `raw_speed` is not a "nice-to-have." It is used for stop duration gating, speed samples in zone, and telemetry (`StopDetector._update_stop_duration`, speed sampling, and spans). Removing it changes recorded metrics and stop/no-stop decisions. See `stopsign/tracking.py`.
- `direction` is not dead. It is used in the debug overlay labels in `stopsign/video_analyzer.py` (`draw_box`, `draw_car_interpolated`). If removed, those labels must be updated or removed.
- `track` is persisted to the DB (`Database.save_car_state` uses `car.state.track`). If you change its structure or stop appending, you'll lose data or break DB assumptions.
- The rewrite assumes `timestamp` behavior is stable. The analyzer can skip frames and perform catch-up; `timestamp` gaps can be large. Time-based debouncing and Kalman velocity will be sensitive to this unless you clamp or guard large `dt` values.
- There is a `tracking.use_kalman_filter` config flag, but it is not used anywhere. This is an easy safety valve for Phase 1 and is currently missing from the plan.

**Risk Assessment: Phase 1 (Kalman Velocity)**
- With `process_noise=0.1` and `measurement_noise=1.0`, the Kalman filter is tuned for smooth position, not responsive velocity. At ~10 FPS (`config/config.yaml`), the computed Q becomes very small. That makes the filter strongly favor a constant-velocity model and slow to react to stops. Expect undercounted stop durations and more "no stop" outcomes.
- If measurement noise is too low for actual YOLO jitter (likely several pixels), the velocity estimate will be noisy, which can also inflate or fragment stop durations. Either way, this is a high-impact change because `stop_speed_threshold` and `min_speed_in_zone` are currently calibrated on `raw_speed`.
- Kalman velocity will change interpolation behavior (`get_interpolated_bbox`) and the debug overlay. The current velocity is derived from a buggy time-diff computation (sign is likely inverted). Switching to Kalman velocity will be visible and is a behavioral change even if core metrics stay stable.
- Bottom line: Phase 1 is not safe as a first move. It should be behind a flag, validated against stop durations and counts, and likely needs retuning (`process_noise` higher, `measurement_noise` higher) or a hybrid approach (median filter on measurements + Kalman on positions, velocity from smoothed positions).

**Phase Ordering**
Proposed order is not optimal. I recommend:

1. Phase 2 (image module extraction): low risk, pure move.
2. Phase 6 (explicit timestamps): remove the ordering landmine early, before larger refactors.
3. Phase 5 (direction removal) only after updating debug overlays or if you explicitly want to drop that overlay data.
4. Phase 3 (CarState grouping): large surface area; do this after you have the ordering dependency removed.
5. Phase 4 (time-based debouncing): needs config updates + tuning; isolate it from Phase 1 to avoid compounding effects.
6. Phase 1 (Kalman velocity) last, behind a flag.

**Missing Concerns**
- No plan to re-tune thresholds (`stop_speed_threshold`, `max_movement_speed`, `unparked_speed_threshold`) if the speed model changes. That is mandatory for Phase 1.
- No validation of stop counts, stop duration distribution, or min speed distribution. Comparing only speed percentiles is insufficient because the business outcome is "did the car stop."
- No warm-up handling for Kalman velocity. The filter starts with `v=0` and large covariance. Using `kf.x[2:4]` immediately will bias early frames unless you gate on track length.
- Time-based debouncing should explicitly handle large `dt` (frame drops, analyzer catch-up). Otherwise, you can enter or exit the zone incorrectly in a single large step.
- If you remove `direction`, you must update `video_analyzer` overlays; otherwise you'll break debug UI expectations.

**Over-Engineering**
- Phase 3 (sub-state grouping) is readability-only and high-touch. It's fine long-term but not urgent; it can wait until after you remove the ordering dependency and harden the velocity model.
- Phase 4 (time-based debouncing) is reasonable but only if you can tune and monitor it. If FPS is stable in production, the gain may be marginal relative to the risk.

**Direct Calls**
- Phase 1 as written is a bad idea to ship immediately. With current Kalman tuning, it is likely to be less responsive than `raw_speed`, and stop detection will regress. Do it last, behind a flag, and validate with real stop-duration outcomes.
- Phase 6 should move earlier. The current ordering dependency is a real footgun and has nothing to do with the CarState refactor.

**Concrete Recommendations**
- Add a config flag for `use_kalman_velocity` and keep `raw_speed` as the default until validated.
- If you proceed with Kalman velocity, retune noise parameters and gate early frames (e.g., require N updates before using velocity for stop logic).
- Expand validation to include stop counts, stop durations, and min speed distributions, not just speed histograms.
- Update `video_analyzer` overlay logic if `direction` is removed.
