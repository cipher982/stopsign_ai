# Stop Calibration & Noise Floor Analysis

**Date**: 2026-02-21
**Commit**: `7d27e93`
**Dataset**: ~53,138 clean passes (53,363 total minus tiz outliers and anomaly window)
**Status**: Snapshot — thresholds derived here may need revisiting as data grows or camera/zone changes

---

## Motivation

`min_speed` is stored in px/s (pixel space, no real-world unit conversion). Before building any scoring or labeling system on top of it, we needed to understand the noise floor: what does "0 mph" actually look like in this detector?

---

## Method: Parked Cars as Ground Truth

Cars with `time_in_zone >= 60s` are almost certainly parked or street-parked. Their `min_speed` distribution represents pure detection noise at zero real velocity.

### Results

| Category | n | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|
| Likely parked (60–300s) | 63 | 0.09 | 0.18 | 0.58 | 2.26 | 6.09 |
| Certainly parked (>300s) | 59 | 0.07 | 0.14 | 0.29 | 1.01 | 5.64 |

**Noise floor ceiling: ~6 px/s.** Zero real-world vehicles above 10 px/s in either parked group.

---

## Speed Distribution (clean data, tiz 3–8s)

Histogram of `min_speed` for vehicles spending 3–8s in zone reveals **weak bimodality**:

- **Peak 1**: 0–5 px/s (2,507 vehicles) — true stoppers, at/below noise floor
- **Dip**: 10–15 px/s (1,658) — natural population separator
- **Peak 2**: 25–45 px/s (~2,400–2,500 each bucket) — slow rollers

The dip at 10–15 px/s, combined with the parked-car ceiling of 6 px/s, makes **10 px/s a defensible hard threshold** for "stopped."

---

## Temporal Anomalies (exclude from analysis)

| Issue | Window | Records | Effect |
|---|---|---|---|
| Zone in intersection | 2026-02-19 18:00 – 2026-02-21 14:00 | 39 | Elevated speed median (~64 vs 51 typical) |
| System outage | 2026-02-20 (all day) | 0 | Complete gap — no passes recorded |
| Parked car tracking bug | `time_in_zone >= 30s` | 191 | tiz up to 4.7 hours, min_speed ≈ 0 |

Standard clean data filter for any analysis query:
```sql
WHERE time_in_zone < 30
  AND NOT (timestamp BETWEEN '2026-02-19 18:00' AND '2026-02-21 14:00')
```

---

## Traffic Behavior Breakdown (clean data)

| Category | Criteria | % |
|---|---|---|
| Full stop | min_speed < 10 AND tiz ≥ 3s | **10.8%** |
| Functional stop | min_speed < 20 AND tiz ≥ 2s | **17.9%** |
| Blow-through | tiz < 1.5s | **5.1%** |
| Rolling (remainder) | — | ~76% |

Prior estimate (based on traffic research) was 25–30% full stop. Calibrated result is ~11%, consistent with residential stop sign literature (10–15%).

### Overall `time_in_zone` distribution
| p5 | p10 | p25 | median | p75 | p90 | p95 |
|---|---|---|---|---|---|---|
| 1.47s | 1.94s | 2.66s | 3.48s | 4.54s | 5.88s | 7.13s |

### Overall `min_speed` distribution
| p5 | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|
| 1.6 | 7.9 | 27.3 | 50.8 | 79.1 | 115.0 |

Median min_speed of 50.8 px/s reflects that most vehicles are still moving when they enter the zone (zone captures approach + stop + departure, not just the stop bar).

---

## Calibrated Thresholds

| Signal | Value | Basis |
|---|---|---|
| Noise floor ceiling | 6 px/s | parked car max |
| "Stopped" gate | < 10 px/s | 1.7× noise ceiling; below bimodal dip |
| Full stop definition | min_speed < 10 AND tiz ≥ 3s | calibrated + meaningful dwell time |
| Exclude tracking artifacts | tiz ≥ 30s | clearly not a real pass |

These thresholds assume the current camera position and YOLO model. If the camera shifts or the model changes, re-run the parked car analysis.

---

## Open Questions / Future Work

- Kalman filter does not currently smooth speed calculations (only display position). If fixed, re-run this analysis — noise floor may drop, shifting the 10 px/s threshold downward.
- Thresholds should be re-validated once dataset exceeds ~100k passes or after any camera/zone configuration change.
- See `docs/specs/stop-scoring.md` for how these thresholds feed into the scoring system.
