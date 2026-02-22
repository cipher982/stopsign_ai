# Stop Scoring System — Design Spec

**Status**: Design complete, not yet implemented
**Prerequisite analysis**: `docs/analysis/2026-02-21-stop-calibration.md`

---

## Problem

`min_speed` (px/s) and `time_in_zone` (seconds) are currently displayed raw in the UI. Neither is meaningful to a casual viewer:
- px/s has no real-world unit
- Raw seconds without context don't convey "good" vs "bad"

---

## Design Goals

1. **One number** a viewer can immediately understand (0–100 score)
2. **One label** conveying the objective verdict (Full Stop / Rolling Stop / No Stop)
3. Score is self-updating — improves automatically as dataset grows
4. No hardcoded per-intersection thresholds in the score itself
5. Thresholds for labels are data-calibrated, not guessed

---

## Scoring Architecture

### Primary Metric: `time_in_zone` Percentile

```sql
stop_score = ROUND(PERCENT_RANK() OVER (ORDER BY time_in_zone) * 100)
```

Computed over clean data only:
```sql
WHERE time_in_zone < 30
  AND NOT (timestamp BETWEEN '2026-02-19 18:00' AND '2026-02-21 14:00')
```

**Why tiz as primary**: Jitter-resistant (duration doesn't fluctuate with bbox noise), human-interpretable ("stopped for X seconds"), legally meaningful.

**Why percentile**: Self-calibrating, no hardcoded thresholds in the score, "Score 80" always means "stopped longer than 80% of vehicles at this intersection."

### Why Not Speed-Based Scoring

`min_speed` is not suitable as a continuous scoring signal:
- Bimodal distribution (stopped vs moving) — better as a binary gate than a scaler
- px/s is camera-angle dependent, not transferable across intersections
- Noise floor (~6 px/s) means the bottom 10–15% of values are indistinguishable

### Verdict Labels (Hybrid: Percentile Score + Absolute Gate)

Labels use `time_in_zone` absolute thresholds **plus** `min_speed` as a stop-confirmation gate. Labels convey objective truth; score conveys relative rank.

| Label | Criteria | ~% of traffic |
|---|---|---|
| **Full Stop** | tiz ≥ 3s AND min_speed < 10 px/s | ~11% |
| **Rolling Stop** | tiz ≥ 1.5s (didn't fully stop or tiz too short) | ~84% |
| **No Stop** | tiz < 1.5s | ~5% |
| *(filter)* | tiz ≥ 30s | Exclude — tracking artifact |

The `min_speed < 10` gate is hardware-calibrated from parked car noise floor analysis. See analysis doc for justification.

**Why hybrid (not pure percentile labels)**: If this intersection's drivers are uniformly bad, a "Full Stop" label based purely on percentile would be misleading — someone in the top 33% might still be rolling through. The absolute tiz threshold ensures the label reflects real behavior.

---

## Implementation Notes

### Compute On the Fly vs Stored

Prefer computing `stop_score` on the fly with a window function in the query. This keeps scores live as data accumulates. If query performance becomes an issue, materialize as a nightly-computed column.

### Clean Data Filter

Any query computing scores or percentiles should apply:
```sql
WHERE time_in_zone < 30
  AND timestamp NOT BETWEEN '2026-02-19 18:00' AND '2026-02-21 14:00'
```

Add future anomaly windows to this filter (document in `docs/analysis/`).

### Display

- Show score as `82` (integer), not `82/100` or `82%` — cleaner
- Label takes color: Full Stop = green, Rolling = amber, No Stop = red
- Raw values (`time_in_zone`, `min_speed`) available on detail page for the curious

---

## What This Replaces

Currently `pass_item.html` shows `{{ min_speed }} · {{ time_in_zone }}s`. These raw values:
- Have no unit (mph label was wrong and was removed 2026-02-21)
- Give no verdict — viewer must interpret meaning
- Are colored independently with no shared legend

The score + label replaces both numbers in the list view. Raw values remain on the detail page.
