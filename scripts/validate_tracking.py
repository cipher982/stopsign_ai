"""Tracking quality validation and monitoring script.

Usage:
    uv run python scripts/validate_tracking.py snapshot   # Save 7-day baseline to tracking_baseline.json
    uv run python scripts/validate_tracking.py compare    # Compare current metrics against saved baseline
    uv run python scripts/validate_tracking.py health     # Run diagnostic health checks
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import psycopg2

DB_CONFIG = {
    "host": "***REMOVED***",
    "port": ***REMOVED***,
    "dbname": "stopsign",
    "user": "postgres",
    "password": "***REMOVED***",
}

BASELINE_PATH = Path(__file__).parent / "tracking_baseline.json"

STOP_SPEED_THRESHOLD = 20  # px/s — below this is "stopped"


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def query_7d_passes(conn):
    """Fetch last 7 days of vehicle_passes."""
    cutoff = datetime.now() - timedelta(days=7)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT min_speed, stop_duration, time_in_zone
            FROM vehicle_passes
            WHERE timestamp >= %s
            ORDER BY timestamp DESC
            """,
            (cutoff,),
        )
        rows = cur.fetchall()
    return [{"min_speed": r[0], "stop_duration": r[1], "time_in_zone": r[2]} for r in rows]


def compute_metrics(passes):
    """Compute all tracking quality metrics from a list of passes."""
    total = len(passes)
    if total == 0:
        return {
            "snapshot_time": datetime.now().isoformat(timespec="seconds"),
            "total_passes_7d": 0,
            "passes_with_stop_duration": 0,
            "avg_min_speed_7d": 0.0,
            "median_min_speed_7d": 0.0,
            "pct_below_20pxs": 0.0,
            "pct_below_50pxs": 0.0,
            "avg_time_in_zone": 0.0,
            "compliance_rate_2s": 0.0,
            "speed_histogram": {},
        }

    speeds = [p["min_speed"] for p in passes if p["min_speed"] is not None]
    times = [p["time_in_zone"] for p in passes if p["time_in_zone"] is not None]
    stops = [p for p in passes if p["stop_duration"] is not None and p["stop_duration"] > 0]

    # Speed histogram: 10px/s buckets up to 200+
    histogram = {}
    for lo in range(0, 200, 10):
        hi = lo + 10
        label = f"{lo}-{hi}"
        histogram[label] = sum(1 for s in speeds if lo <= s < hi)
    histogram["200+"] = sum(1 for s in speeds if s >= 200)

    return {
        "snapshot_time": datetime.now().isoformat(timespec="seconds"),
        "total_passes_7d": total,
        "passes_with_stop_duration": len(stops),
        "avg_min_speed_7d": round(statistics.mean(speeds), 2) if speeds else 0.0,
        "median_min_speed_7d": round(statistics.median(speeds), 2) if speeds else 0.0,
        "pct_below_20pxs": round(100 * sum(1 for s in speeds if s < 20) / len(speeds), 1) if speeds else 0.0,
        "pct_below_50pxs": round(100 * sum(1 for s in speeds if s < 50) / len(speeds), 1) if speeds else 0.0,
        "avg_time_in_zone": round(statistics.mean(times), 2) if times else 0.0,
        "compliance_rate_2s": round(100 * sum(1 for t in times if t >= 2.0) / len(times), 1) if times else 0.0,
        "speed_histogram": histogram,
    }


def cmd_snapshot(_args):
    """Save current 7-day metrics as baseline."""
    print("Querying last 7 days of vehicle passes...")
    conn = get_connection()
    try:
        passes = query_7d_passes(conn)
    finally:
        conn.close()

    metrics = compute_metrics(passes)

    BASELINE_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Baseline saved to {BASELINE_PATH}")
    print(f"  Total passes (7d):        {metrics['total_passes_7d']}")
    print(f"  With stop_duration > 0:   {metrics['passes_with_stop_duration']}")
    print(f"  Avg min_speed:            {metrics['avg_min_speed_7d']} px/s")
    print(f"  Median min_speed:         {metrics['median_min_speed_7d']} px/s")
    print(f"  % below 20 px/s:          {metrics['pct_below_20pxs']}%")
    print(f"  % below 50 px/s:          {metrics['pct_below_50pxs']}%")
    print(f"  Avg time_in_zone:         {metrics['avg_time_in_zone']}s")
    print(f"  Compliance (≥2s):         {metrics['compliance_rate_2s']}%")


def cmd_compare(_args):
    """Compare current metrics against saved baseline."""
    if not BASELINE_PATH.exists():
        print(f"ERROR: No baseline found at {BASELINE_PATH}")
        print("Run 'snapshot' mode first to create a baseline.")
        sys.exit(1)

    baseline = json.loads(BASELINE_PATH.read_text())
    print(f"Baseline from: {baseline['snapshot_time']}")
    print("Querying current 7-day metrics...\n")

    conn = get_connection()
    try:
        passes = query_7d_passes(conn)
    finally:
        conn.close()

    current = compute_metrics(passes)

    # Define metrics to compare and whether "higher is better"
    comparisons = [
        ("total_passes_7d", "Total passes (7d)", None),  # neutral
        ("passes_with_stop_duration", "With stop_duration > 0", "higher"),
        ("avg_min_speed_7d", "Avg min_speed (px/s)", "lower"),
        ("median_min_speed_7d", "Median min_speed (px/s)", "lower"),
        ("pct_below_20pxs", "% below 20 px/s", "higher"),
        ("pct_below_50pxs", "% below 50 px/s", "higher"),
        ("avg_time_in_zone", "Avg time_in_zone (s)", "higher"),
        ("compliance_rate_2s", "Compliance ≥2s (%)", "higher"),
    ]

    header = f"{'Metric':<30} {'Baseline':>12} {'Current':>12} {'Delta':>12}  Dir"
    print(header)
    print("-" * len(header))

    for key, label, direction in comparisons:
        old_val = baseline.get(key, 0)
        new_val = current.get(key, 0)
        delta = new_val - old_val

        # Determine arrow
        if direction is None or delta == 0:
            arrow = " "
        elif direction == "higher":
            arrow = "↑" if delta > 0 else "↓"
        else:  # lower is better
            arrow = "↑" if delta < 0 else "↓"

        # Color the arrow: ↑ = better (green context), ↓ = worse (red context)
        delta_str = f"{delta:+.2f}" if isinstance(new_val, float) else f"{delta:+d}"
        print(f"{label:<30} {old_val:>12} {new_val:>12} {delta_str:>12}  {arrow}")

    # Speed histogram comparison
    if "speed_histogram" in baseline and "speed_histogram" in current:
        print("\nSpeed Histogram (10 px/s buckets):")
        print(f"  {'Bucket':<10} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
        print(f"  {'-'*42}")
        all_buckets = list(baseline["speed_histogram"].keys())
        for bucket in all_buckets:
            old_c = baseline["speed_histogram"].get(bucket, 0)
            new_c = current["speed_histogram"].get(bucket, 0)
            d = new_c - old_c
            d_str = f"{d:+d}" if d != 0 else "—"
            print(f"  {bucket:<10} {old_c:>10} {new_c:>10} {d_str:>10}")


def cmd_health(_args):
    """Run diagnostic health checks."""
    conn = get_connection()
    try:
        passes = query_7d_passes(conn)

        # 24h pass count
        cutoff_24h = datetime.now() - timedelta(hours=24)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM vehicle_passes WHERE timestamp >= %s", (cutoff_24h,))
            passes_24h = cur.fetchone()[0]

        # Noise floor: parked vehicles (time_in_zone > 10s) — their min_speed reflects sensor noise
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT min_speed FROM vehicle_passes
                WHERE timestamp >= %s AND time_in_zone > 10 AND min_speed IS NOT NULL
                """,
                (datetime.now() - timedelta(days=7),),
            )
            parked_speeds = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()

    metrics = compute_metrics(passes)
    issues = []

    print("=" * 60)
    print("  TRACKING HEALTH CHECK")
    print("=" * 60)
    print()

    # Check 1: stop_duration population
    if metrics["passes_with_stop_duration"] == 0 and metrics["total_passes_7d"] > 0:
        pct = 0.0
        issues.append("FAIL")
        print("  FAIL: 0% of passes have stop_duration > 0")
    elif metrics["total_passes_7d"] > 0:
        pct = 100 * metrics["passes_with_stop_duration"] / metrics["total_passes_7d"]
        if pct < 5:
            issues.append("WARN")
            print(f"  WARN: Only {pct:.1f}% of passes have stop_duration > 0")
        else:
            print(
                f"  OK:   {pct:.1f}% of passes have stop_duration > 0 "
                f"({metrics['passes_with_stop_duration']}/{metrics['total_passes_7d']})"
            )
    else:
        issues.append("WARN")
        print("  WARN: No passes in last 7 days")

    # Check 2: median speed vs stop threshold
    if metrics["median_min_speed_7d"] > STOP_SPEED_THRESHOLD:
        issues.append("WARN")
        print(
            f"  WARN: Median speed {metrics['median_min_speed_7d']} px/s "
            f"is above stop threshold {STOP_SPEED_THRESHOLD} px/s"
        )
    else:
        print(f"  OK:   Median speed {metrics['median_min_speed_7d']} px/s is at or below stop threshold")

    # Check 3: speed noise floor from parked vehicles
    if parked_speeds:
        noise_floor = round(statistics.median(parked_speeds), 1)
        print(f"  WARN: Speed noise floor ~{noise_floor}px/s " f"(from {len(parked_speeds)} long-dwell vehicles)")
        if noise_floor > 10:
            issues.append("WARN")
    else:
        print("  INFO: No long-dwell vehicles (>10s) found to estimate noise floor")

    # Check 4: compliance rate
    print(f"  OK:   {metrics['compliance_rate_2s']}% compliance rate (time_in_zone ≥ 2s)")

    # Check 5: 24h volume
    print(f"  INFO: {passes_24h} passes in last 24h")

    print()
    print("-" * 60)
    if "FAIL" in issues:
        print("  STATUS: ISSUES DETECTED — review FAIL items above")
        sys.exit(1)
    elif "WARN" in issues:
        print("  STATUS: WARNINGS — review items above")
    else:
        print("  STATUS: ALL OK")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate tracking quality metrics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("snapshot", help="Save 7-day baseline metrics")
    subparsers.add_parser("compare", help="Compare current metrics to baseline")
    subparsers.add_parser("health", help="Run diagnostic health checks")

    args = parser.parse_args()

    commands = {
        "snapshot": cmd_snapshot,
        "compare": cmd_compare,
        "health": cmd_health,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
