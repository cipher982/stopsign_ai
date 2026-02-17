#!/usr/bin/env python3
"""Verify raw tracking payloads are being captured correctly.

Run periodically after deployment to confirm Phase 1 data pipeline works.
Exit codes: 0 = all checks pass, 1 = failures found, 2 = no data yet (retry later).
"""

import os
import sys

import psycopg2

DB_URL = os.environ.get("DB_URL")
if not DB_URL:
    print("ERROR: DB_URL environment variable is required")
    sys.exit(1)

REQUIRED_PAYLOAD_KEYS = {
    "version",
    "coordinate_space",
    "sample_schema",
    "samples",
    "summary",
    "dimensions",
    "config_snapshot",
    "model_snapshot",
    "raw_complete",
}
REQUIRED_SUMMARY_KEYS = {"entry_time", "exit_time", "time_in_zone", "stop_duration", "min_speed", "image_path"}
EXPECTED_SAMPLE_SCHEMA = ["t", "x", "y", "x1", "y1", "x2", "y2", "raw_speed", "speed"]


def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # 1. Check table exists
    cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'vehicle_pass_raw'")
    if cur.fetchone()[0] == 0:
        print("FAIL: vehicle_pass_raw table does not exist")
        return 1

    # 2. Count raw payloads
    cur.execute("SELECT COUNT(*) FROM vehicle_pass_raw")
    total_raw = cur.fetchone()[0]

    # 3. Recent passes vs raw coverage
    cur.execute("""
        SELECT
            COUNT(*) as total_passes,
            COUNT(vpr.id) as with_raw,
            COUNT(*) - COUNT(vpr.id) as missing_raw
        FROM vehicle_passes vp
        LEFT JOIN vehicle_pass_raw vpr ON vpr.vehicle_pass_id = vp.id
        WHERE vp.timestamp > NOW() - INTERVAL '1 hour'
    """)
    row = cur.fetchone()
    passes_1h, with_raw_1h, missing_raw_1h = row

    print(f"Total raw payloads ever: {total_raw}")
    print(f"Passes last hour: {passes_1h} (with raw: {with_raw_1h}, missing: {missing_raw_1h})")

    if total_raw == 0:
        print("\nNO DATA YET — no raw payloads recorded. Retry after more passes come in.")
        conn.close()
        return 2

    # 4. Validate payload structure on most recent raw records
    cur.execute("""
        SELECT vpr.vehicle_pass_id, vpr.raw_payload, vpr.sample_count, vpr.raw_complete,
               vp.time_in_zone, vp.stop_duration, vp.min_speed
        FROM vehicle_pass_raw vpr
        JOIN vehicle_passes vp ON vp.id = vpr.vehicle_pass_id
        ORDER BY vpr.created_at DESC
        LIMIT 5
    """)
    rows = cur.fetchall()

    failures = []
    for pass_id, payload, sample_count, raw_complete, time_in_zone, stop_duration, min_speed in rows:
        prefix = f"Pass #{pass_id}"

        if not isinstance(payload, dict):
            failures.append(f"{prefix}: raw_payload is not a dict (type={type(payload).__name__})")
            continue

        # Check top-level keys
        missing_keys = REQUIRED_PAYLOAD_KEYS - set(payload.keys())
        if missing_keys:
            failures.append(f"{prefix}: missing top-level keys: {missing_keys}")

        # Check version
        if payload.get("version") != 1:
            failures.append(f"{prefix}: version={payload.get('version')}, expected 1")

        # Check sample_schema matches expected
        if payload.get("sample_schema") != EXPECTED_SAMPLE_SCHEMA:
            failures.append(f"{prefix}: sample_schema mismatch: {payload.get('sample_schema')}")

        # Check samples exist and match count
        samples = payload.get("samples", [])
        if len(samples) == 0:
            failures.append(f"{prefix}: zero samples")
        if len(samples) != sample_count:
            failures.append(f"{prefix}: sample_count mismatch: payload has {len(samples)}, column says {sample_count}")

        # Check sample tuple width matches schema
        if samples:
            expected_width = len(EXPECTED_SAMPLE_SCHEMA)
            bad_widths = [i for i, s in enumerate(samples[:10]) if len(s) != expected_width]
            if bad_widths:
                failures.append(f"{prefix}: sample width mismatch at indices {bad_widths} (expected {expected_width})")

        # Check summary
        summary = payload.get("summary", {})
        missing_summary = REQUIRED_SUMMARY_KEYS - set(summary.keys())
        if missing_summary:
            failures.append(f"{prefix}: missing summary keys: {missing_summary}")

        # Cross-check summary vs vehicle_passes
        if summary.get("time_in_zone") is not None and abs(summary["time_in_zone"] - time_in_zone) > 0.01:
            failures.append(
                f"{prefix}: summary.time_in_zone ({summary['time_in_zone']}) != passes.time_in_zone ({time_in_zone})"
            )

        # Check dimensions
        dims = payload.get("dimensions", {})
        if not dims:
            failures.append(f"{prefix}: dimensions empty")
        elif not all(k in dims for k in ("raw", "processed")):
            failures.append(f"{prefix}: dimensions missing raw or processed")

        # Check config_snapshot is non-empty
        if not payload.get("config_snapshot"):
            failures.append(f"{prefix}: config_snapshot is empty")

        # Check model_snapshot is non-empty
        model = payload.get("model_snapshot", {})
        if not model:
            failures.append(f"{prefix}: model_snapshot is empty")
        elif not model.get("model_name"):
            failures.append(f"{prefix}: model_snapshot missing model_name")

        # Check raw_complete
        if not raw_complete:
            failures.append(f"{prefix}: raw_complete is False")
        if not payload.get("raw_complete"):
            failures.append(f"{prefix}: payload raw_complete is False")

        if not any(f.startswith(prefix) for f in failures):
            print(f"  {prefix}: OK ({sample_count} samples, {time_in_zone:.1f}s in zone)")

    # 5. Check for orphaned passes AFTER first raw payload (excludes pre-deployment passes)
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(vpr.id) as with_raw
        FROM vehicle_passes vp
        LEFT JOIN vehicle_pass_raw vpr ON vpr.vehicle_pass_id = vp.id
        WHERE vp.timestamp > (SELECT MIN(created_at) FROM vehicle_pass_raw)
    """)
    post_deploy_total, post_deploy_with_raw = cur.fetchone()
    if post_deploy_total > 3:
        ratio = post_deploy_with_raw / post_deploy_total
        if ratio < 0.9:
            failures.append(
                f"Post-deploy raw coverage {ratio:.0%} ({post_deploy_with_raw}/{post_deploy_total}) — expected >90%"
            )

    conn.close()

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\nALL CHECKS PASSED ({total_raw} raw payloads verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
