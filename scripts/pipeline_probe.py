#!/usr/bin/env python3
"""
pipeline_probe.py — Read-only Redis pipeline observer for Stop Sign AI.

Measures frame drops, processing lag, queue depths, and YOLO throughput
without consuming or modifying any data.

Wire format for raw_frames (RAW_FRAME_KEY):
  magic    : 4 bytes  b'SSFM'
  version  : 1 byte   (currently 1)
  json_len : 4 bytes  big-endian uint32
  json     : json_len UTF-8 bytes  (at least {"ts": <float>})
  jpeg     : remaining bytes

FRAME_METADATA_KEY: a Redis string (SET) containing JSON with at least:
  {"capture_timestamp": <float>, "latency_sec": <float>, "frame_count": <int>, ...}

Reads only via: LRANGE, LINDEX, LLEN, GET — never RPOP/BRPOP/DEL.

Usage:
  python3 scripts/pipeline_probe.py

Environment variables (all have defaults matching production):
  REDIS_URL              redis://redis:6379/0
  RAW_FRAME_KEY          raw_frames
  PROCESSED_FRAME_KEY    processed_frames
  FRAME_METADATA_KEY     frame_metadata_latest
  PROBE_DURATION_SEC     60   (total run time)
  PROBE_REPORT_INTERVAL  5    (summary print interval)
"""

import json
import os
import sys
import time
from collections import deque
from typing import Optional

# ---------------------------------------------------------------------------
# Dependency check — redis is the only non-stdlib requirement
# ---------------------------------------------------------------------------
try:
    import redis
except ImportError:
    sys.exit("ERROR: 'redis' package not found.\n" "Install with:  pip install redis   or   uv add redis")

# ---------------------------------------------------------------------------
# Configuration from environment (same variable names as production)
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
RAW_FRAME_KEY = os.getenv("RAW_FRAME_KEY", "raw_frames")
PROCESSED_FRAME_KEY = os.getenv("PROCESSED_FRAME_KEY", "processed_frames")
FRAME_METADATA_KEY = os.getenv("FRAME_METADATA_KEY", "frame_metadata_latest")

PROBE_DURATION_SEC = float(os.getenv("PROBE_DURATION_SEC", "60"))
PROBE_REPORT_INTERVAL = float(os.getenv("PROBE_REPORT_INTERVAL", "5"))

# Frame budget: 15 fps = 66.7 ms per frame; >80 ms gap == dropped frame(s)
FRAME_BUDGET_MS = 80.0
TARGET_FPS = 15.0

# Header constants (must match video_analyzer.py exactly)
RAW_HEADER_MAGIC = b"SSFM"
RAW_HEADER_MIN_LEN = 9  # 4 magic + 1 version + 4 json_len


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def parse_capture_ts(data: bytes) -> Optional[float]:
    """
    Extract capture_ts from a raw frame blob using the SSFM header format.
    Returns None if the header is absent or malformed.
    """
    if len(data) < RAW_HEADER_MIN_LEN:
        return None
    if data[0:4] != RAW_HEADER_MAGIC:
        return None
    # version = data[4]  (unused here)
    json_len = int.from_bytes(data[5:9], "big")
    meta_start = 9
    meta_end = meta_start + json_len
    if json_len < 0 or meta_end > len(data):
        return None
    try:
        meta = json.loads(data[meta_start:meta_end].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(meta, dict) or "ts" not in meta:
        return None
    try:
        return float(meta["ts"])
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Rolling statistics helpers
# ---------------------------------------------------------------------------


class RollingWindow:
    """Keeps the last N values; computes avg/p95/max on demand."""

    def __init__(self, maxlen: int = 1000):
        self._data: deque = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        self._data.append(value)

    def count(self) -> int:
        return len(self._data)

    def avg(self) -> float:
        if not self._data:
            return 0.0
        return sum(self._data) / len(self._data)

    def p95(self) -> float:
        if not self._data:
            return 0.0
        s = sorted(self._data)
        idx = max(0, int(len(s) * 0.95) - 1)
        return s[idx]

    def maximum(self) -> float:
        if not self._data:
            return 0.0
        return max(self._data)

    def all_values(self) -> list:
        return list(self._data)


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------


def run_probe() -> None:
    # ------------------------------------------------------------------ setup
    print("=" * 70)
    print("  Stop Sign AI — Pipeline Probe")
    print("=" * 70)
    print(f"  Redis URL      : {REDIS_URL}")
    print(f"  raw_frames key : {RAW_FRAME_KEY}")
    print(f"  processed key  : {PROCESSED_FRAME_KEY}")
    print(f"  metadata key   : {FRAME_METADATA_KEY}")
    print(f"  Duration       : {PROBE_DURATION_SEC:.0f}s")
    print(f"  Report interval: {PROBE_REPORT_INTERVAL:.0f}s")
    print(f"  Frame budget   : {FRAME_BUDGET_MS:.0f}ms  (drop threshold)")
    print(f"  Target FPS     : {TARGET_FPS:.0f}")
    print()
    print("  READ-ONLY: uses LRANGE/LINDEX/LLEN/GET — never pops.")
    print("=" * 70)
    print()

    # Connect
    try:
        r = redis.from_url(REDIS_URL, socket_timeout=5, socket_connect_timeout=5)
        r.ping()
        print(f"  Connected to Redis at {REDIS_URL}")
    except redis.exceptions.ConnectionError as exc:
        sys.exit(f"ERROR: Cannot connect to Redis: {exc}")
    except redis.exceptions.ResponseError as exc:
        sys.exit(f"ERROR: Redis error: {exc}")

    print()

    # Accumulators
    raw_drop_count = 0  # capture_ts gap events
    raw_frames_seen = 0  # distinct capture_ts values observed at raw_frames
    processed_frames_seen = 0  # frames observed at processed_frames (via llen delta)

    prev_raw_ts: Optional[float] = None  # last capture_ts seen at raw oldest frame
    prev_raw_llen = 0
    prev_proc_llen = 0

    # Detect LTRIM drops: track how many times the oldest frame *didn't* advance
    # even though llen stayed constant (producer is pushing but consumer not reading)
    raw_llen_stable_count = 0  # consecutive polls where raw llen didn't change
    raw_oldest_ts_stable_count = 0  # consecutive polls where oldest ts didn't change

    # Max queue depths seen
    max_raw_depth = 0
    max_proc_depth = 0

    # YOLO / processed-frame timing
    yolo_intervals = RollingWindow(maxlen=5000)
    prev_metadata_frame_count: Optional[int] = None
    prev_metadata_ts: Optional[float] = None  # capture_timestamp from last metadata read

    # Processing latency (from metadata latency_sec field)
    proc_latency_window = RollingWindow(maxlen=5000)

    # Lag tracking (wall time - capture_ts of oldest frame)
    raw_lag_window = RollingWindow(maxlen=300)
    proc_lag_window = RollingWindow(maxlen=300)

    # FPS counters (reset every report interval)
    interval_raw_in = 0  # new raw frames produced since last report
    interval_proc_out = 0  # new processed frames since last report
    interval_start = time.monotonic()

    probe_start = time.monotonic()
    last_report = probe_start

    print("  Sampling... (Ctrl-C to abort early)\n")

    # ------------------------------------------------------------------ loop
    while True:
        now_mono = time.monotonic()
        elapsed = now_mono - probe_start

        if elapsed >= PROBE_DURATION_SEC:
            break

        wall_time = time.time()

        # ---- raw_frames queue ----------------------------------------
        raw_llen = r.llen(RAW_FRAME_KEY)
        if raw_llen is None:
            raw_llen = 0
        raw_llen = int(raw_llen)

        if raw_llen > max_raw_depth:
            max_raw_depth = raw_llen

        # Count new frames pushed since last sample (producer side)
        # LPUSH + LTRIM means llen can go up by N then get trimmed back.
        # Positive delta = new frames arrived.
        raw_delta = raw_llen - prev_raw_llen
        if raw_delta > 0:
            interval_raw_in += raw_delta
            raw_frames_seen += raw_delta

        # Peek at the oldest raw frame (rightmost = RPOP end = oldest in LPUSH queue)
        raw_oldest_blob = r.lindex(RAW_FRAME_KEY, -1)  # index -1 = tail = oldest
        raw_oldest_ts: Optional[float] = None
        raw_oldest_age: Optional[float] = None

        if raw_oldest_blob:
            raw_oldest_ts = parse_capture_ts(raw_oldest_blob)
            if raw_oldest_ts is not None:
                raw_oldest_age = wall_time - raw_oldest_ts
                raw_lag_window.add(max(0.0, raw_oldest_age))

        # Detect capture_ts gaps (dropped frames at capture)
        if raw_oldest_ts is not None and prev_raw_ts is not None:
            gap_ms = (raw_oldest_ts - prev_raw_ts) * 1000.0
            # Only check forward gaps (ts moved forward)
            if gap_ms > FRAME_BUDGET_MS:
                missed = max(0, int(gap_ms / (1000.0 / TARGET_FPS)) - 1)
                if missed > 0:
                    raw_drop_count += missed

        # Detect LTRIM drops: oldest ts not advancing despite non-empty queue
        if raw_oldest_ts is not None and raw_oldest_ts == prev_raw_ts and raw_llen > 0:
            raw_oldest_ts_stable_count += 1
        else:
            raw_oldest_ts_stable_count = 0

        if raw_llen == prev_raw_llen and raw_llen > 0:
            raw_llen_stable_count += 1
        else:
            raw_llen_stable_count = 0

        if raw_oldest_ts is not None:
            prev_raw_ts = raw_oldest_ts
        prev_raw_llen = raw_llen

        # ---- processed_frames queue ----------------------------------
        proc_llen = r.llen(PROCESSED_FRAME_KEY)
        if proc_llen is None:
            proc_llen = 0
        proc_llen = int(proc_llen)

        if proc_llen > max_proc_depth:
            max_proc_depth = proc_llen

        proc_delta = proc_llen - prev_proc_llen
        if proc_delta > 0:
            interval_proc_out += proc_delta
            processed_frames_seen += proc_delta

        prev_proc_llen = proc_llen

        # processed_frames contains raw BGR bytes (no timestamp header) so we
        # cannot extract a capture_ts from them directly. We use the metadata
        # key for all processed-stage timing instead (see below).

        # ---- FRAME_METADATA_KEY (latest processed frame metadata) ----
        meta_raw = r.get(FRAME_METADATA_KEY)
        meta_capture_ts: Optional[float] = None
        meta_frame_count: Optional[int] = None
        meta_latency: Optional[float] = None

        if meta_raw:
            try:
                meta = json.loads(meta_raw)
                meta_capture_ts = float(meta.get("capture_timestamp", 0) or 0) or None
                meta_frame_count = int(meta.get("frame_count", 0) or 0) or None
                meta_latency = float(meta.get("latency_sec", 0) or 0) or None
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if meta_capture_ts is not None:
            proc_lag = wall_time - meta_capture_ts
            proc_lag_window.add(max(0.0, proc_lag))

        if meta_latency is not None and meta_latency >= 0:
            proc_latency_window.add(meta_latency)

        # YOLO inter-frame interval: time between successive frame_count advances
        if meta_frame_count is not None and meta_capture_ts is not None:
            if prev_metadata_frame_count is not None and prev_metadata_ts is not None:
                frame_advance = meta_frame_count - prev_metadata_frame_count
                if frame_advance > 0:
                    ts_diff = meta_capture_ts - prev_metadata_ts
                    if ts_diff > 0:
                        per_frame_ms = (ts_diff / frame_advance) * 1000.0
                        yolo_intervals.add(per_frame_ms)
            prev_metadata_frame_count = meta_frame_count
            prev_metadata_ts = meta_capture_ts

        # ---- periodic report -----------------------------------------
        now_mono = time.monotonic()
        if (now_mono - last_report) >= PROBE_REPORT_INTERVAL:
            interval_elapsed = now_mono - interval_start

            raw_fps_in = interval_raw_in / interval_elapsed if interval_elapsed > 0 else 0.0
            proc_fps_out = interval_proc_out / interval_elapsed if interval_elapsed > 0 else 0.0

            interval_raw_in = 0
            interval_proc_out = 0
            interval_start = now_mono
            last_report = now_mono

            elapsed_display = now_mono - probe_start

            # Age of oldest raw frame
            raw_age_str = f"{raw_oldest_age * 1000:.0f}ms" if raw_oldest_age is not None else "n/a"

            # Lag at processed stage (from metadata)
            proc_lag_avg = proc_lag_window.avg()
            raw_lag_avg = raw_lag_window.avg()

            # YOLO throughput
            if yolo_intervals.count() > 0:
                yolo_eff_fps = 1000.0 / yolo_intervals.avg() if yolo_intervals.avg() > 0 else 0.0
                yolo_interval_str = f"{yolo_intervals.avg():.1f}ms avg interval ({yolo_eff_fps:.1f} fps)"
            else:
                yolo_interval_str = "no data yet"

            # LTRIM / stall warning
            ltrim_warning = ""
            if raw_oldest_ts_stable_count > 10 and raw_llen > 0:
                ltrim_warning = "  [WARN] Oldest raw frame ts not advancing — possible LTRIM churn or stall"

            print(f"  [{elapsed_display:5.1f}s] ---- SNAPSHOT ----")
            print(
                f"  raw_frames   : depth={raw_llen:4d} | oldest frame age={raw_age_str:>8s} "
                f"| in  {raw_fps_in:5.1f} fps"
            )
            print(f"  proc_frames  : depth={proc_llen:4d}                          | out {proc_fps_out:5.1f} fps")
            print(f"  Drops (ts gaps)   : {raw_drop_count} estimated dropped frames")
            print(f"  YOLO throughput   : {yolo_interval_str}")
            print(
                f"  Pipeline latency  : raw stage lag {raw_lag_avg * 1000:.0f}ms avg "
                f"| processed stage lag {proc_lag_avg * 1000:.0f}ms avg"
            )
            if proc_latency_window.count() > 0:
                print(
                    f"  YOLO+proc latency : {proc_latency_window.avg() * 1000:.0f}ms avg  "
                    f"{proc_latency_window.p95() * 1000:.0f}ms p95  "
                    f"{proc_latency_window.maximum() * 1000:.0f}ms max"
                )
            if ltrim_warning:
                print(ltrim_warning)
            print()

        time.sleep(0.1)  # 100ms poll cadence

    # ------------------------------------------------------------------ final
    total_elapsed = time.monotonic() - probe_start

    print()
    print("=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    print(f"  Observation window    : {total_elapsed:.1f}s")
    print()
    print(f"  raw_frames seen       : {raw_frames_seen:,}")
    print(f"  processed frames seen : {processed_frames_seen:,}")
    print(f"  Estimated drop events : {raw_drop_count:,}")
    if raw_frames_seen > 0:
        drop_rate = 100.0 * raw_drop_count / max(1, raw_frames_seen + raw_drop_count)
        print(f"  Estimated drop rate   : {drop_rate:.1f}%")
    else:
        print("  Estimated drop rate   : n/a (no raw frames observed)")
    print()
    print(f"  Max raw_frames depth  : {max_raw_depth}")
    print(f"  Max proc_frames depth : {max_proc_depth}")
    print()

    if yolo_intervals.count() > 0:
        yolo_eff_fps = 1000.0 / yolo_intervals.avg() if yolo_intervals.avg() > 0 else 0.0
        print(f"  YOLO effective FPS    : {yolo_eff_fps:.1f}  (target {TARGET_FPS:.0f})")
        print(
            f"  YOLO inter-frame      : avg={yolo_intervals.avg():.1f}ms  "
            f"p95={yolo_intervals.p95():.1f}ms  max={yolo_intervals.maximum():.1f}ms"
        )
    else:
        print("  YOLO metrics          : no data (metadata key never set or stale)")

    if proc_latency_window.count() > 0:
        print(
            f"  Capture-to-output lag : avg={proc_latency_window.avg() * 1000:.0f}ms  "
            f"p95={proc_latency_window.p95() * 1000:.0f}ms  "
            f"max={proc_latency_window.maximum() * 1000:.0f}ms"
        )
    else:
        print("  Capture-to-output lag : no data")

    if raw_lag_window.count() > 0:
        print(
            f"  raw queue oldest age  : avg={raw_lag_window.avg() * 1000:.0f}ms  "
            f"max={raw_lag_window.maximum() * 1000:.0f}ms"
        )
    if proc_lag_window.count() > 0:
        print(
            f"  proc queue oldest age : avg={proc_lag_window.avg() * 1000:.0f}ms  "
            f"max={proc_lag_window.maximum() * 1000:.0f}ms"
        )

    print()
    print("  Notes:")
    print("   - Drop count is estimated from capture_ts gaps > 80ms in raw_frames.")
    print("   - YOLO metrics come from FRAME_METADATA_KEY (latest processed frame).")
    print("   - processed_frames contains raw BGR bytes with no timestamp header;")
    print("     its age is inferred from the metadata key, not the queue itself.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_probe()
    except KeyboardInterrupt:
        print("\n\n  [Interrupted by user]")
        sys.exit(0)
