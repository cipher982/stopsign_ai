#!/usr/bin/env python3
"""Server-side companion logger for the stream debug harness.

Records, every `--interval` seconds, what the production pipeline was doing:
Redis queue depths, ffmpeg/analyzer health keys, and HLS segment production.
Output: JSONL at /app/data/debug-logs/server_<epoch>.jsonl

Run inside a stopsign container that has Redis access and the /app/data volume:
    docker exec is844go80k088kcgo88s8cs8-web_server-1 \
        python scripts/debug_server_logger.py --seconds 900 --interval 2

Then fetch the log back to the analysis machine:
    docker exec <container> cat /app/data/debug-logs/server_*.jsonl > server.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import redis

REDIS_URL = os.getenv("REDIS_URL")
HLS_DIR = "/app/data/stream"
OUT_DIR = "/app/data/debug-logs"

# Keys written by the analyzer / ffmpeg_service (see stopsign/settings.py)
FFMPEG_HEALTH_KEY = os.getenv("FFMPEG_HEALTH_KEY", "stopsign.ffmpeg.health")
ANALYZER_LAST_FRAME_AT_KEY = os.getenv("ANALYZER_LAST_FRAME_AT_KEY", "stopsign.analyzer.last_frame_at")
ANALYZER_STALL_KEY = os.getenv("ANALYZER_STALL_KEY", "stopsign.analyzer.stall")

QUEUE_PATTERNS = ["raw_frame*", "raw_frames*", "processed_frame*", "processed_frames*"]


def redis_snapshot(r: redis.Redis, rec: dict) -> None:
    try:
        rec["queues"] = {}
        seen = set()
        for pat in QUEUE_PATTERNS:
            for key in r.keys(pat):
                k = key.decode(errors="replace")
                if k not in seen:
                    seen.add(k)
                    rec["queues"][k] = r.llen(key)
    except Exception as e:  # noqa: BLE001
        rec["redis_error"] = str(e)
    for name, key in (
        ("ffmpeg_health", FFMPEG_HEALTH_KEY),
        ("analyzer_last_frame_at", ANALYZER_LAST_FRAME_AT_KEY),
        ("analyzer_stall", ANALYZER_STALL_KEY),
    ):
        try:
            raw = r.get(key)
            if raw is None:
                continue
            try:
                rec[name] = json.loads(raw)
            except Exception:  # noqa: BLE001
                rec[name] = raw.decode(errors="replace")
        except Exception as e:  # noqa: BLE001
            rec[f"{name}_error"] = str(e)


def hls_snapshot(rec: dict) -> None:
    try:
        files = sorted(f for f in os.listdir(HLS_DIR) if f.startswith("stream") and f.endswith(".ts"))
        rec["segment_count"] = len(files)
        if files:
            nums = [int(f[len("stream") : -len(".ts")]) for f in files]
            rec["oldest_segment"] = min(nums)
            rec["newest_segment"] = max(nums)
            newest_path = os.path.join(HLS_DIR, f"stream{rec['newest_segment']}.ts")
            rec["newest_mtime"] = os.path.getmtime(newest_path)
            oldest_path = os.path.join(HLS_DIR, f"stream{rec['oldest_segment']}.ts")
            rec["oldest_mtime"] = os.path.getmtime(oldest_path)
        m3u8_path = os.path.join(HLS_DIR, "stream.m3u8")
        rec["m3u8_mtime"] = os.path.getmtime(m3u8_path)
    except Exception as e:  # noqa: BLE001
        rec["hls_error"] = str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=900, help="total run duration")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between records")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"server_{int(time.time())}.jsonl")

    r = redis.from_url(REDIS_URL, socket_connect_timeout=5)
    start = time.time()
    deadline = start + args.seconds
    with open(out_path, "w") as f:
        print(f"writing {out_path} for {args.seconds}s every {args.interval}s", flush=True)
        while time.time() < deadline:
            rec = {
                "wall": time.time(),
                "wall_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
            }
            redis_snapshot(r, rec)
            hls_snapshot(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            time.sleep(args.interval)
    print(f"done: {out_path}", flush=True)


if __name__ == "__main__":
    main()
