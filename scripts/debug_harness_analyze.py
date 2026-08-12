#!/usr/bin/env python3
"""Merge a browser debug-harness capture with the server-side logger and
classify every timestamp jump (skip) observed on the client.

Usage:
    python3 scripts/debug_harness_analyze.py browser.json server.jsonl \
        [--epoch "YYYY-MM-DD HH:MM:SS"] [--export-snaps DIR] [--out DIR]

--epoch: wall-clock capture time (America/Chicago, as burned into the overlay)
         of mediaTime 0. Optional; without it, analysis is relative only.
         Derive it by OCR'ing a frameSnap from the capture (they carry the
         burned-in YYYY-MM-DD HH:MM:SS) or a server-extracted segment frame.

Output:
    printed summary + skip_table.csv + timeline.csv in --out (default cwd).

Skip classification (per skip event, using the nearest server record):
    pipeline_gap  server dup_pct high or analyzer last-frame age large
                  (content was never produced -> upstream stall)
    fetch_gap     segments spanning the skip were never fetched or took >2s
                  (delivery failure on the viewer's network path)
    render_skip   content produced and fetched, but frames were not rendered
                  (player-side buffer/seek behavior)
    unknown       not enough data
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime
import json
import os
from typing import Optional
from zoneinfo import ZoneInfo


def load_browser(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_server(path: str) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def parse_epoch(spec: str) -> float:
    try:
        return float(spec)
    except ValueError:
        pass
    dt = datetime.datetime.strptime(spec, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("America/Chicago"))
    return dt.timestamp()


def nearest_server(recs: list[dict], wall_sec: float) -> Optional[dict]:
    best, best_d = None, None
    for r in recs:
        d = abs(r["wall"] - wall_sec)
        if best_d is None or d < best_d:
            best, best_d = r, d
    return best


def classify(skip: dict, srv: Optional[dict], resources: list[dict]) -> str:
    """Classify one skip against nearest server state and the fetch record."""
    if srv is None:
        return "unknown"
    dup = None
    if isinstance(srv.get("ffmpeg_health"), dict):
        dup = srv["ffmpeg_health"].get("dup_pct")
    analyzer_age = None
    if srv.get("analyzer_last_frame_at"):
        try:
            analyzer_age = srv["wall"] - float(srv["analyzer_last_frame_at"])
        except (TypeError, ValueError):
            pass
    if (dup is not None and dup > 10) or (analyzer_age is not None and analyzer_age > 3):
        return "pipeline_gap"

    # Fetch coverage for the skipped media range
    lo, hi = skip["from"], skip["to"]
    need_sn = list(range(int(lo // 2), int(hi // 2) + 1))
    fetched = set()
    slow = 0
    for r in resources:
        if r["name"].startswith("stream") and r["name"].endswith(".ts"):
            try:
                sn = int(r["name"][len("stream") : -len(".ts")])
            except ValueError:
                continue
            fetched.add(sn)
            if r.get("duration", 0) > 2:
                slow += 1
    missing = [sn for sn in need_sn if sn not in fetched]
    if missing or slow:
        return "fetch_gap"
    return "render_skip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("browser_log")
    ap.add_argument("server_log")
    ap.add_argument("--epoch", default=None, help='e.g. "2026-08-12 09:31:00" (America/Chicago)')
    ap.add_argument("--export-snaps", default=None, help="dump frameSnap images to DIR for OCR")
    ap.add_argument("--out", default=".", help="output directory for CSVs")
    args = ap.parse_args()

    b = load_browser(args.browser_log)
    srv = load_server(args.server_log)
    log = b.get("log", [])
    epoch = parse_epoch(args.epoch) if args.epoch else None
    print(f"browser log: {len(log)} entries | server log: {len(srv)} records | epoch: {args.epoch or 'relative'}")

    resources = [e for e in log if e["type"] == "resource"]

    # Re-derive skips from rvfc (independent of the harness's own detector)
    rvfc = [e for e in log if e["type"] == "rvfc"]
    derived_skips = []
    prev = None
    for e in rvfc:
        if prev is not None:
            gap = e["mediaTime"] - prev
            if gap > 1.5:
                derived_skips.append({"from": prev, "to": e["mediaTime"], "gapSec": gap, "wall": e["wall"]})
        prev = e["mediaTime"]
    harness_skips = [e for e in log if e["type"] == "skip"]
    print(f"rvfc frames: {len(rvfc)} | harness skips: {len(harness_skips)} | derived skips: {len(derived_skips)}")
    if len(harness_skips) != len(derived_skips):
        print("WARNING: harness skip count differs from rvfc-derived count; using derived")

    # Latency curve (wall - capture time), requires --epoch
    if epoch is not None:
        lats = [e["wall"] / 1000 - (epoch + e["mediaTime"]) for e in rvfc]
        if lats:
            lats_sorted = sorted(lats)
            print(
                "latency (wall - capture): "
                f"min {lats_sorted[0]:.2f}s p50 {lats_sorted[len(lats_sorted)//2]:.2f}s "
                f"p95 {lats_sorted[int(len(lats_sorted)*0.95)]:.2f}s max {lats_sorted[-1]:.2f}s"
            )

    # Per-skip classification
    rows = []
    for sk in derived_skips:
        wall_sec = sk["wall"] / 1000
        srv_near = nearest_server(srv, wall_sec)
        cls = classify(sk, srv_near, resources)
        rows.append((sk, srv_near, cls))
    print("\n" + f"{'gap(s)':>7} {'from':>9} {'to':>9} {'wall':>12}  classification")
    for sk, srv_near, cls in rows:
        wall_fmt = datetime.datetime.fromtimestamp(sk["wall"] / 1000).strftime("%H:%M:%S.%f")[:-3]
        print(f"{sk['gapSec']:7.2f} {sk['from']:9.2f} {sk['to']:9.2f} {wall_fmt:>12}  {cls}")

    for sk, srv_near, cls in rows:
        if srv_near is None:
            continue
        fh = srv_near.get("ffmpeg_health") or {}
        age = None
        if srv_near.get("analyzer_last_frame_at"):
            try:
                age = srv_near["wall"] - float(srv_near["analyzer_last_frame_at"])
            except (TypeError, ValueError):
                pass
        wall_fmt = datetime.datetime.fromtimestamp(sk["wall"] / 1000).strftime("%H:%M:%S")
        if age is not None:
            print(
                f"\nskip {sk['gapSec']:.2f}s at {wall_fmt} ({cls}): "
                f"queues={srv_near.get('queues')} dup_pct={fh.get('dup_pct')} "
                f"segments={srv_near.get('oldest_segment')}..{srv_near.get('newest_segment')} "
                f"analyzer_age={age:.1f}s"
            )
        else:
            print(f"\nskip {sk['gapSec']:.2f}s at {wall_fmt} ({cls}): no analyzer key")

    # CSVs
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "skip_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gap_sec", "from_media", "to_media", "wall_iso", "classification"])
        for sk, srv_near, cls in rows:
            wall_fmt = datetime.datetime.fromtimestamp(sk["wall"] / 1000).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            w.writerow([f"{sk['gapSec']:.3f}", f"{sk['from']:.3f}", f"{sk['to']:.3f}", wall_fmt, cls])
    with open(os.path.join(args.out, "timeline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_ms", "type", "detail"])
        for e in log:
            if e["type"] in ("rvfc", "skip", "state", "resource", "hlsEvent"):
                detail = {k: v for k, v in e.items() if k not in ("type", "t", "wall")}
                w.writerow([e["wall"], e["type"], json.dumps(detail)])
    print(f"\nwrote {os.path.join(args.out, 'skip_table.csv')} and {os.path.join(args.out, 'timeline.csv')}")

    # Export frame snapshots for OCR calibration
    if args.export_snaps:
        os.makedirs(args.export_snaps, exist_ok=True)
        n = 0
        for i, e in enumerate(log):
            if e["type"] == "frameSnap" and e.get("dataUrl"):
                raw = e["dataUrl"].split(",", 1)[1]
                with open(os.path.join(args.export_snaps, f"snap_{i:04d}_mt{e['mediaTime']:.2f}.jpg"), "wb") as f:
                    f.write(base64.b64decode(raw))
                n += 1
        print(f"exported {n} frame snapshots to {args.export_snaps}")


if __name__ == "__main__":
    main()
