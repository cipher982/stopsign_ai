#!/usr/bin/env python3
"""Audit replay clip alignment using burned-in clip timestamps.

This script avoids manual website QA by:
1) querying recent vehicle passes with clips from Postgres
2) extracting a near-tail frame from each clip
3) OCR'ing the top-right burned timestamp
4) comparing observed clip time vs expected pass time

Usage:
  uv run python scripts/audit_clip_alignment.py --limit 20
  uv run python scripts/audit_clip_alignment.py --pass-id 55070 --pass-id 55071
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import psycopg2
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps

DEFAULT_CUBE_HOST = "cube"
DEFAULT_REMOTE_CLIP_DIR = "/mnt/hls-stream/clips"
DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "stopsign_clip_audit")
DEFAULT_TIMEZONE = "America/Chicago"


@dataclass
class PassRow:
    pass_id: int
    exit_time: float
    clip_path: str
    clip_status: str
    time_in_zone: float | None
    stop_duration: float | None


@dataclass
class AuditResult:
    pass_id: int
    clip_path: str
    status: str
    duration_sec: float | None
    observed_clip_ts: datetime | None
    expected_clip_ts: datetime
    error_sec: float | None
    note: str


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _get_db_url(args: argparse.Namespace) -> str:
    if args.db_url:
        return args.db_url
    env_url = os.environ.get("DB_URL")
    if env_url:
        return env_url
    if not args.auto_fetch_db_url:
        raise RuntimeError("DB_URL not set. Provide --db-url or set DB_URL.")

    cmd = [
        "ssh",
        args.cube_host,
        ("docker exec $(docker ps --format '{{.Names}}' | grep web_server | head -n1) sh -lc 'echo $DB_URL'"),
    ]
    out = _run(cmd)
    db_url = out.stdout.strip()
    if not db_url:
        raise RuntimeError("Could not auto-fetch DB_URL from cube web_server container.")
    return db_url


def _redact_db_url(db_url: str) -> str:
    # postgresql://user:pass@host/db -> postgresql://user:***@host/db
    return re.sub(r"://([^:/]+):([^@]+)@", r"://\1:***@", db_url)


def _query_passes(db_url: str, limit: int, pass_ids: list[int] | None) -> list[PassRow]:
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            if pass_ids:
                cur.execute(
                    """
                    SELECT id, exit_time, clip_path, clip_status, time_in_zone, stop_duration
                    FROM vehicle_passes
                    WHERE id = ANY(%s)
                      AND clip_path IS NOT NULL
                      AND clip_status IN ('ready','local')
                    ORDER BY id DESC
                    """,
                    (pass_ids,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, exit_time, clip_path, clip_status, time_in_zone, stop_duration
                    FROM vehicle_passes
                    WHERE clip_path IS NOT NULL
                      AND clip_status IN ('ready','local')
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
        return [
            PassRow(
                pass_id=int(r[0]),
                exit_time=float(r[1]),
                clip_path=str(r[2]),
                clip_status=str(r[3]),
                time_in_zone=float(r[4]) if r[4] is not None else None,
                stop_duration=float(r[5]) if r[5] is not None else None,
            )
            for r in rows
        ]
    finally:
        conn.close()


def _ensure_clip_local(clip_path: str, args: argparse.Namespace) -> Path:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / clip_path
    if local_path.exists():
        return local_path

    remote = f"{args.cube_host}:{args.remote_clip_dir.rstrip('/')}/{clip_path}"
    cmd = ["scp", remote, str(local_path)]
    _run(cmd)
    return local_path


def _ffprobe_duration_sec(clip_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(clip_path),
    ]
    out = _run(cmd)
    return float(out.stdout.strip())


def _extract_frame(clip_path: Path, out_path: Path, seek_sec: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{seek_sec:.3f}",
        "-i",
        str(clip_path),
        "-frames:v",
        "1",
        str(out_path),
    ]
    _run(cmd)


def _ocr_text(image_path: Path, psm: int = 7) -> str:
    cmd = [
        "tesseract",
        str(image_path),
        "stdout",
        "--psm",
        str(psm),
        "-l",
        "eng",
        "-c",
        "tessedit_char_whitelist=0123456789:-/ ",
    ]
    out = _run(cmd, check=False)
    return (out.stdout or "").strip()


def _parse_overlay_datetime(text: str, tz: ZoneInfo) -> datetime | None:
    cleaned = " ".join(text.replace("\n", " ").split())
    m = re.search(r"(20\d{2})[-/](\d{2})[-/](\d{2})\s+(\d{2}):(\d{2}):(\d{2})", cleaned)
    if not m:
        return None

    year, month, day, hour, minute, second = map(int, m.groups())
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=tz)
    except ValueError:
        return None


def _extract_overlay_ts_candidates(frame_path: Path, tz: ZoneInfo, cache_dir: Path) -> list[datetime]:
    with Image.open(frame_path) as img:
        w, h = img.size
        # Top-right timestamp lives in this area in 1920x1080 output.
        roi = img.crop((int(0.60 * w), 0, w, int(0.14 * h)))
        roi = roi.resize((roi.width * 2, roi.height * 2))
        gray = ImageOps.grayscale(roi)

        variants: list[tuple[Image.Image, int]] = []
        variants.append((ImageOps.autocontrast(gray), 6))
        sharp = ImageEnhance.Sharpness(gray).enhance(2.5)
        variants.append((ImageOps.autocontrast(sharp), 6))
        variants.append((gray.point(lambda p: 255 if p > 150 else 0), 6))
        variants.append((gray.point(lambda p: 255 if p > 160 else 0), 6))
        variants.append((gray.point(lambda p: 255 if p > 170 else 0), 6))
        variants.append((ImageOps.autocontrast(gray), 11))
        variants.append((gray.point(lambda p: 255 if p > 160 else 0), 11))

        candidates: list[datetime] = []
        cache_dir.mkdir(parents=True, exist_ok=True)
        for variant, psm in variants:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=cache_dir) as tmp:
                tmp_path = Path(tmp.name)
            try:
                variant.save(tmp_path)
                text = _ocr_text(tmp_path, psm=psm)
                parsed = _parse_overlay_datetime(text, tz)
                if parsed is not None:
                    candidates.append(parsed)
            finally:
                tmp_path.unlink(missing_ok=True)

        return candidates
    return []


def _select_observed_ts(
    candidates: list[datetime], expected: datetime, max_plausible_sec: float
) -> tuple[datetime | None, str]:
    if not candidates:
        return None, "ocr_parse_failed"

    expected_epoch = int(expected.timestamp())
    epochs = [int(dt.timestamp()) for dt in candidates]
    counts = Counter(epochs)
    median_epoch = int(statistics.median(epochs))
    ranked = sorted(
        counts.keys(),
        key=lambda epoch: (-counts[epoch], abs(epoch - median_epoch), -epoch),
    )
    best_epoch = ranked[0]
    best_diff = abs(best_epoch - expected_epoch)
    if best_diff <= max_plausible_sec:
        return datetime.fromtimestamp(best_epoch, tz=expected.tzinfo), ""

    plausible = [epoch for epoch in counts if abs(epoch - expected_epoch) <= max_plausible_sec]
    if plausible:
        chosen = min(
            plausible,
            key=lambda epoch: (abs(epoch - expected_epoch), -counts[epoch], -epoch),
        )
        return datetime.fromtimestamp(chosen, tz=expected.tzinfo), "ocr_disambiguated"

    return datetime.fromtimestamp(best_epoch, tz=expected.tzinfo), "ocr_unplausible"


def _pctl(values: Iterable[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    idx = (len(vals) - 1) * q
    lo = int(idx)
    hi = min(len(vals) - 1, lo + 1)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def _audit_one(pass_row: PassRow, args: argparse.Namespace, tz: ZoneInfo) -> AuditResult:
    expected = datetime.fromtimestamp(pass_row.exit_time + args.postpad_sec, tz=tz)
    try:
        clip_local = _ensure_clip_local(pass_row.clip_path, args)
    except Exception as e:
        return AuditResult(
            pass_id=pass_row.pass_id,
            clip_path=pass_row.clip_path,
            status="error",
            duration_sec=None,
            observed_clip_ts=None,
            expected_clip_ts=expected,
            error_sec=None,
            note=f"clip_download_failed: {e}",
        )

    try:
        duration = _ffprobe_duration_sec(clip_local)
    except Exception as e:
        return AuditResult(
            pass_id=pass_row.pass_id,
            clip_path=pass_row.clip_path,
            status="error",
            duration_sec=None,
            observed_clip_ts=None,
            expected_clip_ts=expected,
            error_sec=None,
            note=f"ffprobe_failed: {e}",
        )

    sample_sec = max(0.0, duration - args.tail_seconds)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=cache_dir) as tmp:
        frame_path = Path(tmp.name)
    try:
        _extract_frame(clip_local, frame_path, sample_sec)
        candidates = _extract_overlay_ts_candidates(frame_path, tz, cache_dir)
    except Exception as e:
        return AuditResult(
            pass_id=pass_row.pass_id,
            clip_path=pass_row.clip_path,
            status="error",
            duration_sec=duration,
            observed_clip_ts=None,
            expected_clip_ts=expected,
            error_sec=None,
            note=f"frame_or_ocr_failed: {e}",
        )
    finally:
        frame_path.unlink(missing_ok=True)

    observed, ocr_note = _select_observed_ts(candidates, expected, args.max_plausible_sec)
    if observed is None:
        return AuditResult(
            pass_id=pass_row.pass_id,
            clip_path=pass_row.clip_path,
            status="unknown",
            duration_sec=duration,
            observed_clip_ts=None,
            expected_clip_ts=expected,
            error_sec=None,
            note=ocr_note,
        )

    error_sec = (observed - expected).total_seconds()
    if abs(error_sec) <= args.threshold_sec:
        status = "ok"
    elif error_sec < 0:
        status = "early"
    else:
        status = "late"

    return AuditResult(
        pass_id=pass_row.pass_id,
        clip_path=pass_row.clip_path,
        status=status,
        duration_sec=duration,
        observed_clip_ts=observed,
        expected_clip_ts=expected,
        error_sec=error_sec,
        note=ocr_note,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit replay clip alignment with OCR.")
    parser.add_argument("--db-url", help="Postgres URL. Defaults to DB_URL env or auto-fetch from cube.")
    parser.add_argument("--cube-host", default=DEFAULT_CUBE_HOST, help="SSH host for clip fetch + DB_URL auto-fetch.")
    parser.add_argument("--remote-clip-dir", default=DEFAULT_REMOTE_CLIP_DIR, help="Remote clip directory on cube.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Local clip cache directory.")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent passes to audit (ignored with --pass-id).",
    )
    parser.add_argument("--pass-id", type=int, action="append", help="Specific pass ID(s) to audit.")
    parser.add_argument("--postpad-sec", type=float, default=2.0, help="Expected clip postpad seconds.")
    parser.add_argument("--tail-seconds", type=float, default=0.25, help="Sample this far from clip end.")
    parser.add_argument("--threshold-sec", type=float, default=3.0, help="Abs error threshold for mismatch.")
    parser.add_argument(
        "--max-plausible-sec",
        type=float,
        default=600.0,
        help="If OCR majority result is farther than this from expected, prefer closer parsed candidate if present.",
    )
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help="Overlay timezone (default America/Chicago).")
    parser.add_argument(
        "--auto-fetch-db-url",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-fetch DB_URL from cube web_server when DB_URL not provided.",
    )
    args = parser.parse_args()

    tz = ZoneInfo(args.timezone)
    db_url = _get_db_url(args)
    print(f"DB: {_redact_db_url(db_url)}")

    passes = _query_passes(db_url, args.limit, args.pass_id)
    if not passes:
        print("No clip-ready passes found for audit.")
        return 2

    print(f"Auditing {len(passes)} pass(es)...")
    results: list[AuditResult] = []
    for p in passes:
        r = _audit_one(p, args, tz)
        results.append(r)
        err_str = "n/a" if r.error_sec is None else f"{r.error_sec:+.2f}s"
        obs = r.observed_clip_ts.strftime("%Y-%m-%d %H:%M:%S") if r.observed_clip_ts else "n/a"
        exp = r.expected_clip_ts.strftime("%Y-%m-%d %H:%M:%S")
        print(f"pass={r.pass_id} status={r.status:<7} err={err_str:<8} obs={obs} exp={exp} {r.note}")

    parsed = [r for r in results if r.error_sec is not None]
    early = [r for r in parsed if r.status == "early"]
    late = [r for r in parsed if r.status == "late"]
    abs_errors = [abs(r.error_sec) for r in parsed if r.error_sec is not None]
    parse_rate = len(parsed) / len(results)

    print("\nSummary")
    print(f"total={len(results)} parsed={len(parsed)} parse_rate={parse_rate:.0%}")
    print(f"ok={sum(1 for r in parsed if r.status == 'ok')} early={len(early)} late={len(late)}")
    if abs_errors:
        print(f"abs_err_median={statistics.median(abs_errors):.2f}s abs_err_p95={_pctl(abs_errors, 0.95):.2f}s")

    mismatches = [r for r in parsed if abs(r.error_sec or 0.0) > args.threshold_sec]
    if mismatches or parse_rate < 0.7:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
