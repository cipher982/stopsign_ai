"""Clip building worker for vehicle pass replay videos."""

import logging
import os
import subprocess
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path

from stopsign.database import Database
from stopsign.settings import DB_URL

logger = logging.getLogger(__name__)

CLIP_DIR = "/app/data/stream/clips"
CLIP_MAX_SEC = float(os.getenv("CLIP_MAX_SEC", "120"))
CLIP_PREPAD_SEC = float(os.getenv("CLIP_PREPAD_SEC", "2"))
CLIP_POSTPAD_SEC = float(os.getenv("CLIP_POSTPAD_SEC", "2"))
CLIP_PARKED_SEC = float(os.getenv("CLIP_PARKED_SEC", "300"))
CLIP_WORKER_INTERVAL_SEC = float(os.getenv("CLIP_WORKER_INTERVAL_SEC", "10"))
CLIP_MIN_AGE_SEC = float(os.getenv("CLIP_MIN_AGE_SEC", "2"))
_CLIP_EXPIRY_DEFAULT = max(180.0, CLIP_MAX_SEC + 30.0)
CLIP_EXPIRY_SEC = float(os.getenv("CLIP_EXPIRY_SEC", str(_CLIP_EXPIRY_DEFAULT)))

STREAM_FS_PATH = "/app/data/stream/stream.m3u8"


def _parse_program_date_time(value: str) -> float | None:
    """Parse an ISO 8601 timestamp from #EXT-X-PROGRAM-DATE-TIME to epoch float."""
    try:
        # Handle 'Z' suffix which fromisoformat doesn't accept in Python <3.11
        cleaned = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, OverflowError):
        return None


def _load_hls_segments(playlist_path: str):
    """Parse HLS playlist and return segment timing metadata.

    Uses #EXT-X-PROGRAM-DATE-TIME for accurate segment start times when
    available, falling back to file mtime otherwise.
    """
    if not os.path.exists(playlist_path):
        return []

    segments = []
    stream_dir = os.path.dirname(playlist_path)
    try:
        with open(playlist_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        logger.warning("Failed to read HLS playlist: %s", e)
        return []

    current_duration = None
    current_pdt = None  # program date-time epoch for current segment
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            current_pdt = _parse_program_date_time(line.split(":", 1)[1])
            continue
        if line.startswith("#EXTINF:"):
            try:
                current_duration = float(line.split(":", 1)[1].split(",", 1)[0])
            except ValueError:
                current_duration = None
            continue
        if line.startswith("#"):
            continue
        if current_duration is None:
            continue

        seg_path = os.path.join(stream_dir, line)
        if os.path.exists(seg_path):
            try:
                if current_pdt is not None:
                    # Accurate: use program date-time as segment start
                    start_ts = current_pdt
                    end_ts = start_ts + current_duration
                else:
                    # Fallback: derive from file mtime
                    end_ts = os.path.getmtime(seg_path)
                    start_ts = end_ts - current_duration
                segments.append(
                    {
                        "path": seg_path,
                        "start": start_ts,
                        "end": end_ts,
                        "duration": current_duration,
                    }
                )
            except OSError:
                pass
        current_duration = None
        current_pdt = None

    segments.sort(key=lambda s: s["start"])
    return segments


def _select_segments_for_window(segments, start_ts: float, end_ts: float):
    return [seg for seg in segments if seg["end"] >= start_ts and seg["start"] <= end_ts]


def _build_clip_for_pass(app, pass_data) -> bool:
    """Build an MP4 clip for a completed pass. Returns True on success."""
    if not hasattr(app.state, "db"):
        app.state.db = Database(db_url=DB_URL)

    entry_ts = getattr(pass_data, "entry_time", None)
    exit_ts = getattr(pass_data, "exit_time", None)

    if not entry_ts or not exit_ts:
        app.state.db.update_clip_status(pass_data.id, "failed", clip_error="missing_entry_exit")
        return False

    clip_start = entry_ts - CLIP_PREPAD_SEC
    clip_end = exit_ts + CLIP_POSTPAD_SEC

    stop_duration = getattr(pass_data, "stop_duration", 0.0) or 0.0
    if stop_duration >= CLIP_PARKED_SEC:
        clip_start = clip_end - CLIP_MAX_SEC

    if (clip_end - clip_start) > CLIP_MAX_SEC:
        clip_start = clip_end - CLIP_MAX_SEC

    clip_start = max(0.0, clip_start)

    clip_filename = f"clip_{pass_data.id}_{int(exit_ts)}.mp4"
    clip_path = os.path.join(CLIP_DIR, clip_filename)

    if os.path.exists(clip_path):
        app.state.db.update_clip_status(pass_data.id, "ready", clip_path=clip_filename)
        return True

    segments = _load_hls_segments(STREAM_FS_PATH)
    selected = _select_segments_for_window(segments, clip_start, clip_end)

    if not selected:
        seg_window = f"{segments[0]['start']:.0f}-{segments[-1]['end']:.0f}" if segments else "none"
        logger.debug(
            "No segments for pass %s (clip=[%.0f,%.0f] segs=%s)",
            pass_data.id,
            clip_start,
            clip_end,
            seg_window,
        )
        if time.time() - exit_ts > CLIP_EXPIRY_SEC:
            app.state.db.update_clip_status(pass_data.id, "expired", clip_error="segments_expired")
        return False

    concat_path = os.path.join(CLIP_DIR, f"concat_{pass_data.id}.txt")
    tmp_path = os.path.join(CLIP_DIR, f".{clip_filename}.tmp")
    base_ts = selected[0]["start"]
    offset_start = max(0.0, clip_start - base_ts)
    offset_end = max(offset_start, clip_end - base_ts)
    duration = max(0.1, offset_end - offset_start)

    try:
        with open(concat_path, "w", encoding="utf-8") as f:
            for seg in selected:
                f.write(f"file '{seg['path']}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-ss",
            f"{offset_start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-an",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0 or not os.path.exists(tmp_path):
            error_blob = result.stderr or result.stdout or "ffmpeg_failed"
            error = error_blob[-500:]
            logger.warning("Clip build failed for pass %s: %s", pass_data.id, error)
            app.state.db.update_clip_status(pass_data.id, "failed", clip_error=error)
            return False

        Path(tmp_path).rename(clip_path)
        app.state.db.update_clip_status(pass_data.id, "ready", clip_path=clip_filename)
        logger.info("Clip ready: pass %s -> %s (%.1fs)", pass_data.id, clip_filename, duration)
        return True
    except Exception as e:
        app.state.db.update_clip_status(pass_data.id, "failed", clip_error=str(e)[:500])
        return False
    finally:
        try:
            if os.path.exists(concat_path):
                os.remove(concat_path)
        except OSError:
            pass


def clip_worker_loop(app):
    """Background worker that builds clips for completed passes."""
    logger.info("Clip worker started (max=%ss, parked=%ss)", CLIP_MAX_SEC, CLIP_PARKED_SEC)
    while True:
        try:
            if not hasattr(app.state, "db"):
                app.state.db = Database(db_url=DB_URL)
            if app.state.db.read_only_mode:
                time.sleep(CLIP_WORKER_INTERVAL_SEC)
                continue

            pending = app.state.db.get_passes_missing_clips(limit=5, min_exit_age_sec=CLIP_MIN_AGE_SEC)
            for pass_data in pending:
                _build_clip_for_pass(app, pass_data)
        except Exception as e:
            logger.error("Clip worker error: %s", e)
        time.sleep(CLIP_WORKER_INTERVAL_SEC)
