"""Clip building worker for vehicle pass replay videos."""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path

import redis
from minio import Minio

from stopsign.database import Database
from stopsign.settings import BREMEN_MINIO_ACCESS_KEY
from stopsign.settings import BREMEN_MINIO_CLIP_BUCKET
from stopsign.settings import BREMEN_MINIO_ENDPOINT
from stopsign.settings import BREMEN_MINIO_SECRET_KEY
from stopsign.settings import DB_URL
from stopsign.settings import FRAME_METADATA_KEY
from stopsign.settings import PROCESSED_FRAME_KEY
from stopsign.settings import REDIS_URL

logger = logging.getLogger(__name__)

CLIP_DIR = "/app/data/stream/clips"
CLIP_MAX_SEC = float(os.getenv("CLIP_MAX_SEC", "120"))
CLIP_PREPAD_SEC = float(os.getenv("CLIP_PREPAD_SEC", "2"))
CLIP_POSTPAD_SEC = float(os.getenv("CLIP_POSTPAD_SEC", "2"))
CLIP_PARKED_SEC = float(os.getenv("CLIP_PARKED_SEC", "300"))
CLIP_WORKER_INTERVAL_SEC = float(os.getenv("CLIP_WORKER_INTERVAL_SEC", "10"))
CLIP_MIN_AGE_SEC = float(os.getenv("CLIP_MIN_AGE_SEC", "2"))
CLIP_MAX_RETRIES = int(os.getenv("CLIP_MAX_RETRIES", "3"))
# Segment window: HLS_LIST_SIZE * 2s per segment. Passes older than this can't have clips built.
_HLS_LIST_SIZE = int(os.getenv("HLS_LIST_SIZE", "450"))
CLIP_SEGMENT_WINDOW_SEC = _HLS_LIST_SIZE * 2 + 60  # segments window + buffer
# Local clip retention: keep last N clips on disk (older ones pruned after MinIO upload)
LOCAL_CLIP_MAX_COUNT = int(os.getenv("LOCAL_CLIP_MAX_COUNT", "200"))
# Calibrated on production (March 2, 2026): remove env-driven drift for this
# side project and keep alignment behavior explicit in code.
CLIP_TIME_OFFSET_SEC = 5.5
CLIP_DYNAMIC_LAG_ENABLED = os.getenv("CLIP_DYNAMIC_LAG_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
CLIP_DYNAMIC_LAG_MIN_SEC = float(os.getenv("CLIP_DYNAMIC_LAG_MIN_SEC", "-5"))
CLIP_DYNAMIC_LAG_MAX_SEC = float(os.getenv("CLIP_DYNAMIC_LAG_MAX_SEC", "60"))
CLIP_DYNAMIC_LAG_FALLBACK_ENABLED = False
CLIP_PASS_LAG_ENABLED = True
CLIP_PASS_LAG_MIN_SEC = 0.0
CLIP_PASS_LAG_MAX_SEC = 45.0
CLIP_QUEUE_LAG_ENABLED = os.getenv("CLIP_QUEUE_LAG_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
CLIP_QUEUE_LAG_FPS = max(1.0, float(os.getenv("CLIP_QUEUE_LAG_FPS", "15")))
CLIP_QUEUE_LAG_BASE_SEC = float(os.getenv("CLIP_QUEUE_LAG_BASE_SEC", "2.0"))

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
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            continue  # PDT parsed but unused — see mtime comment below
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
                # Always use file mtime as segment end time.
                # PDT timestamps are derived from FFmpeg PTS which can drift
                # significantly behind wall clock (measured: ~47 min lag after
                # 24h of operation at slightly-below-target fps). File mtime is
                # always wall-clock accurate and matches vehicle_pass.exit_time
                # which is also from time.time(). PDT is parsed but not used.
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

    segments.sort(key=lambda s: s["start"])
    return segments


def _select_segments_for_window(segments, start_ts: float, end_ts: float):
    return [seg for seg in segments if seg["end"] >= start_ts and seg["start"] <= end_ts]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _estimate_dynamic_lag_sec(segments) -> float:
    """Estimate segment-clock minus capture-clock lag from live metadata."""
    if not CLIP_DYNAMIC_LAG_ENABLED or not segments:
        return 0.0

    try:
        client = redis.from_url(REDIS_URL, socket_connect_timeout=0.2, socket_timeout=0.2)
        latest_seg_end = float(segments[-1]["end"])

        # 1) Metadata-based lag (wall-clock segment time minus most recent capture timestamp)
        meta_lag = 0.0
        raw = client.get(FRAME_METADATA_KEY)
        if raw:
            payload = json.loads(raw)
            capture_ts = float(payload.get("capture_timestamp", 0.0) or 0.0)
            if capture_ts > 0:
                meta_lag = latest_seg_end - capture_ts

        # 2) Queue-depth lag (primary for FIFO backlog): queued frames/fps + HLS close latency
        queue_lag = 0.0
        if CLIP_QUEUE_LAG_ENABLED:
            try:
                depth = int(client.llen(PROCESSED_FRAME_KEY) or 0)
                queue_lag = (depth / CLIP_QUEUE_LAG_FPS) + CLIP_QUEUE_LAG_BASE_SEC
            except Exception:
                queue_lag = 0.0

        # Use the larger positive lag signal; this avoids under-correcting when
        # ffmpeg is behind due buffered backlog.
        lag = max(meta_lag, queue_lag, 0.0)
        return _clamp(lag, CLIP_DYNAMIC_LAG_MIN_SEC, CLIP_DYNAMIC_LAG_MAX_SEC)
    except Exception:
        return 0.0


def _estimate_pass_lag_sec(pass_data) -> float | None:
    """Estimate clip shift from per-pass metadata captured at zone exit."""
    if not CLIP_PASS_LAG_ENABLED:
        return None

    try:
        lag_est = getattr(pass_data, "stream_lag_est_sec", None)
        if lag_est is not None:
            lag = float(lag_est)
            return _clamp(max(0.0, lag), CLIP_PASS_LAG_MIN_SEC, CLIP_PASS_LAG_MAX_SEC)
    except Exception:
        pass

    try:
        depth = getattr(pass_data, "stream_queue_depth_exit", None)
        if depth is None:
            return None
        lag = max(0.0, float(depth) / CLIP_QUEUE_LAG_FPS)
        return _clamp(lag, CLIP_PASS_LAG_MIN_SEC, CLIP_PASS_LAG_MAX_SEC)
    except Exception:
        return None


def _timeline_offset_sec(segments, target_ts: float) -> float:
    """Map wall-clock timestamp into concat timeline seconds (gap-aware)."""
    elapsed = 0.0
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        dur = float(seg["duration"])
        if target_ts <= start:
            return elapsed
        if target_ts < end:
            return elapsed + max(0.0, target_ts - start)
        elapsed += max(0.0, dur)
    return elapsed


def _upload_clip_to_minio(clip_path: str, object_name: str) -> bool:
    """Upload a clip to MinIO with retry. Returns True on success."""
    if not BREMEN_MINIO_SECRET_KEY:
        logger.debug("MinIO secret key not set, skipping clip upload")
        return False

    for attempt in range(3):
        try:
            client = Minio(
                BREMEN_MINIO_ENDPOINT,
                access_key=BREMEN_MINIO_ACCESS_KEY,
                secret_key=BREMEN_MINIO_SECRET_KEY,
                secure=False,
            )
            client.fput_object(
                BREMEN_MINIO_CLIP_BUCKET,
                object_name,
                clip_path,
                content_type="video/mp4",
            )
            logger.info("Clip uploaded to MinIO: %s/%s", BREMEN_MINIO_CLIP_BUCKET, object_name)
            return True
        except Exception as e:
            logger.warning("MinIO upload attempt %d failed for %s: %s", attempt + 1, object_name, e)
            if attempt < 2:
                time.sleep(1 << attempt)  # 1s, 2s
    return False


def _download_clip_from_minio(object_name: str, local_path: str) -> bool:
    """Download a clip from MinIO to local disk. Returns True on success."""
    if not BREMEN_MINIO_SECRET_KEY:
        return False
    try:
        client = Minio(
            BREMEN_MINIO_ENDPOINT,
            access_key=BREMEN_MINIO_ACCESS_KEY,
            secret_key=BREMEN_MINIO_SECRET_KEY,
            secure=False,
        )
        client.fget_object(BREMEN_MINIO_CLIP_BUCKET, object_name, local_path)
        logger.info("Clip restored from MinIO: %s", object_name)
        return True
    except Exception as e:
        logger.warning("Failed to restore clip %s from MinIO: %s", object_name, e)
        return False


def _ensure_clip_bucket():
    """Create the clip bucket on MinIO if it doesn't exist."""
    if not BREMEN_MINIO_SECRET_KEY:
        return
    try:
        client = Minio(
            BREMEN_MINIO_ENDPOINT,
            access_key=BREMEN_MINIO_ACCESS_KEY,
            secret_key=BREMEN_MINIO_SECRET_KEY,
            secure=False,
        )
        if not client.bucket_exists(BREMEN_MINIO_CLIP_BUCKET):
            client.make_bucket(BREMEN_MINIO_CLIP_BUCKET)
            logger.info("Created MinIO bucket: %s", BREMEN_MINIO_CLIP_BUCKET)
    except Exception as e:
        logger.warning("Failed to ensure clip bucket: %s", e)


def _prune_local_clips():
    """Remove oldest local clips to stay within LOCAL_CLIP_MAX_COUNT."""
    try:
        clip_dir = Path(CLIP_DIR)
        clips = sorted(clip_dir.glob("clip_*.mp4"), key=lambda p: p.stat().st_mtime)
        if len(clips) > LOCAL_CLIP_MAX_COUNT:
            to_remove = len(clips) - LOCAL_CLIP_MAX_COUNT
            for clip in clips[:to_remove]:
                try:
                    clip.unlink()
                    logger.debug("Pruned old clip: %s", clip.name)
                except OSError as e:
                    logger.warning("Failed to prune %s: %s", clip.name, e)
            logger.info("Pruned %d old clips to maintain %d limit", to_remove, LOCAL_CLIP_MAX_COUNT)
    except Exception as e:
        logger.warning("Clip pruning error: %s", e)


def _get_retry_count(clip_error: str | None) -> int:
    """Extract retry count from clip_error field (format: 'retry:N|...')."""
    if not clip_error or not clip_error.startswith("retry:"):
        return 0
    try:
        return int(clip_error.split("|", 1)[0].split(":", 1)[1])
    except (ValueError, IndexError):
        return 0


def _make_retry_error(count: int, error: str) -> str:
    """Format clip_error with retry count prefix."""
    return f"retry:{count}|{error[:480]}"


def _build_clip_for_pass(app, pass_data) -> bool:
    """Build an MP4 clip for a completed pass. Returns True on success."""
    if not hasattr(app.state, "db"):
        app.state.db = Database(db_url=DB_URL)

    entry_ts = getattr(pass_data, "entry_time", None)
    exit_ts = getattr(pass_data, "exit_time", None)

    if not entry_ts or not exit_ts:
        app.state.db.update_clip_status(pass_data.id, "failed", clip_error="missing_entry_exit")
        return False

    # Check actual segment coverage before applying wall-clock cutoff.
    # After downtime, segments may still be on disk even though wall-clock
    # time exceeds the window.
    segments = _load_hls_segments(STREAM_FS_PATH)
    pass_lag_sec = _estimate_pass_lag_sec(pass_data)
    dynamic_lag_sec = _estimate_dynamic_lag_sec(segments) if CLIP_DYNAMIC_LAG_FALLBACK_ENABLED else 0.0
    use_pass_lag = pass_lag_sec is not None
    align_offset_sec = CLIP_TIME_OFFSET_SEC + (pass_lag_sec if use_pass_lag else dynamic_lag_sec)
    logger.debug(
        "Clip alignment for pass %s: offset=%.2fs (source=%s, pass_lag=%s, dynamic_lag=%.2f, base=%.2f)",
        pass_data.id,
        align_offset_sec,
        "pass_metadata" if use_pass_lag else "dynamic_fallback",
        f"{pass_lag_sec:.2f}" if pass_lag_sec is not None else "none",
        dynamic_lag_sec,
        CLIP_TIME_OFFSET_SEC,
    )
    aligned_entry_ts = entry_ts + align_offset_sec
    aligned_exit_ts = exit_ts + align_offset_sec
    seg_covers_pass = bool(segments) and (segments[0]["start"] <= aligned_entry_ts <= segments[-1]["end"])

    if not seg_covers_pass and time.time() - aligned_exit_ts > CLIP_SEGMENT_WINDOW_SEC:
        app.state.db.update_clip_status(pass_data.id, "no_segments", clip_error="beyond_segment_window")
        return False

    clip_start = aligned_entry_ts - CLIP_PREPAD_SEC
    clip_end = aligned_exit_ts + CLIP_POSTPAD_SEC

    stop_duration = getattr(pass_data, "stop_duration", 0.0) or 0.0
    if stop_duration >= CLIP_PARKED_SEC:
        clip_start = clip_end - CLIP_MAX_SEC

    if (clip_end - clip_start) > CLIP_MAX_SEC:
        clip_start = clip_end - CLIP_MAX_SEC

    clip_start = max(0.0, clip_start)

    clip_filename = f"clip_{pass_data.id}_{int(exit_ts)}.mp4"
    clip_path = os.path.join(CLIP_DIR, clip_filename)

    if os.path.exists(clip_path):
        # File exists locally — try to upload to MinIO if not already done
        uploaded = _upload_clip_to_minio(clip_path, clip_filename)
        if uploaded:
            app.state.db.update_clip_status(pass_data.id, "ready", clip_path=clip_filename)
            _prune_local_clips()
        else:
            # MinIO unavailable — serve locally, worker will retry upload later
            app.state.db.update_clip_status(pass_data.id, "local", clip_path=clip_filename)
        return True

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
        # Don't mark as failed — segments may appear on next poll
        return False

    if selected[0]["start"] > clip_start or selected[-1]["end"] < clip_end:
        # Partial window coverage usually means segment rotation or temporary lag.
        # Retry on next worker cycle rather than building a misleading clip.
        return False

    concat_path = os.path.join(CLIP_DIR, f"concat_{pass_data.id}.txt")
    tmp_path = os.path.join(CLIP_DIR, f".{clip_filename}.tmp")
    offset_start = _timeline_offset_sec(selected, clip_start)
    offset_end = max(offset_start, _timeline_offset_sec(selected, clip_end))
    duration = max(0.1, offset_end - offset_start)

    retry_count = _get_retry_count(getattr(pass_data, "clip_error", None))

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
            next_retry = retry_count + 1
            if next_retry >= CLIP_MAX_RETRIES:
                app.state.db.update_clip_status(pass_data.id, "failed", clip_error=_make_retry_error(next_retry, error))
            else:
                # Mark retryable — worker will pick it up again
                app.state.db.update_clip_status(pass_data.id, "retry", clip_error=_make_retry_error(next_retry, error))
            return False

        Path(tmp_path).rename(clip_path)

        # Upload to MinIO
        uploaded = _upload_clip_to_minio(clip_path, clip_filename)
        if uploaded:
            app.state.db.update_clip_status(pass_data.id, "ready", clip_path=clip_filename)
            _prune_local_clips()
        else:
            # MinIO unavailable — serve locally, worker will retry upload later
            app.state.db.update_clip_status(pass_data.id, "local", clip_path=clip_filename)

        logger.info("Clip ready: pass %s -> %s (%.1fs)", pass_data.id, clip_filename, duration)
        return True
    except Exception as e:
        next_retry = retry_count + 1
        if next_retry >= CLIP_MAX_RETRIES:
            app.state.db.update_clip_status(pass_data.id, "failed", clip_error=_make_retry_error(next_retry, str(e)))
        else:
            app.state.db.update_clip_status(pass_data.id, "retry", clip_error=_make_retry_error(next_retry, str(e)))
        return False
    finally:
        try:
            if os.path.exists(concat_path):
                os.remove(concat_path)
        except OSError:
            pass


def clip_worker_loop(app):
    """Background worker that builds clips for completed passes."""
    logger.info(
        "Clip worker started (max=%ss, segment_window=%ss, minio_bucket=%s)",
        CLIP_MAX_SEC,
        CLIP_SEGMENT_WINDOW_SEC,
        BREMEN_MINIO_CLIP_BUCKET,
    )
    _ensure_clip_bucket()
    while True:
        try:
            if not hasattr(app.state, "db"):
                app.state.db = Database(db_url=DB_URL)
            if app.state.db.read_only_mode:
                time.sleep(CLIP_WORKER_INTERVAL_SEC)
                continue

            # Restore clips that were uploaded to MinIO but lost locally (e.g. after redeploy)
            uploaded_missing = app.state.db.get_clips_missing_locally(limit=5)
            for pass_data in uploaded_missing:
                clip_path = os.path.join(CLIP_DIR, pass_data.clip_path)
                if not os.path.exists(clip_path):
                    _download_clip_from_minio(pass_data.clip_path, clip_path)

            pending = app.state.db.get_passes_needing_clips(limit=5, min_exit_age_sec=CLIP_MIN_AGE_SEC)
            for pass_data in pending:
                _build_clip_for_pass(app, pass_data)
        except Exception as e:
            logger.error("Clip worker error: %s", e)
        time.sleep(CLIP_WORKER_INTERVAL_SEC)
