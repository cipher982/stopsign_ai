import os
import re
import time
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict


def parse_hls_playlist(path: str) -> Dict[str, Any]:
    """Parse HLS playlist and derive dynamic freshness signals.

    Returns keys:
      - exists: bool
      - playlist_mtime: float|None (stat mtime)
      - age_seconds: float|None (now - mtime, used for health checks)
      - pdt_age_seconds: float|None (now - last PDT, informational only)
      - target_duration_sec: float|None
      - playlist_window_sec: float|None (sum of #EXTINF durations)
      - segments_count: int
      - threshold_sec: float (≈ 3×window with 60s floor)

    Note: age_seconds uses file mtime rather than PDT timestamps because
    FFmpeg's internal clock drifts from wall-clock time when frame delivery
    is not perfectly aligned with the declared input FPS.
    """
    now = time.time()
    out: Dict[str, Any] = {
        "exists": False,
        "playlist_mtime": None,
        "age_seconds": None,
        "pdt_age_seconds": None,
        "target_duration_sec": None,
        "playlist_window_sec": None,
        "segments_count": 0,
        "threshold_sec": 60.0,
    }

    if not os.path.exists(path):
        return out

    out["exists"] = True
    try:
        st = os.stat(path)
        out["playlist_mtime"] = st.st_mtime
    except OSError:
        pass

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        # Fall back to mtime-based age only
        if out["playlist_mtime"] is not None:
            out["age_seconds"] = max(0.0, now - out["playlist_mtime"])  # type: ignore[arg-type]
        out["threshold_sec"] = 60.0
        return out

    # Target duration
    m = re.search(r"^#EXT-X-TARGETDURATION:(\d+(?:\.\d+)?)", content, flags=re.MULTILINE)
    if m:
        try:
            out["target_duration_sec"] = float(m.group(1))
        except ValueError:
            pass

    # Segment durations
    extinf = re.findall(r"#EXTINF:([0-9.]+)", content)
    seg_durs = []
    for d in extinf:
        try:
            seg_durs.append(float(d))
        except ValueError:
            continue
    if seg_durs:
        out["playlist_window_sec"] = sum(seg_durs)
        out["segments_count"] = len(seg_durs)

    # Use file mtime for age_seconds (health checks / watchdog).
    # mtime tracks wall-clock time accurately regardless of encoder drift.
    if out["playlist_mtime"] is not None:
        out["age_seconds"] = max(0.0, now - out["playlist_mtime"])  # type: ignore[arg-type]

    # Program date-time (PDT) of last segment — informational only.
    # PDT drifts from wall-clock time because FFmpeg derives it from its
    # internal frame counter, which runs slightly fast when frame delivery
    # is slower than the declared input FPS.
    pdt_matches = re.findall(r"#EXT-X-PROGRAM-DATE-TIME:([^\r\n]+)", content)
    if pdt_matches:
        last = pdt_matches[-1].strip()
        try:
            # Normalize Z to +00:00 for fromisoformat
            if last.endswith("Z"):
                last = last[:-1] + "+00:00"
            dt = datetime.fromisoformat(last)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            out["pdt_age_seconds"] = max(0.0, time.time() - dt.timestamp())
        except Exception:
            pass

    # Threshold: 3×window (floor 60s, cap 180s)
    # Cap prevents large HLS windows (e.g. 15-min for clip retention) from
    # making the health check tolerate unreasonably stale playlists.
    window = out.get("playlist_window_sec")
    if isinstance(window, (int, float)) and window > 0:
        out["threshold_sec"] = max(60.0, min(3.0 * float(window), 180.0))
    else:
        out["threshold_sec"] = 60.0

    # Also consider .ts files present in the same directory (best-effort)
    try:
        stream_dir = os.path.dirname(path)
        if os.path.isdir(stream_dir):
            ts_count = len([f for f in os.listdir(stream_dir) if f.endswith(".ts")])
            out["segments_count"] = max(out.get("segments_count", 0) or 0, ts_count)
    except Exception:
        pass

    return out
