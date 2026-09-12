"""A restart must not take the public stream down with it.

clean_stream_directory used to delete every HLS file at startup, so each deploy (or
crash-restart) 404s the live stream for its duration: the viewer's playlist points at
segments that were just removed. ffmpeg already manages its own window with
-hls_flags delete_segments, so the only files a restart has to clear are the ones an
earlier session left behind - older than the window itself.
"""

import os
import time
from pathlib import Path

from stopsign import ffmpeg_service


def _write(path: Path, age_seconds: float) -> Path:
    path.write_bytes(b"segment")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def test_the_live_window_survives_a_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(ffmpeg_service, "STREAM_DIR", str(tmp_path))
    fresh_playlist = _write(tmp_path / "stream.m3u8", 5)
    fresh_segment = _write(tmp_path / "stream412.ts", 30)

    ffmpeg_service.clean_stream_directory(max_age_seconds=900)

    assert fresh_playlist.exists()
    assert fresh_segment.exists()


def test_a_previous_sessions_leftovers_are_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(ffmpeg_service, "STREAM_DIR", str(tmp_path))
    stale_segment = _write(tmp_path / "stream3.ts", 4000)

    ffmpeg_service.clean_stream_directory(max_age_seconds=900)

    assert not stale_segment.exists()


def test_clips_are_left_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(ffmpeg_service, "STREAM_DIR", str(tmp_path))
    clips = tmp_path / "clips"
    clips.mkdir()
    kept = _write(clips / "pass-clip.mp4", 99999)

    ffmpeg_service.clean_stream_directory(max_age_seconds=900)

    assert kept.exists()


def test_the_default_window_matches_the_configured_playlist(monkeypatch):
    """Pruning must not cut inside the window ffmpeg advertises to players."""
    assert ffmpeg_service.HLS_WINDOW_SECONDS == int(ffmpeg_service.HLS_LIST_SIZE) * 2
