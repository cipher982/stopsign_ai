"""Ingest-rate guard.

The freeze detector answers "is the picture moving?". A camera whose WiFi link is
dropping frames answers that question "yes" while delivering almost nothing, and
on 2026-09-11 that mode starved the whole chain for 90 minutes with every health
signal green. These tests pin the guard that covers it.
"""

import importlib


def _load(monkeypatch, **env):
    monkeypatch.setenv("PROMETHEUS_PORT", "8080")
    monkeypatch.setenv("RTSP_URL", "file:///tmp/example.mp4")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RAW_FRAME_KEY", "raw_frames")
    monkeypatch.setenv("FRAME_BUFFER_SIZE", "10")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    module = importlib.import_module("rtsp_to_redis.rtsp_to_redis")
    return importlib.reload(module)


def _silence_exit(monkeypatch, module):
    exits = []
    monkeypatch.setattr(module.os, "_exit", lambda code: exits.append(code))
    return exits


def test_live_but_starved_camera_escalates_while_no_freeze_is_detected(monkeypatch):
    module = _load(monkeypatch, RTSP_MIN_INPUT_FPS="8", RTSP_LOW_FPS_RECONNECT_SEC="120")
    service = module.RTSPToRedis()
    service.update_status_metric("redis_connected", True)
    exits = _silence_exit(monkeypatch, module)

    t0 = 1_000_000.0
    service._track_input_fps(0.2, t0)

    # Content keeps changing, so the freeze detector reports nothing wrong.
    assert service.freeze_age_sec == 0.0

    # A brief dip is not actionable; a sustained one reconnects.
    assert service._reconnect_reason(t0 + 30) is None
    assert service._reconnect_reason(t0 + 121) is not None

    # Past the exit threshold, capture restarts instead of starving forever.
    service._exit_if_ingest_degraded(t0 + 800)
    assert exits == []
    service._exit_if_ingest_degraded(t0 + 901)
    assert exits == [1]


def test_recovery_clears_the_guard(monkeypatch):
    module = _load(monkeypatch, RTSP_MIN_INPUT_FPS="8")
    service = module.RTSPToRedis()

    service._track_input_fps(0.2, 1_000.0)
    assert service.low_input_fps_since == 1_000.0

    service._track_input_fps(14.9, 1_030.0)
    assert service.low_input_fps_since is None
    assert service._reconnect_reason(5_000.0) is None


def test_guard_does_not_restart_capture_while_redis_is_down(monkeypatch):
    """A Redis outage must not bounce the RTSP container; restarting would not help."""
    module = _load(monkeypatch, RTSP_MIN_INPUT_FPS="8", RTSP_LOW_FPS_EXIT_SEC="900")
    service = module.RTSPToRedis()
    service.update_status_metric("redis_connected", False)
    exits = _silence_exit(monkeypatch, module)

    service._track_input_fps(0.1, 1_000.0)
    service._exit_if_ingest_degraded(9_999.0)

    assert exits == []


def test_readiness_reports_starved_ingest(monkeypatch):
    module = _load(monkeypatch, RTSP_MIN_INPUT_FPS="8")
    service = module.RTSPToRedis()
    service.redis_client = type("_FakeRedis", (), {"ping": lambda self: True})()
    monkeypatch.setattr(service, "get_uptime_seconds", lambda: 10_000.0)

    service._track_input_fps(0.2, 1_000.0)
    report = service.get_readiness_report()

    assert report["input_fps_ok"] is False
    assert report["ready"] is False
    assert report["input_fps_min"] == 8


def test_disabled_guard_leaves_ingest_alone(monkeypatch):
    module = _load(monkeypatch, RTSP_MIN_INPUT_FPS="0", RTSP_LOW_FPS_RECONNECT_SEC="0")
    service = module.RTSPToRedis()
    exits = _silence_exit(monkeypatch, module)

    service._track_input_fps(0.0, 1_000.0)

    assert service.low_input_fps_since is None
    assert service._reconnect_reason(99_999.0) is None
    service._exit_if_ingest_degraded(99_999.0)
    assert exits == []
