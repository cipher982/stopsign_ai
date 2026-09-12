"""Ingest-rate guard.

The freeze detector answers "is the picture moving?". A camera whose WiFi link is
dropping frames answers that question "yes" while delivering almost nothing, and
on 2026-09-11 that mode starved the whole chain for 90 minutes with every health
signal green. These tests pin the guard that covers it.
"""

import importlib
import threading
import time

import numpy as np
import pytest


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


def _live_redis():
    return type("_FakeRedis", (), {"ping": lambda self: True})()


def test_live_but_starved_camera_escalates_while_no_freeze_is_detected(monkeypatch):
    module = _load(monkeypatch, RTSP_MIN_INPUT_FPS="8", RTSP_LOW_FPS_RECONNECT_SEC="120")
    service = module.RTSPToRedis()
    service.redis_client = _live_redis()
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


def test_readiness_gates_on_the_input_rate(monkeypatch):
    module = _load(monkeypatch, RTSP_MIN_INPUT_FPS="8")
    service = module.RTSPToRedis()
    service.redis_client = _live_redis()
    monkeypatch.setattr(service, "get_uptime_seconds", lambda: 10_000.0)

    service._track_input_fps(0.2, 1_000.0)

    # The probe reports the starvation, and does not treat it as healthy.
    assert service.get_readiness_report()["input_fps_ok"] is False


def test_disabled_guard_leaves_ingest_alone(monkeypatch):
    module = _load(monkeypatch, RTSP_MIN_INPUT_FPS="0", RTSP_LOW_FPS_RECONNECT_SEC="0")
    service = module.RTSPToRedis()
    exits = _silence_exit(monkeypatch, module)

    service._track_input_fps(0.0, 1_000.0)

    assert service.low_input_fps_since is None
    assert service._reconnect_reason(99_999.0) is None
    service._exit_if_ingest_degraded(99_999.0)
    assert exits == []


class _ExitRequested(BaseException):
    """Stands in for os._exit tearing the process down mid-loop."""


class _NullSpan:
    def set_attribute(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _NullTracer:
    def start_as_current_span(self, name):
        return _NullSpan()


class _FakePipeline:
    def __init__(self, client):
        self._client = client

    def lpush(self, key, value):
        self._client.pushed.append(value)
        return self

    def ltrim(self, key, *args):
        return self

    def llen(self, key):
        return len(self._client.pushed)

    def execute(self):
        return (1, True, len(self._client.pushed))


class _FakeRedis:
    def __init__(self):
        self.pushed = []

    def ping(self):
        return True

    def pipeline(self):
        return _FakePipeline(self)


class _StarvedCapture:
    """A camera that answers every read, with a live frame, but only rarely."""

    def __init__(self, delay):
        self.delay = delay

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        time.sleep(self.delay)
        return True, np.zeros((32, 32, 3), dtype=np.uint8)

    def release(self):
        pass


def test_guard_does_not_restart_capture_when_redis_is_actually_down(monkeypatch):
    """A stale healthy flag must not be enough to restart capture."""
    module = _load(monkeypatch, RTSP_MIN_INPUT_FPS="6", RTSP_LOW_FPS_EXIT_SEC="900")
    service = module.RTSPToRedis()
    # The cached flag says healthy (set before the outage); the ping is the truth.
    service.update_status_metric("redis_connected", True)

    class _DeadRedis:
        def ping(self):
            raise module.redis.exceptions.ConnectionError("redis is down")

    service.redis_client = _DeadRedis()
    exits = _silence_exit(monkeypatch, module)
    service._track_input_fps(0.1, 1_000.0)

    service._exit_if_ingest_degraded(9_999.0)
    assert exits == []

    # Control: the same state with a reachable Redis does restart capture, so the
    # assertion above cannot pass for want of a firing guard.
    service.redis_client = _live_redis()
    service._exit_if_ingest_degraded(9_999.0)
    assert exits == [1]


def test_a_recovered_stream_is_not_restarted_at_the_deadline(monkeypatch):
    """Rate is recomputed from arrival times, so recovery wins before the guard reads it."""
    module = _load(
        monkeypatch,
        RTSP_MIN_INPUT_FPS="6",
        RTSP_RATE_WINDOW_SEC="10",
        RTSP_LOW_FPS_EXIT_SEC="900",
    )
    exits = _silence_exit(monkeypatch, module)

    # Control: the same starvation with no recovery does reach the exit.
    starved = module.RTSPToRedis()
    starved.redis_client = _live_redis()
    starved._record_arrival(1_000.0)
    starved._exit_if_ingest_degraded(2_000.0)
    assert exits == [1]
    exits.clear()

    # Recovered: a full window of healthy arrivals lands before the guard is asked.
    recovered = module.RTSPToRedis()
    recovered.redis_client = _live_redis()
    recovered._record_arrival(1_000.0)
    for i in range(150):
        recovered._record_arrival(2_000.0 + i / 15.0)

    recovered._exit_if_ingest_degraded(2_010.0)
    assert exits == []


def test_zero_frame_stream_reinitialises_capture_instead_of_starving_forever(monkeypatch):
    """A camera that yields no frames never reaches the rate guard, and does not need to.

    read() fails, the loop re-opens capture, and it keeps re-opening rather than
    exiting the process: re-opening the session is the same remediation the exit
    exists to trigger, minus the restart.
    """
    module = _load(monkeypatch, RTSP_MIN_INPUT_FPS="6", RTSP_LOW_FPS_EXIT_SEC="1")
    exits = _silence_exit(monkeypatch, module)
    opened = []

    class _DeadCapture:
        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    def _capture(*args, **kwargs):
        opened.append(True)
        return _DeadCapture()

    monkeypatch.setattr(module.redis, "from_url", lambda *a, **k: _FakeRedis())
    monkeypatch.setattr(module.cv2, "VideoCapture", _capture)

    service = module.RTSPToRedis()
    service.set_telemetry(None, _NullTracer())

    thread = threading.Thread(target=service.run, daemon=True)
    thread.start()
    time.sleep(2.0)
    service.should_stop.set()
    thread.join(timeout=5)

    assert len(opened) >= 2
    assert exits == []


def test_capture_loop_restarts_capture_when_starvation_persists(monkeypatch):
    """End to end through run(): read -> rate -> guard -> restart request.

    Drives the production call site rather than the private helpers, so it fails if
    the guard stops being wired into the loop or is scored against stale evidence.
    """
    module = _load(
        monkeypatch,
        RTSP_MIN_INPUT_FPS="6",
        RTSP_RATE_WINDOW_SEC="1",
        RTSP_LOW_FPS_RECONNECT_SEC="999",  # isolate the exit stage
        RTSP_LOW_FPS_EXIT_SEC="2",
        RTSP_FREEZE_DETECT_SEC="0",  # freeze detection off: only the rate guard can act
    )
    exits = []

    def _exit(code):
        exits.append(code)
        raise _ExitRequested

    monkeypatch.setattr(module.os, "_exit", _exit)
    monkeypatch.setattr(module.redis, "from_url", lambda *a, **k: _FakeRedis())
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda *a, **k: _StarvedCapture(0.6))

    service = module.RTSPToRedis()
    service.set_telemetry(None, _NullTracer())

    try:
        with pytest.raises(_ExitRequested):
            service.run()
    finally:
        service.should_stop.set()

    assert exits == [1]
