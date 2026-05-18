import importlib


def test_rtsp_jpeg_quality_is_configurable(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_PORT", "8080")
    monkeypatch.setenv("RTSP_URL", "file:///tmp/example.mp4")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RAW_FRAME_KEY", "raw_frames")
    monkeypatch.setenv("FRAME_BUFFER_SIZE", "10")
    monkeypatch.setenv("RTSP_JPEG_QUALITY", "75")

    module = importlib.import_module("rtsp_to_redis.rtsp_to_redis")
    module = importlib.reload(module)
    service = module.RTSPToRedis()

    assert service.jpeg_quality == 75


def test_rtsp_jpeg_quality_is_clamped(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_PORT", "8080")
    monkeypatch.setenv("RTSP_URL", "file:///tmp/example.mp4")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RAW_FRAME_KEY", "raw_frames")
    monkeypatch.setenv("FRAME_BUFFER_SIZE", "10")
    monkeypatch.setenv("RTSP_JPEG_QUALITY", "150")

    module = importlib.import_module("rtsp_to_redis.rtsp_to_redis")
    module = importlib.reload(module)
    service = module.RTSPToRedis()

    assert service.jpeg_quality == 100
