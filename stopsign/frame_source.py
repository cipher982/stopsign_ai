"""Frame source abstraction for video analyzer.

Provides a common interface for reading frames from different sources:
- RedisFrameSource: Reads from Redis queue (legacy pipeline)
- RTSPFrameSource: Reads directly from RTSP stream via MediaMTX
"""

import json
import logging
import os
import time
from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import redis

logger = logging.getLogger(__name__)

# SSFM header format constants (from rtsp_to_redis)
RAW_HEADER_MAGIC = b"SSFM"
RAW_HEADER_MIN_LEN = 9  # magic(4) + version(1) + meta_len(4)


class FrameSource(ABC):
    """Abstract base class for frame sources."""

    @abstractmethod
    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get the next frame with timestamp.

        Returns:
            Tuple of (frame, capture_timestamp) or None if no frame available.
            The capture_timestamp should be Unix epoch seconds.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the frame source is connected and ready."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

    def get_queue_depth(self) -> int:
        """Get the current queue depth (for sources that have a queue).

        Returns 0 for sources without a queue (e.g., RTSP).
        """
        return 0


class RedisFrameSource(FrameSource):
    """Reads frames from Redis queue (legacy pipeline via rtsp_to_redis)."""

    def __init__(self, redis_client: redis.Redis, frame_key: str, catchup_sec: float = 0):
        self.redis_client = redis_client
        self.frame_key = frame_key
        self.catchup_sec = catchup_sec
        self._connected = False

    def _parse_raw_frame(self, data: bytes) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Parse packed RAW frame with SSFM metadata header.

        Returns (frame, capture_ts). If header missing/invalid, returns (None, None).
        """
        capture_ts: Optional[float] = None
        jpeg_bytes = data

        try:
            if len(data) >= RAW_HEADER_MIN_LEN and data[0:4] == RAW_HEADER_MAGIC:
                _version = data[4]
                meta_len = int.from_bytes(data[5:9], "big")
                meta_start = 9
                meta_end = meta_start + meta_len

                if 0 <= meta_len <= len(data) - meta_start:
                    meta_bytes = data[meta_start:meta_end]
                    meta = json.loads(meta_bytes.decode("utf-8"))
                    if isinstance(meta, dict) and "ts" in meta:
                        capture_ts = float(meta.get("ts"))
                        jpeg_bytes = data[meta_end:]
                    else:
                        return None, None
                else:
                    return None, None
            else:
                return None, None
        except Exception:
            return None, None

        nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame, capture_ts

    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get frame from Redis queue with SSFM metadata."""
        try:
            # BRPOP with LPUSH forms a FIFO queue (oldest first)
            frame_data = self.redis_client.brpop([self.frame_key], timeout=1)
            if frame_data:
                self._connected = True
                _, data = frame_data
                frame, capture_ts = self._parse_raw_frame(data)

                if frame is None or capture_ts is None:
                    logger.error("Discarding frame without valid capture timestamp metadata")
                    return None

                # Log high lag for monitoring
                if self.catchup_sec > 0:
                    lag = time.time() - capture_ts
                    if lag > self.catchup_sec:
                        logger.info(
                            "High pipeline lag: %.2fs (threshold %.2fs)",
                            lag,
                            self.catchup_sec,
                        )

                return frame, capture_ts
            return None
        except Exception as e:
            logger.error(f"Error retrieving frame from Redis: {e}")
            self._connected = False
            return None

    def is_connected(self) -> bool:
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False

    def get_queue_depth(self) -> int:
        """Get current Redis queue depth."""
        try:
            return self.redis_client.llen(self.frame_key)
        except Exception:
            return 0

    def close(self) -> None:
        """Redis client is shared, don't close it here."""
        pass


class RTSPFrameSource(FrameSource):
    """Reads frames directly from RTSP stream (e.g., MediaMTX).

    This bypasses the Redis frame queue for lower latency.
    Timestamps are generated at capture time since RTSP may not
    provide reliable timestamps.
    """

    # Max consecutive read failures before forcing reconnect
    MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        rtsp_url: str,
        target_fps: float = 15.0,
        reconnect_delay: float = 3.0,
        read_timeout_sec: float = 5.0,
    ):
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay
        self.read_timeout_sec = read_timeout_sec
        self.frame_interval = 1.0 / target_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0.0
        self._connected = False
        self._consecutive_failures = 0
        self._connect()

    def _build_rtsp_url_with_options(self) -> str:
        """Build RTSP URL with FFmpeg options for TCP transport and timeouts.

        OpenCV's VideoCapture passes options via URL query params to FFmpeg.
        """
        import urllib.parse

        # Parse existing URL
        parsed = urllib.parse.urlparse(self.rtsp_url)

        # FFmpeg options for RTSP:
        # - rtsp_transport=tcp: Use TCP instead of UDP for reliability
        # - stimeout: Socket timeout in microseconds (prevents infinite hangs)
        # - buffer_size: Reduce buffer for lower latency
        timeout_us = int(self.read_timeout_sec * 1_000_000)
        ffmpeg_opts = {
            "rtsp_transport": "tcp",
            "stimeout": str(timeout_us),
            "buffer_size": "524288",  # 512KB buffer
        }

        # Merge with existing query params (if any)
        existing_params = urllib.parse.parse_qs(parsed.query)
        for k, v in ffmpeg_opts.items():
            if k not in existing_params:
                existing_params[k] = [v]

        # Rebuild URL
        new_query = urllib.parse.urlencode(existing_params, doseq=True)
        new_parsed = parsed._replace(query=new_query)
        return urllib.parse.urlunparse(new_parsed)

    def _connect(self) -> bool:
        """Connect to RTSP stream."""
        if self._cap is not None:
            self._cap.release()

        logger.info(f"Connecting to RTSP stream: {self._mask_url(self.rtsp_url)}")

        # Build URL with FFmpeg options for TCP transport and timeouts
        rtsp_url_with_opts = self._build_rtsp_url_with_options()

        # Set environment variable for FFmpeg RTSP transport (belt and suspenders)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        self._cap = cv2.VideoCapture(rtsp_url_with_opts, cv2.CAP_FFMPEG)

        if self._cap is not None and self._cap.isOpened():
            self._connected = True
            self._consecutive_failures = 0
            # Get stream properties
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"RTSP connected: {width}x{height} @ {fps:.1f} FPS")
            return True
        else:
            self._connected = False
            logger.error("Failed to connect to RTSP stream")
            return False

    def _mask_url(self, url: str) -> str:
        """Mask credentials in URL for logging."""
        import re

        return re.sub(r"://[^:]+:[^@]+@", "://***:***@", url)

    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Read frame from RTSP stream."""
        if self._cap is None or not self._cap.isOpened():
            if not self._reconnect():
                return None

        # Rate limiting: sleep if we're ahead of target FPS
        # This prevents draining buffered frames too fast (which distorts timestamps)
        now = time.time()
        elapsed = now - self._last_frame_time
        if elapsed < self.frame_interval and self._last_frame_time > 0:
            sleep_time = self.frame_interval - elapsed
            time.sleep(sleep_time)

        try:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._connected = True
                self._consecutive_failures = 0
                self._last_frame_time = time.time()
                # Use current time as capture timestamp
                # (RTSP timestamps via OpenCV are unreliable)
                capture_ts = self._last_frame_time
                return frame, capture_ts
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    logger.warning(f"RTSP read failed {self._consecutive_failures} times, forcing reconnect")
                    self._connected = False
                    self._cap.release()
                    self._reconnect()
                else:
                    logger.debug(f"RTSP read failed ({self._consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES})")
                return None
        except Exception as e:
            logger.error(f"Error reading RTSP frame: {e}")
            self._connected = False
            self._consecutive_failures += 1
            return None

    def _reconnect(self) -> bool:
        """Attempt to reconnect to RTSP stream."""
        logger.info(f"Attempting RTSP reconnect in {self.reconnect_delay}s...")
        time.sleep(self.reconnect_delay)
        return self._connect()

    def is_connected(self) -> bool:
        return self._connected and self._cap is not None and self._cap.isOpened()

    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False


def create_frame_source(
    source_type: str,
    redis_client: Optional[redis.Redis] = None,
    frame_key: str = "raw_frames",
    rtsp_url: Optional[str] = None,
    catchup_sec: float = 0,
    target_fps: float = 15.0,
) -> FrameSource:
    """Factory function to create the appropriate frame source.

    Args:
        source_type: "redis" or "rtsp"
        redis_client: Redis client (required for redis source)
        frame_key: Redis key for frames (for redis source)
        rtsp_url: RTSP URL (required for rtsp source)
        catchup_sec: Lag threshold for logging (redis source)
        target_fps: Target frame rate (rtsp source)

    Returns:
        Configured FrameSource instance
    """
    if source_type == "redis":
        if redis_client is None:
            raise ValueError("redis_client required for redis frame source")
        return RedisFrameSource(redis_client, frame_key, catchup_sec)

    elif source_type == "rtsp":
        if rtsp_url is None:
            raise ValueError("rtsp_url required for rtsp frame source")
        return RTSPFrameSource(rtsp_url, target_fps)

    else:
        raise ValueError(f"Unknown frame source type: {source_type}")
