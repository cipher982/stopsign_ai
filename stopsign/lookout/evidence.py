"""Evidence capture: sampled ring, decisive frames, timestamped filmstrip.

Review rules kept here: evidence must not depend on clip generation; the ring is
bounded and sampled rather than continuous; the decisive images are persisted
before anything is enriched; and retention is enforced in code, not by hope.
Evidence of a public street is a liability, so the default window is short.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from stopsign.lookout.types import Box


@dataclass(slots=True)
class RingFrame:
    ts: float
    path: str


class RingBuffer:
    """A bounded, sampled sequence of small frames for one watch.

    Sampling instead of recording keeps memory and disk predictable, and gives
    the filmstrip its time axis: each stored frame is stamped with its capture
    time, so the evidence shows *when*, not just *what*.
    """

    def __init__(self, directory: str, *, max_frames: int, interval_sec: float, width: int, quality: int) -> None:
        self.directory = directory
        self.max_frames = max(2, max_frames)
        self.interval_sec = max(0.1, interval_sec)
        self.width = width
        self.quality = quality
        self._frames: list[RingFrame] = []
        self._next_capture_ts = 0.0
        self._counter = 0
        os.makedirs(self.directory, exist_ok=True)

    def maybe_add(self, ts: float, frame: np.ndarray) -> None:
        if ts < self._next_capture_ts:
            return
        self._next_capture_ts = ts + self.interval_sec
        small = downscale(frame, self.width)
        if small is None:
            return
        self._counter += 1
        path = os.path.join(self.directory, f"{self._counter:04d}.jpg")
        if not cv2.imwrite(path, small, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]):
            return
        self._frames.append(RingFrame(ts=ts, path=path))
        while len(self._frames) > self.max_frames:
            stale = self._frames.pop(0)
            _unlink(stale.path)

    def frames(self) -> list[RingFrame]:
        return list(self._frames)

    def clear(self) -> None:
        for item in self._frames:
            _unlink(item.path)
        self._frames.clear()


class EvidenceWriter:
    """Writes and expires per-event evidence directories."""

    def __init__(self, root: str, *, max_events: int, max_age_days: float, quality: int = 82) -> None:
        self.root = root
        self.max_events = max(1, max_events)
        self.max_age_sec = max(0.0, max_age_days) * 86400.0
        self.quality = quality
        os.makedirs(self.root, exist_ok=True)

    def write(
        self,
        event_id: str,
        *,
        frame: Optional[np.ndarray],
        box: Box,
        ring: list[RingFrame],
        capture_ts: float,
        wall_ts: float,
        summary: dict[str, object],
    ) -> Optional[str]:
        """Persist the decisive frame, the watched crop, and a filmstrip.

        Frames are written before anything is enriched or delivered: an alert
        whose evidence failed to save is worse than no alert.
        """
        target_dir = os.path.join(self.root, event_id)
        try:
            os.makedirs(target_dir, exist_ok=True)
            if frame is not None and frame.size:
                annotated = frame.copy()
                _draw_marker(annotated, box)
                cv2.imwrite(
                    os.path.join(target_dir, "full.jpg"), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                )
                crop = _crop(frame, box)
                if crop is not None and crop.size:
                    cv2.imwrite(
                        os.path.join(target_dir, "target.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                    )
            self._write_filmstrip(target_dir, ring, box)
            _write_summary(target_dir, {**summary, "capture_ts": capture_ts, "wall_ts": wall_ts, "box": list(box)})
            return target_dir
        except (OSError, cv2.error):
            return None

    def _write_filmstrip(self, target_dir: str, ring: list[RingFrame], box: Box) -> Optional[str]:
        tiles: list[np.ndarray] = []
        for item in ring:
            image = cv2.imread(item.path)
            if image is None or image.size == 0:
                continue
            source_width = max(1, image.shape[1])
            tile = downscale(image, 320)
            if tile is None:
                continue
            scale = tile.shape[1] / float(source_width)
            _draw_marker(tile, tuple(value * scale for value in box))  # type: ignore[arg-type]
            _stamp(tile, item.ts)
            tiles.append(tile)
        if not tiles:
            return None
        height = max(tile.shape[0] for tile in tiles)
        width = max(tile.shape[1] for tile in tiles)
        padded: list[np.ndarray] = []
        for tile in tiles:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[: tile.shape[0], : tile.shape[1]] = tile
            padded.append(canvas)
        strip = np.hstack(padded)
        if strip.size == 0:
            return None
        cv2.imwrite(os.path.join(target_dir, "strip.jpg"), strip, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        return os.path.join(target_dir, "strip.jpg")

    def sweep(self, *, now: Optional[float] = None) -> int:
        """Delete event directories beyond the retention budget. Returns removals."""
        current = time.time() if now is None else now
        try:
            names = [name for name in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, name))]
        except OSError:
            return 0
        stamped: list[tuple[float, str]] = []
        for name in names:
            path = os.path.join(self.root, name)
            try:
                stamped.append((os.path.getmtime(path), path))
            except OSError:
                continue
        stamped.sort(reverse=True)
        removed = 0
        for index, (mtime, path) in enumerate(stamped):
            too_many = index >= self.max_events
            too_old = self.max_age_sec > 0 and (current - mtime) > self.max_age_sec
            if not (too_many or too_old):
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        return removed


def downscale(frame: np.ndarray, width: int) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None
    height, current_width = frame.shape[:2]
    if current_width <= 0 or height <= 0:
        return None
    if current_width <= width:
        return frame.copy()
    scale = width / float(current_width)
    return cv2.resize(frame, (width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def _crop(frame: np.ndarray, box: Box) -> Optional[np.ndarray]:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(round(box[0]))))
    y1 = max(0, min(height - 1, int(round(box[1]))))
    x2 = max(x1 + 1, min(width, int(round(box[2]))))
    y2 = max(y1 + 1, min(height, int(round(box[3]))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2]


def _draw_marker(frame: np.ndarray, box: Box) -> None:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(round(box[0]))))
    y1 = max(0, min(height - 1, int(round(box[1]))))
    x2 = max(0, min(width - 1, int(round(box[2]))))
    y2 = max(0, min(height - 1, int(round(box[3]))))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 214, 255), 2)


def _stamp(image: np.ndarray, ts: float) -> None:
    label = time.strftime("%H:%M:%S", time.localtime(ts))
    cv2.rectangle(image, (0, 0), (96, 20), (0, 0, 0), -1)
    cv2.putText(image, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def _write_summary(target_dir: str, summary: dict[str, object]) -> None:
    path = os.path.join(target_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
