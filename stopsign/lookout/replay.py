"""Offline replay: the same evaluator, driven over a recorded sequence.

Two jobs, one code path:

* **Demo** — watch a real minute of footage play out instead of waiting an hour
  for something to happen on the street.
* **Measurement** — score the state machine against known footage, and give the
  decision-model comparison a place to run that is not the alert path.

The timestamps here come from the video's own clock (frame index / fps), so hold
times are measured in *video* seconds. For a sequence recorded at real time — the
clean frames the analyzer publishes — that is the same thing; for an archived
clip with burned-in overlays it is not, and the report says so.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence

import cv2
import numpy as np

from stopsign.lookout.manager import FramePayload
from stopsign.lookout.manager import LookoutManager
from stopsign.lookout.options import LookoutOptions
from stopsign.lookout.store import WatchStore
from stopsign.lookout.types import LookoutEvent
from stopsign.lookout.types import Watch


@dataclass
class ReplayReport:
    source: str
    sampled_frames: int
    video_seconds: float
    fps: float
    frame_size: tuple[int, int]
    events: list[dict[str, Any]] = field(default_factory=list)
    final_states: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def alerting(self) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("alerting")]

    def transitions(self) -> list[str]:
        return [
            f"{event['capture_ts']:.1f}s {event['from_state']}->{event['to_state']} ({event['condition']})"
            for event in self.events
            if not event.get("detail", {}).get("note")
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sampled_frames": self.sampled_frames,
            "video_seconds": round(self.video_seconds, 2),
            "fps": self.fps,
            "frame_size": list(self.frame_size),
            "alerting": self.alerting,
            "events": self.events,
            "final_states": self.final_states,
            "health": self.health,
            "notes": self.notes,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


def write_reference_image(frame: np.ndarray, watch_id: str, storage_root: str) -> str:
    """Persist the exact arming frame so the watch references what the operator saw."""
    directory = os.path.join(storage_root, "references")
    os.makedirs(directory, exist_ok=True)
    relative = os.path.join("references", f"{watch_id}.jpg")
    path = os.path.join(storage_root, relative)
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return relative


def run_replay(
    source: str,
    watches: Sequence[Watch],
    *,
    options: LookoutOptions,
    sample_hz: float = 2.0,
    max_seconds: Optional[float] = None,
    detect: Optional[Callable[[np.ndarray, Sequence[int]], tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    storage_root: Optional[str] = None,
    start_sec: float = 0.0,
    time_base: float = 0.0,
    quiet: bool = False,
) -> ReplayReport:
    """Drive the live evaluator over a video source at a realistic tick rate."""
    root = storage_root or options.storage_root
    opts = dataclasses.replace(options, storage_root=root) if root != options.storage_root else options
    os.makedirs(root, exist_ok=True)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"cannot open video source: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0 or fps > 120:
        fps = 15.0
    step = max(1, int(round(fps / max(0.1, sample_hz))))
    if start_sec > 0:
        # The watch was armed from this instant; earlier frames are not what the
        # operator saw and must not be evaluated as if they were.
        capture.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    store = WatchStore(opts)
    store.save(watches)
    # Evidence and events from a replay are diagnostic, not history: clear them.
    for path in (opts.events_path, opts.arbitration_log):
        try:
            os.unlink(path)
        except OSError:
            pass
    manager = LookoutManager(opts, store=store, redis_client=None, detect=detect)
    manager.sync_now()
    if not manager.armed:
        raise ValueError("no armed watches: every watch was rejected as malformed")

    events: list[dict[str, Any]] = []
    notes: list[str] = []
    started = time.time()
    index = 0
    sampled = 0
    video_seconds = 0.0
    state: tuple[int, int] = (0, 0)
    for watch in watches:
        if watch.reference_image:
            notes.append(
                f"{watch.id}: armed from {watch.reference_image} at {watch.reference_size[0]}x{watch.reference_size[1]}"
            )
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        video_seconds = index / fps
        if max_seconds is not None and video_seconds > max_seconds:
            break
        if (index - 1) % step != 0:
            continue
        sampled += 1
        state = (frame.shape[1], frame.shape[0])
        # time_base anchors the video clock to real capture time, so evidence
        # stamps match the clock burned into the footage.
        payload = FramePayload(
            ts=time_base + video_seconds,
            frame=frame,
            detections=[],
            fresh=True,
            wall_ts=time_base + video_seconds,
            capture_ts=time_base + video_seconds,
            detection_channel=detect is not None,
        )
        for event in manager.process(payload):
            events.append(_event_row(event))
            if not quiet:
                reason = event.detail.get("reason", "")
                print(f"  [{video_seconds:6.1f}s] {event.to_state:10s} {event.condition.value:9s} {reason}")
    capture.release()

    snapshot = manager.snapshot()
    health = snapshot.get("health", {})
    if index and sampled:
        health["sampled_interval_sec"] = round(video_seconds / max(1, sampled), 3)
    notes.append("timestamps are video time (frame index / fps), not wall-clock capture time")
    return ReplayReport(
        source=source,
        sampled_frames=sampled,
        video_seconds=video_seconds,
        fps=fps,
        frame_size=state,
        events=events,
        final_states=snapshot.get("watches", {}),
        health=health,
        notes=notes,
        elapsed_sec=time.time() - started,
    )


def _event_row(event: LookoutEvent) -> dict[str, Any]:
    row = event.to_row()
    row["wording"] = event.wording
    return json.loads(json.dumps(row, default=str))
