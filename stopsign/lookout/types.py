"""Shared types for the Lookout watch evaluator.

Lookout watches a region of a live camera: you arm a watch from an exact frame,
pick what you want to be told, and the evaluator reports state changes with the
evidence that decided them.

Three vocabulary rules, all from review, kept deliberately:

* Coordinates are in *processed* frame space — the space the player displays and
  the operator draws in. Nothing assumes a resolution.
* "Unknown" is not "clear". A watch that cannot see its own subject reports
  ``UNKNOWN`` or ``UNAVAILABLE``; it never reports the nominal condition because
  evidence went missing.
* Durations are wall-clock seconds with an explicit evidence mask, never frame
  counts. The analyzer can drop frames, run YOLO off-rate, and discard backlog
  during catch-up, so "N consecutive frames" means nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Optional
from typing import Tuple

Box = Tuple[float, float, float, float]
Point = Tuple[float, float]


class WatchKind(str, Enum):
    """What the box selects: a movable subject, or an area."""

    OBJECT = "object"
    ZONE = "zone"


class Condition(str, Enum):
    """The state change the operator asked to be told about."""

    GONE = "gone"
    BACK = "back"
    OCCUPIED = "occupied"
    CLEARED = "cleared"

    @property
    def kind(self) -> WatchKind:
        return WatchKind.ZONE if self in _ZONE_CONDITIONS else WatchKind.OBJECT

    @property
    def question(self) -> str:
        return _CONDITION_QUESTIONS[self]

    @property
    def selectable(self) -> bool:
        """Whether the operator can arm a watch for this condition.

        ``BACK`` is reported when it happens but is not an arming choice: it is
        the inverse transition of ``GONE`` and would only double the tuning
        surface.
        """
        return self is not Condition.BACK


_ZONE_CONDITIONS = frozenset({Condition.OCCUPIED, Condition.CLEARED})

_CONDITION_QUESTIONS = {
    Condition.GONE: "Tell me when it is gone",
    Condition.BACK: "Tell me when it is back",
    Condition.OCCUPIED: "Tell me when this area is occupied",
    Condition.CLEARED: "Tell me when this area is clear",
}

# Human-facing wording. Deliberately free of identity claims: a re-appearance is
# "visible again", never "the same object came back".
CONDITION_WORDING = {
    Condition.GONE: "gone",
    Condition.BACK: "visible again",
    Condition.OCCUPIED: "occupied",
    Condition.CLEARED: "clear",
}


class WatchState(str, Enum):
    """Honest live state of a watch.

    ``OCCUPIED`` is the nominal state for an object watch (the subject is
    there); ``OBSERVING`` is the nominal state for a zone watch (the area is
    clear). Both mean "we can see, and the reading is fresh".
    """

    PENDING = "pending"
    OBSERVING = "observing"
    OCCUPIED = "occupied"
    GONE = "gone"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"


# States in which the evaluator is willing to change its mind about the subject.
DECIDABLE_STATES = frozenset({WatchState.OBSERVING, WatchState.OCCUPIED, WatchState.GONE})


@dataclass(slots=True)
class Watch:
    """Operator-defined observation job, armed from an exact frame."""

    id: str
    kind: WatchKind
    label: str
    box: Box
    condition: Condition
    created_at: float
    created_by: str = "web"
    # Processed-space size of the frame this watch was armed from, so the box can
    # be rescaled if the pipeline's crop/scale changes underneath it.
    reference_size: Tuple[int, int] = (0, 0)
    reference_wall_ts: float = 0.0
    reference_capture_ts: float = 0.0
    # Expected subject, taken from the detection under the box at arm time
    # ("car", "person", ...). Empty means "any object".
    subject: str = ""
    # Arming appearance: the frame the operator drew on, relative to the storage
    # root. The box is cropped out of it when the watch is built, so the
    # reference is the picture they actually saw.
    reference_image: str = ""
    # Whether the subject was there when the watch was armed. Decides which
    # reference describes the armed state.
    armed_occupied: bool = True
    options: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    revision: int = 0

    @property
    def size(self) -> Tuple[float, float]:
        return (max(1.0, self.box[2] - self.box[0]), max(1.0, self.box[3] - self.box[1]))

    @property
    def center(self) -> Point:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def area(self) -> float:
        width, height = self.size
        return width * height

    def scaled_box(self, frame_size: Tuple[int, int]) -> Box:
        """Rescale the stored box to a frame of a different resolution."""
        ref_w, ref_h = self.reference_size
        frame_w, frame_h = frame_size
        if ref_w <= 0 or ref_h <= 0 or (ref_w == frame_w and ref_h == frame_h):
            return self.box
        sx = frame_w / float(ref_w)
        sy = frame_h / float(ref_h)
        return (self.box[0] * sx, self.box[1] * sy, self.box[2] * sx, self.box[3] * sy)


@dataclass(slots=True)
class Observation:
    """What one frame actually shows for one watch.

    ``fresh`` is false when the detection stage did not run for this frame. A
    non-fresh observation carries no new negative evidence — it only ages the
    last real reading. That is what stops a YOLO-skipped frame from being
    counted as "the area is empty".
    """

    ts: float
    fresh: bool
    age_sec: float
    present_prob: float
    decisive: bool
    similarity_armed: float
    similarity_opposite: float
    subject_cover: float
    subject_conf: float
    subject_label: Optional[str]
    occluder_cover: float
    occluder_label: Optional[str]
    frame_mean: float
    frame_contrast: float
    # Whether a detection pass actually ran for this frame. False means the
    # detection channel has nothing to say, not that nothing was detected.
    detection_channel: bool
    reason: str


@dataclass(slots=True)
class Decision:
    """Typed answer to "does the evidence still show the subject?"."""

    present_prob: float
    evidence_ok: bool
    source: str
    provisional: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LookoutEvent:
    """A state change worth telling someone about."""

    id: str
    watch_id: str
    label: str
    condition: Condition
    kind: WatchKind
    subject: str
    from_state: str
    to_state: str
    capture_ts: float
    wall_ts: float
    present_prob: float
    evidence_ok: bool
    evidence_fraction: float
    source: str
    alerting: bool
    detail: dict[str, Any] = field(default_factory=dict)
    evidence_dir: Optional[str] = None

    @property
    def wording(self) -> str:
        """What the operator is told. Never stronger than what was measured.

        "Clear" is a claim about the region being empty. Appearance alone can
        only support "no longer looks like what you armed on" — something else
        may be sitting there. With a detection channel available the stronger
        claim is made only when nothing is detected in the region.
        """
        if self.condition is Condition.CLEARED and not self.detail.get("occupancy_confirmed", False):
            return "no longer the same as when armed"
        return CONDITION_WORDING[self.condition]

    def to_row(self) -> dict[str, Any]:
        return {
            "event_id": self.id,
            "watch_id": self.watch_id,
            "label": self.label,
            "condition": self.condition.value,
            "kind": self.kind.value,
            "subject": self.subject,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "capture_ts": self.capture_ts,
            "wall_ts": self.wall_ts,
            "present_prob": self.present_prob,
            "evidence_ok": self.evidence_ok,
            "evidence_fraction": self.evidence_fraction,
            "source": self.source,
            "alerting": self.alerting,
            "detail": self.detail,
            "evidence_dir": self.evidence_dir,
        }


@dataclass(slots=True)
class WatchStatus:
    """Serialisable live status published for the web UI."""

    watch_id: str
    label: str
    kind: str
    condition: str
    subject: str
    state: str
    since_ts: float
    updated_ts: float
    present_prob: float
    evidence_fraction: float
    decisive: bool
    reason: str
    box: Box
    last_frame_ts: float
    armed_wall_ts: float
    alerting_events: int
    history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "watch_id": self.watch_id,
            "label": self.label,
            "kind": self.kind,
            "condition": self.condition,
            "subject": self.subject,
            "state": self.state,
            "since_ts": self.since_ts,
            "updated_ts": self.updated_ts,
            "present_prob": round(self.present_prob, 4),
            "evidence_fraction": round(self.evidence_fraction, 4),
            "decisive": self.decisive,
            "reason": self.reason,
            "box": [round(value, 2) for value in self.box],
            "last_frame_ts": self.last_frame_ts,
            "armed_wall_ts": self.armed_wall_ts,
            "alerting_events": self.alerting_events,
            "history": [round(value, 3) for value in self.history[-60:]],
        }


def area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0
