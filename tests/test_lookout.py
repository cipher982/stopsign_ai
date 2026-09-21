"""Lookout behaviour: what a watch reports, and what it refuses to report.

These are the properties that make the feature usable rather than merely
plausible, so they are asserted directly:

* a state change needs *seconds* of evidence, not frames of it;
* missing evidence produces "cannot tell", never the nominal condition;
* an area is only called clear when emptiness was actually checked for objects;
* the frame path never evaluates anything (the worker does), and its queue is
  bounded when the worker falls behind.

Deterministic and offline: synthetic frames, no detector, no Redis, no network.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("cv2", reason="lookout perception requires the lookout extra")

import cv2  # noqa: E402

from stopsign.lookout.decider import RuleDecider  # noqa: E402
from stopsign.lookout.detections import Detection  # noqa: E402
from stopsign.lookout.evidence import RingBuffer  # noqa: E402
from stopsign.lookout.manager import LookoutManager  # noqa: E402
from stopsign.lookout.manager import arm_watch  # noqa: E402
from stopsign.lookout.options import LookoutOptions  # noqa: E402
from stopsign.lookout.references import ReferenceModel  # noqa: E402
from stopsign.lookout.references import patch_for  # noqa: E402
from stopsign.lookout.runtime import WatchRuntime  # noqa: E402
from stopsign.lookout.store import WatchStore  # noqa: E402
from stopsign.lookout.store import watch_from_dict  # noqa: E402
from stopsign.lookout.store import watch_to_dict  # noqa: E402
from stopsign.lookout.types import Condition  # noqa: E402
from stopsign.lookout.types import WatchState  # noqa: E402

FRAME_W, FRAME_H = 320, 240
BOX = (120.0, 90.0, 200.0, 150.0)  # the watched region, inside the frame
START = 1_000_000.0


def scene(occupied: bool, *, seed: int = 7) -> np.ndarray:
    """A static scene whose watched region is either occupied or empty.

    The background is fixed noise so "empty" is a stable, measurable appearance —
    exactly what a fixed camera sees — and the object is a high-contrast striped
    block, which is what makes the region distinguishable at all.
    """
    rng = np.random.default_rng(seed)
    frame = rng.integers(60, 100, size=(FRAME_H, FRAME_W, 3), dtype=np.uint8)
    if occupied:
        x1, y1, x2, y2 = (int(value) for value in BOX)
        block = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
        block[:, :] = (200, 60, 40)
        block[::4, :] = (240, 240, 240)
        frame[y1:y2, x1:x2] = block
    return frame


def options(**overrides) -> LookoutOptions:
    base = {
        "enabled": True,
        "tick_hz": 2.0,
        "gone_hold_sec": 4.0,
        "return_hold_sec": 2.0,
        "occupied_hold_sec": 6.0,
        "empty_hold_sec": 6.0,
        "arm_confirm_sec": 1.0,
        "min_evidence_fraction": 0.6,
        "alert_cooldown_sec": 60.0,
        "note_cooldown_sec": 30.0,
    }
    base.update(overrides)
    return LookoutOptions(**base)  # type: ignore[arg-type]


def build_runtime(tmp_path, condition: Condition, *, armed_occupied: bool, opts: LookoutOptions):
    watch = arm_watch(
        box=BOX,
        condition=condition,
        label="test region",
        reference_size=(FRAME_W, FRAME_H),
        armed_occupied=armed_occupied,
        now=START,
        watch_id="t1",
    )
    model = ReferenceModel.from_armed_patch(
        patch_for(scene(armed_occupied), BOX),
        armed_occupied=armed_occupied,
        drift=opts.reference_drift,
        decisive_similarity=opts.decisive_similarity,
        present_floor=opts.present_drift_floor,
        absent_ceiling=opts.absent_drift_ceiling,
        min_opposite_samples=opts.min_opposite_samples,
    )
    ring = RingBuffer(str(tmp_path / "ring"), max_frames=4, interval_sec=1.0, width=160, quality=70)
    runtime = WatchRuntime(watch, opts, model=model, ring=ring, now=START)
    return watch, runtime


def drive(runtime, decider, frames, *, start=START, step=0.5, detections=None, detection_channel=False, fresh=True):
    """Feed an alternating sequence and collect every event, in order."""
    events = []
    ts = start
    for frame in frames:
        events.extend(
            runtime.tick(
                frame,
                ts,
                list(detections or []),
                fresh=fresh,
                decider=decider,
                capture_ts=ts,
                detection_channel=detection_channel,
            )
        )
        ts += step
    return events, ts


def hold(frames, repeats):
    return [frame for frame in frames for _ in range(repeats)]


def transitions(events):
    return [(event.condition.value, event.from_state, event.to_state) for event in events]


# --------------------------------------------------------------------- metric
def test_similarity_separates_present_from_absent(tmp_path):
    """The measurement must actually separate the two states, with margin."""
    model = ReferenceModel.from_armed_patch(patch_for(scene(True), BOX), armed_occupied=True)
    present = model.score(patch_for(scene(True), BOX))
    absent = model.score(patch_for(scene(False), BOX))
    assert present.similarity_armed > 0.75
    assert absent.similarity_armed < present.similarity_armed - 0.25
    assert 0.0 <= absent.similarity_armed <= 1.0


def test_missing_measurement_is_not_treated_as_absence(tmp_path):
    """An unmeasurable region must return neither a probability nor a match."""
    model = ReferenceModel.from_armed_patch(patch_for(scene(True), BOX), armed_occupied=True)
    unobtainable = np.zeros((4, 4, 3), dtype=np.uint8)
    patch = patch_for(unobtainable, (0.0, 0.0, 1.0, 1.0))
    assert patch is None
    score = model.score(patch)
    assert score.decisive is False
    assert score.present_prob == pytest.approx(0.5)


# ---------------------------------------------------------------- state machine
def test_object_watch_reports_gone_only_after_confirmed_absence(tmp_path):
    opts = options()
    watch, runtime = build_runtime(tmp_path, Condition.GONE, armed_occupied=True, opts=opts)
    decider = RuleDecider(opts)

    events, ts = drive(runtime, decider, hold([scene(True)], 4))
    assert runtime.state is WatchState.OCCUPIED
    assert transitions(events) == []

    # 3s of absence is short of the 4s hold; nothing may fire yet.
    events, ts = drive(runtime, decider, hold([scene(False)], 6), start=ts)
    assert transitions(events) == []
    assert runtime.state is WatchState.OCCUPIED

    events, ts = drive(runtime, decider, hold([scene(False)], 4), start=ts)
    assert transitions(events) == [("gone", "occupied", "gone")]
    gone = events[0]
    assert gone.alerting is True  # it is the condition the operator asked about
    assert gone.evidence_ok is True
    assert gone.evidence_fraction >= opts.min_evidence_fraction
    assert gone.detail["held_sec"] >= opts.gone_hold_sec


def test_absence_does_not_accumulate_without_usable_evidence(tmp_path):
    """A quantity of unusable frames must never become a confirmed absence."""
    opts = options(gone_hold_sec=2.0)
    watch, runtime = build_runtime(tmp_path, Condition.GONE, armed_occupied=True, opts=opts)
    decider = RuleDecider(opts)

    _, ts = drive(runtime, decider, hold([scene(True)], 4))
    assert runtime.state is WatchState.OCCUPIED

    # The box is squeezed to an unmeasurable size: no reading is possible.
    runtime.watch.box = (10.0, 10.0, 12.0, 12.0)
    events, ts = drive(runtime, decider, hold([scene(False)], 20), start=ts)
    assert runtime.state is WatchState.UNKNOWN
    assert [event.condition.value for event in events] == ["gone"]  # the armed condition, as a note
    assert all(event.alerting is False for event in events)
    assert all(event.detail.get("note") is True for event in events)


def test_zone_clear_reports_whether_emptiness_was_checked(tmp_path):
    """Appearance alone may not claim an area is clear; detections may."""
    opts = options()
    decider = RuleDecider(opts)

    watch, runtime = build_runtime(tmp_path, Condition.CLEARED, armed_occupied=True, opts=opts)
    _, ts = drive(runtime, decider, hold([scene(True)], 4))
    assert runtime.state is WatchState.OCCUPIED
    appearance_only, _ = drive(runtime, decider, hold([scene(False)], 16), start=ts)
    assert transitions(appearance_only) == [("cleared", "occupied", "observing")]
    assert appearance_only[0].detail["occupancy_check"] == "appearance"
    assert appearance_only[0].detail["occupancy_confirmed"] is False
    assert appearance_only[0].wording == "no longer the same as when armed"

    # Same scene, but this time emptiness was checked for objects: the fuller
    # claim is allowed.
    watch, runtime = build_runtime(tmp_path, Condition.CLEARED, armed_occupied=True, opts=opts)
    _, ts = drive(runtime, decider, hold([scene(True)], 4), detection_channel=True)
    checked, _ = drive(runtime, decider, hold([scene(False)], 16), start=ts, detection_channel=True)
    assert transitions(checked) == [("cleared", "occupied", "observing")]
    assert checked[0].detail["occupancy_confirmed"] is True
    assert checked[0].wording == "clear"


def test_zone_keeps_occupied_when_another_object_takes_the_place(tmp_path):
    """The defect real footage exposed: the subject leaves, something else stays.

    A changed appearance is not an empty area. With a detection channel the
    region is still reported occupied, so no false "clear" is emitted.
    """
    opts = options()
    decider = RuleDecider(opts)
    watch, runtime = build_runtime(tmp_path, Condition.CLEARED, armed_occupied=True, opts=opts)
    _, ts = drive(runtime, decider, hold([scene(True)], 4), detection_channel=True)

    # The subject left, but a vehicle is now sitting in the region.
    parked_elsewhere = Detection(box=(130.0, 95.0, 195.0, 145.0), label="car", conf=0.72)
    events, ts = drive(
        runtime,
        decider,
        hold([scene(False)], 30),
        start=ts,
        detections=[parked_elsewhere],
        detection_channel=True,
    )
    assert runtime.state is WatchState.OCCUPIED
    assert transitions(events) == []


def test_watch_that_cannot_see_reports_blind_not_clear(tmp_path):
    opts = options()
    watch, runtime = build_runtime(tmp_path, Condition.CLEARED, armed_occupied=True, opts=opts)
    decider = RuleDecider(opts)
    _, ts = drive(runtime, decider, hold([scene(True)], 4))
    assert runtime.state is WatchState.OCCUPIED

    event = runtime.mark_unavailable(ts)
    assert event is not None
    assert event.to_state == WatchState.UNAVAILABLE.value
    assert event.alerting is False
    assert event.detail["note"] is True
    assert runtime.state is WatchState.UNAVAILABLE
    assert runtime.state is not WatchState.OBSERVING


def test_repeated_reports_respect_the_cooldown(tmp_path):
    opts = options(alert_cooldown_sec=60.0, empty_hold_sec=2.0, return_hold_sec=1.0, occupied_hold_sec=1.0)
    decider = RuleDecider(opts)
    watch, runtime = build_runtime(tmp_path, Condition.CLEARED, armed_occupied=True, opts=opts)
    _, ts = drive(runtime, decider, hold([scene(True)], 4))
    first, ts = drive(runtime, decider, hold([scene(False)], 8), start=ts)
    assert [event.alerting for event in first] == [True]
    _, ts = drive(runtime, decider, hold([scene(True)], 6), start=ts)
    second, ts = drive(runtime, decider, hold([scene(False)], 8), start=ts)
    # The transition still happens and is still recorded; only the *alert* is
    # suppressed while the cooldown is running.
    assert transitions(second), "the state change must still be reported in the feed"
    assert all(event.alerting is False for event in second if event.condition is Condition.CLEARED)


# ---------------------------------------------------------------- frame path
def test_frame_path_queues_without_evaluating_and_drops_oldest(tmp_path):
    """Bounded work off the hot path, verified rather than asserted in prose."""
    # 50 Hz with a 30ms gap: every submission clears the throttle, so the queue
    # is the only thing absorbing a stalled worker — which is what is under test.
    opts = options(tick_hz=50.0, storage_root=str(tmp_path))
    store = WatchStore(opts)
    reference = scene(True)
    directory = tmp_path / "references"
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "t1.jpg"), reference)
    watch = arm_watch(
        box=BOX,
        condition=Condition.GONE,
        label="queued",
        reference_size=(FRAME_W, FRAME_H),
        armed_occupied=True,
        now=START,
        watch_id="t1",
    )
    watch.reference_image = "references/t1.jpg"
    store.save([watch])
    manager = LookoutManager(opts, store=store, redis_client=None)
    manager.sync_now()
    assert manager.armed

    accepted = 0
    for index in range(8):
        if manager.submit_frame(reference, START + index, [], fresh=True):
            accepted += 1
        time.sleep(0.03)

    health = manager.health()
    assert accepted >= 4, "the throttle should accept roughly one frame per tick"
    assert health["queue_drops"] >= 1, "queue must shed load rather than grow"
    assert manager._queue.qsize() <= 2
    # Nothing was evaluated and no state advanced: evaluation is the worker's job.
    assert health["ticks"] == 0
    assert manager.snapshot()["watches"]["t1"]["state"] == WatchState.PENDING.value


def test_manager_publishes_the_clean_frame_for_arming(tmp_path):
    """Arming needs a frame before a watch exists, so publishing is unconditional."""
    opts = options(frame_publish_hz=100.0, storage_root=str(tmp_path))

    class FakeRedis:
        def __init__(self):
            self.values = {}

        def set(self, key, value, ex=None):
            self.values[key] = (value, ex)

        def get(self, key):
            return None

        def hset(self, *args, **kwargs):
            return 1

        def hget(self, *args, **kwargs):
            return None

        def hdel(self, *args, **kwargs):
            return 1

        def expire(self, *args, **kwargs):
            return True

        def pipeline(self):
            raise RuntimeError("not used in this test")

    client = FakeRedis()
    manager = LookoutManager(opts, store=WatchStore(opts), redis_client=client)
    manager.submit_frame(scene(True), START, [], fresh=True)

    from stopsign.frame_codec import unpack_frame

    assert "lookout.frame" in client.values
    payload, ttl = client.values["lookout.frame"]
    decoded = unpack_frame(payload)
    assert decoded is not None
    assert decoded.metadata["w"] == FRAME_W
    assert decoded.metadata["h"] == FRAME_H
    assert decoded.payload[:2] == b"\xff\xd8"  # a real JPEG, not a pickle of one
    assert ttl and ttl > 0


# ------------------------------------------------------------------ storage
def test_watch_file_accepts_the_shape_the_web_writer_produces(tmp_path):
    """The web container writes this file without importing the evaluator."""
    row = {
        "id": "abc123",
        "kind": "zone",
        "label": "stop buffer",
        "box": [854.0, 604.0, 1074.0, 702.0],
        "condition": "cleared",
        "created_at": 1789957284.6,
        "created_by": "web",
        "reference_size": [960, 540],
        "reference_wall_ts": 1789957284.6,
        "reference_capture_ts": 1789957284.1,
        "subject": "car",
        "reference_image": "references/abc123.jpg",
        "armed_occupied": True,
        "options": {"armed_choice": "auto"},
        "active": True,
        "revision": 1789957284001,
    }
    watch = watch_from_dict(row)
    assert watch is not None
    assert watch.condition is Condition.CLEARED
    assert watch.kind.value == "zone"
    assert watch.reference_size == (960, 540)
    assert watch.armed_occupied is True
    # A watch stored at a different resolution is rescaled to the frame it is
    # evaluated on, rather than silently watching the wrong region.
    assert watch.scaled_box((1920, 1080)) == (1708.0, 1208.0, 2148.0, 1404.0)


def test_watch_file_rejects_malformed_rows_without_raising():
    assert watch_from_dict({}) is None
    assert watch_from_dict({"id": "x", "box": [1, 2, 3]}) is None
    assert watch_from_dict({"id": "x", "box": [1, 2, 3, 4], "kind": "nonsense", "condition": "gone"}) is None


def test_watch_round_trip_preserves_the_reference(tmp_path):
    opts = options(storage_root=str(tmp_path))
    store = WatchStore(opts)
    watch = arm_watch(
        box=BOX,
        condition=Condition.OCCUPIED,
        label="driveway",
        reference_size=(960, 540),
        reference_image="references/x.jpg",
        subject="truck",
        now=START,
        watch_id="x",
    )
    store.save([watch])
    reloaded = store.load()
    assert len(reloaded) == 1
    assert reloaded[0].reference_image == "references/x.jpg"
    assert reloaded[0].subject == "truck"
    assert reloaded[0].box == pytest.approx(watch.box)
    assert watch_to_dict(reloaded[0])["condition"] == "occupied"
