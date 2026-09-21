"""Live evaluation: one bounded worker, live state, evidence, and delivery.

The analyzer's only contact with this module is :meth:`LookoutManager.submit_frame`,
called from the frame path. It does no evaluation, takes no locks the worker
holds, and blocks on nothing: at most one downscale per tick, then a
non-blocking put on a two-slot queue that drops its oldest entry under pressure.
Everything expensive — detection, reference scoring, evidence encoding, HTTP
delivery — happens on the worker thread or its side-effect pool.

That split is not decoration. The analyzer skips YOLO when a frame is more than
1.5 frame-budgets stale, so a few tens of milliseconds added to the frame path
does not slow the pipeline down gracefully; it starts switching inference off.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence

import cv2
import numpy as np

from stopsign.frame_codec import pack_frame
from stopsign.lookout import references
from stopsign.lookout.decider import AmbiguityLog
from stopsign.lookout.decider import RuleDecider
from stopsign.lookout.detections import Detection
from stopsign.lookout.detections import detections_from_arrays
from stopsign.lookout.detections import detections_from_boxes
from stopsign.lookout.evidence import EvidenceWriter
from stopsign.lookout.evidence import RingBuffer
from stopsign.lookout.evidence import downscale
from stopsign.lookout.notifier import WebhookNotifier
from stopsign.lookout.options import LookoutOptions
from stopsign.lookout.references import ReferenceModel
from stopsign.lookout.runtime import WatchRuntime
from stopsign.lookout.store import WatchStore
from stopsign.lookout.types import Box
from stopsign.lookout.types import Condition
from stopsign.lookout.types import LookoutEvent
from stopsign.lookout.types import Watch

logger = logging.getLogger(__name__)

STATE_KEY = "lookout.state"
HEALTH_KEY = "lookout.health"
REFERENCE_KEY = "lookout.references"
DELIVERY_KEY = "lookout.delivery"
# Clean frame published for the arming UI. Short TTL: a snapshot nobody renewed
# must expire rather than be served as if it were current.
FRAME_KEY = "lookout.frame"
FRAME_TTL_SEC = 20

DetectCallable = Callable[[np.ndarray, Sequence[int]], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(slots=True)
class FramePayload:
    ts: float
    frame: np.ndarray
    detections: list[Detection]
    fresh: bool
    wall_ts: float
    capture_ts: Optional[float] = None
    # False when no detector is wired up at all (offline replay): absence of
    # detections then carries no information.
    detection_channel: bool = True


ARMED_OCCUPIED_DEFAULT = {
    Condition.GONE: True,
    Condition.CLEARED: True,
    Condition.OCCUPIED: False,
    Condition.BACK: False,
}


def arm_watch(
    *,
    box: Box,
    condition: Condition,
    label: str,
    reference_image: str = "",
    reference_size: tuple[int, int] = (0, 0),
    subject: str = "",
    armed_occupied: Optional[bool] = None,
    created_by: str = "web",
    now: Optional[float] = None,
    watch_id: Optional[str] = None,
) -> Watch:
    """Build a watch from an exact arming frame plus a box.

    ``reference_image`` is a path relative to the Lookout storage root holding the
    frame the operator drew on, so the reference is the picture they actually
    saw — not whatever the pipeline was showing seconds later.
    """
    timestamp = time.time() if now is None else now
    kind = condition.kind
    if armed_occupied is None:
        # The condition says which state the operator expects to find *now*:
        # "tell me when it is gone" is only meaningful because it is here.
        armed_occupied = ARMED_OCCUPIED_DEFAULT[condition]
    return Watch(
        id=watch_id or uuid.uuid4().hex[:12],
        kind=kind,
        label=label.strip() or "watch",
        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
        condition=condition,
        created_at=timestamp,
        created_by=created_by,
        reference_size=(int(reference_size[0]), int(reference_size[1])),
        reference_wall_ts=timestamp,
        reference_capture_ts=timestamp,
        subject=subject.strip(),
        reference_image=reference_image,
        armed_occupied=bool(armed_occupied),
    )


class LookoutManager:
    """Owns armed watches, the worker thread, and everything downstream of it."""

    def __init__(
        self,
        options: LookoutOptions,
        *,
        store: Optional[WatchStore] = None,
        redis_client: Any = None,
        detect: Optional[DetectCallable] = None,
        notifier: Optional[WebhookNotifier] = None,
        decider: Optional[RuleDecider] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.options = options
        self.log = logger_ or logger
        self.store = store or WatchStore(options)
        self.redis = redis_client
        self.detect = detect
        self.notifier = notifier or WebhookNotifier(options.webhook_url, timeout_sec=options.webhook_timeout_sec)
        self.decider = decider or RuleDecider(options)
        self.ambiguity_log = AmbiguityLog(options.arbitration_log, max_per_hour=options.ambiguity_log_per_hour)
        self.evidence = EvidenceWriter(
            options.evidence_dir,
            max_events=options.evidence_max_events,
            max_age_days=options.evidence_max_age_days,
            quality=options.jpeg_quality,
        )
        self._runtimes: tuple[WatchRuntime, ...] = ()
        self._queue: queue.Queue[FramePayload] = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._effects = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lookout-fx")
        self._revision: Optional[str] = None
        self._last_submit = 0.0
        self._last_detect = 0.0
        self._last_publish = 0.0
        self._last_frame_publish = 0.0
        self._last_sweep = time.time()
        self._last_kill_check = 0.0
        self._last_tick_ts = 0.0
        self._killed = False
        self._disabled_detect = False
        self._detect_failures = 0
        self._drops = 0
        self._ticks = 0
        self._tick_ms = 0.0
        self._deliveries: dict[str, str] = {}

    # --------------------------------------------------------------- lifecycle
    @property
    def armed(self) -> bool:
        return bool(self._runtimes)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sync_watches(force=True)
        self._thread = threading.Thread(target=self._run, name="lookout", daemon=True)
        self._thread.start()
        self.log.info("Lookout worker started (armed=%d, tick=%.1fHz)", len(self._runtimes), self.options.tick_hz)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._effects.shutdown(wait=False, cancel_futures=True)

    # ----------------------------------------------------------- analyzer path
    def submit_frame(self, frame: np.ndarray, ts: float, boxes: Sequence[Any], *, fresh: bool) -> bool:
        """Hot path. Bounded work only; never blocks, never raises."""
        if not self.options.enabled:
            return False
        now = time.time()
        # Checked here, not only in the worker: a worker with nothing to do stops
        # running, and a kill switch that cannot be released without a restart is
        # not a switch.
        self._check_kill()
        if self._killed or frame is None or frame.size == 0:
            return False
        # The arming UI needs a clean frame even when nothing is armed yet.
        self._maybe_publish_frame(frame, ts, now)
        if not self._runtimes:
            return False
        interval = 1.0 / max(0.1, self.options.tick_hz)
        if (now - self._last_submit) < interval:
            return False
        self._last_submit = now
        try:
            small = downscale(frame, self.options.ring_width)
        except cv2.error:
            return False
        if small is None:
            return False
        payload = FramePayload(
            ts=ts,
            frame=small,
            detections=detections_from_boxes(boxes, min_conf=self.options.min_det_conf),
            fresh=fresh,
            wall_ts=now,
        )
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._drops += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                self._drops += 1
                return False
        return True

    def _maybe_publish_frame(self, frame: np.ndarray, ts: float, now: float) -> None:
        """Publish a clean processed frame for the arming UI.

        Deliberately the frame *before* visualization burns tracking overlays into
        it: an operator drawing a box on an annotated frame would be selecting
        the pipeline's own drawings as often as the thing they care about.
        """
        if self.redis is None or self.options.frame_publish_hz <= 0:
            return
        interval = 1.0 / max(0.01, self.options.frame_publish_hz)
        if (now - self._last_frame_publish) < interval:
            return
        self._last_frame_publish = now
        try:
            small = downscale(frame, self.options.ring_width)
            if small is None:
                return
            ok, buffer = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), self.options.jpeg_quality])
            if not ok:
                return
            envelope = pack_frame(
                buffer.tobytes(),
                {"ts": float(ts), "w": int(small.shape[1]), "h": int(small.shape[0]), "src": "lookout"},
            )
            self.redis.set(FRAME_KEY, envelope, ex=FRAME_TTL_SEC)
            # Health rides along with the frame: with no watch armed the worker
            # never evaluates anything, so this is the only liveness signal an
            # operator or watchdog can read.
            self.redis.set(HEALTH_KEY, json.dumps(self.health()), ex=FRAME_TTL_SEC * 3)
        except Exception as exc:
            self.log.debug("Lookout frame publish failed: %s", exc)

    def process(self, payload: FramePayload) -> list[LookoutEvent]:
        """Evaluate one frame synchronously. Used by the replay harness."""
        return self._evaluate(payload)

    # -------------------------------------------------------------- worker
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                self._watchdog()
                continue
            try:
                self._evaluate(payload)
            except Exception as exc:  # a watch must never take the analyzer down
                self.log.warning("Lookout evaluation failed: %s", exc, exc_info=True)

    def _evaluate(self, payload: FramePayload) -> list[LookoutEvent]:
        self._sync_watches()
        if self._killed or not self._runtimes:
            return []
        started = time.time()
        detections = list(payload.detections)
        detections.extend(self._extra_detections(payload.frame))
        emitted: list[LookoutEvent] = []
        for runtime in self._runtimes:
            events = runtime.tick(
                payload.frame,
                payload.ts,
                detections,
                fresh=payload.fresh,
                decider=self.decider,
                capture_ts=payload.capture_ts,
                detection_channel=payload.detection_channel or self.detect is not None,
            )
            for event in events:
                self._handle_event(runtime, event, payload.frame)
                emitted.append(event)
            runtime.ring.maybe_add(payload.ts, payload.frame)
        self._ticks += 1
        self._last_tick_ts = time.time()
        self._tick_ms = (self._last_tick_ts - started) * 1000.0
        self._publish()
        self._maybe_sweep()
        return emitted

    def _extra_detections(self, frame: np.ndarray) -> list[Detection]:
        """Detection pass for subjects the stop pipeline filters out.

        Runs on the downscaled frame in the worker, so it never competes with the
        main inference for the frame path. Disables itself after repeated failure.
        """
        if self.detect is None or self._disabled_detect or self.options.detect_hz <= 0:
            return []
        now = time.time()
        if (now - self._last_detect) < (1.0 / max(0.01, self.options.detect_hz)):
            return []
        self._last_detect = now
        try:
            xyxy, conf, class_ids = self.detect(frame, self.options.detect_classes)
        except Exception as exc:
            self._detect_failures += 1
            if self._detect_failures >= 3:
                self._disabled_detect = True
                self.log.warning("Lookout extra detection disabled after repeated failure: %s", exc)
            return []
        self._detect_failures = 0
        return detections_from_arrays(
            xyxy,
            conf,
            class_ids,
            min_conf=self.options.min_det_conf,
            class_filter=self.options.detect_classes,
        )

    # ----------------------------------------------------------- watch sync
    def _sync_watches(self, *, force: bool = False) -> None:
        revision = self.store.revision()
        if not force and revision == self._revision:
            return
        self._revision = revision
        try:
            watches = self.store.load()
        except Exception as exc:
            self.log.warning("Lookout watch load failed: %s", exc)
            return
        if len(watches) > self.options.max_watches:
            self.log.warning(
                "Lookout armed with %d watches; only the newest %d are evaluated",
                len(watches),
                self.options.max_watches,
            )
            watches = sorted(watches, key=lambda item: item.created_at, reverse=True)[: self.options.max_watches]
        existing = {runtime.watch.id: runtime for runtime in self._runtimes}
        updated: list[WatchRuntime] = []
        for watch in watches:
            current = existing.get(watch.id)
            if current is not None and current.watch.revision == watch.revision:
                updated.append(current)
                continue
            if current is not None:
                current.ring.clear()
            runtime = self._build_runtime(watch)
            if runtime is not None:
                updated.append(runtime)
        removed = set(existing) - {runtime.watch.id for runtime in updated}
        for watch_id in removed:
            existing[watch_id].ring.clear()
            self._forget_references(watch_id)
        self._runtimes = tuple(updated)

    def _build_runtime(self, watch: Watch) -> Optional[WatchRuntime]:
        now = time.time()
        patch = self._arming_patch(watch)
        model = ReferenceModel.from_armed_patch(
            patch,
            armed_occupied=watch.armed_occupied,
            drift=self.options.reference_drift,
            decisive_similarity=self.options.decisive_similarity,
            present_floor=self.options.present_drift_floor,
            absent_ceiling=self.options.absent_drift_ceiling,
            min_opposite_samples=self.options.min_opposite_samples,
        )
        self._restore_references(watch.id, model)
        ring = RingBuffer(
            os.path.join(self.options.ring_root, watch.id),
            max_frames=self.options.ring_frames,
            interval_sec=self.options.ring_interval_sec,
            width=self.options.ring_width,
            quality=self.options.jpeg_quality,
        )
        runtime = WatchRuntime(watch, self.options, model=model, ring=ring, now=now)
        if not model.ready:
            runtime.reason = "arming frame unavailable; waiting for evidence"
        self.log.info("Lookout armed watch %s (%s, %s)", watch.id, watch.label, watch.condition.value)
        return runtime

    def _arming_patch(self, watch: Watch) -> Optional[np.ndarray]:
        """Crop the armed reference out of the exact frame the operator drew on."""
        if not watch.reference_image:
            return None
        path = os.path.join(self.options.storage_root, watch.reference_image)
        try:
            image = cv2.imread(path)
        except cv2.error:
            return None
        if image is None or image.size == 0:
            return None
        box = watch.box
        if watch.reference_size and (watch.reference_size[0] > 0):
            box = watch.scaled_box((image.shape[1], image.shape[0]))
        return references.patch_for(image, box)

    # -------------------------------------------------------------- events
    def _handle_event(self, runtime: WatchRuntime, event: LookoutEvent, frame: np.ndarray) -> None:
        box = runtime.watch.scaled_box((frame.shape[1], frame.shape[0]))
        is_note = bool(event.detail.get("note"))
        evidence_dir = self.evidence.write(
            event.id,
            frame=frame,
            box=box,
            ring=runtime.ring.frames(),
            capture_ts=event.capture_ts,
            wall_ts=event.wall_ts,
            summary=event.to_row(),
        )
        event.evidence_dir = evidence_dir
        self.store.append_event(event)
        if event.alerting:
            self._effects.submit(self._deliver, event)
        elif not is_note and runtime.last_decision is not None and not runtime.last_decision.evidence_ok:
            self._log_ambiguity(runtime, event)
        self.log.info(
            "Lookout %s: %s -> %s (%s, prob=%.2f, evidence=%.0f%%)",
            event.watch_id,
            event.from_state,
            event.to_state,
            event.wording,
            event.present_prob,
            event.evidence_fraction * 100,
        )

    def _log_ambiguity(self, runtime: WatchRuntime, event: LookoutEvent) -> None:
        """Record an inconclusive episode so the offline comparison has real data.

        The record carries the measurements the rule saw, so a decision model can
        be replayed against the same features instead of a different question.
        """
        episode_id = f"amb-{event.id}"
        patch = None
        frame = runtime.ring.frames()
        if frame:
            try:
                patch = cv2.imread(frame[-1].path)
            except cv2.error:
                patch = None
        if patch is not None:
            self.evidence.write(
                episode_id,
                frame=patch,
                box=(0.0, 0.0, float(patch.shape[1]), float(patch.shape[0])),
                ring=[],
                capture_ts=event.capture_ts,
                wall_ts=event.wall_ts,
                summary={"kind": "ambiguity", **event.to_row()},
            )
        decision = runtime.last_decision
        self.ambiguity_log.record(
            {
                "watch_id": event.watch_id,
                "kind": runtime.watch.kind.value,
                "subject": runtime.watch.subject,
                "condition": runtime.watch.condition.value,
                "state": runtime.state.value,
                "from_state": event.from_state,
                "to_state": event.to_state,
                "rule_prob": runtime.present_prob,
                "features": (decision.detail if decision else {}),
                "evidence_dir": os.path.join(self.options.evidence_dir, episode_id),
            }
        )

    def _deliver(self, event: LookoutEvent) -> None:
        image_path = os.path.join(event.evidence_dir, "full.jpg") if event.evidence_dir else None
        result = self.notifier.notify(event, image_path=image_path)
        self._deliveries[event.id] = result
        if len(self._deliveries) > 50:
            for stale in list(self._deliveries)[:-25]:
                self._deliveries.pop(stale, None)
        self._publish_delivery(event, result)
        if result.startswith("failed"):
            self.log.warning("Lookout delivery failed for %s: %s", event.id, result)

    def _publish_delivery(self, event: LookoutEvent, result: str) -> None:
        if self.redis is None:
            return
        try:
            self.redis.hset(
                DELIVERY_KEY,
                event.id,
                json.dumps({"watch_id": event.watch_id, "result": result, "ts": time.time()}),
            )
            self.redis.expire(DELIVERY_KEY, 86400)
        except Exception:
            return

    # -------------------------------------------------------------- watchdog
    def _watchdog(self) -> None:
        """Frames stopping is not the same as the area clearing."""
        if not self._runtimes:
            return
        now = time.time()
        if self._last_tick_ts and (now - self._last_tick_ts) > self.options.unavailable_after_sec:
            for runtime in self._runtimes:
                event = runtime.mark_unavailable(now)
                if event is not None:
                    self.store.append_event(event)
                    self.log.warning(
                        "Lookout %s unavailable: no frames for %.0fs", event.watch_id, now - self._last_tick_ts
                    )
            self._last_tick_ts = now
            self._publish()

    def _maybe_sweep(self) -> None:
        now = time.time()
        if (now - self._last_sweep) < 900.0:
            return
        self._last_sweep = now
        try:
            self.store.compact_events()
            self.evidence.sweep(now=now)
        except Exception as exc:
            self.log.debug("Lookout retention sweep failed: %s", exc)

    # ---------------------------------------------------------------- publish
    def _check_kill(self) -> None:
        now = time.time()
        if (now - self._last_kill_check) < 1.0:
            return
        self._last_kill_check = now
        if self.redis is None:
            return
        try:
            raw = self.redis.get(self.options.kill_key)
        except Exception:
            return
        killed = bool(raw) and str(raw).strip() in {"1", "true", "yes", "on"}
        if killed != self._killed:
            self.log.warning("Lookout kill switch %s", "engaged" if killed else "released")
        self._killed = killed

    def snapshot(self) -> dict[str, Any]:
        return {
            "updated_ts": time.time(),
            "killed": self._killed,
            "armed": len(self._runtimes),
            "watches": {runtime.watch.id: runtime.status().to_dict() for runtime in self._runtimes},
            "health": self.health(),
        }

    def health(self) -> dict[str, Any]:
        return {
            "ticks": self._ticks,
            "queue_drops": self._drops,
            "tick_ms": round(self._tick_ms, 2),
            "last_tick_ts": self._last_tick_ts,
            "armed": len(self._runtimes),
            "killed": self._killed,
            "detect_disabled": self._disabled_detect,
            "deliveries": dict(list(self._deliveries.items())[-10:]),
        }

    def _publish(self) -> None:
        self._check_kill()
        now = time.time()
        interval = 1.0 / max(0.1, self.options.publish_hz)
        if (now - self._last_publish) < interval:
            return
        self._last_publish = now
        if self.redis is None:
            return
        try:
            pipeline = self.redis.pipeline()
            for runtime in self._runtimes:
                pipeline.hset(STATE_KEY, runtime.watch.id, json.dumps(runtime.status().to_dict()))
                pipeline.hset(REFERENCE_KEY, runtime.watch.id, json.dumps(self._storable_references(runtime)))
            pipeline.set(HEALTH_KEY, json.dumps(self.health()))
            pipeline.execute()
        except Exception as exc:
            self.log.debug("Lookout publish failed: %s", exc)

    # ------------------------------------------------------------ persistence
    def _storable_references(self, runtime: WatchRuntime) -> dict[str, Any]:
        model = runtime.model
        return {
            "armed": model.armed.to_storable() if model.armed is not None else None,
            "opposite": model.opposite.to_storable() if model.opposite is not None else None,
        }

    def _restore_references(self, watch_id: str, model: ReferenceModel) -> None:
        if self.redis is None:
            return
        try:
            raw = self.redis.hget(REFERENCE_KEY, watch_id)
        except Exception:
            return
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except ValueError:
            return
        model.restore(
            references.Reference.from_storable(payload.get("armed")),
            references.Reference.from_storable(payload.get("opposite")),
        )

    def _forget_references(self, watch_id: str) -> None:
        if self.redis is None:
            return
        try:
            self.redis.hdel(REFERENCE_KEY, watch_id)
            self.redis.hdel(STATE_KEY, watch_id)
        except Exception:
            return

    def sync_now(self) -> None:
        """Force a reload of the armed watches. Used by replay and tests."""
        self._sync_watches(force=True)

    def shutdown_watch(self, watch_id: str) -> None:
        """Called when the web stops a watch: drop its state and learned refs."""
        self._forget_references(watch_id)
        for runtime in self._runtimes:
            if runtime.watch.id == watch_id:
                runtime.stop(time.time())
                runtime.ring.clear()
