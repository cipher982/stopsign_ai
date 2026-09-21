"""Per-watch state machine over wall-clock time with an evidence mask.

The rules the state machine enforces, all of them consequences of review:

* A transition needs the new reading to hold for a number of *seconds*, and the
  span it measures must be mostly evidence-backed. Missing evidence never
  accumulates absence.
* Losing sight of the subject is not a state change. The watch goes ``UNKNOWN``
  and remembers what it believed; when evidence returns, the ordinary hold times
  apply again before anything is reported.
* ``UNKNOWN`` and ``UNAVAILABLE`` are reported as notes, never as the alerting
  condition, and never as the nominal state.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Optional

from stopsign.lookout import perception
from stopsign.lookout.decider import RuleContext
from stopsign.lookout.decider import RuleDecider
from stopsign.lookout.detections import Detection
from stopsign.lookout.evidence import RingBuffer
from stopsign.lookout.options import LookoutOptions
from stopsign.lookout.references import ReferenceModel
from stopsign.lookout.references import patch_for
from stopsign.lookout.types import Condition
from stopsign.lookout.types import Decision
from stopsign.lookout.types import LookoutEvent
from stopsign.lookout.types import Observation
from stopsign.lookout.types import Watch
from stopsign.lookout.types import WatchKind
from stopsign.lookout.types import WatchState
from stopsign.lookout.types import WatchStatus

_PRESENT = "present"
_ABSENT = "absent"
_UNKNOWN = "unknown"


@dataclass(slots=True)
class _Sample:
    ts: float
    kind: str


class WatchRuntime:
    """Live evaluation state for one armed watch."""

    def __init__(
        self,
        watch: Watch,
        options: LookoutOptions,
        *,
        model: ReferenceModel,
        ring: RingBuffer,
        now: float,
    ) -> None:
        self.watch = watch
        self.options = options
        self.model = model
        self.ring = ring
        self.state = WatchState.PENDING
        self.state_since = now
        self.suspended_from = WatchState.PENDING
        self.updated_ts = now
        self.last_frame_ts = 0.0
        self.last_fresh_ts = 0.0
        self.reference_frame_mean = 0.0
        self.present_prob = 0.5
        self.evidence_fraction = 0.0
        self.decisive = False
        self.reason = "armed, waiting for the first reading"
        self.alert_count = 0
        # Negative infinity, not zero: a clock that starts near zero (replay)
        # must not have its first alert swallowed by the cooldown.
        self.last_alert_ts = float("-inf")
        self.last_decision: Optional[Decision] = None
        self.last_observation: Optional[Observation] = None
        self.history: deque[float] = deque(maxlen=120)
        self.samples: deque[_Sample] = deque(maxlen=600)
        self._blind_note_ts = float("-inf")

    # ------------------------------------------------------------------ tick
    def tick(
        self,
        frame,
        ts: float,
        detections: list[Detection],
        *,
        fresh: bool,
        decider: RuleDecider,
        capture_ts: Optional[float] = None,
        detection_channel: bool = True,
    ) -> list[LookoutEvent]:
        """Evaluate one frame. Returns any events the frame produced."""
        started = time.time()
        clock = ts
        self.last_frame_ts = started
        box = self.watch.scaled_box((frame.shape[1], frame.shape[0]))
        patch = patch_for(frame, box)
        frame_mean, frame_contrast = perception.frame_statistics(frame)
        if self.reference_frame_mean <= 0.0 and fresh:
            self.reference_frame_mean = frame_mean

        score = self.model.score(patch)
        metrics = perception.measure_region(box, detections, self.watch.subject)
        age_sec = max(0.0, started - self.last_fresh_ts) if self.last_fresh_ts else 0.0
        if fresh:
            self.last_fresh_ts = started

        obs = Observation(
            ts=capture_ts if capture_ts is not None else started,
            fresh=fresh,
            age_sec=0.0 if fresh else age_sec,
            present_prob=score.present_prob,
            decisive=score.decisive,
            similarity_armed=score.similarity_armed,
            similarity_opposite=score.similarity_opposite,
            subject_cover=metrics.subject_cover,
            subject_conf=metrics.subject_conf,
            subject_label=metrics.subject_label,
            occluder_cover=metrics.occluder_cover,
            occluder_label=metrics.occluder_label,
            frame_mean=frame_mean,
            frame_contrast=frame_contrast,
            detection_channel=detection_channel,
            reason="",
        )

        absent_sec, absent_frac = self._absence_stats(clock)
        present_sec, present_frac = self._presence_stats(clock)
        ctx = RuleContext(
            reference_frame_mean=self.reference_frame_mean,
            evidence_fraction=self.evidence_fraction,
            absent_sec=absent_sec,
            detection_available=detection_channel,
        )
        decision = decider.decide(self.watch, obs, ctx)
        self.last_decision = decision
        self.last_observation = obs
        self.present_prob = decision.present_prob
        self.decisive = obs.decisive
        self.reason = self._describe(decision)

        kind = _reading_kind(decision, self.options)
        self.samples.append(_Sample(ts=clock, kind=kind))
        self.history.append(decision.present_prob)
        self.updated_ts = started

        # Reference drift is only allowed while the reading is confident and the
        # evidence actually supports it — otherwise an occluder teaches the model
        # that a parked car looks like a pedestrian.
        if decision.evidence_ok and obs.decisive:
            self.model.observe(patch, present_prob=decision.present_prob, ts=clock)

        events = self._advance(
            now=clock,
            decision=decision,
            obs=obs,
            absent_sec=absent_sec,
            absent_frac=absent_frac,
            present_sec=present_sec,
            present_frac=present_frac,
            frame=frame,
            box=box,
        )
        self._observe_ms = (time.time() - started) * 1000.0
        return events

    def _advance(
        self,
        *,
        now: float,
        decision: Decision,
        obs: Observation,
        absent_sec: float,
        absent_frac: float,
        present_sec: float,
        present_frac: float,
        frame,
        box,
    ) -> list[LookoutEvent]:
        opts = self.options
        events: list[LookoutEvent] = []
        confident = decision.evidence_ok and obs.decisive
        state = self.state
        self.evidence_fraction = present_frac if self._subject_expected() else absent_frac

        # Blind: no usable evidence right now. Suspend rather than decide.
        if not confident:
            if state in (WatchState.UNKNOWN, WatchState.UNAVAILABLE, WatchState.STOPPED):
                return events
            newly_blind = (now - self._blind_note_ts) >= opts.note_cooldown_sec
            self._set_state(WatchState.UNKNOWN, now)
            self.reason = self.reason or "no usable evidence"
            if newly_blind:
                self._blind_note_ts = now
                events.append(self._note_event(now, obs, decision, "view unavailable or inconclusive"))
            return events

        if state in (WatchState.UNKNOWN, WatchState.UNAVAILABLE):
            # Back from the blind: restore what we believed and keep accumulating.
            self.state = self.suspended_from
            self.state_since = now
            state = self.state
            self.reason = f"evidence restored; last known state was {state.value}"

        nominal = self._nominal_state()
        absent_state = self._absent_state()

        if state is WatchState.PENDING:
            confirm = (
                (present_sec >= opts.arm_confirm_sec and present_frac >= opts.min_evidence_fraction)
                if nominal is WatchState.OCCUPIED
                else (absent_sec >= opts.arm_confirm_sec and absent_frac >= opts.min_evidence_fraction)
            )
            if confirm:
                self._set_state(nominal, now)
                self.reason = f"confirmed {nominal.value} at arm time"
            elif now - self.watch.created_at > opts.arm_timeout_sec:
                self._set_state(WatchState.UNKNOWN, now)
                self.suspended_from = WatchState.PENDING
                self.reason = "never confirmed the armed state"
            return events

        ready_for_present = present_sec >= self._hold_sec(state, WatchState.OCCUPIED) and (
            present_frac >= opts.min_evidence_fraction
        )
        ready_for_absent = absent_sec >= self._hold_sec(state, absent_state) and (
            absent_frac >= opts.min_evidence_fraction
        )

        if state is not WatchState.OCCUPIED and ready_for_present:
            previous = state
            self._set_state(WatchState.OCCUPIED, now)
            condition = Condition.BACK if previous is WatchState.GONE else Condition.OCCUPIED
            if previous in (WatchState.GONE, WatchState.OBSERVING):
                events.append(
                    self._event(
                        condition,
                        previous,
                        WatchState.OCCUPIED,
                        now,
                        obs,
                        decision,
                        present_sec,
                        present_frac,
                    )
                )
        elif state is not absent_state and ready_for_absent:
            previous = state
            self._set_state(absent_state, now)
            condition = Condition.GONE if absent_state is WatchState.GONE else Condition.CLEARED
            if previous in (WatchState.OCCUPIED, WatchState.OBSERVING, WatchState.GONE):
                events.append(
                    self._event(
                        condition,
                        previous,
                        absent_state,
                        now,
                        obs,
                        decision,
                        absent_sec,
                        absent_frac,
                    )
                )
        return events

    # ------------------------------------------------------------- helpers
    def _subject_expected(self) -> bool:
        return self.state in (WatchState.OCCUPIED, WatchState.PENDING) or self._nominal_state() is WatchState.OCCUPIED

    def _nominal_state(self) -> WatchState:
        return WatchState.OCCUPIED if self.watch.armed_occupied else self._absent_state()

    def _absent_state(self) -> WatchState:
        return WatchState.GONE if self.watch.kind is WatchKind.OBJECT else WatchState.OBSERVING

    def _hold_sec(self, state: WatchState, target: WatchState) -> float:
        opts = self.options
        if target is WatchState.OCCUPIED:
            return opts.return_hold_sec if self.watch.kind is WatchKind.OBJECT else opts.occupied_hold_sec
        return opts.gone_hold_sec if self.watch.kind is WatchKind.OBJECT else opts.empty_hold_sec

    def _absence_stats(self, now: float) -> tuple[float, float]:
        """Seconds since presence was last observed, and how much of that span was evidence-backed absence."""
        return self._stats_since(now, trigger=_PRESENT, counted=_ABSENT)

    def _presence_stats(self, now: float) -> tuple[float, float]:
        return self._stats_since(now, trigger=_ABSENT, counted=_PRESENT)

    def _stats_since(self, now: float, *, trigger: str, counted: str) -> tuple[float, float]:
        start: Optional[float] = None
        for sample in reversed(self.samples):
            if sample.kind == trigger:
                start = sample.ts
                break
        if start is None:
            start = self.samples[0].ts if self.samples else now
        counted_n = 0
        total = 0
        for sample in self.samples:
            if sample.ts <= start:
                continue
            total += 1
            if sample.kind == counted:
                counted_n += 1
        fraction = (counted_n / total) if total else 0.0
        return max(0.0, now - start), fraction

    def _set_state(self, state: WatchState, now: float) -> None:
        if state in (WatchState.UNKNOWN, WatchState.UNAVAILABLE):
            self.suspended_from = self.state
        self.state = state
        self.state_since = now

    def _describe(self, decision: Decision) -> str:
        detail = decision.detail
        subject = self.watch.subject or "object"
        if not decision.evidence_ok:
            reasons = detail.get("reasons") or []
            return "; ".join(str(item) for item in reasons) or "evidence insufficient"
        label = self.last_observation.subject_label if self.last_observation else None
        cover = detail.get("subject_cover", 0.0)
        unfamiliar = "" if detail.get("matches_known_state", True) else " — unfamiliar content"
        return f"{label or subject} presence {decision.present_prob:.0%} (coverage {float(cover):.0%}){unfamiliar}"

    def _event(
        self,
        condition: Condition,
        from_state: WatchState,
        to_state: WatchState,
        now: float,
        obs: Observation,
        decision: Decision,
        held_sec: float,
        evidence_fraction: float,
    ) -> LookoutEvent:
        alerting = condition is self.watch.condition and (now - self.last_alert_ts) >= self.options.alert_cooldown_sec
        if alerting:
            self.last_alert_ts = now
            self.alert_count += 1
        return LookoutEvent(
            id=uuid.uuid4().hex[:12],
            watch_id=self.watch.id,
            label=self.watch.label,
            condition=condition,
            kind=self.watch.kind,
            subject=self.watch.subject,
            from_state=from_state.value,
            to_state=to_state.value,
            capture_ts=obs.ts,
            wall_ts=time.time(),
            present_prob=decision.present_prob,
            evidence_ok=decision.evidence_ok,
            evidence_fraction=evidence_fraction,
            source=decision.source,
            alerting=alerting,
            detail={
                "reason": self.reason,
                "held_sec": round(held_sec, 2),
                # Was emptiness actually checked for objects, or only inferred
                # from the region no longer matching the armed appearance?
                "occupancy_check": "detections" if obs.detection_channel else "appearance",
                "occupancy_confirmed": bool(obs.detection_channel and obs.subject_cover <= self.options.zone_cover_lo),
                **{k: v for k, v in decision.detail.items() if k != "reasons"},
            },
        )

    def _note_event(self, now: float, obs: Observation, decision: Decision, reason: str) -> LookoutEvent:
        return LookoutEvent(
            id=uuid.uuid4().hex[:12],
            watch_id=self.watch.id,
            label=self.watch.label,
            condition=self.watch.condition,
            kind=self.watch.kind,
            subject=self.watch.subject,
            from_state=self.suspended_from.value,
            to_state=WatchState.UNKNOWN.value,
            capture_ts=obs.ts,
            wall_ts=time.time(),
            present_prob=decision.present_prob,
            evidence_ok=False,
            evidence_fraction=self.evidence_fraction,
            source=decision.source,
            alerting=False,
            detail={"reason": reason, "note": True},
        )

    def mark_unavailable(self, now: float) -> Optional[LookoutEvent]:
        """Called when frames stop entirely: the watch is blind, not clear."""
        if self.state is WatchState.UNAVAILABLE:
            return None
        previous = self.state
        self._set_state(WatchState.UNAVAILABLE, now)
        self.reason = "no frames from the camera"
        return LookoutEvent(
            id=uuid.uuid4().hex[:12],
            watch_id=self.watch.id,
            label=self.watch.label,
            condition=self.watch.condition,
            kind=self.watch.kind,
            subject=self.watch.subject,
            from_state=previous.value,
            to_state=WatchState.UNAVAILABLE.value,
            capture_ts=now,
            wall_ts=time.time(),
            present_prob=self.present_prob,
            evidence_ok=False,
            evidence_fraction=0.0,
            source="watchdog",
            alerting=False,
            detail={"reason": "no frames", "note": True},
        )

    def stop(self, now: float) -> None:
        self.state = WatchState.STOPPED
        self.state_since = now
        self.reason = "stopped"

    def status(self) -> WatchStatus:
        return WatchStatus(
            watch_id=self.watch.id,
            label=self.watch.label,
            kind=self.watch.kind.value,
            condition=self.watch.condition.value,
            subject=self.watch.subject,
            state=self.state.value,
            since_ts=self.state_since,
            updated_ts=self.updated_ts,
            present_prob=self.present_prob,
            evidence_fraction=self.evidence_fraction,
            decisive=self.decisive,
            reason=self.reason,
            box=self.watch.box,
            last_frame_ts=self.last_frame_ts,
            armed_wall_ts=self.watch.created_at,
            alerting_events=self.alert_count,
            history=list(self.history),
        )

    def tick_cost_ms(self) -> float:
        return getattr(self, "_observe_ms", 0.0)


def _reading_kind(decision: Decision, options: LookoutOptions) -> str:
    """Classify one reading. The middle band is 'unknown', not a weak 'absent'."""
    if not decision.evidence_ok:
        return _UNKNOWN
    if decision.present_prob >= options.presence_hi:
        return _PRESENT
    if decision.present_prob <= options.presence_lo:
        return _ABSENT
    return _UNKNOWN
