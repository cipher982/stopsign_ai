"""Decision layer: measurements in, a typed answer out.

The live authority is the rule decider. It is boring on purpose: it combines the
two-reference similarity with detection evidence and reports, separately, *what
it thinks* and *whether the evidence supports thinking anything at all*.

The VLM and Jev backends exist in this file for one use: replaying the ambiguous
episodes the live system logs, so "does a decision model actually add anything
here?" can be answered with data instead of enthusiasm. They are never on the
alert path. Every arbitration record keeps the features the rule read, so the
comparison is apples to apples.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from typing import Optional

from stopsign.lookout.options import LookoutOptions
from stopsign.lookout.types import Decision
from stopsign.lookout.types import Observation
from stopsign.lookout.types import Watch
from stopsign.lookout.types import WatchKind


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class RuleContext:
    """Everything the live rule is allowed to look at."""

    reference_frame_mean: float = 0.0
    evidence_fraction: float = 1.0
    absent_sec: float = 0.0
    detection_available: bool = True


class RuleDecider:
    """Deterministic live decider over perception measurements."""

    name = "rule"

    def __init__(self, options: LookoutOptions) -> None:
        self.options = options
        self.illumination_limit = options.illumination_limit

    def decide(self, watch: Watch, obs: Observation, ctx: RuleContext) -> Decision:
        opts = self.options
        # Two independent readings of the same question: does the region look
        # like the subject is there, and is the subject's class detected in it?
        reference_p = obs.present_prob
        detection_p: Optional[float] = None
        if ctx.detection_available and (obs.subject_cover > 0.0 or watch.kind is WatchKind.ZONE):
            threshold = opts.zone_cover_hi if watch.kind is WatchKind.ZONE else opts.object_cover_hi
            detection_p = _clamp(obs.subject_cover / max(1e-6, threshold))

        # A detection overlapping the region is direct evidence that something is
        # there, and it wins when it is confident. The *absence* of detections is
        # weak evidence — the class list is finite, and a bike, a bin or a branch
        # is invisible to it — so it must never argue a region is empty. Absence
        # of objects is used only to confirm, at the moment an event is emitted.
        blended = detection_p is not None and detection_p >= 0.5
        present_prob = max(reference_p, detection_p) if blended else reference_p

        # Evidence sufficiency — a separate output, never folded into the
        # probability. A confident number computed from an occluded view is
        # still a guess.
        reasons: list[str] = []
        evidence_ok = True
        # A frame the detector skipped is not evidence of anything: it ages the
        # last real reading instead. Becoming blind on a single skipped frame
        # would make the watch flap, because skipping stale frames is a designed
        # behaviour of this pipeline (~20% of frames), not a fault.
        if obs.age_sec > opts.stale_after_sec:
            evidence_ok = False
            reasons.append(f"no fresh detection for {obs.age_sec:.1f}s")
        if not obs.decisive:
            evidence_ok = False
            reasons.append("no usable measurement of the region")
        if obs.occluder_cover >= opts.occluder_cover_limit:
            evidence_ok = False
            reasons.append(f"{obs.occluder_label or 'object'} blocks the view")
        if self._illumination_invalid(obs, ctx) and present_prob < 0.5:
            evidence_ok = False
            reasons.append("lighting changed")

        return Decision(
            present_prob=present_prob,
            evidence_ok=evidence_ok,
            source=self.name,
            detail={
                "reference_p": round(reference_p, 4),
                "detection_p": None if detection_p is None else round(detection_p, 4),
                "detection_decided": blended,
                "subject_cover": round(obs.subject_cover, 4),
                "occluder_cover": round(obs.occluder_cover, 4),
                "similarity_armed": round(obs.similarity_armed, 4),
                "similarity_opposite": round(obs.similarity_opposite, 4),
                "decisive": obs.decisive,
                "matches_known_state": max(obs.similarity_armed, obs.similarity_opposite)
                >= self.options.decisive_similarity,
                "evidence_fraction": round(ctx.evidence_fraction, 4),
                "absent_sec": round(ctx.absent_sec, 2),
                "reasons": reasons,
            },
        )

    def _illumination_invalid(self, obs: Observation, ctx: RuleContext) -> bool:
        if ctx.reference_frame_mean <= 0.0:
            return False
        return abs(obs.frame_mean - ctx.reference_frame_mean) / 255.0 > self.illumination_limit


class AmbiguityLog:
    """Append-only record of the episodes where a decision model could matter.

    Deliberately capped: logging every ambiguous tick would produce hundreds of
    thousands of rows a day, most of them duplicates, and cost more attention
    than it returns.
    """

    def __init__(self, path: str, *, max_per_hour: int, ring_dir: str = "") -> None:
        self.path = path
        self.max_per_hour = max(1, max_per_hour)
        self.ring_dir = ring_dir
        self._hour_started = 0.0
        self._written_this_hour = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def record(self, payload: dict[str, Any], *, now: Optional[float] = None) -> bool:
        current = time.time() if now is None else now
        if current - self._hour_started >= 3600.0:
            self._hour_started = current
            self._written_this_hour = 0
        if self._written_this_hour >= self.max_per_hour:
            return False
        self._written_this_hour += 1
        row = {"ts": current, "ring_dir": self.ring_dir, **payload}
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
            return True
        except OSError:
            return False


class ShadowDecider:
    """Base for offline-only backends. Never called from the alert path."""

    name = "shadow"

    def __init__(self, options: LookoutOptions) -> None:
        self.options = options

    def decide(self, watch: Watch, obs: Observation, ctx: RuleContext, *, image_jpeg: bytes | None = None) -> Decision:
        raise NotImplementedError

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.options.arbiter_timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))


class JevDecider(ShadowDecider):
    """A structured-state arbiter: the measurements, not the image.

    The state is the perceptual evidence the rule already computed, plus the
    watch's own history. That is the only form of "vision" a text-only decision
    model can be given honestly, and it is exactly the comparison worth running:
    can a calibrated arbiter beat a fixed threshold on the same features?
    """

    name = "jev"

    def decide(self, watch: Watch, obs: Observation, ctx: RuleContext, *, image_jpeg: bytes | None = None) -> Decision:
        del image_jpeg
        state = {
            "watch": {
                "kind": watch.kind.value,
                "subject": watch.subject or "any object",
                "label": watch.label,
                "condition": watch.condition.value,
                "armed_occupied": watch.armed_occupied,
            },
            "measurement": {
                "similarity_to_armed_state": round(obs.similarity_armed, 4),
                "similarity_to_opposite_state": round(obs.similarity_opposite, 4),
                "matches_neither_reference": not obs.decisive,
                "subject_coverage": round(obs.subject_cover, 4),
                "subject_detection_confidence": round(obs.subject_conf, 4),
                "subject_label": obs.subject_label,
                "other_object_coverage": round(obs.occluder_cover, 4),
                "other_object_label": obs.occluder_label,
                "seconds_without_evidence": round(ctx.absent_sec, 2),
                "frame_brightness_delta": None,
            },
            "context": {
                "detection_is_fresh": obs.fresh,
                "detection_age_seconds": round(obs.age_sec, 2),
                "evidence_fraction_of_window": round(ctx.evidence_fraction, 4),
                "note": (
                    "Coverage is a summed upper bound per detection, not a union. "
                    "The region resembles neither stored reference when something "
                    "unfamiliar occupies it."
                ),
            },
        }
        questions = {
            "subject_present": {
                "type": "noul",
                "instructions": (
                    f"Is the watched {watch.kind.value} ({watch.label}) present in the region, "
                    "given these measurements?"
                ),
                "criteria": {
                    "true": "Coverage or reference similarity indicates the subject is present",
                    "false": "Evidence indicates the subject is not present",
                },
            },
            "view_blocked": {
                "type": "noul",
                "instructions": "Is something else blocking the view of the region?",
                "criteria": {
                    "true": "Another object covers the region or the measurement matches neither reference",
                    "false": "The view of the region is unobstructed",
                },
            },
        }
        started = time.time()
        try:
            response = self._post_json(
                f"{self.options.openrouter_base_url.rstrip('/')}/systemone",
                {"state": state, "model": self.options.jev_model, "questions": questions},
                {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}"},
            )
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            return Decision(
                present_prob=obs.present_prob,
                evidence_ok=False,
                source=self.name,
                detail={"error": str(exc)},
            )
        answers = response.get("answers") or {}
        present = _noul(answers, "subject_present")
        blocked = _noul(answers, "view_blocked")
        usage = response.get("usage") or {}
        input_tokens = float(usage.get("input_tokens") or 0)
        return Decision(
            present_prob=present if present is not None else obs.present_prob,
            evidence_ok=(blocked or 0.0) < 0.5 if blocked is not None else True,
            source=f"{self.name}:{response.get('model', self.options.jev_model)}",
            latency_ms=(time.time() - started) * 1000.0,
            cost_usd=input_tokens * self.options.jev_input_usd_per_mtok / 1_000_000.0,
            detail={
                "subject_present": present,
                "view_blocked": blocked,
                "input_tokens": input_tokens,
                "raw_answers": answers,
            },
        )


class VlmDecider(ShadowDecider):
    """A direct-look arbiter: the crop, and one typed question.

    Kept as the honest second opinion for the offline comparison. It sees pixels
    the rule never did, so a win here would be evidence of information the
    structured state threw away — which is the actual question.
    """

    name = "vlm"

    def decide(self, watch: Watch, obs: Observation, ctx: RuleContext, *, image_jpeg: bytes | None = None) -> Decision:
        if not self.options.vlm_model:
            return Decision(
                present_prob=obs.present_prob, evidence_ok=False, source=self.name, detail={"error": "no model"}
            )
        if not image_jpeg:
            return Decision(
                present_prob=obs.present_prob, evidence_ok=False, source=self.name, detail={"error": "no image"}
            )
        prompt = (
            f"You are the visual arbiter for a camera watch. The watched region is the {watch.kind.value} "
            f'labelled "{watch.label}"'
            + (f" (expected subject: {watch.subject})" if watch.subject else "")
            + f". The operator asked: {watch.condition.question}. The region has been unobserved for "
            f"{ctx.absent_sec:.1f}s and {ctx.evidence_fraction:.0%} of that window had usable evidence.\n"
            "Answer only as JSON: "
            '{"presence": "present" | "absent" | "cannot_tell", "confidence": <0-1>, "reason": "<short>"}'
        )
        payload = {
            "model": self.options.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()},
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }
        started = time.time()
        try:
            response = self._post_json(
                f"{self.options.openrouter_base_url.rstrip('/')}/chat/completions",
                payload,
                {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}"},
            )
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            return Decision(
                present_prob=obs.present_prob, evidence_ok=False, source=self.name, detail={"error": str(exc)}
            )
        content = ""
        try:
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, ValueError):
            return Decision(
                present_prob=obs.present_prob,
                evidence_ok=False,
                source=self.name,
                detail={"error": "unparsable", "raw": content[:400]},
            )
        presence = str(parsed.get("presence", "cannot_tell")).lower()
        confidence = _clamp(float(parsed.get("confidence") or 0.0))
        if presence == "present":
            present_prob = confidence
            evidence_ok = True
        elif presence == "absent":
            present_prob = 1.0 - confidence
            evidence_ok = True
        else:
            present_prob = 0.5
            evidence_ok = False
        usage = response.get("usage") or {}
        return Decision(
            present_prob=present_prob,
            evidence_ok=evidence_ok,
            source=f"{self.name}:{response.get('model', self.options.vlm_model)}",
            latency_ms=(time.time() - started) * 1000.0,
            detail={
                "presence": presence,
                "confidence": confidence,
                "reason": parsed.get("reason"),
                "tokens": usage,
                "raw": content[:400],
            },
        )


def _noul(answers: dict[str, Any], key: str) -> Optional[float]:
    entry = answers.get(key)
    if not isinstance(entry, dict):
        return None
    value = entry.get("noul")
    try:
        return _clamp(float(value))
    except (TypeError, ValueError):
        return None


def make_shadow_decider(options: LookoutOptions) -> Optional[ShadowDecider]:
    """Build the offline comparison backend named by ``LOOKOUT_DECIDER``."""
    name = (options.decider or "rule").strip().lower()
    if name == "jev":
        return JevDecider(options)
    if name == "vlm":
        return VlmDecider(options)
    return None
