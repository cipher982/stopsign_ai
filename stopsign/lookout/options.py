"""Lookout runtime options, read from the environment.

Everything is env-driven with conservative defaults so the evaluator can be
turned down, or off, without a deploy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import fields

_STR_FIELDS = frozenset(
    {
        "extra_detect_classes",
        "decider",
        "vlm_model",
        "jev_model",
        "openrouter_base_url",
        "storage_root",
        "access_token",
        "webhook_url",
        "kill_key",
    }
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class LookoutOptions:
    """Thresholds and cadence for watch evaluation.

    Every hold time is in wall-clock seconds with an explicit evidence mask.
    Frame counts are never used: the analyzer drops frames, runs YOLO off-rate,
    and discards backlog during catch-up.
    """

    # --- switches ---------------------------------------------------------
    enabled: bool = True
    # Redis flag; when it reads "1" the evaluator idles without a deploy.
    kill_key: str = "lookout.kill"
    max_watches: int = 6
    tick_hz: float = 2.0
    publish_hz: float = 1.0
    # Extra detection pass for non-vehicle subjects, from the worker thread (Hz).
    detect_hz: float = 1.5
    extra_detect_classes: str = "0,1,2,3,5,7,16"

    # --- observation validity --------------------------------------------
    stale_after_sec: float = 3.0
    unavailable_after_sec: float = 20.0
    # An armed watch must confirm the state it was armed in before it may report
    # a change; otherwise an object armed outside the frame fires "gone" at once.
    arm_confirm_sec: float = 1.0
    arm_timeout_sec: float = 20.0
    # Sliding window of readings used for the "absent for N seconds" test.
    window_sec: float = 60.0
    # Fraction of that window that must carry usable evidence.
    min_evidence_fraction: float = 0.6

    # --- presence decision -------------------------------------------------
    presence_hi: float = 0.62
    presence_lo: float = 0.50
    # Fraction of an object box that a matching detection must cover.
    object_cover_hi: float = 0.25
    # Fraction of a zone that detections must cover to count as occupied.
    zone_cover_hi: float = 0.18
    zone_cover_lo: float = 0.06
    min_det_conf: float = 0.35
    # An object covering this much of the region is blocking the view.
    occluder_cover_limit: float = 0.55
    # Global brightness change (0-1) that invalidates an "absent" reading.
    illumination_limit: float = 0.30

    # --- transition hold times --------------------------------------------
    gone_hold_sec: float = 4.0
    return_hold_sec: float = 2.0
    occupied_hold_sec: float = 20.0
    empty_hold_sec: float = 6.0
    alert_cooldown_sec: float = 60.0

    # --- arbitration (offline comparison only) ----------------------------
    decider: str = "rule"
    arbiter_timeout_sec: float = 12.0
    vlm_model: str = ""
    jev_model: str = "jev-latest"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    jev_input_usd_per_mtok: float = 0.042
    # Cap on ambiguous episodes written to the arbitration log per hour.
    ambiguity_log_per_hour: int = 120

    # --- evidence ---------------------------------------------------------
    storage_root: str = "/app/data/lookout"
    ring_frames: int = 10
    ring_interval_sec: float = 1.5
    ring_width: int = 960
    jpeg_quality: int = 82
    # Reference drift rate while a reading is confidently nominal.
    reference_drift: float = 0.02
    # Similarity below which a region resembles no known state at all.
    decisive_similarity: float = 0.55
    # Reading confidence required before it may teach the reference model.
    present_drift_floor: float = 0.72
    absent_drift_ceiling: float = 0.35
    # Confident observations required before a second reference may veto a reading.
    min_opposite_samples: int = 3
    # Minimum spacing between "cannot see" notes for one watch.
    note_cooldown_sec: float = 30.0
    # Clean-frame snapshot published for the web arming UI (Hz, 0 disables).
    frame_publish_hz: float = 0.5
    # A public street camera: evidence is a liability, so the default is short.
    evidence_max_events: int = 400
    evidence_max_age_days: float = 2.0

    # --- delivery and access ---------------------------------------------
    # Shared secret required by every /lookout route. Empty disables the UI.
    access_token: str = ""
    # POST target for alerting events (ntfy/Slack/Discord compatible).
    webhook_url: str = ""
    webhook_timeout_sec: float = 8.0

    @classmethod
    def from_env(cls) -> LookoutOptions:
        defaults = cls()
        values: dict[str, object] = {}
        for spec in fields(cls):
            name = f"LOOKOUT_{spec.name.upper()}"
            default = getattr(defaults, spec.name)
            if spec.name in _STR_FIELDS:
                values[spec.name] = os.getenv(name, default)
            elif isinstance(default, bool):
                values[spec.name] = _env_bool(name, default)
            elif isinstance(default, int):
                values[spec.name] = _env_int(name, default)
            else:
                values[spec.name] = _env_float(name, default)
        return cls(**values)  # type: ignore[arg-type]

    @property
    def evidence_dir(self) -> str:
        return os.path.join(self.storage_root, "evidence")

    @property
    def ring_root(self) -> str:
        return os.path.join(self.storage_root, "rings")

    @property
    def watches_path(self) -> str:
        return os.path.join(self.storage_root, "watches.json")

    @property
    def events_path(self) -> str:
        return os.path.join(self.storage_root, "events.jsonl")

    @property
    def arbitration_log(self) -> str:
        return os.path.join(self.storage_root, "arbitration.jsonl")

    @property
    def detect_classes(self) -> list[int]:
        out: list[int] = []
        for token in self.extra_detect_classes.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                out.append(int(token))
            except ValueError:
                continue
        return out
