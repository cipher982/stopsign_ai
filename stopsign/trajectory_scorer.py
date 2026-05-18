"""Completed-trajectory scoring for stop-sign passes.

This module is deliberately pure: it reads recorded bbox samples and geometry
and returns what it would have decided. The online detector can run it in
shadow mode without changing pass-recording behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from typing import Sequence

import cv2
import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class TrajectoryScore:
    would_record_pass: bool
    reason: str
    time_in_zone: float = 0.0
    min_speed: float | None = None
    stop_duration: float = 0.0
    approach_progress: float = 0.0
    zone_sample_count: int = 0


@dataclass(frozen=True)
class _Sample:
    t: float
    x: float
    y: float
    x1: float
    y1: float
    x2: float
    y2: float
    raw_speed: float


def _polygon_intersects(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    try:
        area, _ = cv2.intersectConvexConvex(poly1.astype(np.float32), poly2.astype(np.float32))
        return area > 0
    except cv2.error:
        return False


def _bbox_polygon(sample: _Sample) -> np.ndarray:
    return np.array(
        [
            [sample.x1, sample.y1],
            [sample.x2, sample.y1],
            [sample.x2, sample.y2],
            [sample.x1, sample.y2],
        ],
        dtype=np.float32,
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(np.percentile(np.array(values, dtype=float), percentile))


def _parse_samples(samples: Iterable[Sequence[float]], sample_schema: Sequence[str]) -> list[_Sample]:
    idx = {name: pos for pos, name in enumerate(sample_schema)}
    required = ("t", "x", "y", "x1", "y1", "x2", "y2", "raw_speed")
    if any(name not in idx for name in required):
        return []

    parsed: list[_Sample] = []
    for row in samples:
        try:
            parsed.append(
                _Sample(
                    t=float(row[idx["t"]]),
                    x=float(row[idx["x"]]),
                    y=float(row[idx["y"]]),
                    x1=float(row[idx["x1"]]),
                    y1=float(row[idx["y1"]]),
                    x2=float(row[idx["x2"]]),
                    y2=float(row[idx["y2"]]),
                    raw_speed=float(row[idx["raw_speed"]]),
                )
            )
        except (IndexError, TypeError, ValueError):
            continue
    return parsed


def _best_zone_interval(samples: Sequence[_Sample], stop_zone: np.ndarray) -> tuple[int, int] | None:
    intervals: list[tuple[int, int]] = []
    start_idx: int | None = None

    for idx, sample in enumerate(samples):
        in_zone = _polygon_intersects(_bbox_polygon(sample), stop_zone)
        if in_zone:
            if start_idx is None:
                start_idx = idx
        elif start_idx is not None:
            intervals.append((start_idx, idx - 1))
            start_idx = None

    if start_idx is not None:
        intervals.append((start_idx, len(samples) - 1))

    if not intervals:
        return None

    return max(
        intervals,
        key=lambda pair: (
            samples[pair[1]].t - samples[pair[0]].t,
            pair[1] - pair[0],
        ),
    )


def score_samples(
    samples: Iterable[Sequence[float]],
    sample_schema: Sequence[str],
    stop_zone: np.ndarray,
    pre_stop_line: np.ndarray,
    *,
    stop_speed_threshold: float,
    min_zone_time_sec: float = 0.2,
    max_zone_time_sec: float = 60.0,
    min_approach_progress_px: float = 20.0,
    approach_history_samples: int = 30,
) -> TrajectoryScore:
    """Score whether a completed bbox trajectory looks like a vehicle pass."""
    parsed = _parse_samples(samples, sample_schema)
    if len(parsed) < 2:
        return TrajectoryScore(False, "too_few_samples")

    stop_zone = np.asarray(stop_zone, dtype=np.float32).reshape(-1, 2)
    pre_stop_line = np.asarray(pre_stop_line, dtype=np.float32).reshape(-1, 2)
    if stop_zone.shape[0] < 3 or pre_stop_line.shape[0] != 2:
        return TrajectoryScore(False, "bad_geometry")

    interval = _best_zone_interval(parsed, stop_zone)
    if interval is None:
        return TrajectoryScore(False, "never_in_zone")

    start_idx, end_idx = interval
    zone_samples = parsed[start_idx : end_idx + 1]
    time_in_zone = max(0.0, parsed[end_idx].t - parsed[start_idx].t)

    if time_in_zone < min_zone_time_sec:
        return TrajectoryScore(False, "zone_too_short", time_in_zone=time_in_zone)
    if time_in_zone > max_zone_time_sec:
        return TrajectoryScore(False, "zone_timeout", time_in_zone=time_in_zone)

    stop_center = np.mean(stop_zone, axis=0)
    pre_stop_center = np.mean(pre_stop_line, axis=0)
    approach_vector = stop_center - pre_stop_center
    approach_norm = float(np.linalg.norm(approach_vector))
    if approach_norm < 1e-6:
        return TrajectoryScore(False, "bad_approach", time_in_zone=time_in_zone)

    unit_approach = approach_vector / approach_norm
    progress_start_idx = max(0, start_idx - approach_history_samples)
    start = np.array([parsed[progress_start_idx].x, parsed[progress_start_idx].y], dtype=float)
    end = np.array([parsed[end_idx].x, parsed[end_idx].y], dtype=float)
    approach_progress = float(np.dot(end - start, unit_approach))

    if approach_progress < min_approach_progress_px:
        return TrajectoryScore(
            False,
            "insufficient_progress",
            time_in_zone=time_in_zone,
            approach_progress=approach_progress,
            zone_sample_count=len(zone_samples),
        )

    speeds = [sample.raw_speed for sample in zone_samples]
    min_speed = _percentile(speeds, 5)
    stop_duration = 0.0
    prev_t: float | None = None
    for sample in zone_samples:
        if prev_t is not None and sample.raw_speed <= stop_speed_threshold:
            stop_duration += max(0.0, sample.t - prev_t)
        prev_t = sample.t

    return TrajectoryScore(
        True,
        "pass",
        time_in_zone=time_in_zone,
        min_speed=min_speed,
        stop_duration=stop_duration,
        approach_progress=approach_progress,
        zone_sample_count=len(zone_samples),
    )


def score_raw_payload(
    raw_payload: dict,
    *,
    min_approach_progress_px: float = 20.0,
    approach_history_samples: int = 30,
) -> TrajectoryScore:
    """Score a stored VehiclePassRaw payload."""
    config = raw_payload.get("config_snapshot") or {}
    detection = config.get("stopsign_detection") or {}
    video = config.get("video_processing") or {}
    dimensions = raw_payload.get("dimensions") or {}
    raw_dimensions = dimensions.get("raw") or {}

    raw_width = float(raw_dimensions.get("width") or 0)
    raw_height = float(raw_dimensions.get("height") or 0)
    scale = float(video.get("scale") or 1.0)
    crop_top = float(video.get("crop_top") or 0.0)
    crop_side = float(video.get("crop_side") or 0.0)

    def raw_to_processing(point: Sequence[float]) -> Point:
        x, y = float(point[0]), float(point[1])
        if raw_width <= 0 or raw_height <= 0:
            return x, y
        return ((x - raw_width * crop_side) * scale, (y - raw_height * crop_top) * scale)

    stop_zone = np.array([raw_to_processing(point) for point in detection.get("stop_zone", [])], dtype=np.float32)
    pre_stop_line = np.array(
        [raw_to_processing(point) for point in detection.get("pre_stop_line", [])],
        dtype=np.float32,
    )

    return score_samples(
        raw_payload.get("samples") or [],
        raw_payload.get("sample_schema") or [],
        stop_zone,
        pre_stop_line,
        stop_speed_threshold=float(detection.get("stop_speed_threshold") or 20.0),
        min_zone_time_sec=float(detection.get("in_zone_time_threshold") or 0.2),
        min_approach_progress_px=min_approach_progress_px,
        approach_history_samples=approach_history_samples,
    )
