"""Helpers for rendering vehicle-pass cards."""

from __future__ import annotations

import time
from collections.abc import Sequence
from zoneinfo import ZoneInfo

from stopsign.web.services.images import resolve_card_thumbnail_url
from stopsign.web.services.images import resolve_image_url
from stopsign.web.services.scoring import get_speed_color
from stopsign.web.services.scoring import get_time_color
from stopsign.web.services.scoring import get_verdict_color


def format_pass_item(pass_data, vehicle_attrs=None, score=None):
    """Build template context dict for a single pass card."""
    image_url = resolve_image_url(pass_data.image_path)
    thumbnail_url = resolve_card_thumbnail_url(pass_data.image_path)

    if hasattr(pass_data, "timestamp") and pass_data.timestamp:
        utc_time = pass_data.timestamp.replace(tzinfo=ZoneInfo("UTC"))
        chicago_time = utc_time.astimezone(ZoneInfo("America/Chicago"))
        seconds_ago = int(time.time() - utc_time.timestamp())
        if seconds_ago < 60:
            time_str = "just now"
        elif seconds_ago < 3600:
            mins = seconds_ago // 60
            time_str = f"{mins} min{'s' if mins != 1 else ''} ago"
        elif seconds_ago < 86400:
            hrs = seconds_ago // 3600
            time_str = f"{hrs} hr{'s' if hrs != 1 else ''} ago"
        else:
            days = seconds_ago // 86400
            time_str = f"{days} day{'s' if days != 1 else ''} ago"
        time_absolute = chicago_time.strftime("%-I:%M %p")
    else:
        time_str = "N/A"
        time_absolute = ""

    badge_text = ""
    if vehicle_attrs and hasattr(pass_data, "id"):
        attrs = vehicle_attrs.get(pass_data.id)
        if attrs:
            parts = []
            type_color = " ".join(filter(None, [attrs.get("color"), attrs.get("vehicle_type")]))
            if type_color:
                parts.append(type_color.title())
            make_model = attrs.get("make_model")
            if make_model:
                parts.append(make_model)
            badge_text = " | ".join(parts) if parts else ""

    clip_url = None
    clip_status = getattr(pass_data, "clip_status", None)
    clip_path = getattr(pass_data, "clip_path", None)
    if clip_path and clip_status in ("ready", "local"):
        clip_url = f"/clips/{clip_path}"

    verdict = (score or {}).get("verdict", "Rolling Stop")
    stop_score = (score or {}).get("stop_score", 0)

    return {
        "pass_id": getattr(pass_data, "id", None),
        "image_url": image_url,
        "thumbnail_url": thumbnail_url,
        "time_str": time_str,
        "time_absolute": time_absolute,
        "min_speed": pass_data.min_speed,
        "time_in_zone": pass_data.time_in_zone,
        "speed_color": get_speed_color(pass_data.min_speed),
        "time_color": get_time_color(pass_data.time_in_zone),
        "verdict": verdict,
        "verdict_color": get_verdict_color(verdict),
        "stop_score": stop_score,
        "badge_text": badge_text,
        "clip_url": clip_url,
    }


def build_recent_pass_items(db, recent_passes: Sequence) -> list[dict]:
    pass_ids = [p.id for p in recent_passes]
    vehicle_attrs = db.get_vehicle_attributes_for_passes(pass_ids)
    scores = db.get_pass_stop_scores(
        [
            {
                "time_in_zone": p.time_in_zone,
                "min_speed": p.min_speed,
                "stop_duration": p.stop_duration,
                "decel_score": getattr(p, "decel_score", None),
                "track_quality": getattr(p, "track_quality", None),
                "entry_speed": getattr(p, "entry_speed", None),
            }
            for p in recent_passes
        ]
    )
    return [format_pass_item(p, vehicle_attrs, scores[i]) for i, p in enumerate(recent_passes)]
