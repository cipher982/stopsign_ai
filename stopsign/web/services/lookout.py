"""Lookout storage and access control for the web container.

The web container has no OpenCV and must not import the evaluator, so this module
owns the thin slice of Lookout the UI needs: the watch file, the latest clean
frame, live state from Redis, and the event log. The JSON shape written here is
read by ``stopsign.lookout.store.watch_from_dict`` in the analyzer; keep the two
in step.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any
from typing import Optional

import redis

from stopsign.frame_codec import unpack_frame
from stopsign.settings import REDIS_URL

logger = logging.getLogger(__name__)

STORAGE_ROOT = os.getenv("LOOKOUT_STORAGE_ROOT", "/app/data/lookout")
CLEAN_FRAME_KEY = os.getenv("LOOKOUT_FRAME_KEY", "lookout.frame")
STATE_KEY = os.getenv("LOOKOUT_STATE_KEY", "lookout.state")
HEALTH_KEY = os.getenv("LOOKOUT_HEALTH_KEY", "lookout.health")
DELIVERY_KEY = os.getenv("LOOKOUT_DELIVERY_KEY", "lookout.delivery")
ACCESS_TOKEN = os.getenv("LOOKOUT_ACCESS_TOKEN", "")
COOKIE_NAME = "lookout_key"

# Which state the operator expects to find when arming. Mirrors
# ARMED_OCCUPIED_DEFAULT in stopsign/lookout/manager.py.
ARMED_DEFAULT = {"gone": True, "cleared": True, "occupied": False, "back": False}

CONDITIONS = {
    "gone": ("object", "Tell me when it is gone"),
    "occupied": ("zone", "Tell me when this area is occupied"),
    "cleared": ("zone", "Tell me when this area is clear"),
}

SUBJECTS = ["", "car", "truck", "bus", "person", "bicycle", "dog", "cat"]


def access_token() -> str:
    return ACCESS_TOKEN


def authorized(request) -> bool:
    """Fail closed: no configured token means the feature is not exposed."""
    if not ACCESS_TOKEN:
        return False
    supplied = request.query_params.get("key") or request.cookies.get(COOKIE_NAME) or ""
    return supplied == ACCESS_TOKEN


def _redis() -> Optional[redis.Redis]:
    try:
        return redis.from_url(REDIS_URL, socket_connect_timeout=2)
    except Exception as exc:  # a broken UI must not take the site down
        logger.warning("Lookout Redis unavailable: %s", exc)
        return None


def watches_path() -> str:
    return os.path.join(STORAGE_ROOT, "watches.json")


def events_path() -> str:
    return os.path.join(STORAGE_ROOT, "events.jsonl")


def load_watches() -> list[dict[str, Any]]:
    try:
        with open(watches_path(), encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def save_watches(watches: list[dict[str, Any]]) -> None:
    directory = os.path.dirname(watches_path())
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{watches_path()}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(watches, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, watches_path())


def latest_frame() -> Optional[dict[str, Any]]:
    """The most recent clean processed frame, straight from the analyzer."""
    client = _redis()
    if client is None:
        return None
    try:
        raw = client.get(CLEAN_FRAME_KEY)
    except Exception as exc:
        logger.warning("Lookout frame read failed: %s", exc)
        return None
    if not raw:
        return None
    decoded = unpack_frame(bytes(raw))
    if decoded is None or not decoded.payload:
        return None
    metadata = decoded.metadata
    return {
        "jpeg": decoded.payload,
        "width": int(metadata.get("w") or 0),
        "height": int(metadata.get("h") or 0),
        "capture_ts": float(metadata.get("ts") or 0.0),
        "age_sec": max(0.0, time.time() - float(metadata.get("ts") or 0.0)),
    }


def live_state() -> dict[str, Any]:
    client = _redis()
    if client is None:
        return {"available": False, "watches": {}, "health": {}, "deliveries": {}}
    try:
        watches = client.hgetall(STATE_KEY) or {}
        health = client.get(HEALTH_KEY)
        deliveries = client.hgetall(DELIVERY_KEY) or {}
    except Exception as exc:
        logger.warning("Lookout state read failed: %s", exc)
        return {"available": False, "watches": {}, "health": {}, "deliveries": {}}
    return {
        "available": True,
        "watches": {_key(key): _json(value) for key, value in watches.items()},
        "health": _json(health) if health else {},
        "deliveries": {_key(key): _json(value) for key, value in deliveries.items()},
    }


def recent_events(limit: int = 40) -> list[dict[str, Any]]:
    try:
        with open(events_path(), encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in reversed(lines[-limit * 3 :]):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            events.append(row)
        if len(events) >= limit:
            break
    return events


def arm_watch(
    *,
    box: list[float],
    condition: str,
    label: str,
    subject: str = "",
    armed: str = "auto",
) -> dict[str, Any]:
    """Create a watch from the exact frame currently on screen.

    The reference image is the frame the operator drew on — retained whole, so
    the evaluator crops exactly what they saw rather than whatever the stream
    happened to be showing seconds later.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    frame = latest_frame()
    if frame is None:
        raise RuntimeError("no clean frame available yet; try again in a moment")
    x1, y1, x2, y2 = (float(value) for value in box)
    width, height = frame["width"], frame["height"]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box is outside the frame")
    if (x2 - x1) < 24 or (y2 - y1) < 24:
        raise ValueError("box is too small to watch")

    watch_id = uuid.uuid4().hex[:12]
    references_dir = os.path.join(STORAGE_ROOT, "references")
    os.makedirs(references_dir, exist_ok=True)
    relative = os.path.join("references", f"{watch_id}.jpg")
    with open(os.path.join(STORAGE_ROOT, relative), "wb") as handle:
        handle.write(frame["jpeg"])

    if armed == "auto":
        armed_occupied = ARMED_DEFAULT[condition]
    else:
        armed_occupied = armed == "occupied"

    row = {
        "id": watch_id,
        "kind": CONDITIONS[condition][0],
        "label": (label or "watch").strip()[:60],
        "box": [x1, y1, x2, y2],
        "condition": condition,
        "created_at": time.time(),
        "created_by": "web",
        "reference_size": [width, height],
        "reference_wall_ts": time.time(),
        "reference_capture_ts": frame["capture_ts"],
        "subject": subject if subject in SUBJECTS else "",
        "reference_image": relative,
        "reference_histogram": [],
        "armed_occupied": bool(armed_occupied),
        "options": {"armed_choice": armed, "frame_age_sec": round(frame["age_sec"], 2)},
        "active": True,
        "revision": int(time.time() * 1000),
    }
    watches = [item for item in load_watches() if item.get("id") != watch_id]
    watches.append(row)
    save_watches(watches)
    return row


def stop_watch(watch_id: str) -> bool:
    watches = load_watches()
    remaining = [item for item in watches if item.get("id") != watch_id]
    if len(remaining) == len(watches):
        return False
    save_watches(remaining)
    client = _redis()
    if client is not None:
        try:
            client.hdel(STATE_KEY, watch_id)
            client.hdel("lookout.references", watch_id)
        except Exception:
            pass
    return True


def evidence_url(evidence_dir: Optional[str], filename: str = "strip.jpg") -> str:
    """Map a stored evidence directory to a servable URL."""
    if not evidence_dir:
        return ""
    name = os.path.basename(str(evidence_dir).rstrip("/"))
    if not name:
        return ""
    return f"/lookout-evidence/{name}/{filename}"


def _key(raw: Any) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _json(raw: Any) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}
