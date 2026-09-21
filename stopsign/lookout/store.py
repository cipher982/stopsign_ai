"""Durable, boring storage for watch definitions and events.

Review's call: the spike does not need schema migrations. Watches live in one
JSON file on the shared volume, events append to one JSONL, and live state lives
in Redis where the web already reads it. Both containers mount the same volume,
writes are atomic, and a corrupt file degrades to "no watches" instead of taking
the analyzer down.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from typing import Iterable
from typing import Optional

from stopsign.lookout.options import LookoutOptions
from stopsign.lookout.types import Condition
from stopsign.lookout.types import LookoutEvent
from stopsign.lookout.types import Watch
from stopsign.lookout.types import WatchKind

_WATCH_FIELDS = (
    "id",
    "kind",
    "label",
    "box",
    "condition",
    "created_at",
    "created_by",
    "reference_size",
    "reference_wall_ts",
    "reference_capture_ts",
    "subject",
    "reference_image",
    "armed_occupied",
    "options",
    "active",
    "revision",
)


def watch_to_dict(watch: Watch) -> dict[str, Any]:
    payload = {field: getattr(watch, field) for field in _WATCH_FIELDS}
    payload["kind"] = watch.kind.value
    payload["condition"] = watch.condition.value
    payload["box"] = [float(value) for value in watch.box]
    payload["reference_size"] = [int(value) for value in watch.reference_size]
    return payload


def watch_from_dict(payload: dict[str, Any]) -> Optional[Watch]:
    """Parse one stored watch. Returns None for anything malformed."""
    try:
        box = payload["box"]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        reference_size = payload.get("reference_size") or (0, 0)
        return Watch(
            id=str(payload["id"]),
            kind=WatchKind(str(payload["kind"])),
            label=str(payload.get("label") or "watch"),
            box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            condition=Condition(str(payload["condition"])),
            created_at=float(payload.get("created_at") or 0.0),
            created_by=str(payload.get("created_by") or "web"),
            reference_size=(int(reference_size[0]), int(reference_size[1])),
            reference_wall_ts=float(payload.get("reference_wall_ts") or 0.0),
            reference_capture_ts=float(payload.get("reference_capture_ts") or 0.0),
            subject=str(payload.get("subject") or ""),
            reference_image=str(payload.get("reference_image") or ""),
            armed_occupied=bool(payload.get("armed_occupied", True)),
            options=dict(payload.get("options") or {}),
            active=bool(payload.get("active", True)),
            revision=int(payload.get("revision") or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None


class WatchStore:
    """File-backed watch definitions and event history on the shared volume."""

    def __init__(self, options: LookoutOptions, *, max_events: int = 4000) -> None:
        self.options = options
        self.max_events = max_events
        os.makedirs(options.storage_root, exist_ok=True)
        self._writes_since_compact = 0

    # ------------------------------------------------------------- watches
    def revision(self) -> str:
        """A cheap change token for the watches file."""
        try:
            stat = os.stat(self.options.watches_path)
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            return "missing"

    def load(self) -> list[Watch]:
        """Read the armed watches. A half-written or corrupt file reads as empty.

        The web writes this file atomically, so a failure here means something
        else is wrong with the volume — better an empty set than a crash in the
        analyzer's worker loop.
        """
        try:
            with open(self.options.watches_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return []
        if not isinstance(payload, list):
            return []
        watches: list[Watch] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            parsed = watch_from_dict(row)
            if parsed is not None and parsed.active:
                watches.append(parsed)
        return watches

    def save(self, watches: Iterable[Watch]) -> None:
        rows = [watch_to_dict(watch) for watch in watches]
        _atomic_write_json(self.options.watches_path, rows)

    def upsert(self, watch: Watch) -> list[Watch]:
        watches = [item for item in self.load() if item.id != watch.id]
        watches.append(watch)
        self.save(watches)
        return watches

    def remove(self, watch_id: str) -> list[Watch]:
        watches = [item for item in self.load() if item.id != watch_id]
        self.save(watches)
        return watches

    # -------------------------------------------------------------- events
    def append_event(self, event: LookoutEvent) -> None:
        row = event.to_row()
        row["wording"] = event.wording
        try:
            with open(self.options.events_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        except OSError:
            return
        self._writes_since_compact += 1
        if self._writes_since_compact >= 200:
            self._writes_since_compact = 0
            self.compact_events()

    def compact_events(self) -> None:
        path = self.options.events_path
        try:
            with open(path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return
        if len(lines) <= self.max_events:
            return
        keep = lines[-self.max_events :]
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.writelines(keep)
        except OSError:
            return

    def load_events(self, limit: int = 40) -> list[dict[str, Any]]:
        try:
            with open(self.options.events_path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return []
        events: list[dict[str, Any]] = []
        for line in reversed(lines[-max(limit, 1) * 4 :]):
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


def _atomic_write_json(path: str, payload: object) -> None:
    """Write via a temporary file so a reader never sees a half-written file."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
