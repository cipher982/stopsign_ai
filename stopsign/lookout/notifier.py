"""Out-of-page delivery for alerting events.

Review was blunt about this: an alert that only exists in an open browser tab is
not a product. The page can be closed, so the delivery path is a webhook — one
POST that any receiver David already reads can accept (ntfy for phone push,
Slack/Discord for a channel, or a local endpoint for tests).

Delivery is best-effort and never blocks evaluation: it runs on the worker's
side-effect pool, and a failure is recorded on the event rather than raised.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Optional

from stopsign.lookout.types import LookoutEvent


class WebhookNotifier:
    """Posts a multipart alert with the decisive frame attached."""

    def __init__(self, url: str, *, timeout_sec: float = 8.0, public_base_url: str = "") -> None:
        self.url = (url or "").strip()
        self.timeout_sec = timeout_sec
        self.public_base_url = public_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def notify(self, event: LookoutEvent, *, image_path: Optional[str] = None, watch_url: str = "") -> str:
        """Return a short outcome string: 'sent', 'disabled', or 'failed: ...'."""
        if not self.enabled:
            return "disabled"
        text = self._message(event, watch_url)
        fields = {
            "title": f"Lookout: {event.label} is {event.wording}",
            "message": text,
            "text": text,
            "event_id": event.id,
            "watch_id": event.watch_id,
            "label": event.label,
            "condition": event.condition.value,
            "state": event.to_state,
            "capture_ts": f"{event.capture_ts:.3f}",
            "confidence": f"{event.present_prob:.3f}",
            "evidence_ok": "yes" if event.evidence_ok else "no",
            "watch_url": watch_url or self.public_base_url,
        }
        body, content_type = _multipart(fields, image_path)
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": content_type, "User-Agent": "lookout/1"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                return f"sent:{response.status}"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return f"failed: {exc}"

    def _message(self, event: LookoutEvent, watch_url: str) -> str:
        when = time.strftime("%H:%M:%S", time.localtime(event.capture_ts))
        lines = [
            f"{event.label} is {event.wording} ({when}).",
            f"watched as: {event.condition.question.lower()}",
        ]
        if not event.evidence_ok:
            lines.append("partial view: this reading comes from incomplete evidence.")
        lines.append(f"confidence {event.present_prob:.0%}, evidence {event.evidence_fraction:.0%} of the window.")
        if watch_url or self.public_base_url:
            lines.append(watch_url or self.public_base_url)
        return "\n".join(lines)


def _multipart(fields: dict[str, str], image_path: Optional[str]) -> tuple[bytes, str]:
    """Build a multipart/form-data body: plain fields plus an optional file.

    Multipart rather than JSON because the useful receivers all accept a file
    upload, and ntfy accepts the same shape while ignoring what it does not use.
    """
    boundary = f"----lookout{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        if value is None:
            continue
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.append(f"{value}\r\n".encode())
    if image_path:
        try:
            with open(image_path, "rb") as handle:
                payload = handle.read()
        except OSError:
            payload = b""
        if payload:
            chunks.append(f"--{boundary}\r\n".encode())
            chunks.append(
                b'Content-Disposition: form-data; name="file"; filename="evidence.jpg"\r\n'
                b"Content-Type: image/jpeg\r\n\r\n"
            )
            chunks.append(payload)
            chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def read_json(path: str, default: object) -> object:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default
