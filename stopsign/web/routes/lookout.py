"""Lookout routes: arm a watch from an exact frame, and see what it reported.

Token-gated on purpose. This camera watches a public street; "anyone may arm an
arbitrary watch on any region and read the evidence" is surveillance tooling, not
a demo. With no token configured the routes answer 404 rather than degrading to
open access.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import Response

from stopsign.web.app import templates
from stopsign.web.services import lookout as lookout_service

logger = logging.getLogger(__name__)

router = APIRouter()

EVIDENCE_FILENAME_ALLOWLIST = {"strip.jpg", "full.jpg", "target.jpg", "summary.json"}


def _require_access(request: Request) -> None:
    if not lookout_service.access_token():
        raise HTTPException(status_code=404, detail="Not found")
    if lookout_service.authorized(request):
        return
    supplied = request.query_params.get("key")
    if supplied:
        raise HTTPException(status_code=403, detail="Bad key")
    raise HTTPException(status_code=403, detail="Access key required")


def _merge(watches: list[dict], state: dict) -> list[dict]:
    """Definitions with their live status, newest watch first."""
    merged = []
    live = state.get("watches", {})
    for row in watches:
        status = live.get(row.get("id"), {})
        merged.append(
            {
                **{
                    key: row.get(key)
                    for key in ("id", "label", "kind", "condition", "subject", "box", "armed_occupied")
                },
                "reference_size": row.get("reference_size"),
                "reference_capture_ts": row.get("reference_capture_ts"),
                "created_at": row.get("created_at"),
                "status": status or None,
            }
        )
    merged.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return merged


@router.get("/lookout")
async def lookout_page(request: Request):
    """Serve the arming UI, and trade a ?key= for a session cookie.

    The cookie is what the page's own fetches authenticate with, so it has to be
    set on the same response that renders the page. Serving the page on a
    ?key= link without it produces a page whose every request 403s.
    """
    token = lookout_service.access_token()
    if not token:
        raise HTTPException(status_code=404, detail="Not found")
    supplied = request.query_params.get("key") or request.cookies.get(lookout_service.COOKIE_NAME) or ""
    if supplied != token:
        raise HTTPException(status_code=403, detail="Access key required")
    response = templates.TemplateResponse(
        "lookout.html",
        {
            "request": request,
            "conditions": lookout_service.CONDITIONS,
            "subjects": lookout_service.SUBJECTS,
            "meta_description": "Operator watches on the Crestwood camera.",
            "canonical_url": "https://crestwoodstopsign.com/lookout",
        },
    )
    if request.query_params.get("key"):
        response.set_cookie(
            lookout_service.COOKIE_NAME,
            token,
            httponly=True,
            samesite="lax",
            # Secure only over TLS, so a local http check does not silently
            # produce a session-less page.
            secure=request.url.scheme == "https",
            max_age=60 * 60 * 24 * 30,
        )
    return response


@router.get("/api/lookout/state")
async def lookout_state(request: Request):
    _require_access(request)
    state = lookout_service.live_state()
    events = lookout_service.recent_events(limit=25)
    for event in events:
        event["evidence"] = {
            "strip": lookout_service.evidence_url(event.get("evidence_dir"), "strip.jpg"),
            "full": lookout_service.evidence_url(event.get("evidence_dir"), "full.jpg"),
            "target": lookout_service.evidence_url(event.get("evidence_dir"), "target.jpg"),
        }
    watches = _merge(lookout_service.load_watches(), state)
    return JSONResponse(
        {
            "watches": watches,
            "events": events,
            "health": state.get("health", {}),
            "redis_available": state.get("available", False),
        }
    )


@router.get("/api/lookout/frame")
async def lookout_frame(request: Request):
    """The exact clean frame an operator arms against."""
    _require_access(request)
    frame = lookout_service.latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="no clean frame available")
    return Response(
        content=frame["jpeg"],
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Width": str(frame["width"]),
            "X-Frame-Height": str(frame["height"]),
            "X-Frame-Age": f"{frame['age_sec']:.2f}",
            "X-Frame-Capture-Ts": f"{frame['capture_ts']:.3f}",
        },
    )


@router.post("/api/lookout/watches")
async def create_watch(request: Request):
    _require_access(request)
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    box = payload.get("box")
    if not isinstance(box, list) or len(box) != 4:
        raise HTTPException(status_code=400, detail="box must be [x1,y1,x2,y2]")
    try:
        row = lookout_service.arm_watch(
            box=[float(value) for value in box],
            condition=str(payload.get("condition") or "gone"),
            label=str(payload.get("label") or "watch"),
            subject=str(payload.get("subject") or ""),
            armed=str(payload.get("armed") or "auto"),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    logger.info("Lookout watch armed: %s (%s)", row["id"], row["condition"])
    return JSONResponse({"watch": row})


@router.post("/api/lookout/watches/{watch_id}/stop")
async def stop_watch(request: Request, watch_id: str):
    _require_access(request)
    if not lookout_service.stop_watch(watch_id):
        raise HTTPException(status_code=404, detail="no such watch")
    return JSONResponse({"stopped": watch_id})


@router.get("/lookout-evidence/{evidence_id}/{filename}")
async def lookout_evidence(request: Request, evidence_id: str, filename: str):
    """Serve one evidence artefact. Names are validated, not sanitised."""
    _require_access(request)
    if filename not in EVIDENCE_FILENAME_ALLOWLIST:
        raise HTTPException(status_code=404, detail="Not found")
    base = os.path.realpath(os.path.join(lookout_service.STORAGE_ROOT, "evidence"))
    if not evidence_id.isalnum() and not evidence_id.replace("-", "").isalnum():
        raise HTTPException(status_code=404, detail="Not found")
    path = os.path.realpath(os.path.join(base, evidence_id, filename))
    if not path.startswith(base + os.sep) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Not found")
    if filename.endswith(".json"):
        return FileResponse(path, media_type="application/json")
    return FileResponse(path, media_type="image/jpeg")
