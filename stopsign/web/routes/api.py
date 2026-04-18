# ruff: noqa: E501
"""API routes — HTMX partials returning HTML fragments."""

import logging
import time

from fastapi import APIRouter
from fastapi import Request

from stopsign.database import Database
from stopsign.settings import DB_URL
from stopsign.web.app import templates
from stopsign.web.services.images import resolve_image_url
from stopsign.web.services.insights import get_real_insights
from stopsign.web.services.passes import build_recent_pass_items
from stopsign.web.services.passes import format_pass_item
from stopsign.web.services.scoring import COLOR_MAP

logger = logging.getLogger(__name__)

router = APIRouter()


def _ensure_db(request: Request):
    if not hasattr(request.app.state, "db"):
        request.app.state.db = Database(db_url=DB_URL)
    return request.app.state.db


@router.get("/api/live-stats")
async def get_live_stats(request: Request):
    try:
        db = _ensure_db(request)
        total_passes_24h = db.get_total_passes_last_24h()
        recent_passes = db.get_recent_vehicle_passes(limit=100)

        if recent_passes:
            compliant_count = sum(1 for p in recent_passes if p.time_in_zone >= 2.0)
            compliance_rate = round((compliant_count / len(recent_passes)) * 100)
        else:
            compliance_rate = 0

        last_detection = "N/A"
        if recent_passes:
            last_time = recent_passes[0].timestamp
            minutes_ago = int((time.time() - last_time.timestamp()) / 60)
            if minutes_ago < 60:
                last_detection = f"{minutes_ago}m ago"
            else:
                hours_ago = int(minutes_ago / 60)
                last_detection = f"{hours_ago}h ago"

        trend_arrow = "\u2192"
        if len(recent_passes) >= 20:
            recent_10 = recent_passes[:10]
            prev_10 = recent_passes[10:20]
            recent_compliance = sum(1 for p in recent_10 if p.time_in_zone >= 2.0) / 10
            prev_compliance = sum(1 for p in prev_10 if p.time_in_zone >= 2.0) / 10
            if recent_compliance > prev_compliance:
                trend_arrow = "\u2197"
            elif recent_compliance < prev_compliance:
                trend_arrow = "\u2198"

        rotating_insight = get_real_insights(db, recent_passes)
        violation_count = total_passes_24h - int(total_passes_24h * compliance_rate / 100)

        return templates.TemplateResponse(
            "partials/live_stats.html",
            {
                "request": request,
                "compliance_rate": compliance_rate,
                "violation_count": violation_count,
                "vehicle_count": total_passes_24h,
                "last_detection": last_detection,
                "trend_arrow": trend_arrow,
                "rotating_insight": rotating_insight,
            },
        )
    except Exception as e:
        logger.error(f"Error in get_live_stats: {str(e)}")
        from fastapi.responses import HTMLResponse

        return HTMLResponse('<div id="statsError"><p>Stats unavailable</p></div>')


@router.get("/api/recent-vehicle-passes")
async def get_recent_vehicle_passes(request: Request):
    try:
        db = _ensure_db(request)
        recent_passes = db.get_recent_vehicle_passes(limit=30)
        passes = build_recent_pass_items(db, recent_passes)

        return templates.TemplateResponse(
            "partials/recent_passes.html",
            {
                "request": request,
                "passes": passes,
            },
        )
    except Exception as e:
        logger.error(f"Error in get_recent_vehicle_passes: {str(e)}")
        from fastapi.responses import HTMLResponse

        return HTMLResponse(f'<div id="recentPasses"><p>Error: {str(e)}</p></div>')


@router.get("/api/worst-passes")
async def get_worst_passes(request: Request):
    try:
        db = _ensure_db(request)
        hours = 168
        worst_speed_passes = db.get_extreme_passes("min_speed", "DESC", 5, hours)
        worst_time_passes = db.get_extreme_passes("time_in_zone", "ASC", 5, hours)

        all_passes = worst_speed_passes + worst_time_passes
        pass_ids = [p.id for p in all_passes]
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
                for p in all_passes
            ]
        )
        n = len(worst_speed_passes)

        return templates.TemplateResponse(
            "partials/pass_list.html",
            {
                "request": request,
                "title": "Worst Passes — Last 7 Days",
                "div_id": "worstPasses",
                "speed_passes": [
                    format_pass_item(p, vehicle_attrs, scores[i]) for i, p in enumerate(worst_speed_passes)
                ],
                "time_passes": [
                    format_pass_item(p, vehicle_attrs, scores[n + i]) for i, p in enumerate(worst_time_passes)
                ],
            },
        )
    except Exception as e:
        logger.error(f"Error in get_worst_passes: {str(e)}")
        from fastapi.responses import HTMLResponse

        return HTMLResponse(f'<div id="worstPasses"><p>Error: {str(e)}</p></div>')


@router.get("/api/best-passes")
async def get_best_passes(request: Request):
    try:
        db = _ensure_db(request)
        hours = 168
        best_speed_passes = db.get_extreme_passes("min_speed", "ASC", 5, hours)
        best_time_passes = db.get_extreme_passes("time_in_zone", "DESC", 5, hours)

        all_passes = best_speed_passes + best_time_passes
        pass_ids = [p.id for p in all_passes]
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
                for p in all_passes
            ]
        )
        n = len(best_speed_passes)

        return templates.TemplateResponse(
            "partials/pass_list.html",
            {
                "request": request,
                "title": "Best Passes — Last 7 Days",
                "div_id": "bestPasses",
                "speed_passes": [
                    format_pass_item(p, vehicle_attrs, scores[i]) for i, p in enumerate(best_speed_passes)
                ],
                "time_passes": [
                    format_pass_item(p, vehicle_attrs, scores[n + i]) for i, p in enumerate(best_time_passes)
                ],
            },
        )
    except Exception as e:
        logger.error(f"Error in get_best_passes: {str(e)}")
        from fastapi.responses import HTMLResponse

        return HTMLResponse(f'<div id="bestPasses"><p>Error: {str(e)}</p></div>')


@router.get("/api/vehicles/summary")
async def api_vehicles_summary(request: Request):
    try:
        db = _ensure_db(request)
        summary = db.get_vehicle_stats_summary()
        if not summary:
            from fastapi.responses import HTMLResponse

            return HTMLResponse('<div id="vehicleSummary"><p>Vehicle classification data not yet available.</p></div>')

        class SummaryObj:
            def __init__(self, d):
                self.total_classified = d["total_classified"]
                self.cluster_count = d["cluster_count"]
                self.coverage_pct = d["coverage_pct"]

        return templates.TemplateResponse(
            "partials/vehicle_summary.html",
            {
                "request": request,
                "summary": SummaryObj(summary),
            },
        )
    except Exception as e:
        logger.error(f"Error in api_vehicles_summary: {e}")
        from fastapi.responses import HTMLResponse

        return HTMLResponse('<div id="vehicleSummary"><p>Summary unavailable.</p></div>')


@router.get("/api/vehicles/types")
async def api_vehicles_types(request: Request):
    try:
        db = _ensure_db(request)
        items = db.get_vehicle_type_distribution()
        if not items:
            from fastapi.responses import HTMLResponse

            return HTMLResponse('<div id="vehicleTypes"><p>No vehicle type data.</p></div>')

        max_val = max(item["count"] for item in items) if items else 1
        chart_items = [
            {"label": item["vehicle_type"], "value": item["count"], "pct": round(item["count"] / max_val * 100, 1)}
            for item in items
        ]
        return templates.TemplateResponse(
            "partials/bar_chart.html",
            {
                "request": request,
                "title": "By Type",
                "div_id": "vehicleTypes",
                "items": chart_items,
            },
        )
    except Exception as e:
        logger.error(f"Error in api_vehicles_types: {e}")
        from fastapi.responses import HTMLResponse

        return HTMLResponse('<div id="vehicleTypes"><p>Type data unavailable.</p></div>')


@router.get("/api/vehicles/colors")
async def api_vehicles_colors(request: Request):
    try:
        db = _ensure_db(request)
        items = db.get_vehicle_color_distribution()
        if not items:
            from fastapi.responses import HTMLResponse

            return HTMLResponse('<div id="vehicleColors"><p>No vehicle color data.</p></div>')

        max_val = max(item["count"] for item in items) if items else 1
        chart_items = [
            {
                "label": item["color"],
                "value": item["count"],
                "pct": round(item["count"] / max_val * 100, 1),
                "color": COLOR_MAP.get(item["color"].lower(), "#000080"),
            }
            for item in items
        ]
        return templates.TemplateResponse(
            "partials/bar_chart.html",
            {
                "request": request,
                "title": "By Color",
                "div_id": "vehicleColors",
                "items": chart_items,
            },
        )
    except Exception as e:
        logger.error(f"Error in api_vehicles_colors: {e}")
        from fastapi.responses import HTMLResponse

        return HTMLResponse('<div id="vehicleColors"><p>Color data unavailable.</p></div>')


@router.get("/api/vehicles/make-models")
async def api_vehicles_make_models(request: Request):
    try:
        db = _ensure_db(request)
        items = db.get_top_make_models(limit=15)
        if not items:
            from fastapi.responses import HTMLResponse

            return HTMLResponse('<div id="vehicleMakeModels"><p>No make/model data.</p></div>')

        template_items = [
            {
                "make_model": item["make_model"],
                "count": item["count"],
                "image_url": resolve_image_url(item.get("image_path")),
            }
            for item in items
        ]
        return templates.TemplateResponse(
            "partials/make_models.html",
            {
                "request": request,
                "items": template_items,
            },
        )
    except Exception as e:
        logger.error(f"Error in api_vehicles_make_models: {e}")
        from fastapi.responses import HTMLResponse

        return HTMLResponse('<div id="vehicleMakeModels"><p>Make/model data unavailable.</p></div>')


@router.get("/api/vehicles/gallery")
async def api_vehicles_gallery(request: Request):
    try:
        db = _ensure_db(request)
        items = db.get_cluster_gallery(limit=30)
        if not items:
            from fastapi.responses import HTMLResponse

            return HTMLResponse('<div id="vehicleGallery"><p>No cluster gallery data.</p></div>')

        template_items = []
        for item in items:
            label_parts = list(filter(None, [item.get("color"), item.get("vehicle_type")]))
            label = " ".join(label_parts).title() if label_parts else "Unknown"
            template_items.append(
                {
                    "image_url": resolve_image_url(item.get("image_path")),
                    "label": label,
                    "make_model": item.get("make_model") or "",
                    "size": item.get("cluster_size") or 0,
                }
            )

        return templates.TemplateResponse(
            "partials/gallery.html",
            {
                "request": request,
                "items": template_items,
            },
        )
    except Exception as e:
        logger.error(f"Error in api_vehicles_gallery: {e}")
        from fastapi.responses import HTMLResponse

        return HTMLResponse('<div id="vehicleGallery"><p>Gallery unavailable.</p></div>')
