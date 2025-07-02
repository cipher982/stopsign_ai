# ruff: noqa: E501
import logging
import os
import time

import redis
import uvicorn
from fasthtml.common import H2
from fasthtml.common import H3
from fasthtml.common import H4
from fasthtml.common import Body
from fasthtml.common import Div
from fasthtml.common import FastHTML
from fasthtml.common import FileResponse
from fasthtml.common import Html
from fasthtml.common import HTMLResponse
from fasthtml.common import Iframe
from fasthtml.common import Img
from fasthtml.common import Li
from fasthtml.common import Link
from fasthtml.common import P
from fasthtml.common import Script
from fasthtml.common import Span
from fasthtml.common import StaticFiles
from fasthtml.common import StreamingResponse
from fasthtml.common import Ul
from fasthtml.common import Video
from minio import Minio
from sqlalchemy import text

from stopsign.components import common_footer_component
from stopsign.components import common_header_component
from stopsign.components import debug_controls_component
from stopsign.components import debug_tools_component
from stopsign.components import debug_video_component
from stopsign.components import main_layout_component
from stopsign.components import page_head_component
from stopsign.config import Config
from stopsign.database import Database
from stopsign.settings import DB_URL
from stopsign.settings import MINIO_ACCESS_KEY
from stopsign.settings import MINIO_BUCKET
from stopsign.settings import MINIO_ENDPOINT
from stopsign.settings import MINIO_PUBLIC_URL
from stopsign.settings import MINIO_SECRET_KEY
from stopsign.settings import REDIS_URL

logger = logging.getLogger(__name__)

# Optional Grafana for local development
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080

STREAM_FS_PATH = "/app/data/stream/stream.m3u8"  # filesystem path
STREAM_URL = "/stream/stream.m3u8"  # URL path


def get_minio_client():
    logger.debug(f"Creating Minio client with endpoint: {MINIO_ENDPOINT}, secure: True")
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=True,  # Set to True if using HTTPS
        # cert_check=True is default, so we can remove the argument
    )


# Initialize FastHTML app
app = FastHTML(
    ws_hdr=True,
    pico=False,
    hdrs=(
        Link(rel="stylesheet", href="/static/base.css"),
        Script(src="https://unpkg.com/htmx.org@1.9.4"),
    ),
)


app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/stream", StaticFiles(directory="/app/data/stream"), name="stream")


# Add cache headers for images
@app.middleware("http")
async def add_cache_headers(request, call_next):
    response = await call_next(request)
    # Add cache headers for static images and assets
    if request.url.path.startswith("/static/") or request.url.path.startswith("/vehicle-image/"):
        if any(request.url.path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".css", ".js"]):
            response.headers["Cache-Control"] = "public, max-age=3600"  # 1 hour cache
            response.headers["ETag"] = f'"{hash(request.url.path)}"'
    return response


@app.get("/vehicle-image/{object_name:path}")  # type: ignore
def get_image(object_name: str):
    # Validate object_name before trying to fetch
    if not object_name or not isinstance(object_name, str) or object_name.strip() == "":
        logger.warning(f"Invalid object_name requested: {object_name}")
        return HTMLResponse("Invalid image request", status_code=400)

    try:
        client = get_minio_client()
        data = client.get_object(MINIO_BUCKET, object_name)

        # Create response with proper cache headers
        response = StreamingResponse(data, media_type="image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=86400"  # 24-hour cache for images
        response.headers["ETag"] = f'"{hash(object_name)}"'
        return response

    except Exception as e:
        logger.error(f"Error fetching image from Minio: {str(e)}", exc_info=True)
        return HTMLResponse("Image not found", status_code=404)


@app.get("/favicon.ico")  # type: ignore
async def favicon():
    return FileResponse("static/favicon.ico")


@app.get("/")  # type: ignore
def home():
    return Html(
        page_head_component("Stop Sign Nanny", include_video_deps=True, page_type="home"),
        Body(
            common_header_component("Stop Sign Nanny"),
            main_layout_component(),
            common_footer_component(),
        ),
    )


@app.post("/api/update-stop-zone")  # type: ignore
async def update_stop_zone(request):
    data = await request.json()
    points = data["points"]
    stop_line = ((points[0]["x"], points[0]["y"]), (points[1]["x"], points[1]["y"]))
    stop_box_tolerance = [10, 10]
    min_stop_duration = 2.0

    new_config = {
        "stop_line": stop_line,
        "stop_box_tolerance": stop_box_tolerance,
        "min_stop_duration": min_stop_duration,
    }

    # Update config file directly - video analyzer will pick up changes
    config = Config("./config.yaml")
    config.update_stop_zone(new_config)

    return {"status": "success"}


@app.get("/api/coordinate-info")  # type: ignore
async def get_coordinate_info():
    """Get current coordinate system information for coordinate transformations."""
    try:
        # Read config directly since we're in separate containers
        config = Config("./config.yaml")

        # Get actual video dimensions from Redis metadata
        redis_client = redis.from_url(REDIS_URL)

        try:
            # Check if there's any frame metadata available
            metadata_keys = redis_client.keys("frame_metadata:*")
            if not metadata_keys:
                return {"error": "No video metadata available. Video analyzer may not be running."}

            # Get the most recent metadata
            latest_metadata = redis_client.get(metadata_keys[-1])
            if not latest_metadata:
                return {"error": "Could not read video metadata."}

            import json

            metadata = json.loads(latest_metadata)
            if "raw_video_dimensions" not in metadata:
                return {"error": "Video dimensions not available in metadata."}

            raw_width = metadata["raw_video_dimensions"]["width"]
            raw_height = metadata["raw_video_dimensions"]["height"]

            if not raw_width or not raw_height:
                return {"error": "Invalid video dimensions in metadata."}

        except Exception as e:
            logger.error(f"Failed to get video dimensions from metadata: {e}")
            return {"error": f"Failed to get video dimensions: {str(e)}"}

        # Calculate coordinate system information
        crop_side_pixels = int(raw_width * config.crop_side)
        crop_top_pixels = int(raw_height * config.crop_top)
        cropped_width = raw_width - (2 * crop_side_pixels)
        cropped_height = raw_height - crop_top_pixels
        processing_width = int(cropped_width * config.scale)
        processing_height = int(cropped_height * config.scale)

        return {
            "current_stop_line": {"coordinates": list(config.stop_line), "coordinate_system": "raw_coordinates"},
            "coordinate_system_info": {
                "raw_resolution": f"{raw_width}x{raw_height}",
                "cropped_resolution": f"{cropped_width}x{cropped_height}",
                "processing_resolution": f"{processing_width}x{processing_height}",
                "config_values": {
                    "scale": config.scale,
                    "crop_top": config.crop_top,
                    "crop_side": config.crop_side,
                },
                "transformation_chain": f"{raw_width}x{raw_height} → crop → {cropped_width}x{cropped_height} → scale({config.scale}) → {processing_width}x{processing_height}",
            },
        }
    except Exception as e:
        logger.error(f"Error getting coordinate info: {str(e)}")
        return {"error": str(e)}


@app.post("/api/update-stop-zone-from-display")  # type: ignore
async def update_stop_zone_from_display(request):
    """Update stop zone using display coordinates (from click-to-set UI)."""
    try:
        data = await request.json()

        # Extract the display click coordinates provided by the UI and the size of the
        # <video> element that was clicked.  The latter is useful if, in the future,
        # we decide to scale the browser-space coordinates into the processing
        # coordinate system.  For now we simply echo it back so the variable is
        # genuinely "used" and does not trip Ruff's F841 rule.

        display_points = data["display_points"]  # [{"x": px, "y": py}, {"x": px, "y": py}]
        video_element_size = data["video_element_size"]  # {"width": px, "height": px}
        # NOTE:
        # -----
        # The client sends `actual_video_size` (HTMLVideoElement.videoWidth/Height).
        # Those dimensions correspond to the *processed* frame (after the
        # crop/scale operations that the VideoAnalyzer performs) – currently
        # 1440×810 for a 1920×1080 source with a 0.75 scale factor.
        #
        # For correct round-tripping we need the *raw* dimensions so that the
        # stop-line coordinates we persist line-up with the pixels drawn by the
        # analyzer *before* it crops/scales the frame.  Therefore we **ignore**
        # the client-supplied value and instead look it up from the metadata
        # that the analyzer publishes to Redis.

        config = Config("./config.yaml")

        # Must get raw dimensions from Redis metadata – this guarantees that we
        # are using the same numbers as the analyzer.
        redis_client = redis.from_url(REDIS_URL)

        try:
            metadata_keys = redis_client.keys("frame_metadata:*")
            if not metadata_keys:
                return {"error": "No video metadata available. Video analyzer must be running first."}

            latest_metadata = redis_client.get(metadata_keys[-1])
            if not latest_metadata:
                return {"error": "Could not read video metadata."}

            import json

            metadata = json.loads(latest_metadata)
            if "raw_video_dimensions" not in metadata:
                return {"error": "Video dimensions not available in metadata."}

            raw_width = metadata["raw_video_dimensions"]["width"]
            raw_height = metadata["raw_video_dimensions"]["height"]

            if not raw_width or not raw_height:
                return {"error": "Invalid video dimensions in metadata."}

        except Exception as e:
            logger.error(f"Failed to get video dimensions: {e}")
            return {"error": f"Failed to get video dimensions: {str(e)}"}

        # Scale browser coordinates directly to actual video coordinates
        # The video analyzer will draw stop lines on frames BEFORE crop/scale
        scale_x = raw_width / video_element_size["width"]
        scale_y = raw_height / video_element_size["height"]

        raw_points = []
        for p in display_points:
            raw_x = p["x"] * scale_x
            raw_y = p["y"] * scale_y

            # Validate coordinates are within raw frame bounds
            if raw_x < 0 or raw_x > raw_width or raw_y < 0 or raw_y > raw_height:
                logger.warning(
                    f"Coordinate ({raw_x:.1f}, {raw_y:.1f}) is outside raw frame bounds ({raw_width}x{raw_height})"
                )

            raw_points.append({"x": raw_x, "y": raw_y})

        # Persist the new stop-line coordinates to the YAML config so that the
        # video-analyzer container can pick them up automatically.
        # Now using raw coordinates instead of processing coordinates

        if len(raw_points) < 2:
            return {"error": "Need exactly 2 points"}

        stop_line = (
            (raw_points[0]["x"], raw_points[0]["y"]),
            (raw_points[1]["x"], raw_points[1]["y"]),
        )

        try:
            # Debug what we're sending to config
            logger.info(f"Sending to config - stop_line: {stop_line}, type: {type(stop_line)}")
            for i, point in enumerate(stop_line):
                logger.info(f"Point {i}: {point}, type: {type(point)}")

            config.update_stop_zone(
                {
                    "stop_line": stop_line,
                    "stop_box_tolerance": [10, 10],
                    "min_stop_duration": 2.0,
                }
            )
        except Exception as e:
            logger.error(f"Error in update_stop_zone: {str(e)}")
            return {"error": f"Config update failed: {str(e)}"}

        # Respond with details that might be useful to the frontend for
        # confirmation or debugging.
        return {
            "status": "success",
            "display_coordinates": display_points,
            "raw_coordinates": raw_points,
            "video_element_size": video_element_size,
            "scaling_info": {
                "coordinate_system": "raw_frame_coordinates",
                "raw_resolution": f"{raw_width}x{raw_height}",
                "scale_factors": f"x={scale_x:.3f}, y={scale_y:.3f}",
                "browser_video_size": f"{video_element_size['width']}x{video_element_size['height']}",
                "note": "Using raw frame coordinates (Option 2) - stop lines drawn before crop/scale",
            },
        }

    except Exception as e:
        logger.error(f"Error updating stop zone from display: {str(e)}")
        return {"error": str(e)}


@app.post("/api/debug-coordinates")  # type: ignore
async def debug_coordinates(request):
    """Debug endpoint for testing coordinate transformations with Option 2 (raw coordinates)."""
    try:
        data = await request.json()

        display_points = data.get("display_points", [])
        video_element_size = data.get("video_element_size", {})

        config = Config("./config.yaml")

        # Get actual dimensions from Redis - no fallbacks
        redis_client = redis.from_url(REDIS_URL)

        try:
            metadata_keys = redis_client.keys("frame_metadata:*")
            if not metadata_keys:
                return {"error": "No video metadata available. Video analyzer must be running."}

            latest_metadata = redis_client.get(metadata_keys[-1])
            if not latest_metadata:
                return {"error": "Could not read video metadata."}

            import json

            metadata = json.loads(latest_metadata)
            if "raw_video_dimensions" not in metadata:
                return {"error": "Video dimensions not available in metadata."}

            raw_width = metadata["raw_video_dimensions"]["width"]
            raw_height = metadata["raw_video_dimensions"]["height"]

            if not raw_width or not raw_height:
                return {"error": "Invalid video dimensions in metadata."}

        except Exception as e:
            logger.error(f"Failed to get video dimensions: {e}")
            return {"error": f"Failed to get video dimensions: {str(e)}"}

        if video_element_size:
            scale_x = raw_width / video_element_size["width"]
            scale_y = raw_height / video_element_size["height"]
        else:
            scale_x = scale_y = 1.0

        # Transform all provided points to raw coordinates
        transformations = []
        for i, dp in enumerate(display_points):
            raw_x = dp["x"] * scale_x
            raw_y = dp["y"] * scale_y

            # Simple roundtrip test
            back_x = raw_x / scale_x
            back_y = raw_y / scale_y

            transformations.append(
                {
                    "point_index": i,
                    "display_input": {"x": dp["x"], "y": dp["y"]},
                    "raw_output": {"x": raw_x, "y": raw_y},
                    "display_roundtrip": {"x": back_x, "y": back_y},
                    "roundtrip_error": {"x": abs(dp["x"] - back_x), "y": abs(dp["y"] - back_y)},
                    "bounds_check": {
                        "display_valid": True,  # Simplified
                        "raw_valid": 0 <= raw_x <= raw_width and 0 <= raw_y <= raw_height,
                    },
                }
            )

        return {
            "coordinate_system": "raw_frame_coordinates",
            "current_stop_line": {"coordinates": list(config.stop_line), "coordinate_system": "raw_coordinates"},
            "scaling_info": {
                "raw_resolution": f"{raw_width}x{raw_height}",
                "scale_factors": f"x={scale_x:.3f}, y={scale_y:.3f}",
                "browser_video_size": f"{video_element_size.get('width', 'unknown')}x{video_element_size.get('height', 'unknown')}",
            },
            "point_transformations": transformations,
        }

    except Exception as e:
        logger.error(f"Error in debug coordinates: {str(e)}")
        return {"error": str(e)}


@app.post("/api/toggle-debug-zones")  # type: ignore
async def toggle_debug_zones(request):
    """Toggle debug zones visibility via Redis flag."""
    try:
        data = await request.json()
        enabled = data.get("enabled", False)

        redis_client = redis.from_url(REDIS_URL)
        redis_client.set("debug_zones_enabled", "1" if enabled else "0")

        return {"status": "success", "debug_zones_enabled": enabled}
    except Exception as e:
        logger.error(f"Error toggling debug zones: {str(e)}")
        return {"error": str(e)}


@app.post("/api/update-zone-from-display")  # type: ignore
async def update_zone_from_display(request):
    """Update any zone type using display coordinates."""
    try:
        data = await request.json()
        zone_type = data.get("zone_type", "stop-line")
        display_points = data["display_points"]
        video_element_size = data["video_element_size"]

        config = Config("./config.yaml")

        # Get raw dimensions from Redis metadata
        redis_client = redis.from_url(REDIS_URL)

        try:
            metadata_keys = redis_client.keys("frame_metadata:*")
            if not metadata_keys:
                return {"error": "No video metadata available. Video analyzer must be running first."}

            latest_metadata = redis_client.get(metadata_keys[-1])
            if not latest_metadata:
                return {"error": "Could not read video metadata."}

            import json

            metadata = json.loads(latest_metadata)
            if "raw_video_dimensions" not in metadata:
                return {"error": "Video dimensions not available in metadata."}

            raw_width = metadata["raw_video_dimensions"]["width"]
            raw_height = metadata["raw_video_dimensions"]["height"]

            if not raw_width or not raw_height:
                return {"error": "Invalid video dimensions in metadata."}

        except Exception as e:
            logger.error(f"Failed to get video dimensions: {e}")
            return {"error": f"Failed to get video dimensions: {str(e)}"}

        # Convert display coordinates to processing coordinates
        scale_x = raw_width / video_element_size["width"]
        scale_y = raw_height / video_element_size["height"]

        # Handle different zone types
        if zone_type == "stop-line":
            # Stop line needs two points in raw coordinates
            if len(display_points) != 2:
                return {"error": "Stop line requires exactly 2 points"}

            raw_points = []
            for p in display_points:
                raw_x = p["x"] * scale_x
                raw_y = p["y"] * scale_y
                raw_points.append((raw_x, raw_y))

            stop_line = tuple(raw_points)
            config.update_stop_zone(
                {
                    "stop_line": stop_line,
                    "stop_box_tolerance": config.stop_box_tolerance,
                    "min_stop_duration": 2.0,
                }
            )

        elif zone_type in ["pre-stop", "capture"]:
            # X-range zones need two points for X coordinates
            if len(display_points) != 2:
                return {"error": f"{zone_type} zone requires exactly 2 points"}

            # Convert to processing coordinates (these zones work in processing space)
            proc_x_coords = []
            for p in display_points:
                # Convert to raw coordinates first
                raw_x = p["x"] * scale_x
                raw_y = p["y"] * scale_y

                # Then convert to processing coordinates using the same logic as video analyzer
                crop_side_pixels = int(raw_width * config.crop_side)

                cropped_x = raw_x - crop_side_pixels
                processing_x = cropped_x * config.scale

                proc_x_coords.append(processing_x)

            # Ensure proper ordering (left to right)
            x_range = tuple(sorted(proc_x_coords))

            # Basic validation - ensure zones are reasonable
            zone_width = abs(x_range[1] - x_range[0])
            if zone_width < 10:
                return {"error": f"Zone width ({zone_width:.1f}) is too small. Minimum width is 10 pixels."}
            if zone_width > 500:
                return {"error": f"Zone width ({zone_width:.1f}) is too large. Maximum width is 500 pixels."}

            # Update the appropriate zone using the new method
            if zone_type == "pre-stop":
                config.update_zone("pre_stop", x_range)
            elif zone_type == "capture":
                config.update_zone("capture", x_range)

        # Signal video analyzer to reload config
        try:
            redis_client.set("config_updated", "1")
        except Exception as e:
            logger.warning(f"Could not signal config update: {e}")

        return {
            "status": "success",
            "zone_type": zone_type,
            "display_coordinates": display_points,
            "video_element_size": video_element_size,
            "message": f"{zone_type} zone updated successfully",
        }

    except Exception as e:
        logger.error(f"Error updating zone from display: {str(e)}")
        return {"error": str(e)}


@app.get("/debug")  # type: ignore
def debug_page():
    """Simple debug page for stop line adjustment - access via /debug URL"""
    from fasthtml.common import H1

    return Html(
        page_head_component("Stop Sign Debug", include_video_deps=True, page_type="debug"),
        Body(
            Div(
                H1("Stop Sign Debug Interface"),
                # Main layout: Video left, Controls right
                Div(
                    Div(
                        debug_video_component(),
                        cls="content-primary",
                    ),
                    Div(
                        debug_controls_component(),
                        cls="content-secondary",
                    ),
                    cls="content-grid",
                ),
                debug_tools_component(),
                cls="app-layout",
            )
        ),
    )


@app.get("/debug-perf")  # type: ignore
def debug_perf_page():
    """Performance debugging page"""
    from fasthtml.common import H1
    from fasthtml.common import Button
    from fasthtml.common import Code
    from fasthtml.common import Pre

    return Html(
        page_head_component("Performance Debug"),
        Body(
            Div(
                H1("Performance Debugging"),
                Button(
                    "Run Performance Test",
                    hx_get="/debug-performance",
                    hx_target="#perf-results",
                    hx_indicator="#loading",
                ),
                Div("Loading...", id="loading", style="display: none;"),
                Pre(
                    Code(id="perf-results"),
                    style="background: #f5f5f5; padding: 15px; border-radius: 5px; margin-top: 20px;",
                ),
                Script("""
                    document.body.addEventListener('htmx:afterRequest', function(evt) {
                        if (evt.detail.target.id === 'perf-results') {
                            try {
                                const data = JSON.parse(evt.detail.xhr.response);
                                evt.detail.target.textContent = JSON.stringify(data, null, 2);
                            } catch (e) {
                                evt.detail.target.textContent = evt.detail.xhr.response;
                            }
                        }
                    });
                """),
                cls="window",
            )
        ),
    )


@app.get("/api/live-stats")  # type: ignore
async def get_live_stats():
    """Get live statistics for the stats panel"""
    try:
        if not hasattr(app.state, "db"):
            app.state.db = Database(db_url=DB_URL)

        # Get today's data

        # Get 24h stats
        total_passes_24h = app.state.db.get_total_passes_last_24h()

        # Get recent passes for compliance calculation
        recent_passes = app.state.db.get_recent_vehicle_passes(limit=100)

        # Calculate compliance rate (vehicles with time_in_zone > 2.0 seconds)
        if recent_passes:
            compliant_count = sum(1 for p in recent_passes if p.time_in_zone >= 2.0)
            compliance_rate = round((compliant_count / len(recent_passes)) * 100)
        else:
            compliance_rate = 0

        # Get last detection time
        last_detection = "N/A"
        if recent_passes:
            import time

            last_time = recent_passes[0].timestamp
            minutes_ago = int((time.time() - last_time.timestamp()) / 60)
            if minutes_ago < 60:
                last_detection = f"{minutes_ago}m ago"
            else:
                hours_ago = int(minutes_ago / 60)
                last_detection = f"{hours_ago}h ago"

        # Calculate trend (compare last 10 vs previous 10)
        trend_arrow = "→"
        if len(recent_passes) >= 20:
            recent_10 = recent_passes[:10]
            prev_10 = recent_passes[10:20]
            recent_compliance = sum(1 for p in recent_10 if p.time_in_zone >= 2.0) / 10
            prev_compliance = sum(1 for p in prev_10 if p.time_in_zone >= 2.0) / 10
            if recent_compliance > prev_compliance:
                trend_arrow = "↗"
            elif recent_compliance < prev_compliance:
                trend_arrow = "↘"

        # Rotating insights
        insights = [
            "Peak hour today: 8-9 AM",
            "Average stop time: 3.2s",
            f"Fastest vehicle: {max((p.min_speed for p in recent_passes), default=0):.1f} px/s",
            "Best compliance streak: 7 vehicles",
        ]
        import random

        rotating_insight = random.choice(insights)

        return Div(
            Div(f"{compliance_rate}%", id="complianceData"),
            Div(f"{total_passes_24h - int(total_passes_24h * compliance_rate / 100)}", id="violationData"),
            Div(f"{total_passes_24h}", id="vehicleData"),
            Div(last_detection, id="lastDetectionData"),
            Div(trend_arrow, id="trendData"),
            Div(rotating_insight, id="insightData"),
        )

    except Exception as e:
        logger.error(f"Error in get_live_stats: {str(e)}")
        return Div(P("Stats unavailable"), id="statsError")


@app.get("/api/recent-vehicle-passes")  # type: ignore
async def get_recent_vehicle_passes():
    try:
        if not hasattr(app.state, "db"):
            app.state.db = Database(db_url=DB_URL)

        recent_passes = app.state.db.get_recent_vehicle_passes(limit=10)

        # Get all scores at once
        scores = app.state.db.get_bulk_scores(
            [{"min_speed": p.min_speed, "time_in_zone": p.time_in_zone} for p in recent_passes]
        )

        # Create a dict to map scores to passes
        scores_dict = {(score["min_speed"], score["time_in_zone"]): score for score in scores}

        passes_list = Ul(
            *[
                Li(
                    create_pass_item(pass_data, scores_dict[(pass_data.min_speed, pass_data.time_in_zone)]),
                    id=f"pass-{pass_data.id}",  # Add unique ID for each pass
                )
                for pass_data in recent_passes
            ],
            id="passes-list",
        )

        return Div(
            passes_list,
            id="recentPasses",
        )

    except Exception as e:
        logger.error(f"Error in get_recent_vehicle_passes: {str(e)}")
        return Div(P(f"Error: {str(e)}"), id="recentPasses")


@app.get("/api/new-vehicles")  # type: ignore
async def get_new_vehicles(since: int = 0):
    """Get only new vehicle passes since timestamp for incremental updates"""
    try:
        if not hasattr(app.state, "db"):
            app.state.db = Database(db_url=DB_URL)

        import time

        current_time = int(time.time())

        # Get recent passes
        all_passes = app.state.db.get_recent_vehicle_passes(limit=10)

        # Filter for new passes since timestamp
        if since > 0:
            new_passes = [p for p in all_passes if int(p.timestamp.timestamp()) > since]
        else:
            # First load - return top 3 most recent
            new_passes = all_passes[:3]

        if not new_passes:
            # Return empty response with timestamp for next check
            return Div(hx_vals=f'{{"last_check": {current_time}}}', style="display: none;")

        # Get scores for new passes
        scores = app.state.db.get_bulk_scores(
            [{"min_speed": p.min_speed, "time_in_zone": p.time_in_zone} for p in new_passes]
        )
        scores_dict = {(score["min_speed"], score["time_in_zone"]): score for score in scores}

        # Create new vehicle items with out-of-band swap to prepend to list
        new_items = []
        for pass_data in new_passes:
            new_items.append(
                Li(
                    create_pass_item(pass_data, scores_dict[(pass_data.min_speed, pass_data.time_in_zone)]),
                    id=f"pass-{pass_data.id}",
                    hx_swap_oob="afterbegin:#passes-list",
                )
            )

        # Also include timestamp for next check
        response_div = Div(*new_items, Div(hx_vals=f'{{"last_check": {current_time}}}', style="display: none;"))

        return response_div

    except Exception as e:
        logger.error(f"Error in get_new_vehicles: {str(e)}")
        return Div(hx_vals=f'{{"last_check": {int(time.time())}}}', style="display: none;")


@app.get("/records")  # type: ignore
def records():
    from fasthtml.common import Main

    return Html(
        page_head_component("Records"),
        Body(
            common_header_component("Records"),
            Main(
                Div(
                    H2("Vehicle Records"),
                    Div(
                        Div(id="worstPasses", hx_get="/api/worst-passes", hx_trigger="load"),
                        Div(id="bestPasses", hx_get="/api/best-passes", hx_trigger="load"),
                        cls="two-col",
                    ),
                    cls="window",
                ),
            ),
            common_footer_component(),
        ),
    )


@app.get("/api/worst-passes")  # type: ignore
async def get_worst_passes():
    try:
        hours = 168  # Increased time window to 7 days
        worst_speed_passes = app.state.db.get_extreme_passes("min_speed", "DESC", 5, hours)
        worst_time_passes = app.state.db.get_extreme_passes("time_in_zone", "ASC", 5, hours)

        # Get scores for all passes at once
        all_passes = worst_speed_passes + worst_time_passes
        scores = app.state.db.get_bulk_scores(
            [{"min_speed": p.min_speed, "time_in_zone": p.time_in_zone} for p in all_passes]
        )
        scores_dict = {(score["min_speed"], score["time_in_zone"]): score for score in scores}

        return create_pass_list(
            "Worst Passes (7 Days)", worst_speed_passes, worst_time_passes, "worstPasses", scores_dict
        )
    except Exception as e:
        logger.error(f"Error in get_worst_passes: {str(e)}")
        return Div(P(f"Error: {str(e)}"), id="worstPasses")


@app.get("/api/best-passes")  # type: ignore
async def get_best_passes():
    try:
        hours = 168  # Increased time window to 7 days
        best_speed_passes = app.state.db.get_extreme_passes("min_speed", "ASC", 5, hours)
        best_time_passes = app.state.db.get_extreme_passes("time_in_zone", "DESC", 5, hours)

        # Get scores for all passes at once
        all_passes = best_speed_passes + best_time_passes
        scores = app.state.db.get_bulk_scores(
            [{"min_speed": p.min_speed, "time_in_zone": p.time_in_zone} for p in all_passes]
        )
        scores_dict = {(score["min_speed"], score["time_in_zone"]): score for score in scores}

        return create_pass_list("Best Passes (7 Days)", best_speed_passes, best_time_passes, "bestPasses", scores_dict)
    except Exception as e:
        logger.error(f"Error in get_best_passes: {str(e)}")
        return Div(P(f"Error: {str(e)}"), id="bestPasses")


def get_minio_object_name(minio_path: str) -> str | None:
    """Extract object name from MinIO path, return None if invalid."""
    if not minio_path or not isinstance(minio_path, str):
        return None
    if not minio_path.startswith("minio://"):
        return None
    parts = minio_path.split("/", 3)
    return parts[3] if len(parts) >= 4 else None


def create_pass_list(title, speed_passes, time_passes, div_id, scores_dict):
    return Div(
        H3(title),
        H4("Minimum Speed"),
        Div(
            *[
                create_pass_item(pass_data, scores_dict[(pass_data.min_speed, pass_data.time_in_zone)])
                for pass_data in speed_passes
            ],
        ),
        H4("Time in Stop Zone"),
        Div(
            *[
                create_pass_item(pass_data, scores_dict[(pass_data.min_speed, pass_data.time_in_zone)])
                for pass_data in time_passes
            ],
        ),
        id=div_id,
        cls="window",
    )


def create_pass_item(pass_data, scores):
    # Extract just the filename from minio://bucket/filename, with validation
    image_url = "/static/placeholder.jpg"  # Default fallback image
    if pass_data.image_path and isinstance(pass_data.image_path, str) and pass_data.image_path.startswith("minio://"):
        parts = pass_data.image_path.split("/", 3)
        if len(parts) >= 4 and parts[3]:
            object_name = parts[3]
            # Use direct MinIO URL for now - proxy endpoint has issues
            image_url = f"{MINIO_PUBLIC_URL}/{MINIO_BUCKET}/{object_name}"

    # Format timestamp
    time_str = pass_data.created_at.strftime("%H:%M:%S") if hasattr(pass_data, "created_at") else "N/A"

    # Create data visualization squares based on percentiles
    speed_percentile = int(scores.get("speed_score", 0))
    time_percentile = int(scores.get("time_score", 0))

    # Color squares based on percentile ranges (retro style)
    def get_retro_color(percentile):
        if percentile >= 90:
            return "#FF0000"  # Red - high values
        elif percentile >= 70:
            return "#FF8000"  # Orange
        elif percentile >= 50:
            return "#FFFF00"  # Yellow
        elif percentile >= 30:
            return "#80FF00"  # Light green
        else:
            return "#00FF00"  # Green - low values

    return Div(
        Img(
            src=image_url,
            alt="Vehicle",
            cls="activity-feed__image sunken",
            loading="lazy",
            decoding="async",
        ),
        Div(
            Div(time_str, cls="activity-feed__time"),
            Div(
                # Speed data with color square
                Span(cls="data-square", style=f"background-color: {get_retro_color(speed_percentile)};"),
                Span(f"{pass_data.min_speed:.2f} px/s", cls="activity-feed__data"),
                # Time data with color square
                Span(
                    cls="data-square", style=f"background-color: {get_retro_color(time_percentile)}; margin-left: 8px;"
                ),
                Span(f"{pass_data.time_in_zone:.2f}s", cls="activity-feed__data"),
                cls="activity-feed__metrics",
            ),
            cls="activity-feed__content",
        ),
        cls="activity-feed__item",
    )


@app.get("/statistics")  # type: ignore
def statistics():
    from fasthtml.common import Main

    return Html(
        page_head_component("Statistics"),
        Body(
            common_header_component("Statistics"),
            Main(
                Div(
                    Iframe(
                        src=GRAFANA_URL,
                        width="100%",
                        height="600",
                        frameborder="0",
                    ),
                    cls="window",
                ),
            ),
            common_footer_component(),
        ),
    )


@app.get("/about")  # type: ignore
def about():
    from fasthtml.common import Main

    with open("static/summary.md", "r") as file:
        summary_content = file.read()
    return Html(
        page_head_component("About"),
        Body(
            common_header_component("About"),
            Main(
                Div(
                    H2("Project Summary"),
                    P(summary_content),
                    cls="window",
                ),
            ),
            common_footer_component(),
        ),
    )


@app.get("/check-stream")  # type: ignore
async def check_stream():
    if os.path.exists(STREAM_FS_PATH):
        logger.info(f"Stream file exists: {STREAM_FS_PATH}")
        with open(STREAM_FS_PATH, "r") as f:
            content = f.read()
        logger.debug(f"Stream file content:\n{content}")
        return {"status": "exists", "content": content}
    else:
        logger.warning(f"Stream file does not exist: {STREAM_FS_PATH}")
        stream_dir = os.path.dirname(STREAM_FS_PATH)
        if os.path.exists(stream_dir):
            logger.warning(f"Files in stream directory: {os.listdir(stream_dir)}")
        else:
            logger.warning(f"Stream directory does not exist: {stream_dir}")
        return {"status": f"HLS file not found at {STREAM_FS_PATH}"}


@app.get("/debug-performance")  # type: ignore
async def debug_performance():
    """Debug endpoint to time various operations"""
    import os
    import time

    import psutil

    results = {}

    # Test database queries
    if hasattr(app.state, "db"):
        # Test recent passes query
        start = time.time()
        try:
            recent_passes = app.state.db.get_recent_vehicle_passes(limit=10)
            results["db_recent_passes"] = {
                "time_ms": round((time.time() - start) * 1000),
                "count": len(recent_passes),
                "status": "success",
            }
        except Exception as e:
            results["db_recent_passes"] = {
                "time_ms": round((time.time() - start) * 1000),
                "error": str(e),
                "status": "error",
            }

        # Test bulk scoring
        start = time.time()
        try:
            test_passes = [{"min_speed": 10.0, "time_in_zone": 2.5}]
            scores = app.state.db.get_bulk_scores(test_passes)
            results["db_bulk_scores"] = {"time_ms": round((time.time() - start) * 1000), "status": "success"}
        except Exception as e:
            results["db_bulk_scores"] = {
                "time_ms": round((time.time() - start) * 1000),
                "error": str(e),
                "status": "error",
            }

        # Test stats query
        start = time.time()
        try:
            total = app.state.db.get_total_passes_last_24h()
            results["db_stats"] = {
                "time_ms": round((time.time() - start) * 1000),
                "total_passes": total,
                "status": "success",
            }
        except Exception as e:
            results["db_stats"] = {"time_ms": round((time.time() - start) * 1000), "error": str(e), "status": "error"}

    # Test MinIO connection
    start = time.time()
    try:
        client = get_minio_client()
        # Try to list first few objects
        objects = list(client.list_objects(MINIO_BUCKET, max_keys=1))
        results["minio_connection"] = {
            "time_ms": round((time.time() - start) * 1000),
            "status": "success",
            "objects_found": len(objects),
        }
    except Exception as e:
        results["minio_connection"] = {
            "time_ms": round((time.time() - start) * 1000),
            "error": str(e),
            "status": "error",
        }

    # Test Redis connection
    start = time.time()
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        results["redis_connection"] = {"time_ms": round((time.time() - start) * 1000), "status": "success"}
    except Exception as e:
        results["redis_connection"] = {
            "time_ms": round((time.time() - start) * 1000),
            "error": str(e),
            "status": "error",
        }

    # System info
    try:
        results["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else "N/A",
        }
    except Exception as e:
        results["system"] = {"error": str(e)}

    # HLS stream check
    start = time.time()
    stream_exists = os.path.exists(STREAM_FS_PATH)
    results["hls_stream"] = {
        "time_ms": round((time.time() - start) * 1000),
        "exists": stream_exists,
        "path": STREAM_FS_PATH,
    }

    if stream_exists:
        try:
            stat = os.stat(STREAM_FS_PATH)
            results["hls_stream"]["size_bytes"] = stat.st_size
            results["hls_stream"]["last_modified"] = stat.st_mtime
            results["hls_stream"]["age_seconds"] = time.time() - stat.st_mtime
        except Exception as e:
            results["hls_stream"]["stat_error"] = str(e)

    return results


@app.get("/load-video")  # type: ignore
def load_video():
    return Div(
        Video(
            id="videoPlayer",
            controls=True,
            autoplay=True,
            muted=True,
        ),
    )


def calculate_time_in_zone_score(time_in_zone):
    percentile = app.state.db.get_time_in_zone_percentile(time_in_zone)
    return round(percentile / 10)


def calculate_speed_score(min_speed):
    percentile = app.state.db.get_min_speed_percentile(min_speed)
    return round((100 - percentile) / 10)


class DBHealthTracker:
    def __init__(self):
        self.last_failure_time = None
        self.failure_count = 0
        self.max_failure_duration = 300  # 5 minutes in seconds

    def record_failure(self):
        current_time = time.time()
        if self.last_failure_time is None:
            self.last_failure_time = current_time
        self.failure_count += 1

    def record_success(self):
        self.last_failure_time = None
        self.failure_count = 0

    def is_failure_persistent(self) -> bool:
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) > self.max_failure_duration


# Initialize the health tracker
db_health_tracker = DBHealthTracker()


@app.get("/health")  # type: ignore
async def health():
    try:
        if not hasattr(app.state, "db"):
            app.state.db = Database(db_url=DB_URL)

        with app.state.db.Session() as session:
            session.execute(text("SELECT 1 /* health check */"), execution_options={"timeout": 5}).scalar()
            db_health_tracker.record_success()
            return HTMLResponse(status_code=200, content="Healthy: Database connection verified")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        db_health_tracker.record_failure()

        # Only return unhealthy if failures have persisted
        if db_health_tracker.is_failure_persistent():
            return HTMLResponse(
                status_code=503,  # Service Unavailable
                content=f"Unhealthy: Database connection issues for over 5 minutes - {str(e)}",
            )
        else:
            # Still return healthy if this is a temporary blip
            return HTMLResponse(
                status_code=200,
                content="Healthy: Tolerating temporary database connectivity issue",
            )


def main():
    """Main entry point for the web server."""
    db_connected = False
    db_init_attempts = 0
    max_db_init_attempts = 10
    db_init_delay = 5  # seconds

    while not db_connected and db_init_attempts < max_db_init_attempts:
        try:
            app.state.db = Database(db_url=DB_URL)
            # A simple query to ensure the database is responsive
            with app.state.db.Session() as session:
                session.execute(text("SELECT 1"))
            db_connected = True
            logger.info("Database connection successful.")
        except Exception as e:
            db_init_attempts += 1
            logger.warning(
                f"Database connection attempt {db_init_attempts} failed: {e}. "
                f"Retrying in {db_init_delay} seconds..."
            )
            time.sleep(db_init_delay)

    if not db_connected:
        logger.error("Failed to connect to the database after several attempts. Exiting.")
        # Fallback to YAML config if the database is unavailable
        logger.warning("Config: Database connection failed, using YAML fallback")
        # Depending on requirements, you might exit here or continue with limited functionality
        # For now, we'll continue and let endpoints handle the lack of a DB connection.

    try:
        uvicorn.run(
            "stopsign.web_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="warning",
            reload_dirs=["./stopsign"],
            reload_excludes=["./app/data/*"],
        )
    except Exception as e:
        logger.error(f"Error in web server: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
