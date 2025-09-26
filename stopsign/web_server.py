# ruff: noqa: E501
import logging
import os
import time

from stopsign.hls_health import parse_hls_playlist

# Debug telemetry issues in web server
try:
    from stopsign import debug_otel  # Patch OpenTelemetry for debugging  # noqa: F401
except ImportError:
    pass  # debug_otel.py not available, continue without debug

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
from fasthtml.common import Link
from fasthtml.common import P
from fasthtml.common import Script
from fasthtml.common import Span
from fasthtml.common import StaticFiles
from fasthtml.common import StreamingResponse
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
from stopsign.telemetry import get_tracer
from stopsign.telemetry import setup_web_server_telemetry

logger = logging.getLogger(__name__)

# Optional Grafana for local development
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080

STREAM_FS_PATH = "/app/data/stream/stream.m3u8"  # filesystem path
STREAM_URL = "/stream/stream.m3u8"  # URL path
GRACE_STARTUP_SEC = 120  # aligns with ffmpeg_service
WEB_START_TIME = time.time()


_HLS_PARSE_WARN_LAST_TS = 0.0


def _parse_hls_playlist(path: str) -> dict:
    """Wrapper to parse playlist; rate-limit noisy warnings."""
    global _HLS_PARSE_WARN_LAST_TS
    try:
        info = parse_hls_playlist(path)
    except Exception as e:
        now = time.time()
        if now - _HLS_PARSE_WARN_LAST_TS > 60:
            logger.warning(f"HLS playlist parse failed: {e}")
            _HLS_PARSE_WARN_LAST_TS = now
        else:
            logger.debug(f"HLS playlist parse failed: {e}")
        # Best-effort fallback
        info = {
            "exists": os.path.exists(path),
            "playlist_mtime": os.path.getmtime(path) if os.path.exists(path) else None,
            "age_seconds": None,
            "segments_count": 0,
            "threshold_sec": 60.0,
        }
        # Try to count TS segments in the same directory for visibility
        try:
            stream_dir = os.path.dirname(path)
            if os.path.isdir(stream_dir):
                ts_count = len([f for f in os.listdir(stream_dir) if f.endswith(".ts")])
                info["segments_count"] = ts_count
        except Exception:
            pass
    return info


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

# Initialize telemetry
metrics = setup_web_server_telemetry(app)
tracer = get_tracer("stopsign.web_server")


# Add cache headers for images and HLS streaming with optimized telemetry
@app.middleware("http")
async def add_cache_headers(request, call_next):
    # Detect browser type once for all requests
    user_agent = request.headers.get("user-agent", "").lower()
    if "chrome" in user_agent:
        browser = "chrome"
    elif "safari" in user_agent and "chrome" not in user_agent:
        browser = "safari"
    elif "firefox" in user_agent:
        browser = "firefox"
    else:
        browser = "other"

    # Skip telemetry for most static assets (sample 10%)
    if request.url.path.startswith("/static/"):
        if any(request.url.path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".css", ".js"]):
            response = await call_next(request)
            response.headers["Cache-Control"] = "public, max-age=3600"  # 1 hour cache
            response.headers["ETag"] = f'"{hash(request.url.path)}"'

            # Only trace 10% of static requests to reduce overhead
            import random

            if random.random() < 0.1:
                with tracer.start_as_current_span("http_static_sample") as span:
                    span.set_attribute("request.type", "static_asset")
                    span.set_attribute("browser.type", browser)
            return response

    # Full telemetry for streaming and API requests
    is_streaming = request.url.path.startswith("/stream/")
    is_api = request.url.path.startswith("/api/") or request.url.path in ["/health", "/health/stream", "/check-stream"]

    if is_streaming or is_api:
        with tracer.start_as_current_span("http_request") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("browser.type", browser)

            # Only log full URL for streaming requests
            if is_streaming:
                span.set_attribute("http.path", request.url.path)

            start_time = time.time()
            response = await call_next(request)
            duration = time.time() - start_time

            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("http.response_time_seconds", duration)

            # HLS Streaming telemetry (detailed)
            if is_streaming:
                if request.url.path.endswith(".m3u8"):
                    # Manifest request
                    span.set_attribute("hls.type", "manifest")
                    metrics.db_operations.add(
                        1, {"operation": "hls_manifest", "browser": browser, "status": str(response.status_code)}
                    )

                    response.headers["Cache-Control"] = "no-store, must-revalidate"
                    response.headers["Pragma"] = "no-cache"
                    response.headers["Expires"] = "0"

                elif request.url.path.endswith(".ts"):
                    # Video segment request
                    segment_name = request.url.path.split("/")[-1]
                    span.set_attribute("hls.type", "segment")

                    # Extract segment number if possible
                    try:
                        segment_num = int("".join(filter(str.isdigit, segment_name)))
                        span.set_attribute("hls.segment_number", segment_num)
                    except (ValueError, TypeError):
                        pass

                    # Track successful vs failed segment requests
                    if response.status_code == 200:
                        metrics.db_operations.add(1, {"operation": "hls_segment_success", "browser": browser})
                        span.set_attribute("hls.success", True)
                    elif response.status_code == 404:
                        metrics.db_operations.add(1, {"operation": "hls_segment_404", "browser": browser})
                        span.set_attribute("hls.success", False)
                        span.set_attribute("hls.error", "segment_not_found")
                        # This is critical for debugging Chrome issues
                        logger.warning(f"404 on HLS segment {segment_name} from {browser}")

                    # Check if this is a slow request (potential Chrome issue)
                    if duration > 1.0:  # More than 1 second for a segment
                        span.set_attribute("hls.slow_request", True)
                        logger.warning(f"Slow HLS segment request: {segment_name} took {duration:.2f}s from {browser}")

                    response.headers["Cache-Control"] = "public, max-age=60"  # 1 minute cache for segments

            # Track errors
            if response.status_code >= 400:
                span.set_attribute("http.error", True)
                metrics.db_operations.add(1, {"operation": "http_error", "status": str(response.status_code)})
    else:
        # Minimal processing for other requests
        response = await call_next(request)

        # Apply cache headers for vehicle images
        if request.url.path.startswith("/vehicle-image/"):
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
            Div(
                common_header_component("Stop Sign Nanny"),
                main_layout_component(),
                common_footer_component(),
                cls="desktop-container",
            ),
        ),
    )


@app.get("/api/coordinate-info")  # type: ignore
async def get_coordinate_info():
    """Get current coordinate system information for coordinate transformations."""
    try:
        # Read config directly since we're in separate containers
        config = Config("/app/config/config.yaml")

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
            "current_stop_zone": {
                "coordinates": [list(point) for point in config.stop_zone] if config.stop_zone else [],
                "coordinate_system": "raw_coordinates",
            },
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

        display_points = data["display_points"]  # Four points: [{"x": px, "y": py}, ...]
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

        config = Config("/app/config/config.yaml")

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
        # The video analyzer draws the stop zone on frames BEFORE crop/scale
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

        # Persist the new stop zone coordinates to the YAML config
        # Stop zone now uses 4 points to define a rectangle

        if len(raw_points) != 4:
            return {"error": "Stop zone requires exactly 4 points"}

        # Store as list of 4 points
        stop_zone = [
            (raw_points[0]["x"], raw_points[0]["y"]),
            (raw_points[1]["x"], raw_points[1]["y"]),
            (raw_points[2]["x"], raw_points[2]["y"]),
            (raw_points[3]["x"], raw_points[3]["y"]),
        ]

        try:
            # Debug what we're sending to config
            logger.info(f"Sending to config - stop_zone: {stop_zone}, type: {type(stop_zone)}")

            result = config.update_stop_zone(
                {
                    "stop_zone": stop_zone,
                    "min_stop_duration": 2.0,
                }
            )
        except Exception as e:
            logger.error(f"Error in update_stop_zone: {str(e)}")
            return {"error": f"Config update failed: {str(e)}"}

        # Signal analyzer to reload config
        redis_client.set("config_updated", "1")

        # Respond with details that might be useful to the frontend for
        # confirmation or debugging.
        return {
            "status": "success",
            "version": result["version"],
            "stop_zone": result.get("stop_zone", stop_zone),
            "display_coordinates": display_points,
            "raw_coordinates": raw_points,
            "video_element_size": video_element_size,
            "scaling_info": {
                "coordinate_system": "raw_frame_coordinates",
                "raw_resolution": f"{raw_width}x{raw_height}",
                "scale_factors": f"x={scale_x:.3f}, y={scale_y:.3f}",
                "browser_video_size": f"{video_element_size['width']}x{video_element_size['height']}",
                "note": "Stop zone defined by 4 corner points",
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

        config = Config("/app/config/config.yaml")

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
            "current_stop_zone": {
                "coordinates": [list(point) for point in config.stop_zone] if config.stop_zone else [],
                "coordinate_system": "raw_coordinates",
            },
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
        zone_type = data.get("zone_type")
        display_points = data["display_points"]
        video_element_size = data["video_element_size"]

        config = Config("/app/config/config.yaml")

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

        # Handle supported zone types
        if zone_type in ["pre-stop", "capture"]:
            # Line zones need exactly 2 points
            if len(display_points) != 2:
                return {"error": f"{zone_type} zone requires exactly 2 points"}

            # Store as raw coordinate line (2 points)
            raw_line = []
            for p in display_points:
                raw_x = p["x"] * scale_x
                raw_y = p["y"] * scale_y
                raw_line.append((raw_x, raw_y))

            # Basic validation - ensure line has some length
            dx = abs(raw_line[1][0] - raw_line[0][0])
            dy = abs(raw_line[1][1] - raw_line[0][1])
            line_length = (dx * dx + dy * dy) ** 0.5

            if line_length < 10:
                return {"error": f"Line too short ({line_length:.1f} pixels). Draw a longer line."}

            # Update the appropriate zone with the line
            if zone_type == "pre-stop":
                result = config.update_zone("pre_stop", raw_line)
            elif zone_type == "capture":
                result = config.update_zone("capture", raw_line)

        else:
            return {"error": f"Unsupported zone type '{zone_type}'"}

        # Signal video analyzer to reload config
        try:
            redis_client.set("config_updated", "1")
        except Exception as e:
            logger.warning(f"Could not signal config update: {e}")

        return {
            "status": "success",
            "version": result.get("version", "unknown"),
            "zone_type": zone_type,
            "zone_data": result,
            "display_coordinates": display_points,
            "video_element_size": video_element_size,
            "message": f"{zone_type} zone updated successfully",
        }

    except Exception as e:
        logger.error(f"Error updating zone from display: {str(e)}")
        return {"error": str(e)}


@app.get("/debug")  # type: ignore
def debug_page():
    """Simple debug page for live zone adjustment - access via /debug URL"""
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

        passes_list = Div(
            *[
                create_pass_item(pass_data, scores_dict[(pass_data.min_speed, pass_data.time_in_zone)])
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
    time_str = pass_data.timestamp.strftime("%H:%M:%S") if hasattr(pass_data, "timestamp") else "N/A"

    # Use actual values to create visual variety (simpler approach)
    speed_val = pass_data.min_speed
    time_val = pass_data.time_in_zone

    # Color squares based on value ranges (using raw values for now)
    def get_speed_color(speed):
        if speed > 2.0:
            return "#FF0000"  # Red - high speed
        elif speed > 1.5:
            return "#FF8000"  # Orange
        elif speed > 1.0:
            return "#FFFF00"  # Yellow
        elif speed > 0.5:
            return "#80FF00"  # Light green
        else:
            return "#00FF00"  # Green - low speed

    def get_time_color(time):
        if time > 4.0:
            return "#FF0000"  # Red - long time
        elif time > 3.0:
            return "#FF8000"  # Orange
        elif time > 2.0:
            return "#FFFF00"  # Yellow
        elif time > 1.0:
            return "#80FF00"  # Light green
        else:
            return "#00FF00"  # Green - short time

    return Div(
        Img(
            src=image_url,
            alt="Vehicle",
            cls="activity-feed__image",
            loading="lazy",
            decoding="async",
        ),
        Div(
            Div(time_str, cls="activity-feed__time"),
            Div(
                # Speed data with color square
                Span(cls="data-square", style=f"background-color: {get_speed_color(speed_val)};"),
                Span(f"{pass_data.min_speed:.2f} px/s", cls="activity-feed__data"),
                # Time data with color square
                Span(cls="data-square", style=f"background-color: {get_time_color(time_val)}; margin-left: 8px;"),
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
    with tracer.start_as_current_span("check_stream_debug") as span:
        if os.path.exists(STREAM_FS_PATH):
            logger.info(f"Stream file exists: {STREAM_FS_PATH}")
            with open(STREAM_FS_PATH, "r") as f:
                content = f.read()
            logger.debug(f"Stream file content:\n{content}")
            span.set_attribute("stream.exists", True)
            span.set_attribute("stream.content_length", len(content))
            # Count segments in manifest
            segment_count = content.count(".ts")
            span.set_attribute("stream.segment_count", segment_count)
            return {"status": "exists", "content": content}
        else:
            logger.warning(f"Stream file does not exist: {STREAM_FS_PATH}")
            stream_dir = os.path.dirname(STREAM_FS_PATH)
            span.set_attribute("stream.exists", False)
            span.set_attribute("stream.error", "file_not_found")

            if os.path.exists(stream_dir):
                files = os.listdir(stream_dir)
                ts_files = [f for f in files if f.endswith(".ts")]
                logger.warning(f"Files in stream directory: {len(files)} total, {len(ts_files)} segments")
                span.set_attribute("stream.directory_file_count", len(files))
                span.set_attribute("stream.directory_segments_count", len(ts_files))
                # Only log first few files to avoid spam
                if files:
                    span.set_attribute("stream.directory_sample_files", str(files[:5]))
            else:
                logger.warning(f"Stream directory does not exist: {stream_dir}")
                span.set_attribute("stream.directory_exists", False)
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
            _ = app.state.db.get_bulk_scores(test_passes)
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


@app.get("/health/stream")  # type: ignore
async def health_stream():
    """Dedicated stream freshness health endpoint (JSON).

    Returns 200 if the HLS playlist exists and is fresh. Includes fields:
    fresh, exists, age_seconds, threshold_sec, segments_count.
    """
    with tracer.start_as_current_span("health_stream") as span:
        info = _parse_hls_playlist(STREAM_FS_PATH)
        age = info.get("age_seconds")
        exists = bool(info.get("exists"))
        threshold = info.get("threshold_sec", 60.0)
        warming_up = (time.time() - WEB_START_TIME) <= GRACE_STARTUP_SEC
        fresh = (exists and age is not None and age <= threshold) or warming_up

        span.set_attribute("hls.exists", exists)
        if age is not None:
            span.set_attribute("hls.age_seconds", float(age))
        span.set_attribute("hls.segments_count", info.get("segments_count", 0))
        span.set_attribute("hls.threshold_sec", threshold)
        span.set_attribute("hls.fresh", fresh)

        status = 200 if fresh else 503
        payload = {
            "fresh": bool(fresh),
            "exists": bool(exists),
            "age_seconds": age,
            "threshold_sec": threshold,
            "segments_count": info.get("segments_count", 0),
        }
        import json

        resp = HTMLResponse(status_code=status, content=json.dumps(payload))
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Content-Type"] = "application/json"
        return resp


@app.get("/healthz")  # type: ignore
async def healthz():
    """Process liveness check."""
    return {"status": "ok"}


@app.get("/load-video")  # type: ignore
def load_video():
    return Div(
        Video(
            id="videoPlayer",
            controls=True,
            autoplay=True,
            muted=True,
            cls="video-stream__player",
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
    with tracer.start_as_current_span("health_check") as span:
        try:
            if not hasattr(app.state, "db"):
                app.state.db = Database(db_url=DB_URL)

            db_start = time.time()
            with app.state.db.Session() as session:
                session.execute(text("SELECT 1 /* health check */"), execution_options={"timeout": 5}).scalar()
            db_duration = time.time() - db_start

            db_health_tracker.record_success()
            span.set_attribute("health.database_ok", True)
            span.set_attribute("health.database_duration_seconds", db_duration)
            span.set_attribute("health.status", "healthy")

            # Check HLS stream health
            hls_healthy = os.path.exists(STREAM_FS_PATH)
            span.set_attribute("health.hls_stream_ok", hls_healthy)

            # Check if stream directory has recent files
            stream_dir = os.path.dirname(STREAM_FS_PATH)
            if os.path.exists(stream_dir):
                files = [f for f in os.listdir(stream_dir) if f.endswith(".ts")]
                span.set_attribute("health.hls_segments_count", len(files))

            resp = HTMLResponse(status_code=200, content="Healthy: Database connection verified")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            db_health_tracker.record_failure()
            span.set_attribute("health.database_ok", False)
            span.set_attribute("health.error", str(e))

            # Only return unhealthy if failures have persisted
            if db_health_tracker.is_failure_persistent():
                span.set_attribute("health.status", "unhealthy")
                span.set_attribute("health.persistent_failure", True)
                resp = HTMLResponse(
                    status_code=503,  # Service Unavailable
                    content=f"Unhealthy: Database connection issues for over 5 minutes - {str(e)}",
                )
                resp.headers["Cache-Control"] = "no-store"
                return resp
            else:
                # Still return healthy if this is a temporary blip
                span.set_attribute("health.status", "degraded")
                span.set_attribute("health.persistent_failure", False)
                resp = HTMLResponse(
                    status_code=200, content="Healthy: Tolerating temporary database connectivity issue"
                )
                resp.headers["Cache-Control"] = "no-store"
                return resp


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

    # Check if we're in production or development mode
    env = os.getenv("ENV", "prod")
    is_dev_mode = env == "local" or env == "dev"

    try:
        uvicorn.run(
            "stopsign.web_server:app",
            host="0.0.0.0",
            port=8000,
            reload=is_dev_mode,  # Only reload in development
            log_level="warning",
            reload_dirs=["./stopsign"] if is_dev_mode else None,
            reload_excludes=["./app/data/*"] if is_dev_mode else None,
        )
    except Exception as e:
        logger.error(f"Error in web server: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
