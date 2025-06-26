# ruff: noqa: E501
import logging
import os
import time

import redis
import uvicorn
from fasthtml.common import H1
from fasthtml.common import H2
from fasthtml.common import H3
from fasthtml.common import H4
from fasthtml.common import A
from fasthtml.common import Body
from fasthtml.common import Button
from fasthtml.common import Details
from fasthtml.common import Div
from fasthtml.common import FastHTML
from fasthtml.common import FileResponse
from fasthtml.common import Footer
from fasthtml.common import Head
from fasthtml.common import Header
from fasthtml.common import Html
from fasthtml.common import HTMLResponse
from fasthtml.common import Iframe
from fasthtml.common import Img
from fasthtml.common import Input
from fasthtml.common import Label
from fasthtml.common import Li
from fasthtml.common import Link
from fasthtml.common import Main
from fasthtml.common import Nav
from fasthtml.common import P
from fasthtml.common import Script
from fasthtml.common import Span
from fasthtml.common import StaticFiles
from fasthtml.common import StreamingResponse
from fasthtml.common import Summary
from fasthtml.common import Title
from fasthtml.common import Ul
from fasthtml.common import Video
from minio import Minio
from sqlalchemy import text

from stopsign.config import Config
from stopsign.database import Database
from stopsign.settings import DB_URL
from stopsign.settings import MINIO_ACCESS_KEY
from stopsign.settings import MINIO_BUCKET
from stopsign.settings import MINIO_ENDPOINT
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


def get_common_header(title):
    return Header(
        Div(title, cls="title-bar"),
        Nav(
            A("Home", href="/"),
            A("Records", href="/records"),
            A("About", href="/about"),
            A("GitHub", href="https://github.com/cipher982/stopsign_ai", target="_blank"),
        ),
        cls="window",
    )


def get_common_footer():
    return Footer(
        P("By David Rose"),
        cls="window",
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


@app.get("/vehicle-image/{object_name:path}")  # type: ignore
def get_image(object_name: str):
    # Validate object_name before trying to fetch
    if not object_name or not isinstance(object_name, str) or object_name.strip() == "":
        logger.warning(f"Invalid object_name requested: {object_name}")
        return HTMLResponse("Invalid image request", status_code=400)

    try:
        client = get_minio_client()
        data = client.get_object(MINIO_BUCKET, object_name)
        return StreamingResponse(data, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error fetching image from Minio: {str(e)}", exc_info=True)
        return HTMLResponse("Image not found", status_code=404)


@app.get("/favicon.ico")  # type: ignore
async def favicon():
    return FileResponse("static/favicon.ico")


@app.get("/")  # type: ignore
def home():
    return Html(
        Head(
            Title("Stop Sign Nanny"),
            Script(src="https://unpkg.com/htmx.org@1.9.4"),
            Script(src="https://cdn.jsdelivr.net/npm/hls.js@latest"),
            Script("""
                // Stop zone adjustment state
                let adjustmentMode = false;
                let clickedPoints = [];
                let coordinateInfo = null;
                let currentZoneType = 'stop-line';  // Default zone type
                
                function updateStopZone() {
                    const x1 = document.getElementById('x1').value;
                    const y1 = document.getElementById('y1').value;
                    const x2 = document.getElementById('x2').value;
                    const y2 = document.getElementById('y2').value;
                    
                    const points = [
                        {x: parseFloat(x1), y: parseFloat(y1)},
                        {x: parseFloat(x2), y: parseFloat(y2)}
                    ];
                    
                    fetch('/api/update-stop-zone', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({points: points}),
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log('Success:', data);
                        document.getElementById("status").innerText = 'Stop zone updated successfully!';
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                        document.getElementById("status").innerText = 'Failed to update stop zone.';
                    });
                }
                
                // Load coordinate system information
                function loadCoordinateInfo() {
                    fetch('/api/coordinate-info')
                        .then(response => response.json())
                        .then(data => {
                            coordinateInfo = data;
                            console.log('Coordinate info loaded:', data);
                        })
                        .catch(error => {
                            console.error('Error loading coordinate info:', error);
                        });
                }
                
                // Toggle click-to-set adjustment mode
                function toggleAdjustmentMode() {
                    adjustmentMode = !adjustmentMode;
                    clickedPoints = [];
                    
                    const video = document.getElementById('videoPlayer');
                    const button = document.getElementById('adjustmentModeBtn');
                    const status = document.getElementById('status');
                    
                    if (adjustmentMode) {
                        video.style.cursor = 'crosshair';
                        video.style.outline = '3px solid #ff0000';
                        button.innerText = 'Cancel Adjustment';
                        status.innerText = 'ADJUSTMENT MODE: Click two points on the video to set the stop line';
                        loadCoordinateInfo();
                    } else {
                        video.style.cursor = 'default';
                        video.style.outline = 'none';
                        button.innerText = 'Adjust Stop Line';
                        status.innerText = '';
                        clearClickMarkers();
                    }
                }
                
                // Handle video clicks for stop line adjustment
                function handleVideoClick(event) {
                    if (!adjustmentMode) return;
                    
                    const video = event.target;
                    const rect = video.getBoundingClientRect();
                    
                    // Get click coordinates relative to video element
                    const x = event.clientX - rect.left;
                    const y = event.clientY - rect.top;
                    
                    // Add click point
                    clickedPoints.push({x: x, y: y});
                    
                    // Add visual marker
                    addClickMarker(event.clientX, event.clientY, clickedPoints.length);
                    
                    const status = document.getElementById('status');
                    
                    if (clickedPoints.length === 1) {
                        status.innerText = 'Good! Now click the second point for the stop line.';
                    } else if (clickedPoints.length === 2) {
                        // Send both points to update the stop zone
                        updateStopZoneFromClicks();
                    }
                }
                
                // Update stop zone using clicked coordinates
                function updateStopZoneFromClicks() {
                    const video = document.getElementById('videoPlayer');
                    
                    const data = {
                        display_points: clickedPoints,
                        video_element_size: {
                            width: video.clientWidth,
                            height: video.clientHeight
                        },
                        actual_video_size: {
                            width: video.videoWidth,
                            height: video.videoHeight
                        }
                    };
                    
                    fetch('/api/update-stop-zone-from-display', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data),
                    })
                    .then(response => response.json())
                    .then(data => {
                        const status = document.getElementById('status');
                        if (data.status === 'success') {
                            status.innerText = 'Stop line updated successfully! Coordinates transformed from display to processing system.';
                            console.log('Coordinate transformation details:', data);
                            
                            // Exit adjustment mode
                            setTimeout(() => {
                                toggleAdjustmentMode();
                            }, 2000);
                        } else {
                            status.innerText = 'Error: ' + (data.error || 'Unknown error occurred');
                            console.error('Update error:', data);
                        }
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                        document.getElementById('status').innerText = 'Network error occurred.';
                    });
                }
                
                // Add visual click markers
                function addClickMarker(pageX, pageY, pointNumber) {
                    const video = document.getElementById('videoPlayer');
                    const rect = video.getBoundingClientRect();
                    
                    // Ensure video has a wrapper for positioning
                    let wrapper = video.parentElement;
                    if (!wrapper || !wrapper.classList.contains('video-wrapper')) {
                        wrapper = document.createElement('div');
                        wrapper.className = 'video-wrapper';
                        wrapper.style.position = 'relative';
                        wrapper.style.display = 'inline-block';
                        video.parentNode.insertBefore(wrapper, video);
                        wrapper.appendChild(video);
                    }
                    
                    // Calculate position relative to video element
                    const relativeX = pageX - rect.left;
                    const relativeY = pageY - rect.top;
                    
                    const marker = document.createElement('div');
                    marker.className = 'click-marker';
                    marker.innerHTML = pointNumber;
                    marker.style.position = 'absolute';
                    marker.style.left = (relativeX - 15) + 'px';
                    marker.style.top = (relativeY - 15) + 'px';
                    marker.style.width = '30px';
                    marker.style.height = '30px';
                    marker.style.backgroundColor = color;
                    marker.style.color = 'white';
                    marker.style.borderRadius = '50%';
                    marker.style.display = 'flex';
                    marker.style.alignItems = 'center';
                    marker.style.justifyContent = 'center';
                    marker.style.fontWeight = 'bold';
                    marker.style.fontSize = '14px';
                    marker.style.zIndex = '9999';
                    marker.style.pointerEvents = 'none';
                    wrapper.appendChild(marker);
                }
                
                // Clear all click markers
                function clearClickMarkers() {
                    const markers = document.querySelectorAll('.click-marker');
                    markers.forEach(marker => marker.remove());
                }

                function fetchRecentPasses() {
                    fetch('/api/recent-vehicle-passes')
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`HTTP error! status: ${response.status}`);
                            }
                            return response.text();
                        })
                        .then(html => {
                            document.getElementById('recentPasses').innerHTML = html;
                        })
                        .catch(error => {
                            console.error('Error fetching recent passes:', error);
                            document.getElementById('recentPasses').innerHTML = `<p>Error fetching data: ${error.message}</p>`;
                        });
                }

                document.addEventListener('DOMContentLoaded', function() {
                    fetchRecentPasses();
                    setInterval(fetchRecentPasses, 30000);
                    
                    // Initialize video event listeners
                    setTimeout(() => {
                        const video = document.getElementById('videoPlayer');
                        if (video) {
                            video.addEventListener('click', handleVideoClick);
                            console.log('Video click handler attached');
                        }
                        loadCoordinateInfo();
                    }, 1000);
                });
                
                // Debug mode for coordinate testing (Ctrl+Shift+D)
                function debugCoordinateTransform(testPoints) {
                    const video = document.getElementById('videoPlayer');
                    if (!video) {
                        console.error('Video element not found');
                        return;
                    }
                    
                    const data = {
                        display_points: testPoints || [
                            {x: 100, y: 100},
                            {x: 500, y: 300}
                        ],
                        video_element_size: {
                            width: video.clientWidth,
                            height: video.clientHeight
                        }
                    };
                    
                    fetch('/api/debug-coordinates', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data),
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log('=== COORDINATE TRANSFORMATION DEBUG ===');
                        console.log('Coordinate System Info:', data.coordinate_system_info);
                        console.log('Transformation Chain:', data.transformation_debug);
                        console.log('Point Transformations:', data.point_transformations);
                        console.log('Current Stop Line (Display Coords):', data.current_stop_line_display);
                        
                        // Show roundtrip errors
                        data.point_transformations.forEach(pt => {
                            const error = Math.sqrt(pt.roundtrip_error.x ** 2 + pt.roundtrip_error.y ** 2);
                            console.log(`Point ${pt.point_index + 1} roundtrip error: ${error.toFixed(2)}px`);
                        });
                    })
                    .catch(error => {
                        console.error('Debug error:', error);
                    });
                }
                
                // Simple setup - no keyboard shortcuts needed
            """),
        ),
        Body(
            get_common_header("Stop Sign Nanny"),
            Main(
                Div(
                    Div(
                        Div(
                            id="videoContainer",
                            hx_get="/load-video",
                            hx_trigger="load",
                        ),
                        cls="sunken",
                    ),
                    Div(
                        H2("Recent Vehicle Passes"),
                        Div(id="recentPasses"),
                        cls="window",
                    ),
                    cls="two-col",
                ),
                # Stop line adjustment panel (hidden by default, activated by Ctrl+Shift+A)
                Div(
                    H3("Stop Line Adjustment"),
                    # Click-to-set interface
                    Div(
                        Button(
                            "Adjust Stop Line",
                            id="adjustmentModeBtn",
                            onclick="toggleAdjustmentMode()",
                        ),
                        P(
                            "Click the button above, then click two points on the video to set the new stop line position.",
                        ),
                    ),
                    # Manual coordinate input (legacy interface)
                    Details(
                        Summary("Manual Coordinate Input"),
                        Div(
                            Div(
                                Label("Point 1 - X:"),
                                Input(type="number", id="x1", value="550"),
                                Label("Y:"),
                                Input(type="number", id="y1", value="500"),
                            ),
                            Div(
                                Label("Point 2 - X:"),
                                Input(type="number", id="x2", value="400"),
                                Label("Y:"),
                                Input(type="number", id="y2", value="550"),
                            ),
                            Button(
                                "Update Stop Zone",
                                onclick="updateStopZone()",
                            ),
                            cls="sunken",
                        ),
                    ),
                    # Status display
                    Div(
                        id="status",
                    ),
                    id="adjustmentPanel",
                    style="display: none;",
                    cls="window",
                ),
            ),
            get_common_footer(),
        ),
    )


@app.post("/api/update-stop-zone")  # type: ignore
async def update_stop_zone(request):
    data = await request.json()
    points = data["points"]
    stop_line = ((points[0]["x"], points[0]["y"]), (points[1]["x"], points[1]["y"]))
    stop_box_tolerance = 10
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
                    "stop_box_tolerance": 10,
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
    return Html(
        Head(
            Title("Stop Sign Debug"),
            Script(src="https://unpkg.com/htmx.org@1.9.4"),
            Script(src="https://cdn.jsdelivr.net/npm/hls.js@latest"),
            Script(src="/static/debug.js"),
        ),
        Body(
            Div(
                H1("Stop Sign Debug Interface"),
                # Main layout: Video left, Controls right
                Div(
                    # Left side - Video
                    Div(
                        H2("Video Stream"),
                        Div(hx_get="/load-video", hx_trigger="load"),
                        cls="debug-card",
                    ),
                    # Right side - All controls stacked vertically
                    Div(
                        # Zone Selection
                        Div(
                            H3("1. Select Zone"),
                            Div(
                                Button(
                                    "Stop Line",
                                    id="zone-stop-line",
                                    onclick="selectZoneType('stop-line')",
                                    cls="zone-selector active",
                                ),
                                Button(
                                    "Pre-Stop",
                                    id="zone-pre-stop",
                                    onclick="selectZoneType('pre-stop')",
                                    cls="zone-selector",
                                ),
                                Button(
                                    "Capture",
                                    id="zone-capture",
                                    onclick="selectZoneType('capture')",
                                    cls="zone-selector",
                                ),
                            ),
                            P(id="zone-instructions"),
                        ),
                        # Visualization
                        Div(
                            H3("2. Visualization"),
                            Button("Show Zones", id="debugZonesBtn", onclick="toggleDebugZones()"),
                        ),
                        # Actions
                        Div(
                            H3("3. Edit Zone"),
                            Button("Adjust", id="adjustmentModeBtn", onclick="toggleAdjustmentMode()"),
                            Button("Reset", onclick="resetPoints()"),
                            Button(
                                "SUBMIT",
                                id="submitBtn",
                                onclick="updateZoneFromClicks()",
                                style="display: none;",
                                disabled=True,
                            ),
                        ),
                        # Status
                        Div(
                            H3("4. Status"),
                            Div(id="status"),
                            Div(
                                P(
                                    "Select zone → Show zones → Adjust → Click 2 points → Submit",
                                ),
                            ),
                        ),
                        cls="debug-card",
                    ),
                    cls="two-col",
                ),
                # Debug tools in compact form
                Div(
                    H3("Debug Tools"),
                    Button("Coord Info", onclick="showCoordinateInfo()"),
                    Button("Debug Transforms", onclick="debugCoordinates()"),
                    Div(id="coordOutput"),
                    Div(id="debugOutput"),
                    cls="window",
                ),
            )
        ),
    )


@app.get("/api/recent-vehicle-passes")  # type: ignore
async def get_recent_vehicle_passes():
    try:
        if not hasattr(app.state, "db"):
            app.state.db = Database(db_url=DB_URL)

        recent_passes = app.state.db.get_recent_vehicle_passes(limit=50)

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
                )
                for pass_data in recent_passes
            ],
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
    return (
        get_common_header("Records"),
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
        get_common_footer(),
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
            image_url = f"/vehicle-image/{object_name}"

    return Div(
        Div(
            Img(
                src=image_url,
                alt="Vehicle Image",
                cls="sunken",
            ),
        ),
        Div(
            Div(
                Div(
                    Span("Speed Score: "),
                    Span(
                        f"{scores['speed_score']}",
                    ),
                ),
                Div(
                    f"Min Speed: {pass_data.min_speed:.2f} pixels/sec",
                ),
            ),
            Div(
                Div(
                    Span("Time Score: "),
                    Span(
                        f"{scores['time_score']}",
                    ),
                ),
                Div(
                    f"Time in Zone: {pass_data.time_in_zone:.2f} seconds",
                ),
            ),
        ),
        cls="two-col",
    )


@app.get("/statistics")  # type: ignore
def statistics():
    return (
        get_common_header("Statistics"),
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
        get_common_footer(),
    )


@app.get("/about")  # type: ignore
def about():
    with open("static/summary.md", "r") as file:
        summary_content = file.read()
    return (
        get_common_header("About"),
        Main(
            Div(
                H2("Project Summary"),
                P(summary_content),
                cls="window",
            ),
        ),
        get_common_footer(),
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


@app.get("/load-video")  # type: ignore
def load_video():
    return Div(
        Video(
            id="videoPlayer",
            controls=True,
            autoplay=True,
            muted=True,
        ),
        Script(f"""
            var video = document.getElementById('videoPlayer');
            console.log('Attempting to load video');
            if (Hls.isSupported()) {{
                console.log('HLS is supported');
                var hls = new Hls({{debug: false}});
                hls.loadSource('{STREAM_URL}');
                hls.attachMedia(video);
                hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                    console.log('Manifest parsed, attempting to play');
                    video.play();
                }});
                hls.on(Hls.Events.ERROR, function(event, data) {{
                    console.error('HLS error:', event, data);
                }});
            }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                console.log('HLS not supported, but can play HLS natively');
                video.src = '{STREAM_URL}';
                video.addEventListener('loadedmetadata', function() {{
                    console.log('Metadata loaded, attempting to play');
                    video.play();
                }});
            }} else {{
                console.error('HLS is not supported and cannot play HLS natively');
            }}
            video.addEventListener('error', function(e) {{
                console.error('Video error:', e);
            }});
            
            // Log video dimensions when metadata is loaded
            video.addEventListener('loadedmetadata', function() {{
                console.log('Video dimensions:', {{
                    videoWidth: video.videoWidth,
                    videoHeight: video.videoHeight,
                    displayWidth: video.clientWidth,
                    displayHeight: video.clientHeight
                }});
            }});
        """),
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
