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
from fasthtml.common import Style
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


def get_common_styles():
    return Style("""
            :root {
                --bg-color: #1a1a1a;
                --text-color: #e0e0e0;
                --accent-color: #ff6600;
                --secondary-color: #8b00ff;
                --card-bg: #2a2a2a;
            }
            body {
                font-family: 'Roboto', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                line-height: 1.6;
            }
            header {
                background-color: var(--bg-color);
                border-bottom: 1px solid var(--accent-color);
            }
            h1, h2, h3 {
                color: var(--accent-color);
                text-shadow: 2px 2px 4px rgba(255, 102, 0, 0.5);
            }
            a {
                color: var(--secondary-color);
                text-decoration: none;
                transition: color 0.3s ease;
            }
            a:hover {
                color: var(--accent-color);
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }
            .card {
                background-color: var(--card-bg);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(255, 102, 0, 0.2);
                transition: transform 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
            }
            button {
                background-color: var(--accent-color);
                color: var(--bg-color);
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            button:hover {
                background-color: var(--secondary-color);
            }
            code {
                font-family: 'Roboto Mono', monospace;
                background-color: #3a3a3a;
                padding: 2px 4px;
                border-radius: 4px;
            }
            nav a {
                color: var(--accent-color);
                text-decoration: none;
                transition: color 0.3s ease;
                margin-left: 15px;
            }
            nav a:hover {
                color: var(--secondary-color);
            }
            ::-webkit-scrollbar {
                width: 10px;
            }

            ::-webkit-scrollbar-track {
                background: var(--bg-color);
                border-radius: 5px;
            }

            ::-webkit-scrollbar-thumb {
                background: var(--accent-color);
                border-radius: 5px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: var(--secondary-color);
            }

            /* For Firefox */
            * {
                scrollbar-width: thin;
                scrollbar-color: var(--accent-color) var(--bg-color);
            }
        """)


def get_common_header(title):
    return Header(
        H1(title, style="text-align: center; flex-grow: 1;"),
        Nav(
            A("Home", href="/"),
            A("Records", href="/records"),
            # A("Statistics", href="/statistics"),
            A("About", href="/about"),
            A("GitHub", href="https://github.com/cipher982/stopsign_ai", target="_blank"),
            style="margin-left: 20px;",
        ),
        style="padding: 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--accent-color);",
    )


def get_common_footer():
    return Footer(
        P("By David Rose"),
        style="padding: 10px; text-align: center; border-top: 1px solid var(--accent-color);",
    )


# Initialize FastHTML app
app = FastHTML(
    ws_hdr=True,
    pico=False,
    hdrs=(
        Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Roboto+Mono&display=swap",
        ),
        get_common_styles(),
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
            Style("""
                .content-wrapper {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .video-container {
                    position: relative;
                    width: 100%;
                }
                #videoFrame {
                    width: 100%;
                    height: auto;
                    border: 1px solid var(--accent-color);
                    border-radius: 8px;
                }
                #videoPlayer {
                    width: 100%;
                    height: auto;
                    border: 1px solid var(--accent-color);
                    border-radius: 8px;
                }
                .video-container {
                    position: relative;
                    display: inline-block;
                }
                #selectionCanvas {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                }
                .recent-passes {
                    width: 100%;
                    max-width: 500px;
                }
                @media (min-width: 768px) {
                    .content-wrapper {
                        flex-direction: row;
                    }
                    .video-container, .recent-passes {
                        flex: 1;
                    }
                }
            """),
        ),
        Body(
            get_common_header("Stop Sign Nanny"),
            Main(
                Div(
                    Div(
                        id="videoContainer",
                        hx_get="/load-video",
                        hx_trigger="load",
                    ),
                    cls="video-container",
                ),
                # Stop line adjustment panel (hidden by default, activated by Ctrl+Shift+A)
                Div(
                    H3("Stop Line Adjustment", style="margin-bottom: 15px; color: var(--accent-color);"),
                    # Click-to-set interface
                    Div(
                        Button(
                            "Adjust Stop Line",
                            id="adjustmentModeBtn",
                            onclick="toggleAdjustmentMode()",
                            style="padding: 10px 20px; margin: 10px 0; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;",
                        ),
                        P(
                            "Click the button above, then click two points on the video to set the new stop line position.",
                            style="margin: 10px 0; font-size: 14px; color: var(--text-color);",
                        ),
                        style="margin-bottom: 20px;",
                    ),
                    # Manual coordinate input (legacy interface)
                    Details(
                        Summary(
                            "Manual Coordinate Input", style="cursor: pointer; font-weight: bold; margin-bottom: 10px;"
                        ),
                        Div(
                            Div(
                                Label("Point 1 - X:", style="display: inline-block; width: 80px;"),
                                Input(type="number", id="x1", value="550", style="width: 80px; margin: 5px;"),
                                Label("Y:", style="margin-left: 10px;"),
                                Input(type="number", id="y1", value="500", style="width: 80px; margin: 5px;"),
                                style="margin-bottom: 10px;",
                            ),
                            Div(
                                Label("Point 2 - X:", style="display: inline-block; width: 80px;"),
                                Input(type="number", id="x2", value="400", style="width: 80px; margin: 5px;"),
                                Label("Y:", style="margin-left: 10px;"),
                                Input(type="number", id="y2", value="550", style="width: 80px; margin: 5px;"),
                                style="margin-bottom: 15px;",
                            ),
                            Button(
                                "Update Stop Zone",
                                onclick="updateStopZone()",
                                style="padding: 8px 16px; background-color: var(--accent-color); color: white; border: none; border-radius: 4px; cursor: pointer;",
                            ),
                            style="padding: 10px; background-color: var(--card-bg); border-radius: 5px;",
                        ),
                    ),
                    # Status display
                    Div(
                        id="status",
                        style="margin-top: 15px; padding: 10px; border-radius: 5px; background-color: var(--card-bg); min-height: 20px; font-weight: bold;",
                    ),
                    id="adjustmentPanel",
                    style="display: none; margin: 20px 0; padding: 20px; background-color: var(--bg-secondary); border-radius: 10px; border: 2px solid var(--accent-color);",
                ),
                Div(
                    H2("Recent Vehicle Passes"),
                    Div(
                        Div(id="recentPasses"),
                        style="overflow-y: auto; max-height: 70vh;",
                    ),
                    cls="recent-passes",
                ),
                cls="content-wrapper",
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
            Script("""
                let adjustmentMode = false;
                let clickedPoints = [];
                let currentZoneType = 'stop-line';  // Current zone being edited
                let debugZonesVisible = false;
                
                // Zone configuration
                const zoneConfig = {
                    'stop-line': {
                        name: 'Stop Line',
                        color: '#c0c0c0',
                        bgColor: '#c0c0c0',
                        clicksRequired: 2,
                        description: 'Click two points to define where vehicles must stop'
                    },
                    'pre-stop': {
                        name: 'Pre-Stop Zone',
                        color: '#c0c0c0',
                        bgColor: '#c0c0c0',
                        clicksRequired: 2,
                        description: 'Click two points to set detection range for approaching vehicles'
                    },
                    'capture': {
                        name: 'Image Capture Zone',
                        color: '#c0c0c0',
                        bgColor: '#c0c0c0',
                        clicksRequired: 2,
                        description: 'Click two points to set optimal photo capture range'
                    }
                };
                
                function selectZoneType(zoneType) {
                    currentZoneType = zoneType;
                    clickedPoints = [];
                    clearClickMarkers();
                    
                    // Update zone selector buttons
                    const buttons = document.querySelectorAll('.zone-selector');
                    buttons.forEach(btn => {
                        btn.classList.remove('active');
                        btn.style.backgroundColor = '';
                        btn.style.color = '';
                    });
                    
                    const activeButton = document.getElementById('zone-' + zoneType);
                    activeButton.classList.add('active');
                    
                    // Update instructions
                    const instructions = document.getElementById('zone-instructions');
                    instructions.innerText = zoneConfig[zoneType].description;
                    
                    // Update status
                    const status = document.getElementById('status');
                    status.innerText = `Ready to adjust ${zoneConfig[zoneType].name}`;
                    
                    // Reset adjustment mode button
                    const adjustBtn = document.getElementById('adjustmentModeBtn');
                    adjustBtn.innerText = `Adjust ${zoneConfig[zoneType].name}`;
                }
                
                function toggleDebugZones() {
                    debugZonesVisible = !debugZonesVisible;
                    
                    fetch('/api/toggle-debug-zones', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ enabled: debugZonesVisible })
                    })
                    .then(response => response.json())
                    .then(data => {
                        const button = document.getElementById('debugZonesBtn');
                        button.innerText = debugZonesVisible ? 'Hide Debug Zones' : 'Show Debug Zones';
                        
                        const status = document.getElementById('status');
                        status.innerText = debugZonesVisible ? 'Debug zones visible on video' : 'Debug zones hidden';
                    })
                    .catch(error => {
                        console.error('Error toggling debug zones:', error);
                    });
                }
                
                function toggleAdjustmentMode() {
                    adjustmentMode = !adjustmentMode;
                    clickedPoints = [];
                    
                    const video = document.getElementById('videoPlayer');
                    const button = document.getElementById('adjustmentModeBtn');
                    const status = document.getElementById('status');
                    const config = zoneConfig[currentZoneType];
                    
                    if (adjustmentMode) {
                        video.style.cursor = 'crosshair';
                        video.style.outline = `3px solid ${config.color}`;
                        button.innerText = 'Cancel Adjustment';
                        status.innerText = `ADJUSTMENT MODE: ${config.description}`;
                    } else {
                        video.style.cursor = 'default';
                        video.style.outline = 'none';
                        button.innerText = `Adjust ${config.name}`;
                        status.innerText = '';
                        clearClickMarkers();
                    }
                }
                
                function handleVideoClick(event) {
                    if (!adjustmentMode) return;
                    
                    const config = zoneConfig[currentZoneType];
                    
                    // Prevent more clicks than required
                    if (clickedPoints.length >= config.clicksRequired) return;
                    
                    const video = event.target;
                    const rect = video.getBoundingClientRect();
                    
                    // Get click coordinates relative to video element
                    const x = event.clientX - rect.left;
                    const y = event.clientY - rect.top;
                    
                    console.log('Click debug:', {
                        zoneType: currentZoneType,
                        browserClick: { x, y },
                        videoElement: { width: rect.width, height: rect.height },
                        actualVideo: { width: video.videoWidth, height: video.videoHeight }
                    });
                    
                    clickedPoints.push({x: x, y: y});
                    
                    // Position marker at exact click location with zone color
                    addClickMarker(rect.left + x, rect.top + y, clickedPoints.length, config.color);
                    
                    const status = document.getElementById('status');
                    const submitBtn = document.getElementById('submitBtn');
                    
                    if (clickedPoints.length === 1) {
                        status.innerText = `POINT 1 SET ✓ - Now click the second point for ${config.name}.`;
                    } else if (clickedPoints.length === config.clicksRequired) {
                        status.innerText = `ALL POINTS SET ✓ - Click SUBMIT to update ${config.name}.`;
                        submitBtn.style.display = 'inline-block';
                        submitBtn.disabled = false;
                    }
                }
                
                function updateZoneFromClicks() {
                    const video = document.getElementById('videoPlayer');
                    const config = zoneConfig[currentZoneType];
                    
                    const data = {
                        zone_type: currentZoneType,
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
                    
                    const endpoint = currentZoneType === 'stop-line' ? 
                        '/api/update-stop-zone-from-display' : 
                        '/api/update-zone-from-display';
                    
                    fetch(endpoint, {
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
                            status.innerText = `✅ SUCCESS! ${config.name} updated successfully.`;
                            
                            // Show coordinate details
                            console.log('Coordinate transformation details:', data);
                            
                            setTimeout(() => {
                                status.innerText = 'Ready for next adjustment.';
                                toggleAdjustmentMode();
                            }, 3000);
                        } else {
                            status.innerText = '❌ ERROR: ' + (data.error || 'Unknown error occurred');
                        }
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                        const status = document.getElementById('status');
                        status.innerText = '❌ NETWORK ERROR: Could not update stop line.';
                    });
                }
                
                function resetPoints() {
                    clickedPoints = [];
                    clearClickMarkers();
                    const status = document.getElementById('status');
                    const submitBtn = document.getElementById('submitBtn');
                    status.innerText = 'Points cleared. Click two new points on the video.';
                    submitBtn.style.display = 'none';
                    submitBtn.disabled = true;
                }
                
                function addClickMarker(pageX, pageY, pointNumber, color = '#ff0000') {
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
                
                function clearClickMarkers() {
                    const markers = document.querySelectorAll('.click-marker');
                    markers.forEach(marker => marker.remove());
                }
                
                function debugCoordinates() {
                    const video = document.getElementById('videoPlayer');
                    if (!video) return;
                    
                    const data = {
                        display_points: [{x: 100, y: 100}, {x: 500, y: 300}],
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
                        document.getElementById('debugOutput').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    });
                }
                
                function showCoordinateInfo() {
                    fetch('/api/coordinate-info')
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                document.getElementById('coordOutput').innerHTML = '<div style="color: #ff6b6b; padding: 10px; background: #2a1f1f; border-radius: 5px;">Error: ' + data.error + '<br><br>This usually means the video analyzer is not running yet. Try starting the video processing service first.</div>';
                            } else {
                                document.getElementById('coordOutput').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                            }
                        })
                        .catch(error => {
                            document.getElementById('coordOutput').innerHTML = '<div style="color: #ff6b6b;">Network error: ' + error.message + '</div>';
                        });
                }
                
                document.addEventListener('DOMContentLoaded', function() {
                    // Initialize the interface
                    selectZoneType('stop-line'); // Set default zone
                    
                    setTimeout(() => {
                        const video = document.getElementById('videoPlayer');
                        if (video) {
                            video.addEventListener('click', handleVideoClick);
                        }
                    }, 1000);
                });
            """),
            Style("""
                body { 
                    font-family: 'MS Sans Serif', sans-serif; 
                    margin: 0; 
                    padding: 6px; 
                    background: #c0c0c0;
                    font-size: 13px;
                    color: black; 
                }
                .container { max-width: none; margin: 0; }
                
                h1 { 
                    text-align: center; 
                    color: black; 
                    font-size: 18px; 
                    margin: 4px 0 6px 0; 
                    font-weight: bold;
                }
                
                h2, h3 { 
                    font-size: 14px; 
                    margin: 3px 0; 
                    font-weight: bold;
                    color: #000080;
                }
                
                .debug-card { 
                    background: #f0f0f0; 
                    border: 2px inset #c0c0c0; 
                    padding: 6px; 
                    margin: 3px; 
                    display: inline-block;
                    vertical-align: top;
                    width: calc(50% - 12px);
                }
                
                .full-width { width: calc(100% - 12px); }
                
                .zone-selector { 
                    padding: 4px 12px; 
                    border: 1px outset #c0c0c0; 
                    background: #c0c0c0;
                    color: black;
                    cursor: pointer; 
                    font-size: 13px;
                    margin: 2px;
                    min-width: 90px;
                }
                
                .zone-selector:active { 
                    border: 1px inset #c0c0c0; 
                }
                
                .zone-selector.active {
                    border: 1px inset #c0c0c0;
                    background: #a0a0a0;  /* Darker pressed look */
                }
                
                .viz-btn, .action-btn { 
                    padding: 4px 12px; 
                    border: 1px outset #c0c0c0; 
                    background: #c0c0c0;
                    cursor: pointer; 
                    font-size: 13px;
                    margin: 2px;
                }
                
                .viz-btn:active, .action-btn:active { 
                    border: 1px inset #c0c0c0; 
                }
                
                .action-btn.submit { 
                    background: #008000; 
                    color: white;
                }
                
                video { 
                    max-width: 100%; 
                    border: 1px solid #808080; 
                    display: block; 
                }
                
                .video-wrapper { position: relative; display: inline-block; }
                .click-marker { position: absolute; pointer-events: none; }
                
                pre { 
                    background: black; 
                    color: #00ff00; 
                    padding: 6px; 
                    font-size: 11px; 
                    line-height: 1.2;
                    font-family: 'Courier New', monospace;
                    border: 1px inset #c0c0c0;
                    height: 140px;
                    overflow-y: scroll;
                }
                
                .debug-tools { 
                    background: #f0f0f0; 
                    border: 2px inset #c0c0c0; 
                    padding: 6px; 
                    margin: 3px;
                }
                
                .debug-tools button { 
                    background: #c0c0c0; 
                    border: 1px outset #c0c0c0; 
                    padding: 3px 8px; 
                    margin: 2px; 
                    cursor: pointer;
                    font-size: 12px;
                }
                
                .debug-tools button:active { 
                    border: 1px inset #c0c0c0; 
                }
                
                p { 
                    margin: 3px 0; 
                    font-size: 12px; 
                    line-height: 1.3;
                }
                
                #status { 
                    background: black; 
                    color: #00ff00; 
                    padding: 4px; 
                    font-size: 12px; 
                    border: 1px inset #c0c0c0;
                    font-family: 'Courier New', monospace;
                    min-height: 20px;
                }
                
                .instructions-box {
                    background: #ffffcc;
                    border: 1px solid #808080;
                    padding: 4px;
                    font-size: 12px;
                    margin: 3px 0;
                }
            """),
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
                        style="width: 70%;",
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
                                style="border: 1px inset #c0c0c0; padding: 4px; margin: 2px 0; background: #e0e0e0;",
                            ),
                            P(id="zone-instructions", style="font-style: italic; margin: 2px 0; font-size: 11px;"),
                            style="margin-bottom: 4px;",
                        ),
                        # Visualization
                        Div(
                            H3("2. Visualization"),
                            Button("Show Zones", id="debugZonesBtn", onclick="toggleDebugZones()", cls="viz-btn"),
                            style="margin-bottom: 4px;",
                        ),
                        # Actions
                        Div(
                            H3("3. Edit Zone"),
                            Button(
                                "Adjust", id="adjustmentModeBtn", onclick="toggleAdjustmentMode()", cls="action-btn"
                            ),
                            Button("Reset", onclick="resetPoints()", cls="action-btn"),
                            Button(
                                "SUBMIT",
                                id="submitBtn",
                                onclick="updateZoneFromClicks()",
                                cls="action-btn",
                                style="display: none;",
                                disabled=True,
                            ),
                            style="margin-bottom: 4px;",
                        ),
                        # Status
                        Div(
                            H3("4. Status"),
                            Div(id="status", style="height: 40px;"),
                            Div(
                                P(
                                    "Select zone → Show zones → Adjust → Click 2 points → Submit",
                                    cls="instructions-box",
                                ),
                            ),
                        ),
                        cls="debug-card",
                        style="width: 28%;",
                    ),
                    style="display: flex; gap: 6px; align-items: flex-start;",
                ),
                # Debug tools in compact form
                Div(
                    H3("Debug Tools"),
                    Button("Coord Info", onclick="showCoordinateInfo()"),
                    Button("Debug Transforms", onclick="debugCoordinates()"),
                    Div(id="coordOutput"),
                    Div(id="debugOutput"),
                    cls="debug-tools full-width",
                ),
                cls="container",
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
                    style="margin-bottom: 20px; background-color: var(--card-bg); padding: 15px; border-radius: 8px;",
                )
                for pass_data in recent_passes
            ],
            style="list-style-type: none; padding: 0; margin: 0;",
        )

        return Div(
            passes_list,
            id="recentPasses",
            style="background-color: var(--card-bg); padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,255,157,0.1);",
        )
    except Exception as e:
        logger.error(f"Error in get_recent_vehicle_passes: {str(e)}")
        return Div(P(f"Error: {str(e)}"), id="recentPasses")


@app.get("/records")  # type: ignore
def records():
    return Html(
        Head(
            Title("Records - Stop Sign Nanny"),
            Script(src="https://unpkg.com/htmx.org@1.9.4"),
        ),
        Body(
            get_common_header("Records"),
            get_common_styles(),
            Main(
                Div(
                    H2("Vehicle Records", style="text-align: center;"),
                    Div(
                        Div(id="worstPasses", hx_get="/api/worst-passes", hx_trigger="load"),
                        Div(id="bestPasses", hx_get="/api/best-passes", hx_trigger="load"),
                        style="display: flex; justify-content: center; gap: 20px;",
                    ),
                    cls="container",
                    style="margin: 20px auto; max-width: 1200px;",
                ),
            ),
            get_common_footer(),
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
            style="display: flex; flex-direction: column; gap: 20px;",
        ),
        H4("Time in Stop Zone"),
        Div(
            *[
                create_pass_item(pass_data, scores_dict[(pass_data.min_speed, pass_data.time_in_zone)])
                for pass_data in time_passes
            ],
            style="display: flex; flex-direction: column; gap: 20px;",
        ),
        id=div_id,
        cls="card",
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
                style="width: 200px; height: auto; border-radius: 5px;",
            ),
            style="flex: 0 0 200px; margin-right: 15px;",
        ),
        Div(
            Div(
                Div(
                    Span("Speed Score: ", style="font-weight: bold; font-size: 1.1em;"),
                    Span(
                        f"{scores['speed_score']}",
                        style=f"font-weight: bold; font-size: 1.1em; color: hsl({scores['speed_score'] * 12}, 100%, 50%);",
                    ),
                ),
                Div(
                    f"Min Speed: {pass_data.min_speed:.2f} pixels/sec",
                    style="font-size: 0.9em; margin-left: 10px;",
                ),
                style="margin-bottom: 10px;",
            ),
            Div(
                Div(
                    Span("Time Score: ", style="font-weight: bold; font-size: 1.1em;"),
                    Span(
                        f"{scores['time_score']}",
                        style=f"font-weight: bold; font-size: 1.1em; color: hsl({scores['time_score'] * 12}, 100%, 50%);",
                    ),
                ),
                Div(
                    f"Time in Zone: {pass_data.time_in_zone:.2f} seconds",
                    style="font-size: 0.9em; margin-left: 10px;",
                ),
                style="margin-bottom: 3px;",
            ),
            style="flex: 1;",
        ),
        style="display: flex; align-items: flex-start;",
    )


@app.get("/statistics")  # type: ignore
def statistics():
    return Html(
        Head(
            Title("Statistics - Stop Sign Nanny"),
        ),
        Body(
            get_common_header("Statistics"),
            get_common_styles(),
            Main(
                Div(
                    Iframe(
                        src=GRAFANA_URL,
                        width="100%",
                        height="600",
                        frameborder="0",
                    ),
                    cls="container",
                ),
            ),
            get_common_footer(),
        ),
    )


@app.get("/about")  # type: ignore
def about():
    with open("static/summary.md", "r") as file:
        summary_content = file.read()
    return Html(
        Head(
            Title("About - Stop Sign Nanny"),
        ),
        Body(
            get_common_header("About"),
            get_common_styles(),
            Main(
                Div(
                    H2("Project Summary"),
                    P(summary_content),
                    cls="container",
                    style="margin: 20px auto; padding: 20px; background-color: var(--card-bg); border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 255, 157, 0.1); white-space: pre-wrap; max-width: 800px;",
                ),
            ),
            get_common_footer(),
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


def main(config: Config):
    try:
        app.state.db = Database(db_url=DB_URL)
        uvicorn.run(
            "stopsign.web_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for live development
            log_level="warning",
            reload_dirs=["./stopsign"],  # Watch for changes in stopsign directory
            reload_excludes=["./app/data/*"],  # Exclude data files from watching
        )
    except Exception as e:
        logger.error(f"Error in web server: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    config = Config("./config.yaml")
    main(config)
