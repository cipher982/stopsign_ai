# ruff: noqa: E501

import asyncio
import json
import logging
import os
import time

import redis
import uvicorn
from fasthtml.common import H1
from fasthtml.common import A
from fasthtml.common import Body
from fasthtml.common import Button
from fasthtml.common import Canvas
from fasthtml.common import Div
from fasthtml.common import FastHTML
from fasthtml.common import Footer
from fasthtml.common import Head
from fasthtml.common import Header
from fasthtml.common import Html
from fasthtml.common import Iframe
from fasthtml.common import Img
from fasthtml.common import Main
from fasthtml.common import Nav
from fasthtml.common import P
from fasthtml.common import Script
from fasthtml.common import StaticFiles
from fasthtml.common import Title

from stopsign.config import Config

logger = logging.getLogger(__name__)

# Initialize FastHTML app
app = FastHTML(ws_hdr=True)

# Initialize Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
)

# Global variables
original_width = 1920
original_height = 1080

# Grafana dashboard URL
GRAFANA_URL = os.getenv("GRAFANA_URL")


async def frame_loop(send):
    last_frame_time = time.time()
    last_error_time = 0
    frames_sent = 0
    error_count = 0
    while True:
        try:
            # Get the latest frame from Redis
            frame_data = redis_client.lindex("processed_frame_buffer", 0)
            if frame_data:
                frame_dict = json.loads(frame_data)  # type: ignore # noqa: F841
                frame = frame_dict["frame"]
                await send(f"data:image/jpeg;base64,{frame}")
                frames_sent += 1

                current_time = time.time()
                if current_time - last_frame_time >= 60:
                    fps = frames_sent / (current_time - last_frame_time)
                    buffer_length = redis_client.llen("processed_frame_buffer")
                    logger.info(f"Web server sending rate: {fps:.2f} fps, Buffer length: {buffer_length}")
                    frames_sent = 0
                    last_frame_time = current_time
                    error_count = 0  # Reset error count every minute if streaming is working
            else:
                logger.warning("No frames available in buffer")
            await asyncio.sleep(0.01)  # Short sleep to prevent busy waiting
        except Exception as e:
            current_time = time.time()
            error_count += 1
            if current_time - last_error_time >= 60:
                logger.error(f"Error in frame loop (occurred {error_count} times in the last minute): {str(e)}")
                last_error_time = current_time
                error_count = 0
            await asyncio.sleep(0.1)  # Slightly longer sleep on error, but not too long


async def on_connect(send):
    logger.info("WebSocket connected")
    dimensions = {"type": "dimensions", "width": original_width, "height": original_height}
    await send(json.dumps(dimensions))
    # Start the message loop
    asyncio.create_task(frame_loop(send))


async def on_disconnect():
    logger.info("WebSocket disconnected")


@app.ws("/ws", conn=on_connect, disconn=on_disconnect)
async def ws_handler(websocket):
    """
    This is required to be defined, but does not work or get called upon ws connection.
    Keep all functionality in the on_connect function.
    """
    logger.info("WebSocket handler started")


# @app.get("/static/summary.txt") # type: ignore
# async def get_summary():
#     return FileResponse("/static/summary.txt")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")  # type: ignore
def home():
    return Html(
        Head(
            Title("Stop Sign Nanny"),
            Script(src="https://unpkg.com/htmx.org@1.9.4"),
            Script("""
                let ws;
                let isSelecting = false;
                let points = [];

                function connectWebSocket() {
                    ws = new WebSocket('ws://' + location.host + '/ws');
                    ws.onmessage = function(event) {
                        if (event.data.startsWith('data:image')) {
                            document.getElementById('videoFrame').src = event.data;
                        } else {
                            const data = JSON.parse(event.data);
                            if (data.type === 'dimensions') {
                                const img = document.getElementById('videoFrame');
                                img.width = data.width;
                                img.height = data.height;
                                const canvas = document.getElementById('selectionCanvas');
                                canvas.width = data.width;
                                canvas.height = data.height;
                            }
                        }
                    };
                    ws.onclose = function() {
                        setTimeout(connectWebSocket, 1000);
                    };
                }

                function toggleSelection() {
                    isSelecting = !isSelecting;
                    document.getElementById('toggleButton').innerText = isSelecting ? 'Finish Selection' : 'Select Stop Zone';
                    if (!isSelecting && points.length === 4) {
                        sendPointsToServer(points);
                    }
                }

                function initCanvas() {
                    const canvas = document.getElementById('selectionCanvas');
                    const video = document.getElementById('videoFrame');
                    canvas.width = video.width;
                    canvas.height = video.height;

                    canvas.onclick = function(e) {
                        if (!isSelecting) return;
                        const rect = canvas.getBoundingClientRect();
                        const x = e.clientX - rect.left;
                        const y = e.clientY - rect.top;
                        points.push({x, y});
                        drawPoints();
                        if (points.length === 4) {
                            toggleSelection();
                        }
                    }
                }

                function drawPoints() {
                    const canvas = document.getElementById('selectionCanvas');
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    points.forEach((point, index) => {
                        ctx.beginPath();
                        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
                        ctx.fillStyle = 'red';
                        ctx.fill();
                        ctx.fillStyle = 'white';
                        ctx.fillText(index + 1, point.x + 10, point.y + 10);
                    });
                }

                function sendPointsToServer(points) {
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
                        alert('Stop zone updated successfully!');
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                        alert('Failed to update stop zone.');
                    });
                }
                   
                function fetchRecentPasses() {
                    fetch('/api/recent-vehicle-passes')
                        .then(response => response.json())
                        .then(data => {
                            const passesDiv = document.getElementById('recentPasses');
                            passesDiv.innerHTML = '';
                            data.forEach(pass => {
                                const passDiv = document.createElement('div');
                                passDiv.innerHTML = `
                                    <img src="/static/${pass.image_path}" alt="Vehicle ${pass.vehicle_id}" style="width: 100px; height: auto;">
                                    <p>Vehicle ID: ${pass.vehicle_id}</p>
                                    <p>Stop Score: ${pass.stop_score}</p>
                                    <p>Stop Duration: ${pass.stop_duration.toFixed(2)}s</p>
                                    <p>Min Speed: ${pass.min_speed.toFixed(2)} px/s</p>
                                    <hr>
                                `;
                                passesDiv.appendChild(passDiv);
                            });
                        })
                        .catch(error => console.error('Error fetching recent passes:', error));
                }

                // Call fetchRecentPasses initially and then every 30 seconds
                fetchRecentPasses();
                setInterval(fetchRecentPasses, 30000);

                document.addEventListener('DOMContentLoaded', function() {
                    connectWebSocket();
                    initCanvas();
                });
            """),
        ),
        Body(
            Header(
                H1("Stop Sign Nanny"),
                Nav(
                    A("Home", href="/"),
                    A("Statistics", href="/statistics"),
                    A("About", href="/about"),
                    style="margin-left: 20px;",
                ),
                style="background-color: #f0f0f0; padding: 20px; display: flex; justify-content: space-between; align-items: center;",
            ),
            Main(
                Div(
                    Img(id="videoFrame", style="max-width: 100%; height: auto; border: 1px solid black;"),
                    Canvas(id="selectionCanvas", style="position: absolute; top: 0; left: 0; pointer-events: none;"),
                    Button(
                        "Select Stop Zone",
                        id="toggleButton",
                        onclick="toggleSelection()",
                        style="margin-top: 10px;",
                    ),
                    Div(id="status"),
                    style="margin: 20px 0; position: relative;",
                ),
                Div(
                    H1("Recent Vehicle Passes"),
                    Div(id="recentPasses"),
                    style="margin-top: 20px;",
                ),
            ),
            Footer(P("By David Rose"), style="background-color: #f0f0f0; padding: 10px; text-align: center;"),
            style="font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 0 20px;",
        ),
    )


@app.post("/api/update-stop-zone")  # type: ignore
async def update_stop_zone(request):
    data = await request.json()
    points = data["points"]
    # Convert to the format expected by your backend
    stop_line = ((points[0]["x"], points[0]["y"]), (points[1]["x"], points[1]["y"]))
    # You might want to calculate these based on the points or allow user input
    stop_box_tolerance = 10
    min_stop_duration = 2.0

    new_config = {
        "stop_line": stop_line,
        "stop_box_tolerance": stop_box_tolerance,
        "min_stop_duration": min_stop_duration,
    }

    # Update the configuration in your StreamProcessor
    app.state.stream_processor.reload_stop_zone_config(new_config)

    return {"status": "success"}


@app.get("/api/recent-vehicle-passes")  # type: ignore
async def get_recent_vehicle_passes():
    recent_passes = app.state.db.get_recent_vehicle_passes()
    return [
        {
            "id": pass_data[0],
            "timestamp": pass_data[1],
            "vehicle_id": pass_data[2],
            "stop_score": pass_data[3],
            "stop_duration": pass_data[4],
            "min_speed": pass_data[5],
            "image_path": pass_data[6],
        }
        for pass_data in recent_passes
    ]


@app.get("/statistics")  # type: ignore
def statistics():
    return Html(
        Head(
            Title("Statistics - Stop Sign Nanny"),
        ),
        Body(
            Header(
                H1("Statistics"),
                Nav(
                    A("Home", href="/"),
                    A("Statistics", href="/statistics"),
                    A("About", href="/about"),
                    style="margin-left: 20px;",
                ),
                style="background-color: #f0f0f0; padding: 20px; display: flex; justify-content: space-between; align-items: center;",
            ),
            Main(
                Iframe(
                    src=GRAFANA_URL,
                    width="100%",
                    height="600",
                    frameborder="0",
                )
            ),
            Footer(P("By David Rose"), style="background-color: #f0f0f0; padding: 10px; text-align: center;"),
            style="font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 0 20px;",
        ),
    )


@app.get("/about")  # type: ignore
def about():
    return Html(
        Head(Title("About - Stop Sign Nanny")),
        Body(
            Header(
                H1("About"),
                Nav(
                    A("Home", href="/"),
                    A("Statistics", href="/statistics"),
                    A("About", href="/about"),
                    style="margin-left: 20px;",
                ),
                style="background-color: #f0f0f0; padding: 20px; display: flex; justify-content: space-between; align-items: center;",
            ),
            Main(
                Div(
                    id="summary",
                    style="margin: 20px 0; padding: 20px; background-color: #f9f9f9; border-radius: 5px; white-space: pre-wrap;",
                ),
                Script("""
                        fetch('/static/summary.txt')
                            .then(response => response.text())
                            .then(data => {
                                document.getElementById('summary').textContent = data;
                            })
                            .catch(error => console.error('Error fetching summary:', error));
                    """),
            ),
            Footer(P("By David Rose"), style="background-color: #f0f0f0; padding: 10px; text-align: center;"),
            style="font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 0 20px;",
        ),
    )


def main(config: Config):
    try:
        # app.state.db = db
        # app.state.stream_processor = StreamProcessor(config)
        uvicorn.run("stopsign.web_server:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Error in web server: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = Config("./config.yaml")
    main(config)
