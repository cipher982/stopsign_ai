# ruff: noqa: E501

import asyncio
import json
import logging
import os
import time

import redis
import uvicorn
from fasthtml.common import H1
from fasthtml.common import H2
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
from fasthtml.common import Li
from fasthtml.common import Link
from fasthtml.common import Main
from fasthtml.common import Nav
from fasthtml.common import P
from fasthtml.common import Script
from fasthtml.common import StaticFiles
from fasthtml.common import Style
from fasthtml.common import Title
from fasthtml.common import Ul
from uvicorn.config import Config as UvicornConfig

from stopsign.config import Config
from stopsign.database import Database

logger = logging.getLogger(__name__)


def get_common_styles():
    return Style("""
            :root {
                --bg-color: #0a0a0a;
                --text-color: #e0e0e0;
                --accent-color: #00ff9d;
                --secondary-color: #ff00ff;
                --card-bg: #1a1a1a;
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
                box-shadow: 0 4px 6px rgba(0, 255, 157, 0.1);
                transition: transform 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 0 20px;
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
                background-color: #2a2a2a;
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
            A("Statistics", href="/statistics"),
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
    pico=False,  # We'll use our own styles instead of Pico CSS
    hdrs=(
        Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Roboto+Mono&display=swap",
        ),
        get_common_styles(),
    ),
)

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


app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/app/data", StaticFiles(directory="data"), name="data")


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
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`HTTP error! status: ${response.status}`);
                            }
                            return response.text();  // Get the response as text
                        })
                        .then(html => {
                            document.getElementById('recentPasses').innerHTML = html;  // Insert the HTML directly
                        })
                        .catch(error => {
                            console.error('Error fetching recent passes:', error);
                            document.getElementById('recentPasses').innerHTML = `<p>Error fetching data: ${error.message}</p>`;
                        });
                }

                // Call fetchRecentPasses initially and then every 30 seconds
                fetchRecentPasses();
                setInterval(fetchRecentPasses, 30000);

                document.addEventListener('DOMContentLoaded', function() {
                    connectWebSocket();
                    initCanvas();
                });
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
                        Div(
                            Img(id="videoFrame"),
                            Canvas(id="selectionCanvas"),
                            cls="video-container",
                        ),
                        Button(
                            "Select Stop Zone",
                            id="toggleButton",
                            onclick="toggleSelection()",
                            style="margin-top: 10px;",
                        ),
                        Div(id="status"),
                        cls="video-container",
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
                cls="container",
            ),
            get_common_footer(),
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
    try:
        if not hasattr(app.state, "db"):
            app.state.db = Database(db_file=str(os.getenv("SQL_DB_PATH")))
        recent_passes = app.state.db.get_recent_vehicle_passes()

        # Create a styled list of recent passes
        passes_list = Ul(
            *[
                Li(
                    Div(
                        Img(src=pass_data[6], alt="Vehicle Image", style="max-width: 200px; border-radius: 5px;"),
                        Div(
                            P(f"Timestamp: {pass_data[1]}", style="font-weight: bold;"),
                            P(f"Vehicle ID: {pass_data[2]}"),
                            P(f"Stop Score: {pass_data[3]}"),
                            P(f"Stop Duration: {pass_data[4]} seconds"),
                            P(f"Min Speed: {pass_data[5]:.2f} km/h"),
                            style="margin-left: 20px;",
                        ),
                        style="display: flex; align-items: center; background-color: var(--card-bg); padding: 15px; border-radius: 10px; margin-bottom: 15px;",
                    )
                )
                for pass_data in recent_passes
            ],
            style="list-style-type: none; padding: 0;",
        )

        return Div(
            passes_list,
            id="recentPasses",
            style="background-color: var(--card-bg); padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,255,157,0.1);",
        )
    except Exception as e:
        logger.error(f"Error in get_recent_vehicle_passes: {str(e)}")
        return Div(P(f"Error: {str(e)}"), id="recentPasses")


@app.get("/statistics")  # type: ignore
def statistics():
    return Html(
        Head(
            Title("Statistics - Stop Sign Nanny"),
            Script(src="https://unpkg.com/htmx.org@1.9.4"),
        ),
        Body(
            get_common_header("Statistics"),
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
                style="padding: 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--accent-color);",
            ),
            Main(
                Div(
                    id="summary",
                    style="margin: 20px 0; padding: 20px; background-color: var(--card-bg); border-radius: 5px; white-space: pre-wrap;",
                ),
                Script("""
                        fetch('/static/summary.txt')
                            .then(response => response.text())
                            .then(data => {
                                document.getElementById('summary').textContent = data;
                            })
                            .catch(error => console.error('Error fetching summary:', error));
                    """),
                cls="container",
            ),
            Footer(
                P("By David Rose"),
                style="padding: 10px; text-align: center; border-top: 1px solid var(--accent-color);",
            ),
        ),
    )


def main(config: Config):
    try:
        db = Database(db_file=str(os.getenv("SQL_DB_PATH")))
        app.state.db = db

        uvicorn_config = UvicornConfig(
            "stopsign.web_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_excludes=["/app/data"],
        )
        server = uvicorn.Server(uvicorn_config)
        server.run()
    except Exception as e:
        logger.error(f"Error in web server: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = Config("./config.yaml")
    main(config)
