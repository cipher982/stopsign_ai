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
        Script(src="https://unpkg.com/htmx.org@1.9.4"),
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
    logger.info("Starting frame loop")
    while True:
        try:
            # Get the latest frame from Redis
            frame_data = redis_client.lindex("processed_frame_buffer", 0)
            if frame_data:
                frame_dict = json.loads(frame_data)  # type: ignore # noqa: F841
                frame = frame_dict["frame"]
                width = frame_dict.get("width", original_width)
                height = frame_dict.get("height", original_height)

                # Create an HTML snippet to replace the Img tag
                img_html = f"""
                <img id="videoFrame" src="data:image/jpeg;base64,{frame}" width="{width}" height="{height}" alt="Live Stream" class="video-container" />
                """

                logger.info(f"Sending frame: width={width}, height={height}, data length={len(frame)}")
                await send(img_html)  # Send the HTML snippet

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
            logger.error(f"Error in frame loop: {str(e)}")
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
    async for message in websocket.iter_text():
        # Handle incoming messages if needed
        pass


app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/app/data", StaticFiles(directory="data"), name="data")


@app.get("/")  # type: ignore
def home():
    return Html(
        Head(
            Title("Stop Sign Nanny"),
            Script(src="https://unpkg.com/htmx.org@1.9.4"),
            Script(src="https://unpkg.com/htmx.org/dist/ext/ws.js"),  # Add WebSocket extension
            Script("""
                htmx.on("htmx:wsConnected", function(event) {
                    console.log("WebSocket Connected!");
                });
                htmx.on("htmx:wsError", function(event) {
                    console.error("WebSocket Error:", event.detail.error);
                });
            """),
            Script("""
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
                    Img(
                        id="videoFrame",
                        src="",  # Initial empty source
                        alt="Live Stream",
                        cls="video-container",
                        hx_ext="ws",  # Enable WebSocket extension
                        hx_ws="connect:/ws",  # Establish WebSocket connection to /ws
                        hx_swap="outerHTML",  # Replace the entire Img tag on message
                        # hx_trigger="message",  # Trigger on incoming WebSocket messages
                        hx_trigger="load, every 100ms",  # Trigger on load and every 100ms
                    ),
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


def main(config: Config):
    try:
        db = Database(db_file=str(os.getenv("SQL_DB_PATH")))
        app.state.db = db

        uvicorn.run(
            "stopsign.web_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["stopsign"],
            reload_excludes=["data/*"],
        )
    except Exception as e:
        logger.error(f"Error in web server: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = Config("./config.yaml")
    main(config)
