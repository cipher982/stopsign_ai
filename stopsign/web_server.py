# ruff: noqa: E501

import asyncio
import json
import logging
import os
import time

import redis
import uvicorn
from fasthtml import H1
from fasthtml import A
from fasthtml import Body
from fasthtml import Div
from fasthtml import FastHTML
from fasthtml import Footer
from fasthtml import Head
from fasthtml import Header
from fasthtml import Html
from fasthtml import Img
from fasthtml import Main
from fasthtml import Nav
from fasthtml import P
from fasthtml import Script
from fasthtml import StaticFiles
from fasthtml import Title

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


async def frame_loop(send):
    last_frame_time = time.time()
    last_error_time = 0
    frames_sent = 0
    error_count = 0
    while True:
        try:
            # Get the latest frame from Redis
            frame_data = redis_client.lindex("frame_buffer", 0)
            if frame_data:
                frame_dict = json.loads(frame_data)  # type: ignore # noqa: F841
                frame = frame_dict["frame"]
                await send(f"data:image/jpeg;base64,{frame}")
                frames_sent += 1

                current_time = time.time()
                if current_time - last_frame_time >= 60:
                    fps = frames_sent / (current_time - last_frame_time)
                    buffer_length = redis_client.llen("frame_buffer")
                    logger.info(f"Web server sending rate: {fps:.2f} fps, Buffer length: {buffer_length}")
                    frames_sent = 0
                    last_frame_time = current_time
                    # Reset error count every minute if streaming is working
                    error_count = 0
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
                    document.addEventListener('DOMContentLoaded', function() {
                        var socket = new WebSocket('ws://' + window.location.host + '/ws');
                        socket.onopen = function(event) {
                            console.log("WebSocket connection established");
                        };
                        socket.onmessage = function(event) {
                            var data = event.data;
                            if (data.startsWith('data:image')) {
                                document.getElementById('videoFrame').src = data;
                            } else {
                                try {
                                    var message = JSON.parse(data);
                                    if (message.type === 'status') {
                                        document.getElementById('status').innerText = message.content;
                                    } else if (message.type === 'dimensions') {
                                        console.log("Received dimensions:", message);
                                    }
                                } catch (e) {
                                    console.error("Error parsing message:", e);
                                }
                            }
                        };
                        socket.onerror = function(error) {
                            console.error("WebSocket error:", error);
                        };
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
                    Div(id="status"),
                    style="margin: 20px 0;",
                ),
            ),
            Footer(P("By David Rose"), style="background-color: #f0f0f0; padding: 10px; text-align: center;"),
            style="font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 0 20px;",
        ),
    )


@app.get("/statistics")  # type: ignore
def statistics():
    return Html(
        Head(Title("Statistics - Stop Sign Nanny")),
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
            Main(P("Statistics about stop sign compliance will be displayed here."), style="margin: 20px 0;"),
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


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main(config: Config):
    try:
        run_server()
    except Exception as e:
        logger.error(f"Error in web server: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = Config("./config.yaml")
    main(config)
