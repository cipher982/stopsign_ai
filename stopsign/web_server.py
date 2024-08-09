import asyncio
import json
import logging
import os

import redis
import uvicorn
from fasthtml import Body
from fasthtml import Div
from fasthtml import FastHTML
from fasthtml import Head
from fasthtml import Html
from fasthtml import Img
from fasthtml import Script
from fasthtml import Title

from stopsign.config import Config

logger = logging.getLogger(__name__)

# Initialize FastHTML app
app = FastHTML(ws_hdr=True)

# Initialize Redis client
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), db=0)

# Global variables
original_width = 1920
original_height = 1080


async def frame_loop(send):
    while True:
        try:
            # Get the latest frame from Redis
            frame_data = redis_client.lindex("frame_buffer", 0)
            if frame_data:
                frame_dict = json.loads(frame_data)  # type: ignore
                frame = frame_dict["frame"]
                await send(f"data:image/jpeg;base64,{frame}")
            await asyncio.sleep(0.01)  # Short sleep to prevent busy waiting
        except Exception as e:
            logger.error(f"Error in frame loop: {str(e)}")
            await asyncio.sleep(1)  # Longer sleep on error


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


@app.get("/")  # type: ignore
def home():
    return (
        Title("Stop Sign Compliance"),
        Html(
            Head(
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
                Img(id="videoFrame", style="max-width: 100%; height: auto; border: 1px solid black;"),
                Div(id="status"),
            ),
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
