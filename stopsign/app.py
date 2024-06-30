import asyncio
import base64
import os

import cv2
import dotenv
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.responses import HTMLResponse

from stopsign.main import YOLO
from stopsign.main import Config
from stopsign.main import process_frame
from stopsign.main import visualize

dotenv.load_dotenv()

app = FastAPI()

# Load configuration and model
config = Config("./config.yaml")
model = YOLO(os.getenv("YOLO_MODEL_PATH"))  # type: ignore

# Initialize video capture
cap = cv2.VideoCapture(os.getenv("RTSP_URL"))  # type: ignore


@app.get("/")
async def get():
    with open("index.html", "r") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cars = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame, boxes = process_frame(
            model=model,
            frame=frame,
            scale=config.scale,
            crop_top_ratio=config.crop_top,
            crop_side_ratio=config.crop_side,
            vehicle_classes=config.vehicle_classes,
        )

        # Visualize
        annotated_frame = visualize(
            processed_frame,
            cars,
            boxes,
            config.stop_zone,
            frame_count,
        )

        # Encode and send frame
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_data = base64.b64encode(buffer).decode("utf-8")
        await websocket.send_text(frame_data)

        frame_count += 1
        await asyncio.sleep(0.1)  # Adjust as needed


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
