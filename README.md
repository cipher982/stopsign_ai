# stopsign_ai
Tracking stop sign behavior with a camera and AI

🌐 **[Live Demo: crestwoodstopsign.com](https://crestwoodstopsign.com)**
[![Status](https://img.shields.io/uptimerobot/status/m797914657-f517a98377b6b7a2e883d57a)](https://stats.uptimerobot.com/m797914657-f517a98377b6b7a2e883d57a)
[![Uptime](https://img.shields.io/uptimerobot/ratio/30/m797914657-f517a98377b6b7a2e883d57a)](https://stats.uptimerobot.com/m797914657-f517a98377b6b7a2e883d57a)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![UV](https://img.shields.io/badge/package%20manager-uv-4A4A4A.svg)](https://github.com/astral-sh/uv)
[![YOLOv11-XL](https://img.shields.io/badge/AI-YOLOv11--XL-green.svg)](https://github.com/ultralytics/ultralytics)
[![FastHTML](https://img.shields.io/badge/frontend-FastHTML-orange.svg)](https://github.com/davidrose/fasthtml)
[![Redis](https://img.shields.io/badge/cache-Redis-red.svg)](https://redis.io/)
[![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL-blue.svg)](https://www.postgresql.org/)
[![MinIO](https://img.shields.io/badge/storage-MinIO-00C6FF.svg)](https://min.io/)

## Objective:

- **Monitor** a street intersection with a camera streaming video via RTSP.
- **Analyze** the feed in real-time using AI to detect vehicles and evaluate their stop sign behavior.
- **Store** processed images and metadata for each vehicle pass.
- **Display** recent vehicle passes and statistics on a web interface.

### Screenshot
![Night Screenshot](./static/screenshot_afternoon.png)

## System Components and Flow

### 1. RTSP Stream Capture (`rtsp_to_redis.py`)
- **Function:** Captures video frames from the RTSP IP camera.
- **Process:**
  - Connects to the RTSP stream.
  - Encodes frames as JPEG.
  - Pushes frames to Redis.

### 2. Video Analysis (`video_analyzer.py`)
- **Function:** Processes frames from Redis, performs object detection, tracking, and stop sign behavior analysis.
- **Process:**
  - Retrieves frames from Redis.
  - Uses YOLO AI model for vehicle detection.
  - Tracks detected vehicles and analyzes their stop sign behavior.
  - Annotates frames and pushes to new Redis queue.
  - Store data in Postgres, and images in self-hosted S3-style file store (MinIO).

### 3. FFmpeg Streaming (`ffmpeg_service.py`)
- **Function:** Converts processed frames into an HLS video stream for web display.
- **Process:**
  - Consumes annotated frames from Redis.
  - Feeds frames into FFmpeg to generate HLS stream.

### 4. Web Server (`web_server.py`)
- **Function:** Serves the web interface and live video stream.
- **Features:**
  - Displays live HLS video stream.
  - Shows recent vehicle passes with images and scores.
  - Integrates with Grafana for detailed statistics.

### 5. Monitoring and Metrics
- **Prometheus Integration:**
  - All services expose metrics to Prometheus for monitoring.
  - Metrics include frame processing times, FPS, memory usage, and more.
- **Grafana Dashboards:**
  - Visualize collected metrics for real-time monitoring and long-term trends.
