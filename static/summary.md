# stopsign_ai
Tracking stop sign behavior with an IP camera and AI

### Objective:
- Monitor a street intersection with an IP camera streaming video feed via RTSP to a server.
- Analyze the feed in real-time using AI to detect vehicles and evaluate their stop sign behavior.
- Store processed images and metadata for each vehicle pass.
- Display recent vehicle passes and statistics on a web interface.
- Allow user interaction to adjust the stop zone and view analytics.

### System Components and Flow

**IP Camera:**
- Function: Capture and stream video feed.
- Protocol: RTSP (Real-Time Streaming Protocol).

**Backend Server:**
- Function: Process the video feed, detect vehicles, and analyze stop behavior.
- Components:
  - Stream Processor: Handles video processing, object detection, and vehicle tracking.
  - Web Server: Serves the web interface and handles API requests.
- Technologies: Python, OpenCV, YOLO, Redis, SQLite, FastAPI

**AI Model:**
- Function: Detect vehicles in video frames.
- Model: YOLOv8
- Libraries: Ultralytics YOLO, OpenCV

**Storage System:**
- Function: Store processed frames, vehicle images, and metadata.
- Technologies: Redis for frame buffering, SQLite for persistent storage

**Frontend:**
- Function: Display live video feed, recent vehicle passes, and statistics.
- Technologies: FastHTML, WebSocket for real-time updates

### Workflow

**Video Processing:**
- The Stream Processor receives video frames from the RTSP stream.
- Frames are processed using YOLO for vehicle detection.
- Detected vehicles are tracked across frames to analyze their behavior in the stop zone.

**Stop Behavior Analysis:**
- A configurable stop zone is defined in the video frame.
- Vehicle speed and position are monitored as they approach and pass through the stop zone.
- Each vehicle pass is scored based on stopping behavior, duration, and position.

**Data Storage:**
- Processed frames are temporarily stored in Redis for efficient retrieval.
- Vehicle pass data, including scores and cropped images, are stored in SQLite.

**Web Interface:**
- Displays live video feed with overlaid detection and tracking information.
- Shows a list of recent vehicle passes with images and scores.
- Provides an interface to adjust the stop zone.
- Includes a statistics page with embedded Grafana dashboard.

**Monitoring and Analytics:**
- Prometheus metrics are collected for system performance and vehicle statistics.
- Grafana dashboard visualizes long-term trends and real-time data.

### Setup and Deployment

The project is containerized using Docker for easy deployment:

- `Dockerfile.processor`: Builds the container for the Stream Processor.
- `Dockerfile.web`: Builds the container for the Web Server.

Use Docker Compose to orchestrate the full system deployment, including Redis and other necessary services.

### Future Enhancements
- speed up with TensorRT
- ???
