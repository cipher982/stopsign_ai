Stop Sign Nanny – Real-Time Stop Behavior Monitoring

Objective:
• Monitor a busy intersection using an IP camera streaming via RTSP.
• Analyze the video stream in real time to detect, track, and assess vehicle stop behavior.
• Record processed images and metadata for vehicles that fail to stop properly.
• Serve a live video feed (via an HLS stream) alongside interactive records and dashboards.

System Components & Flow:
1. RTSP Stream Capture:
   – A dedicated service connects to an IP camera through RTSP, encodes frames (JPEG), and pushes them into a Redis-backed frame buffer.
   – This service is containerized (see Dockerfile in rtsp_to_redis) and exposes health and Prometheus metrics.

2. Video Analysis Pipeline:
   – Frames are retrieved from Redis and processed by a fully containerized analyzer.
   – YOLO models (ranging from YOLOv8 to YOLOv11 variants available in the repo) are used for real‐time vehicle detection.
   – The pipeline integrates tracking (using both the ByteTrack configuration and an internal Kalman filter implementation) to follow vehicles across frames.
   – A configurable “stop zone” is overlaid on the frame. When vehicles pass through, their speed, dwell time, and overall behavior are measured. Raw image clips are captured and later stored in a MinIO S3-compatible bucket while metadata goes into PostgreSQL.
   – Detailed metrics at every stage (object detection, tracking, and Redis operations) are exported to Prometheus for real-time monitoring.

3. FFmpeg Streaming Service:
   – A separate container uses FFmpeg (with GPU acceleration via h264_nvenc) to convert processed frames into an HLS video stream.
   – This service continuously monitors the processed frame Redis queue, writes frames via FFmpeg into a streaming directory, and keeps the stream updated for web viewers.

4. Web Server and Dashboard:
   – Built on FastHTML, the web server serves the live HLS stream (with automatic client-side handling via HLS.js) and provides pages to view recent vehicle passes.
   – Detailed records (including speed and time scores derived from historical data) are presented interactively.
   – There’s also integration with Grafana for displaying comprehensive system statistics and performance metrics.
   – The About page (updated here) now reflects the expanded functionality and containerized architecture.

5. Deployment and Monitoring:
   – The entire system is orchestrated with Docker Compose. Three dedicated Dockerfiles are available: one for video processing, one for the FFmpeg-based streaming service, and one for the web server.
   – Redis is used as the central frame buffer, while PostgreSQL (accessed via SQLAlchemy) stores vehicle pass histories.
   – System health and performance—including CPU, memory, and temperature metrics—is continuously tracked and exposed to Prometheus.