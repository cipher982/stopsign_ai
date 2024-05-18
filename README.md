# stopsign_ai
Tracking stopsign runners with an IP camera and AI

### Objective:
- Set up an IP camera to monitor the street and stream/record the video feed via RTSP 24/7 to a server.
- The server will analyze the feed in real-time using an AI model to detect and classify stop sign runners.
- Images will be saved to storage.
- The stored images and classifications will be displayed on a website to show all the stop sign runners that have been recently seen. 
    - optional: classify the cars and group them over time.

### System Components and Flow
**IP Camera:**
- Function: Capture and stream video feed.
- Protocol: RTSP (Real-Time Streaming Protocol).
- Location: Positioned on front porch.

**Backend Server:**
- Function: Receive and record the RTSP feed.
- Tools: FFmpef for streaming and recording, FastAPI for modeling and serving.

**AI Model:**
- Function: Analyze the video feed in real-time to detect and classify the cars.
- Frameworks: PyTorch for model implementation.
- Models: Pre-trained YOLO models?
- Libraries: OpenCV for video processing, TensorRT for optimized inference on NVIDIA GPUs in the backend (3090).

**Storage System:**
- Function: Save images of detected cars and related metadata.
Options:
    - Local or cloud, depending on the scale of the project.

**Frontend:**
- Function: Display the recently detected and classified cars.
Frameworks: React.


### Workflow
**Video Capture and Streaming:**
- The IP camera captures the street's video feed and streams it via RTSP.

**Streaming Server:**
- The server receives the RTSP stream and records the video continuously.
- It processes the video feed in real-time using an AI model.
- The model will detect and classify the cars in the frames and classify whether they stop, roll, or dont slow down at all in the intersection.

**Storage:**
- Images and metadata are stored in a designated storage system.
- Options include local storage or cloud based.

**Web Server and Database:**
- Metadata Storage: Metadata is stored in a database.
- Web Server: The web server handles API requests and serves the frontend application.

**Frontend Web Application:**
- The frontend application fetches data from the web server.
- Displays the detected car images and their classifications on the website.