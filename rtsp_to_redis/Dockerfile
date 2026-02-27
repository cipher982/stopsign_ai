FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 curl

RUN pip install --upgrade pip
RUN pip install redis opencv-python prometheus_client opentelemetry-distro opentelemetry-exporter-otlp opentelemetry-instrumentation-redis opentelemetry-instrumentation-requests

WORKDIR /app

# Copy the RTSP service files
COPY rtsp_to_redis/ ./rtsp_to_redis/

# Copy required modules from stopsign directory
COPY stopsign/telemetry.py ./stopsign/
COPY stopsign/service_status.py ./stopsign/
COPY stopsign/freeze_detector.py ./stopsign/
COPY stopsign/settings/ ./stopsign/settings/
RUN touch ./stopsign/__init__.py

# Add /app to Python path so stopsign module can be imported
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Set the working directory to rtsp_to_redis and run the service
WORKDIR /app/rtsp_to_redis
CMD ["python", "rtsp_to_redis.py"]
