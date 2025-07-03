FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 curl

RUN pip install --upgrade pip
RUN pip install redis opencv-python prometheus_client opentelemetry-distro opentelemetry-exporter-otlp

WORKDIR /app

# Copy the RTSP service files
COPY . .

# Copy the telemetry module from parent directory
COPY ../stopsign/telemetry.py ./stopsign/
RUN touch ./stopsign/__init__.py

CMD ["python", "rtsp_to_redis.py"]