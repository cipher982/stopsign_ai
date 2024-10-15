FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 curl

RUN pip install --upgrade pip
RUN pip install redis opencv-python prometheus_client

WORKDIR /app
COPY . .

CMD ["python", "rtsp_to_redis.py"]