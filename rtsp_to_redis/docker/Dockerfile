FROM python:3.9-slim

RUN apt-get update && apt-get install -y libgl1

RUN pip install --upgrade pip
RUN pip install redis opencv-python prometheus_client

WORKDIR /app
COPY app/ .

CMD ["python", "rtsp_to_redis.py"]