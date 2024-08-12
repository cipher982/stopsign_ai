FROM python:3.9-slim

RUN pip install --upgrade pip
RUN pip install redis

WORKDIR /app
COPY app/ .

CMD ["python", "rtsp_to_redis.py"]