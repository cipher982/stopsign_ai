"""Stream and video routes."""

import os

from fastapi import APIRouter
from fastapi import Request

from stopsign.web.app import templates

router = APIRouter()


@router.get("/load-video")
async def load_video(request: Request):
    return templates.TemplateResponse("partials/video.html", {"request": request})


@router.get("/check-stream")
async def check_stream(request: Request):
    from stopsign.web.app import STREAM_FS_PATH

    tracer = request.app.state.tracer

    with tracer.start_as_current_span("check_stream_debug") as span:
        if os.path.exists(STREAM_FS_PATH):
            with open(STREAM_FS_PATH, "r") as f:
                content = f.read()
            span.set_attribute("stream.exists", True)
            span.set_attribute("stream.content_length", len(content))
            segment_count = content.count(".ts")
            span.set_attribute("stream.segment_count", segment_count)
            return {"status": "exists", "content": content}
        else:
            stream_dir = os.path.dirname(STREAM_FS_PATH)
            span.set_attribute("stream.exists", False)
            span.set_attribute("stream.error", "file_not_found")

            if os.path.exists(stream_dir):
                files = os.listdir(stream_dir)
                ts_files = [f for f in files if f.endswith(".ts")]
                span.set_attribute("stream.directory_file_count", len(files))
                span.set_attribute("stream.directory_segments_count", len(ts_files))
                if files:
                    span.set_attribute("stream.directory_sample_files", str(files[:5]))
            else:
                span.set_attribute("stream.directory_exists", False)
            return {"status": f"HLS file not found at {STREAM_FS_PATH}"}
