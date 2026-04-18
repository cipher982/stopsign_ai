# ruff: noqa: E501
"""Infrastructure routes — static files, image proxy, performance debug."""

import hashlib
import logging
import os
import time
from io import BytesIO
from pathlib import Path

import redis
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from minio import Minio
from PIL import Image
from PIL import ImageOps

from stopsign.settings import BREMEN_MINIO_ACCESS_KEY
from stopsign.settings import BREMEN_MINIO_BUCKET
from stopsign.settings import BREMEN_MINIO_ENDPOINT
from stopsign.settings import BREMEN_MINIO_SECRET_KEY
from stopsign.settings import LOCAL_IMAGE_DIR
from stopsign.settings import MINIO_ACCESS_KEY
from stopsign.settings import MINIO_BUCKET
from stopsign.settings import MINIO_ENDPOINT
from stopsign.settings import MINIO_SECRET_KEY
from stopsign.settings import REDIS_URL

logger = logging.getLogger(__name__)

router = APIRouter()

THUMBNAIL_SIZE = (112, 80)
THUMBNAIL_QUALITY = 82
THUMBNAIL_CACHE_DIR = Path(LOCAL_IMAGE_DIR) / ".thumb-cache"


def get_minio_client():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=True)


def get_bremen_minio_client():
    return Minio(
        BREMEN_MINIO_ENDPOINT, access_key=BREMEN_MINIO_ACCESS_KEY, secret_key=BREMEN_MINIO_SECRET_KEY, secure=False
    )


def _normalize_object_name(object_name: str) -> str:
    if not object_name or not isinstance(object_name, str) or object_name.strip() == "":
        raise ValueError("Invalid image request")

    normalized = Path(object_name)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("Invalid image request")

    return normalized.as_posix().lstrip("./")


def _thumbnail_cache_path(object_name: str) -> Path:
    cache_key = hashlib.sha256(f"{THUMBNAIL_SIZE}:{object_name}".encode("utf-8")).hexdigest()[:24]
    stem = Path(object_name).stem[:32] or "thumb"
    return THUMBNAIL_CACHE_DIR / f"{stem}-{cache_key}.jpg"


def _read_source_image_bytes(object_name: str) -> bytes:
    local_path = Path(LOCAL_IMAGE_DIR) / object_name
    if local_path.exists():
        return local_path.read_bytes()

    data = get_bremen_minio_client().get_object(BREMEN_MINIO_BUCKET, object_name)
    try:
        return data.read()
    finally:
        data.close()
        data.release_conn()


def _build_thumbnail_bytes(source_bytes: bytes) -> bytes:
    with Image.open(BytesIO(source_bytes)) as image:
        fitted = ImageOps.fit(
            image.convert("RGB"),
            THUMBNAIL_SIZE,
            method=Image.Resampling.LANCZOS,
        )
        output = BytesIO()
        fitted.save(output, format="JPEG", quality=THUMBNAIL_QUALITY, optimize=True)
        return output.getvalue()


def _thumbnail_response_headers(object_name: str) -> dict[str, str]:
    digest = hashlib.sha256(f"{THUMBNAIL_SIZE}:{object_name}".encode("utf-8")).hexdigest()[:24]
    return {
        "Cache-Control": "public, max-age=31536000, immutable",
        "ETag": f'"thumb-{digest}"',
    }


@router.get("/vehicle-image/{object_name:path}")
def get_image(object_name: str):
    if not object_name or not isinstance(object_name, str) or object_name.strip() == "":
        logger.warning(f"Invalid object_name requested: {object_name}")
        return HTMLResponse("Invalid image request", status_code=400)

    try:
        client = get_bremen_minio_client()
        data = client.get_object(BREMEN_MINIO_BUCKET, object_name)
        response = StreamingResponse(data, media_type="image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["ETag"] = f'"{hash(object_name)}"'
        return response
    except Exception as e:
        logger.error(f"Error fetching image from Bremen MinIO: {str(e)}", exc_info=True)
        return HTMLResponse("Image not found", status_code=404)


@router.get("/vehicle-thumb/{object_name:path}")
def get_thumbnail(object_name: str):
    try:
        normalized_name = _normalize_object_name(object_name)
    except ValueError:
        logger.warning(f"Invalid thumbnail object requested: {object_name}")
        return HTMLResponse("Invalid image request", status_code=400)

    cache_path = _thumbnail_cache_path(normalized_name)
    headers = _thumbnail_response_headers(normalized_name)

    try:
        if cache_path.exists():
            response = FileResponse(cache_path, media_type="image/jpeg")
            response.headers.update(headers)
            return response

        source_bytes = _read_source_image_bytes(normalized_name)
        thumbnail_bytes = _build_thumbnail_bytes(source_bytes)

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(thumbnail_bytes)
        except OSError as cache_error:
            logger.warning(f"Could not persist thumbnail cache for {normalized_name}: {cache_error}")

        return Response(content=thumbnail_bytes, media_type="image/jpeg", headers=headers)
    except Exception as e:
        logger.error(f"Error building thumbnail for {normalized_name}: {str(e)}", exc_info=True)
        return HTMLResponse("Image not found", status_code=404)


@router.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")


@router.get("/robots.txt")
async def robots():
    return FileResponse("static/robots.txt", media_type="text/plain")


@router.get("/sitemap.xml")
async def sitemap():
    return FileResponse("static/sitemap.xml", media_type="application/xml")


@router.get("/debug-performance")
async def debug_performance(request: Request):
    import psutil

    results = {}

    if hasattr(request.app.state, "db"):
        start = time.time()
        try:
            recent_passes = request.app.state.db.get_recent_vehicle_passes(limit=10)
            results["db_recent_passes"] = {
                "time_ms": round((time.time() - start) * 1000),
                "count": len(recent_passes),
                "status": "success",
            }
        except Exception as e:
            results["db_recent_passes"] = {
                "time_ms": round((time.time() - start) * 1000),
                "error": str(e),
                "status": "error",
            }

        start = time.time()
        try:
            test_passes = [{"min_speed": 10.0, "time_in_zone": 2.5, "stop_duration": 1.0}]
            _ = request.app.state.db.get_pass_stop_scores(test_passes)
            results["db_bulk_scores"] = {"time_ms": round((time.time() - start) * 1000), "status": "success"}
        except Exception as e:
            results["db_bulk_scores"] = {
                "time_ms": round((time.time() - start) * 1000),
                "error": str(e),
                "status": "error",
            }

        start = time.time()
        try:
            total = request.app.state.db.get_total_passes_last_24h()
            results["db_stats"] = {
                "time_ms": round((time.time() - start) * 1000),
                "total_passes": total,
                "status": "success",
            }
        except Exception as e:
            results["db_stats"] = {"time_ms": round((time.time() - start) * 1000), "error": str(e), "status": "error"}

    start = time.time()
    try:
        client = get_minio_client()
        objects = list(client.list_objects(MINIO_BUCKET, max_keys=1))
        results["minio_connection"] = {
            "time_ms": round((time.time() - start) * 1000),
            "status": "success",
            "objects_found": len(objects),
        }
    except Exception as e:
        results["minio_connection"] = {
            "time_ms": round((time.time() - start) * 1000),
            "error": str(e),
            "status": "error",
        }

    start = time.time()
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        results["redis_connection"] = {"time_ms": round((time.time() - start) * 1000), "status": "success"}
    except Exception as e:
        results["redis_connection"] = {
            "time_ms": round((time.time() - start) * 1000),
            "error": str(e),
            "status": "error",
        }

    try:
        results["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else "N/A",
        }
    except Exception as e:
        results["system"] = {"error": str(e)}

    from stopsign.web.app import STREAM_FS_PATH

    start = time.time()
    stream_exists = os.path.exists(STREAM_FS_PATH)
    results["hls_stream"] = {
        "time_ms": round((time.time() - start) * 1000),
        "exists": stream_exists,
        "path": STREAM_FS_PATH,
    }

    if stream_exists:
        try:
            stat = os.stat(STREAM_FS_PATH)
            results["hls_stream"]["size_bytes"] = stat.st_size
            results["hls_stream"]["last_modified"] = stat.st_mtime
            results["hls_stream"]["age_seconds"] = time.time() - stat.st_mtime
        except Exception as e:
            results["hls_stream"]["stat_error"] = str(e)

    return results
