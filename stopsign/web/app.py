# ruff: noqa: E501
import logging
import os
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text

from stopsign.database import Database
from stopsign.settings import DB_URL
from stopsign.settings import LOCAL_IMAGE_DIR
from stopsign.telemetry import get_tracer
from stopsign.telemetry import setup_web_server_telemetry
from stopsign.web.middleware import add_cache_headers
from stopsign.web.services.clips import CLIP_DIR
from stopsign.web.services.clips import clip_worker_loop

logger = logging.getLogger(__name__)

# Constants
STREAM_FS_PATH = "/app/data/stream/stream.m3u8"
STREAM_URL = "/stream/stream.m3u8"
WEB_START_TIME = time.time()
ASSET_VERSION = os.getenv("ASSET_VERSION") or str(int(WEB_START_TIME))

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
templates.env.globals["asset_version"] = ASSET_VERSION


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(docs_url=None, redoc_url=None)

    # Static mounts
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # These directories only exist in Docker containers; create if possible, skip if not
    for mount_path, directory, name in [
        ("/stream", "/app/data/stream", "stream"),
        ("/vehicle-images", LOCAL_IMAGE_DIR, "vehicle-images"),
        ("/clips", CLIP_DIR, "clips"),
    ]:
        try:
            os.makedirs(directory, exist_ok=True)
            app.mount(mount_path, StaticFiles(directory=directory), name=name)
        except OSError:
            logger.warning("Could not mount %s -> %s (directory not available)", mount_path, directory)

    # Telemetry
    metrics = setup_web_server_telemetry(app)
    tracer = get_tracer("stopsign.web_server")

    # Store on app state for access by routes
    app.state.metrics = metrics
    app.state.tracer = tracer

    # Middleware
    app.middleware("http")(add_cache_headers)

    # Register routes
    from stopsign.web.routes.api import router as api_router
    from stopsign.web.routes.debug_api import router as debug_api_router
    from stopsign.web.routes.health import router as health_router
    from stopsign.web.routes.infrastructure import router as infra_router
    from stopsign.web.routes.pages import router as pages_router
    from stopsign.web.routes.stream import router as stream_router

    app.include_router(pages_router)
    app.include_router(api_router)
    app.include_router(stream_router)
    app.include_router(health_router)
    app.include_router(debug_api_router)
    app.include_router(infra_router)

    @app.on_event("startup")
    def _startup():
        if not hasattr(app.state, "db"):
            app.state.db = Database(db_url=DB_URL)
        t = threading.Thread(target=clip_worker_loop, args=(app,), daemon=True)
        t.start()

    return app


app = create_app()


def main():
    """Main entry point for the web server."""
    db_connected = False
    db_init_attempts = 0
    max_db_init_attempts = 10
    db_init_delay = 5

    while not db_connected and db_init_attempts < max_db_init_attempts:
        try:
            app.state.db = Database(db_url=DB_URL)
            with app.state.db.Session() as session:
                session.execute(text("SELECT 1"))
            db_connected = True
            logger.info("Database connection successful.")
        except Exception as e:
            db_init_attempts += 1
            logger.warning(
                f"Database connection attempt {db_init_attempts} failed: {e}. "
                f"Retrying in {db_init_delay} seconds..."
            )
            time.sleep(db_init_delay)

    if not db_connected:
        logger.error("Failed to connect to the database after several attempts. Exiting.")
        logger.warning("Config: Database connection failed, using YAML fallback")

    env = os.getenv("ENV", "prod")
    is_dev_mode = env in ("local", "dev")

    try:
        uvicorn.run(
            "stopsign.web.app:app",
            host="0.0.0.0",
            port=8000,
            reload=is_dev_mode,
            log_level="warning",
            reload_dirs=["./stopsign"] if is_dev_mode else None,
            reload_excludes=["./app/data/*"] if is_dev_mode else None,
        )
    except Exception as e:
        logger.error(f"Error in web server: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
