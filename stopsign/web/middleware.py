# ruff: noqa: E501
import logging
import random
import time

logger = logging.getLogger(__name__)


async def add_cache_headers(request, call_next):
    """HTTP cache headers and telemetry middleware."""
    tracer = request.app.state.tracer
    metrics = request.app.state.metrics

    user_agent = request.headers.get("user-agent", "").lower()
    if "chrome" in user_agent:
        browser = "chrome"
    elif "safari" in user_agent and "chrome" not in user_agent:
        browser = "safari"
    elif "firefox" in user_agent:
        browser = "firefox"
    else:
        browser = "other"

    # Skip telemetry for most static assets (sample 10%)
    if request.url.path.startswith("/static/"):
        if any(request.url.path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".css", ".js"]):
            response = await call_next(request)
            response.headers["Cache-Control"] = "public, max-age=3600"
            response.headers["ETag"] = f'"{hash(request.url.path)}"'
            if random.random() < 0.1:
                with tracer.start_as_current_span("http_static_sample") as span:
                    span.set_attribute("request.type", "static_asset")
                    span.set_attribute("browser.type", browser)
            return response

    # Fast cache headers for local vehicle images
    if request.url.path.startswith("/vehicle-images/"):
        response = await call_next(request)
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

    # Full telemetry for streaming and API requests
    is_streaming = request.url.path.startswith("/stream/")
    is_api = request.url.path.startswith("/api/") or request.url.path in ["/health", "/health/stream", "/check-stream"]

    if is_streaming or is_api:
        with tracer.start_as_current_span("http_request") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("browser.type", browser)

            if is_streaming:
                span.set_attribute("http.path", request.url.path)

            start_time = time.time()
            response = await call_next(request)
            duration = time.time() - start_time

            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("http.response_time_seconds", duration)

            if is_streaming:
                if request.url.path.endswith(".m3u8"):
                    span.set_attribute("hls.type", "manifest")
                    metrics.db_operations.add(
                        1, {"operation": "hls_manifest", "browser": browser, "status": str(response.status_code)}
                    )
                    response.headers["Cache-Control"] = "no-store, must-revalidate"
                    response.headers["Pragma"] = "no-cache"
                    response.headers["Expires"] = "0"

                elif request.url.path.endswith(".ts"):
                    segment_name = request.url.path.split("/")[-1]
                    span.set_attribute("hls.type", "segment")
                    try:
                        segment_num = int("".join(filter(str.isdigit, segment_name)))
                        span.set_attribute("hls.segment_number", segment_num)
                    except (ValueError, TypeError):
                        pass
                    if response.status_code == 200:
                        metrics.db_operations.add(1, {"operation": "hls_segment_success", "browser": browser})
                        span.set_attribute("hls.success", True)
                    elif response.status_code == 404:
                        metrics.db_operations.add(1, {"operation": "hls_segment_404", "browser": browser})
                        span.set_attribute("hls.success", False)
                        span.set_attribute("hls.error", "segment_not_found")
                        logger.warning(f"404 on HLS segment {segment_name} from {browser}")
                    if duration > 1.0:
                        span.set_attribute("hls.slow_request", True)
                        logger.warning(f"Slow HLS segment request: {segment_name} took {duration:.2f}s from {browser}")
                    response.headers["Cache-Control"] = "public, max-age=60"

            if response.status_code >= 400:
                span.set_attribute("http.error", True)
                metrics.db_operations.add(1, {"operation": "http_error", "status": str(response.status_code)})
    else:
        response = await call_next(request)
        if request.url.path.startswith("/vehicle-image/"):
            response.headers["Cache-Control"] = "public, max-age=3600"
            response.headers["ETag"] = f'"{hash(request.url.path)}"'

    return response
