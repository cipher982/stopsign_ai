"""
Debug utility to add detailed logging to span operations
"""

import functools
import logging

from opentelemetry.trace import Span

logger = logging.getLogger("span_debug")
logger.setLevel(logging.DEBUG)

# Add console handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("🔍 SPAN_DEBUG: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def debug_span_context(span_name: str):
    """Decorator to add debug logging around span operations"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Creating span: {span_name}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Span {span_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Span {span_name} failed: {e}")
                raise

        return wrapper

    return decorator


def log_span_attribute(span: Span, key: str, value, context: str = ""):
    """Log span attribute setting with context"""
    try:
        span.set_attribute(key, value)
        logger.debug(f"Set attribute {key}={value} on span {span.name} {context}")
    except Exception as e:
        logger.error(f"FAILED to set attribute {key}={value} on span {span.name}: {e} {context}")
        raise
