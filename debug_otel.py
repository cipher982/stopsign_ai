"""
Debug script to patch OpenTelemetry and show exact span/attribute causing warnings with line numbers
"""

import logging
import traceback

from opentelemetry.sdk.trace import Span

# Set up debug logger with line numbers
logger = logging.getLogger("otel_debug")
logger.setLevel(logging.ERROR)

# Add console handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("🚨 OTEL_DEBUG [%(filename)s:%(lineno)d]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Store original set_attribute method
_original_set_attribute = Span.set_attribute


def debug_set_attribute(self, key, value):
    """Patched set_attribute that shows debug info with line numbers when operations fail"""
    try:
        return _original_set_attribute(self, key, value)
    except Exception as e:
        # Get the actual calling location with line numbers
        stack = traceback.extract_stack()

        # Find the first non-debug frame (skip this function and any wrappers)
        caller_frame = None
        for frame in reversed(stack[:-1]):  # Skip current frame
            if "debug_otel" not in frame.filename and "telemetry" not in frame.filename:
                caller_frame = frame
                break

        # Log with precise location
        if caller_frame:
            logger.error(f"Span attribute FAILED: {key}={value} on span '{self.name}'")
            logger.error(f"   Error: {e}")
            logger.error(f"   Location: {caller_frame.filename}:{caller_frame.lineno} in {caller_frame.name}")
            logger.error(f"   Code: {caller_frame.line}")

            # Show span state for debugging
            span_state = "UNKNOWN"
            if hasattr(self, "_ended"):
                span_state = "ENDED" if getattr(self, "_ended", False) else "ACTIVE"
            logger.error(f"   Span state: {span_state}")
        else:
            logger.error(f"Span attribute FAILED: {key}={value} on span '{self.name}' - Error: {e}")

        # Re-raise the exception
        raise


# Monkey patch the method
Span.set_attribute = debug_set_attribute

print("✅ OpenTelemetry debug patching enabled - will show exact file:line for span attribute failures")
