"""
Debug script to patch OpenTelemetry and show exact span/attribute causing warnings
"""

import traceback

from opentelemetry.sdk.trace import Span

# Store original set_attribute method
_original_set_attribute = Span.set_attribute


def debug_set_attribute(self, key, value):
    """Patched set_attribute that shows debug info when span is ended"""
    if self.status.status_code.name == "UNSET" and hasattr(self, "_ended") and getattr(self, "_ended", False):
        # Span has ended, show debug info
        stack = traceback.extract_stack()
        caller_info = []
        for frame in stack[-5:-1]:  # Show last few stack frames
            caller_info.append(f"  {frame.filename}:{frame.lineno} in {frame.name}")

        print("\n🚨 TELEMETRY DEBUG: Setting attribute on ended span!")
        print(f"   Span name: {self.name}")
        print(f"   Attribute: {key} = {value}")
        print("   Call stack:")
        for info in caller_info:
            print(info)
        print()

    return _original_set_attribute(self, key, value)


# Monkey patch the method
Span.set_attribute = debug_set_attribute

print("✅ OpenTelemetry debug patching enabled - will show exact span/attribute causing warnings")
