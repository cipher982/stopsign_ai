#!/usr/bin/env python3
"""Print configuration details from config.yaml."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from stopsign.config import Config


def main():
    """Print config version, stop-line, and zone ranges."""
    try:
        # Load config from default location
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        config = Config(config_path)

        print("=" * 60)
        print("STOP SIGN CONFIGURATION")
        print("=" * 60)

        # Version
        print(f"\nVersion: {config.version}")

        # Stop zone
        if config.stop_zone:
            print("\nStop Zone (4-point polygon):")
            for i, p in enumerate(config.stop_zone, 1):
                print(f"  Corner {i}: ({p[0]:.2f}, {p[1]:.2f})")
        else:
            print("\nStop Zone: Not configured")

        # Pre-stop line
        if getattr(config, "pre_stop_line", None):
            print("\nPre-Stop Line:")
            (p1x, p1y), (p2x, p2y) = config.pre_stop_line
            print(f"  Point 1: ({p1x:.2f}, {p1y:.2f})")
            print(f"  Point 2: ({p2x:.2f}, {p2y:.2f})")
        else:
            print("\nPre-Stop Line: Not configured")

        # Image capture line
        if getattr(config, "capture_line", None):
            print("\nCapture Line:")
            (c1x, c1y), (c2x, c2y) = config.capture_line
            print(f"  Point 1: ({c1x:.2f}, {c1y:.2f})")
            print(f"  Point 2: ({c2x:.2f}, {c2y:.2f})")
        else:
            print("\nCapture Line: Not configured")

        # Detection parameters
        print("\nDetection Parameters:")
        print(f"  In-zone threshold: {config.in_zone_frame_threshold} frames")
        print(f"  Out-zone threshold: {config.out_zone_frame_threshold} frames")
        print(f"  Stop speed threshold: {config.stop_speed_threshold} px/frame")
        print(f"  Min stop time: {config.min_stop_time} seconds")

        # Video processing
        print("\nVideo Processing:")
        print(f"  Scale: {config.scale}")
        print(f"  Crop top: {config.crop_top}")
        print(f"  Crop side: {config.crop_side}")
        print(f"  FPS: {config.fps}")

        print("\n" + "=" * 60)

    except FileNotFoundError as e:
        print(f"Error: Config file not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
