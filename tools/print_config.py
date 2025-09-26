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

        # Stop line
        if config.stop_line:
            p1, p2 = config.stop_line
            print("\nStop Line:")
            print(f"  Point 1: ({p1[0]:.2f}, {p1[1]:.2f})")
            print(f"  Point 2: ({p2[0]:.2f}, {p2[1]:.2f})")
            print(f"  Tolerance: {config.stop_box_tolerance}")
        else:
            print("\nStop Line: Not configured")

        # Pre-stop zone
        if config.pre_stop_zone:
            print("\nPre-Stop Zone:")
            print(f"  X-Range: {config.pre_stop_zone[0]:.2f} to {config.pre_stop_zone[1]:.2f}")
        else:
            print("\nPre-Stop Zone: Not configured")

        # Image capture zone
        if config.image_capture_zone:
            print("\nImage Capture Zone:")
            print(f"  X-Range: {config.image_capture_zone[0]:.2f} to {config.image_capture_zone[1]:.2f}")
        else:
            print("\nImage Capture Zone: Not configured")

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
