#!/usr/bin/env python3
"""Set stop line coordinates in config.yaml."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stopsign.config import Config


def main():
    """Update stop line coordinates from command line."""
    parser = argparse.ArgumentParser(description="Set stop line coordinates in config.yaml")
    parser.add_argument("--x1", type=float, required=True, help="X coordinate of first point")
    parser.add_argument("--y1", type=float, required=True, help="Y coordinate of first point")
    parser.add_argument("--x2", type=float, required=True, help="X coordinate of second point")
    parser.add_argument("--y2", type=float, required=True, help="Y coordinate of second point")
    parser.add_argument("--tolerance", type=int, default=10, help="Stop box tolerance (default: 10)")

    args = parser.parse_args()

    try:
        # Load config from default location
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        config = Config(config_path)

        # Create stop line tuple
        stop_line = ((args.x1, args.y1), (args.x2, args.y2))

        # Update configuration
        result = config.update_stop_zone(
            {
                "stop_line": stop_line,
                "stop_box_tolerance": (args.tolerance, args.tolerance),
                "min_stop_duration": config.min_stop_time,  # Keep existing value
            }
        )

        print("✅ Stop line updated successfully!")
        print(f"Version: {result['version']}")
        print(f"Stop line: {result['stop_line']}")
        print(f"Tolerance: ({args.tolerance}, {args.tolerance})")

    except FileNotFoundError as e:
        print(f"Error: Config file not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
