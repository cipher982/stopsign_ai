#!/usr/bin/env python3
"""Set stop zone coordinates (4-point polygon) in config.yaml."""

import argparse
import os
import sys
from typing import List
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stopsign.config import Config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set stop zone coordinates (4 corners) in config.yaml")
    parser.add_argument("--x1", type=float, required=True, help="Corner 1 X coordinate")
    parser.add_argument("--y1", type=float, required=True, help="Corner 1 Y coordinate")
    parser.add_argument("--x2", type=float, required=True, help="Corner 2 X coordinate")
    parser.add_argument("--y2", type=float, required=True, help="Corner 2 Y coordinate")
    parser.add_argument("--x3", type=float, required=True, help="Corner 3 X coordinate")
    parser.add_argument("--y3", type=float, required=True, help="Corner 3 Y coordinate")
    parser.add_argument("--x4", type=float, required=True, help="Corner 4 X coordinate")
    parser.add_argument("--y4", type=float, required=True, help="Corner 4 Y coordinate")
    parser.add_argument(
        "--min-stop-duration",
        type=float,
        default=None,
        help="Optional minimum stop duration in seconds",
    )
    return parser.parse_args()


def to_points(args: argparse.Namespace) -> List[Tuple[float, float]]:
    return [
        (args.x1, args.y1),
        (args.x2, args.y2),
        (args.x3, args.y3),
        (args.x4, args.y4),
    ]


def main() -> None:
    args = parse_arguments()

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    config = Config(config_path)

    payload = {"stop_zone": to_points(args)}
    if args.min_stop_duration is not None:
        payload["min_stop_duration"] = args.min_stop_duration

    result = config.update_stop_zone(payload)

    print("✅ Stop zone updated successfully!")
    print(f"Version: {result['version']}")
    for idx, (x, y) in enumerate(result["stop_zone"], start=1):
        print(f"Corner {idx}: ({x:.2f}, {y:.2f})")


if __name__ == "__main__":
    main()
