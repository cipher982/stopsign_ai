import math
import os
import tempfile
import threading
from datetime import datetime
from typing import Any

import yaml

# Shared shutdown flag
shutdown_flag = threading.Event()


class Config:
    """Configuration manager with atomic file operations and version tracking."""

    def __init__(self, config_path: str = "/app/config/config.yaml"):
        """Initialize config from file.

        Args:
            config_path: Path to the configuration YAML file.
                        Defaults to /app/config/config.yaml for container environment.
        """
        self.config_path = config_path
        self._lock = threading.Lock()

        # Ensure config exists before loading
        self._ensure_config_exists()

        self.load_config()

    def _ensure_config_exists(self) -> None:
        """Ensure the config file exists by seeding from an example if available."""
        if os.path.exists(self.config_path):
            return

        config_dir = os.path.dirname(self.config_path) or "."
        os.makedirs(config_dir, exist_ok=True)

        example_candidates = [
            os.path.join(config_dir, "config.example.yaml"),
            "/app/config.example.yaml",
            os.path.join(os.path.dirname(__file__), "..", "config", "config.example.yaml"),
        ]

        for candidate in example_candidates:
            if candidate and os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as src:
                    data = src.read()

                with tempfile.NamedTemporaryFile(mode="w", dir=config_dir, delete=False) as tmp_file:
                    tmp_file.write(data)
                    tmp_path = tmp_file.name

                os.replace(tmp_path, self.config_path)
                print(f"Config seeded from template: {candidate} -> {self.config_path}")
                return

        raise FileNotFoundError(
            f"Config file not found at {self.config_path} and no template available. "
            "Ensure config volume is writable or provide a config example."
        )

    def load_config(self):
        """Load configuration from YAML file and validate."""
        with self._lock:
            with open(self.config_path, "r") as file:
                self._yaml_config = yaml.safe_load(file)

            # Validate config structure
            self._validate()

            # Extract version or initialize
            self.version = self._yaml_config.get("version", "1.0.0")

            # Stream settings
            stream = self._yaml_config.get("stream_settings", {})
            self.input_source = stream.get("input_source")
            self.fps = stream.get("fps")
            self.vehicle_classes = stream.get("vehicle_classes", [])

            # Video processing
            video = self._yaml_config.get("video_processing", {})
            self.scale = video.get("scale")
            self.crop_top = video.get("crop_top")
            self.crop_side = video.get("crop_side")
            self.frame_buffer_size = video.get("frame_buffer_size")

            # Stop sign detection zones
            detection = self._yaml_config.get("stopsign_detection", {})

            # Stop zone - must be defined as 4 corner points
            stop_zone_raw = detection.get("stop_zone")
            if stop_zone_raw is None:
                raise ValueError(
                    "stopsign_detection.stop_zone is required and must contain four corner points. "
                    "Update your configuration via the debug UI to record the new polygon."
                )

            self.stop_zone = self._normalize_stop_zone(stop_zone_raw)

            if detection.get("stop_line") is not None:
                raise ValueError(
                    "Legacy stopsign_detection.stop_line detected. Remove the stop_line field and define "
                    "stopsign_detection.stop_zone with four corner points."
                )

            if detection.get("stop_box_tolerance") is not None:
                raise ValueError(
                    "stopsign_detection.stop_box_tolerance is no longer supported. Remove it from the config."
                )

            if "pre_stop_zone" in detection:
                raise ValueError(
                    "stopsign_detection.pre_stop_zone is no longer supported. "
                    "Record a pre_stop_line with two points."
                )

            if "image_capture_zone" in detection:
                raise ValueError(
                    "stopsign_detection.image_capture_zone is no longer supported. "
                    "Record a capture_line with two points."
                )

            pre_stop_line_raw = detection.get("pre_stop_line")
            if pre_stop_line_raw is None:
                raise ValueError("stopsign_detection.pre_stop_line is required and must contain two [x, y] points.")
            self.pre_stop_line = self._normalize_line(pre_stop_line_raw, "stopsign_detection.pre_stop_line")

            capture_line_raw = detection.get("capture_line")
            if capture_line_raw is None:
                raise ValueError("stopsign_detection.capture_line is required and must contain two [x, y] points.")
            self.capture_line = self._normalize_line(capture_line_raw, "stopsign_detection.capture_line")

            # Detection thresholds
            self.in_zone_frame_threshold = detection.get("in_zone_frame_threshold")
            self.out_zone_frame_threshold = detection.get("out_zone_frame_threshold")
            self.stop_speed_threshold = detection.get("stop_speed_threshold")
            self.max_movement_speed = detection.get("max_movement_speed")
            self.parked_frame_threshold = detection.get("parked_frame_threshold")
            self.unparked_frame_threshold = detection.get("unparked_frame_threshold")
            self.unparked_speed_threshold = detection.get("unparked_speed_threshold")
            self.min_stop_time = detection.get("min_stop_time", 2.0)

            # Tracking
            tracking = self._yaml_config.get("tracking", {})
            self.use_kalman_filter = tracking.get("use_kalman_filter", True)

            # Output
            output = self._yaml_config.get("output", {})
            self.save_video = output.get("save_video", True)
            self.frame_skip = output.get("frame_skip", 3)
            self.jpeg_quality = output.get("jpeg_quality", 85)

            # Visualization
            debug = self._yaml_config.get("debugging_visualization", {})
            self.draw_grid = debug.get("draw_grid", False)
            self.grid_size = debug.get("grid_size", 100)

            print(f"✅ Config loaded successfully. Version: {self.version}")

    def _validate(self):
        """Validate configuration structure and required fields."""
        required_sections = ["stream_settings", "video_processing", "stopsign_detection"]
        for section in required_sections:
            if section not in self._yaml_config:
                raise ValueError(f"Missing required config section: {section}")

        detection = self._yaml_config.get("stopsign_detection", {})

        stop_zone = detection.get("stop_zone")
        if stop_zone is None:
            raise ValueError("stopsign_detection.stop_zone section is missing from config")

        if not isinstance(stop_zone, list) or len(stop_zone) != 4:
            raise ValueError(
                "stopsign_detection.stop_zone must be a list of four [x, y] points " "defining the polygon corners"
            )

        for point in stop_zone:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Each stop_zone point must have [x, y] coordinates")

    def _increment_version(self):
        """Increment the version number."""
        # Use timestamp-based versioning for simplicity
        self.version = datetime.now().strftime("%Y%m%d.%H%M%S")
        return self.version

    def _save_atomic(self, config_data: dict) -> str:
        """Save configuration atomically using temp file + rename.

        Args:
            config_data: Configuration dictionary to save

        Returns:
            New version string
        """
        # Increment version
        new_version = self._increment_version()
        config_data["version"] = new_version

        # Write to temp file in same directory (for atomic rename)
        config_dir = os.path.dirname(self.config_path)
        with tempfile.NamedTemporaryFile(mode="w", dir=config_dir, delete=False) as tmp_file:
            yaml.dump(config_data, tmp_file, default_flow_style=False, sort_keys=False)
            tmp_path = tmp_file.name

        # Atomic rename
        os.replace(tmp_path, self.config_path)

        print(f"✅ Config saved atomically. New version: {new_version}")
        return new_version

    @staticmethod
    def _normalize_stop_zone(points: Any) -> list[tuple[float, float]]:
        if not isinstance(points, (list, tuple)):
            raise ValueError("stop_zone must be a list of four [x, y] points")

        if len(points) != 4:
            raise ValueError(
                f"stop_zone requires exactly four points, received {len(points)}. "
                "Re-record the stop zone via the debug interface."
            )

        normalized = []
        for idx, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"stop_zone point {idx + 1} must be a [x, y] pair")
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"stop_zone point {idx + 1} must contain numeric coordinates") from exc
            normalized.append((x, y))

        # Sort points around centroid to guarantee a consistent winding order
        centroid_x = sum(x for x, _ in normalized) / 4.0
        centroid_y = sum(y for _, y in normalized) / 4.0

        def angle(point: tuple[float, float]) -> float:
            return math.atan2(point[1] - centroid_y, point[0] - centroid_x)

        sorted_points = sorted(normalized, key=angle)

        area = Config._polygon_area(sorted_points)
        if abs(area) < 1e-6:
            raise ValueError(
                "stop_zone points collapse to zero area. Ensure the four clicks are distinct corners "
                "of the stop zone rectangle."
            )

        # Use clockwise winding for consistency
        if area < 0:
            sorted_points.reverse()

        return sorted_points

    @staticmethod
    def _polygon_area(points: list[tuple[float, float]]) -> float:
        area = 0.0
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
            area += x1 * y2 - x2 * y1
        return 0.5 * area

    @staticmethod
    def _normalize_line(points: Any, field_name: str) -> list[tuple[float, float]]:
        if not isinstance(points, (list, tuple)) or len(points) != 2:
            raise ValueError(f"{field_name} must contain exactly two [x, y] points")

        normalized: list[tuple[float, float]] = []
        for idx, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"{field_name} point {idx + 1} must be a [x, y] coordinate pair")
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} point {idx + 1} must contain numeric coordinates") from exc
            normalized.append((x, y))

        if normalized[0] == normalized[1]:
            raise ValueError(f"{field_name} must use two distinct points")

        return normalized

    def update_stop_zone(self, new_config: dict) -> dict:
        """Update stop zone configuration.

        Args:
            new_config: Dictionary with stop_zone (4 points) and optional min_stop_duration

        Returns:
            Dictionary with version and stop_zone
        """
        with self._lock:
            normalized_stop_zone = self._normalize_stop_zone(new_config["stop_zone"])
            self.stop_zone = normalized_stop_zone

            self.min_stop_time = new_config.get("min_stop_duration", self.min_stop_time)

            # Load current config for update
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)

            stop_zone_lists = [list(point) for point in self.stop_zone]
            detection = config["stopsign_detection"]
            detection["stop_zone"] = stop_zone_lists
            detection.pop("stop_line", None)
            detection.pop("stop_box_tolerance", None)
            detection["min_stop_time"] = self.min_stop_time

            # Save atomically
            new_version = self._save_atomic(config)

            # Return response data
            return {"version": new_version, "stop_zone": self.stop_zone}

    def update_zone(self, zone_type: str, zone_data: Any) -> dict:
        """Update a specific zone type (lines) and save to config.

        Args:
            zone_type: Type of zone ('pre_stop', 'capture')
            zone_data: Zone configuration data

        Returns:
            Dictionary with version and updated zone data
        """
        with self._lock:
            # Load current config
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)

            detection = config["stopsign_detection"]

            if zone_type == "pre_stop":
                normalized_line = self._normalize_line(zone_data, "pre_stop_line update")
                self.pre_stop_line = normalized_line
                detection["pre_stop_line"] = [list(point) for point in normalized_line]
                result_data = {"pre_stop_line": [list(point) for point in normalized_line]}

            elif zone_type == "capture":
                normalized_line = self._normalize_line(zone_data, "capture_line update")
                self.capture_line = normalized_line
                detection["capture_line"] = [list(point) for point in normalized_line]
                result_data = {"capture_line": [list(point) for point in normalized_line]}

            else:
                raise ValueError(f"Unknown zone type: {zone_type}")

            # Save atomically
            new_version = self._save_atomic(config)

            print(f"✅ Updated {zone_type} zone: {zone_data}")

            # Return response data
            result_data["version"] = new_version
            return result_data

    def get_file_mtime(self) -> float:
        """Get the modification time of the config file."""
        return os.path.getmtime(self.config_path)
