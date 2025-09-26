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

            # Stop line (raw coordinates)
            stop_line_raw = detection.get("stop_line")
            self.stop_line = tuple(tuple(i) for i in stop_line_raw) if stop_line_raw else None

            # Stop box tolerance
            stop_box_tolerance_raw = detection.get("stop_box_tolerance")
            if stop_box_tolerance_raw:
                if isinstance(stop_box_tolerance_raw, (list, tuple)):
                    self.stop_box_tolerance = tuple(stop_box_tolerance_raw)
                else:
                    self.stop_box_tolerance = (stop_box_tolerance_raw, stop_box_tolerance_raw)
            else:
                self.stop_box_tolerance = None

            # Other zones
            pre_stop_zone_raw = detection.get("pre_stop_zone")
            self.pre_stop_zone = tuple(pre_stop_zone_raw) if pre_stop_zone_raw else None

            image_capture_zone_raw = detection.get("image_capture_zone")
            self.image_capture_zone = tuple(image_capture_zone_raw) if image_capture_zone_raw else None

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

        # Validate stop_line has exactly 2 points if present
        detection = self._yaml_config.get("stopsign_detection", {})
        stop_line = detection.get("stop_line")
        if stop_line:
            if not isinstance(stop_line, list) or len(stop_line) != 2:
                raise ValueError("stop_line must have exactly 2 points")
            for point in stop_line:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Each stop_line point must have [x, y] coordinates")

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

    def update_stop_zone(self, new_config: dict) -> dict:
        """Update stop zone configuration.

        Args:
            new_config: Dictionary with stop_line, stop_box_tolerance, min_stop_duration

        Returns:
            Dictionary with version, stop_line, and raw_points
        """
        with self._lock:
            # Update in-memory state
            self.stop_line = new_config["stop_line"]
            self.stop_box_tolerance = new_config["stop_box_tolerance"]
            self.min_stop_time = new_config.get("min_stop_duration", self.min_stop_time)

            # Load current config for update
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)

            # Update stop detection values
            stop_line_lists = [list(point) for point in self.stop_line]
            config["stopsign_detection"]["stop_line"] = stop_line_lists

            # Handle stop_box_tolerance
            if isinstance(self.stop_box_tolerance, (list, tuple)):
                config["stopsign_detection"]["stop_box_tolerance"] = list(self.stop_box_tolerance)
            else:
                config["stopsign_detection"]["stop_box_tolerance"] = [self.stop_box_tolerance, self.stop_box_tolerance]

            config["stopsign_detection"]["min_stop_time"] = self.min_stop_time

            # Save atomically
            new_version = self._save_atomic(config)

            # Return response data
            return {"version": new_version, "stop_line": self.stop_line, "raw_points": stop_line_lists}

    def update_zone(self, zone_type: str, zone_data: Any) -> dict:
        """Update a specific zone type and save to config.

        Args:
            zone_type: Type of zone ('stop_line', 'pre_stop', 'capture')
            zone_data: Zone configuration data

        Returns:
            Dictionary with version and updated zone data
        """
        with self._lock:
            # Load current config
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)

            # Update based on zone type
            if zone_type == "stop_line":
                self.stop_line = zone_data["stop_line"]
                self.stop_box_tolerance = zone_data.get("stop_box_tolerance", self.stop_box_tolerance)
                if "min_stop_duration" in zone_data:
                    self.min_stop_time = zone_data["min_stop_duration"]

                stop_line_lists = [list(point) for point in self.stop_line]
                config["stopsign_detection"]["stop_line"] = stop_line_lists

                if isinstance(self.stop_box_tolerance, (list, tuple)):
                    config["stopsign_detection"]["stop_box_tolerance"] = list(self.stop_box_tolerance)
                else:
                    config["stopsign_detection"]["stop_box_tolerance"] = [
                        self.stop_box_tolerance,
                        self.stop_box_tolerance,
                    ]

                config["stopsign_detection"]["min_stop_time"] = self.min_stop_time
                result_data = {
                    "stop_line": self.stop_line,
                    "raw_points": stop_line_lists,
                }

            elif zone_type == "pre_stop":
                self.pre_stop_zone = zone_data
                config["stopsign_detection"]["pre_stop_zone"] = list(zone_data)
                result_data = {"pre_stop_zone": zone_data}

            elif zone_type == "capture":
                self.image_capture_zone = zone_data
                config["stopsign_detection"]["image_capture_zone"] = list(zone_data)
                result_data = {"image_capture_zone": zone_data}

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
