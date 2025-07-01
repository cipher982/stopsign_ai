import os
import threading

import yaml

# Shared shutdown flag
shutdown_flag = threading.Event()


class Config:
    def __init__(self, config_path, db_url=None):
        self.config_path = config_path
        self.db_url = db_url or os.getenv("DB_URL")
        self.database = None

        # Try to initialize database connection
        if self.db_url:
            try:
                from .database import Database

                self.database = Database(self.db_url)
                print("✅ Config: Connected to database for configuration")
            except Exception as e:
                print(f"⚠️ Config: Database connection failed, using YAML fallback: {e}")
                self.database = None

        self.load_config()

    def _get_config_value(self, category: str, key: str, yaml_path: list):
        """Get config value from database first, then YAML fallback."""
        if self.database:
            try:
                db_value = self.database.get_config_setting(category, key)
                if db_value is not None:
                    return db_value
            except Exception as e:
                print(f"⚠️ Config: Database read failed for {category}.{key}: {e}")

        # Fallback to YAML
        try:
            config_section = self._yaml_config
            for path_part in yaml_path[:-1]:
                config_section = config_section[path_part]
            return config_section[yaml_path[-1]]
        except (KeyError, TypeError) as e:
            print(f"❌ Config: Missing YAML value for {yaml_path}: {e}")
            return None

    def load_config(self):
        # Always load YAML as fallback
        with open(self.config_path, "r") as file:
            self._yaml_config = yaml.safe_load(file)

        # Stream settings
        self.input_source = self._get_config_value("stream", "input_source", ["stream_settings", "input_source"])
        self.fps = self._get_config_value("stream", "fps", ["stream_settings", "fps"])
        self.vehicle_classes = self._get_config_value(
            "stream", "vehicle_classes", ["stream_settings", "vehicle_classes"]
        )

        # Video processing
        self.scale = self._get_config_value("video_processing", "scale", ["video_processing", "scale"])
        self.crop_top = self._get_config_value("video_processing", "crop_top", ["video_processing", "crop_top"])
        self.crop_side = self._get_config_value("video_processing", "crop_side", ["video_processing", "crop_side"])
        self.frame_buffer_size = self._get_config_value(
            "video_processing", "frame_buffer_size", ["video_processing", "frame_buffer_size"]
        )

        # Zones (with coordinate conversion)
        stop_line_raw = self._get_config_value("zones", "stop_line", ["stopsign_detection", "stop_line"])
        self.stop_line = tuple(tuple(i) for i in stop_line_raw) if stop_line_raw else None

        stop_box_tolerance_raw = self._get_config_value(
            "zones", "stop_box_tolerance", ["stopsign_detection", "stop_box_tolerance"]
        )
        if stop_box_tolerance_raw:
            if isinstance(stop_box_tolerance_raw, (list, tuple)):
                self.stop_box_tolerance = tuple(stop_box_tolerance_raw)
            else:
                # Handle case where it's returned as int instead of list
                self.stop_box_tolerance = (stop_box_tolerance_raw, stop_box_tolerance_raw)
        else:
            self.stop_box_tolerance = None

        pre_stop_zone_raw = self._get_config_value("zones", "pre_stop_zone", ["stopsign_detection", "pre_stop_zone"])
        self.pre_stop_zone = tuple(pre_stop_zone_raw) if pre_stop_zone_raw else None

        image_capture_zone_raw = self._get_config_value(
            "zones", "image_capture_zone", ["stopsign_detection", "image_capture_zone"]
        )
        self.image_capture_zone = tuple(image_capture_zone_raw) if image_capture_zone_raw else None

        # Detection thresholds
        self.in_zone_frame_threshold = self._get_config_value(
            "detection", "in_zone_frame_threshold", ["stopsign_detection", "in_zone_frame_threshold"]
        )
        self.out_zone_frame_threshold = self._get_config_value(
            "detection", "out_zone_frame_threshold", ["stopsign_detection", "out_zone_frame_threshold"]
        )
        self.stop_speed_threshold = self._get_config_value(
            "detection", "stop_speed_threshold", ["stopsign_detection", "stop_speed_threshold"]
        )
        self.max_movement_speed = self._get_config_value(
            "detection", "max_movement_speed", ["stopsign_detection", "max_movement_speed"]
        )
        self.parked_frame_threshold = self._get_config_value(
            "detection", "parked_frame_threshold", ["stopsign_detection", "parked_frame_threshold"]
        )
        self.unparked_frame_threshold = self._get_config_value(
            "detection", "unparked_frame_threshold", ["stopsign_detection", "unparked_frame_threshold"]
        )
        self.unparked_speed_threshold = self._get_config_value(
            "detection", "unparked_speed_threshold", ["stopsign_detection", "unparked_speed_threshold"]
        )

        # Tracking
        self.use_kalman_filter = self._get_config_value(
            "tracking", "use_kalman_filter", ["tracking", "use_kalman_filter"]
        )

        # Output
        self.save_video = self._get_config_value("output", "save_video", ["output", "save_video"])
        self.frame_skip = self._get_config_value("output", "frame_skip", ["output", "frame_skip"])
        self.jpeg_quality = self._get_config_value("output", "jpeg_quality", ["output", "jpeg_quality"])

        # Visualization
        self.draw_grid = self._get_config_value("visualization", "draw_grid", ["debugging_visualization", "draw_grid"])
        self.grid_size = self._get_config_value("visualization", "grid_size", ["debugging_visualization", "grid_size"])

    def update_stop_zone(self, new_config):
        print(f"DEBUG: update_stop_zone called with: {new_config}")
        print(f"DEBUG: stop_line from new_config: {new_config['stop_line']}, type: {type(new_config['stop_line'])}")

        self.stop_line = new_config["stop_line"]
        self.stop_box_tolerance = new_config["stop_box_tolerance"]
        self.min_stop_time = new_config["min_stop_duration"]

        print(f"DEBUG: self.stop_line after assignment: {self.stop_line}, type: {type(self.stop_line)}")

        self._save_config_changes(
            {"zones": {"stop_line": new_config["stop_line"], "stop_box_tolerance": new_config["stop_box_tolerance"]}},
            change_reason="Stop zone updated via debug interface",
        )

    def _save_config_changes(self, updates, change_reason="Config update"):
        """Save config changes to database first, then YAML fallback."""
        saved_to_db = False

        if self.database:
            try:
                for category, settings in updates.items():
                    for key, value in settings.items():
                        # Determine coordinate system for zones
                        coord_system = None
                        if category == "zones":
                            if key == "stop_line":
                                coord_system = "raw"
                            elif key in ["pre_stop_zone", "image_capture_zone"]:
                                coord_system = "processing"

                        self.database.update_config_setting(
                            category=category,
                            key=key,
                            value=value,
                            coordinate_system=coord_system,
                            updated_by="debug_interface",
                            change_reason=change_reason,
                        )
                print("✅ Configuration saved to database")
                saved_to_db = True
            except Exception as e:
                print(f"⚠️ Database save failed, falling back to YAML: {e}")

        if not saved_to_db:
            # Fallback to YAML
            self.save_to_yaml()

    def save_to_yaml(self):
        """Legacy YAML save method (fallback only)."""
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)

        # Update stop sign detection values
        try:
            stop_line_lists = [list(point) for point in self.stop_line]
            config["stopsign_detection"]["stop_line"] = stop_line_lists
        except Exception:
            raise

        # Handle stop_box_tolerance - it might be an int instead of a list/tuple
        if isinstance(self.stop_box_tolerance, (list, tuple)):
            config["stopsign_detection"]["stop_box_tolerance"] = list(self.stop_box_tolerance)
        else:
            # It's probably an int, convert to list format
            config["stopsign_detection"]["stop_box_tolerance"] = [self.stop_box_tolerance, self.stop_box_tolerance]

        # Update zone ranges
        config["stopsign_detection"]["pre_stop_zone"] = list(self.pre_stop_zone)
        config["stopsign_detection"]["image_capture_zone"] = list(self.image_capture_zone)

        # Handle min_stop_time if it exists
        if hasattr(self, "min_stop_time"):
            config["stopsign_detection"]["min_stop_time"] = self.min_stop_time

        with open(self.config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"Configuration updated and saved to {self.config_path}")

    def update_zone(self, zone_type: str, zone_data):
        """Update a specific zone type and save to config."""
        updates = {}

        if zone_type == "stop_line":
            self.stop_line = zone_data["stop_line"]
            self.stop_box_tolerance = zone_data.get("stop_box_tolerance", self.stop_box_tolerance)
            if "min_stop_duration" in zone_data:
                self.min_stop_time = zone_data["min_stop_duration"]
            updates["zones"] = {
                "stop_line": zone_data["stop_line"],
                "stop_box_tolerance": zone_data.get("stop_box_tolerance", self.stop_box_tolerance),
            }
        elif zone_type == "pre_stop":
            self.pre_stop_zone = zone_data
            updates["zones"] = {"pre_stop_zone": zone_data}
        elif zone_type == "capture":
            self.image_capture_zone = zone_data
            updates["zones"] = {"image_capture_zone": zone_data}
        else:
            raise ValueError(f"Unknown zone type: {zone_type}")

        self._save_config_changes(updates, change_reason=f"Updated {zone_type} zone via debug interface")
        print(f"Updated {zone_type} zone: {zone_data}")
