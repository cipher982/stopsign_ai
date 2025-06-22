import threading

import yaml

# Shared shutdown flag
shutdown_flag = threading.Event()


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)

        # new stuff
        self.input_source = config["stream_settings"]["input_source"]

        # Video processing
        self.scale = config["video_processing"]["scale"]
        self.crop_top = config["video_processing"]["crop_top"]
        self.crop_side = config["video_processing"]["crop_side"]
        self.frame_buffer_size = config["video_processing"]["frame_buffer_size"]

        # Stop sign detection
        stop_sign = config["stopsign_detection"]
        self.stop_line = tuple(tuple(i) for i in stop_sign["stop_line"])
        self.stop_box_tolerance = tuple(stop_sign["stop_box_tolerance"])
        self.pre_stop_zone = tuple(stop_sign["pre_stop_zone"])
        self.image_capture_zone = tuple(stop_sign["image_capture_zone"])
        self.in_zone_frame_threshold = stop_sign["in_zone_frame_threshold"]
        self.out_zone_frame_threshold = stop_sign["out_zone_frame_threshold"]
        self.stop_speed_threshold = stop_sign["stop_speed_threshold"]
        self.max_movement_speed = stop_sign["max_movement_speed"]
        self.parked_frame_threshold = stop_sign["parked_frame_threshold"]
        self.unparked_frame_threshold = stop_sign["unparked_frame_threshold"]
        self.unparked_speed_threshold = stop_sign["unparked_speed_threshold"]

        # Tracking
        self.use_kalman_filter = config["tracking"]["use_kalman_filter"]

        # Output
        self.save_video = config["output"]["save_video"]
        self.frame_skip = config["output"]["frame_skip"]
        self.jpeg_quality = config["output"]["jpeg_quality"]

        # Visualization
        self.draw_grid = config["debugging_visualization"]["draw_grid"]
        self.grid_size = config["debugging_visualization"]["grid_size"]

        # Stream settings
        self.fps = config["stream_settings"]["fps"]
        self.vehicle_classes = config["stream_settings"]["vehicle_classes"]

    def update_stop_zone(self, new_config):
        print(f"DEBUG: update_stop_zone called with: {new_config}")
        print(f"DEBUG: stop_line from new_config: {new_config['stop_line']}, type: {type(new_config['stop_line'])}")

        self.stop_line = new_config["stop_line"]
        self.stop_box_tolerance = new_config["stop_box_tolerance"]
        self.min_stop_time = new_config["min_stop_duration"]

        print(f"DEBUG: self.stop_line after assignment: {self.stop_line}, type: {type(self.stop_line)}")

        self.save_to_yaml()

    def save_to_yaml(self):
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)

        # Update only the changed values (convert tuples to lists for YAML)
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
        config["stopsign_detection"]["min_stop_time"] = self.min_stop_time

        with open(self.config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"Configuration updated and saved to {self.config_path}")
