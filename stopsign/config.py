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
        self.stop_box_tolerance = stop_sign["stop_box_tolerance"]
        self.min_stop_time = stop_sign["min_stop_time"]
        self.max_movement_speed = stop_sign["max_movement_speed"]
        self.parked_frame_threshold = stop_sign["parked_frame_threshold"]
        self.unparked_frame_threshold = stop_sign["unparked_frame_threshold"]

        # Car Analysis
        self.image_capture_zone = config["car_analysis"]["image_capture_zone"]
        self.stopsign_entry_zone = config["car_analysis"]["stopsign_entry_zone"]

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
        self.stop_line = new_config["stop_line"]
        self.stop_box_tolerance = new_config["stop_box_tolerance"]
        self.min_stop_time = new_config["min_stop_duration"]
        self.save_to_yaml()

    def save_to_yaml(self):
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)

        # Update only the changed values
        config["stopsign_detection"]["stop_line"] = self.stop_line
        config["stopsign_detection"]["stop_box_tolerance"] = self.stop_box_tolerance
        config["stopsign_detection"]["min_stop_time"] = self.min_stop_time

        with open(self.config_path, "w") as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"Configuration updated and saved to {self.config_path}")
