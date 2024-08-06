import threading

import yaml

# Shared shutdown flag
shutdown_flag = threading.Event()


class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

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
