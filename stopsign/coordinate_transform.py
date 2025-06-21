"""
Coordinate transformation utilities for mapping between different video coordinate systems.
Handles the complex pipeline: Raw → Cropped → Scaled → FFmpeg → HLS → Browser Display
"""

import logging
from dataclasses import dataclass
from typing import Dict
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class Resolution:
    """Represents a resolution with width and height."""

    width: int
    height: int

    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 1.0

    def __str__(self):
        return f"{self.width}x{self.height}"


@dataclass
class CoordinateSystemInfo:
    """Complete information about all coordinate systems in the pipeline."""

    raw_resolution: Resolution  # Original RTSP input (1920x1080)
    cropped_resolution: Resolution  # After cropping
    scaled_resolution: Resolution  # After scaling (processing coords)
    stream_resolution: Resolution  # HLS stream output
    display_resolution: Resolution  # Browser video element size

    # Transformation parameters
    crop_top: float  # Proportion of top to crop (0.0-1.0)
    crop_side: float  # Proportion of sides to crop (0.0-1.0)
    scale_factor: float  # Scale factor applied after cropping

    def get_transform_chain(self) -> str:
        """Get a human-readable description of the transformation chain."""
        return (
            f"{self.raw_resolution} → "
            f"crop({self.crop_top:.2f},{self.crop_side:.2f}) → "
            f"{self.cropped_resolution} → "
            f"scale({self.scale_factor:.2f}) → "
            f"{self.scaled_resolution} → "
            f"stream → {self.stream_resolution} → "
            f"display → {self.display_resolution}"
        )


class CoordinateTransform:
    """Handles coordinate transformations between different video coordinate systems."""

    def __init__(self, coord_info: CoordinateSystemInfo):
        self.info = coord_info

    @classmethod
    def from_config(cls, config, stream_resolution: Resolution, display_resolution: Resolution):
        """Create transformer from config and runtime resolutions."""
        # Calculate resolutions at each step
        raw_res = Resolution(1920, 1080)  # Default, should be detected at runtime

        # Calculate cropped resolution
        crop_width = int(raw_res.width * (1.0 - 2 * config.crop_side))
        crop_height = int(raw_res.height * (1.0 - config.crop_top))
        cropped_res = Resolution(crop_width, crop_height)

        # Calculate scaled resolution (this is where stop zones are defined)
        scaled_width = int(cropped_res.width * config.scale)
        scaled_height = int(cropped_res.height * config.scale)
        scaled_res = Resolution(scaled_width, scaled_height)

        coord_info = CoordinateSystemInfo(
            raw_resolution=raw_res,
            cropped_resolution=cropped_res,
            scaled_resolution=scaled_res,
            stream_resolution=stream_resolution,
            display_resolution=display_resolution,
            crop_top=config.crop_top,
            crop_side=config.crop_side,
            scale_factor=config.scale,
        )

        return cls(coord_info)

    def display_to_processing(self, display_x: float, display_y: float) -> Tuple[float, float]:
        """
        Convert browser display coordinates to processing coordinates (where stop zones are defined).
        This is the key transformation for click-to-set functionality.
        """
        # Step 1: Display → Stream coordinates
        stream_x = display_x * (self.info.stream_resolution.width / self.info.display_resolution.width)
        stream_y = display_y * (self.info.stream_resolution.height / self.info.display_resolution.height)

        # Step 2: Stream → Scaled coordinates (where stop zones are defined)
        # This is complex because FFmpeg may have done additional scaling
        scale_x = stream_x * (self.info.scaled_resolution.width / self.info.stream_resolution.width)
        scale_y = stream_y * (self.info.scaled_resolution.height / self.info.stream_resolution.height)

        return scale_x, scale_y

    def processing_to_display(self, proc_x: float, proc_y: float) -> Tuple[float, float]:
        """
        Convert processing coordinates to browser display coordinates.
        Used for rendering stop zone lines on the display.
        """
        # Step 1: Scaled → Stream coordinates
        stream_x = proc_x * (self.info.stream_resolution.width / self.info.scaled_resolution.width)
        stream_y = proc_y * (self.info.stream_resolution.height / self.info.scaled_resolution.height)

        # Step 2: Stream → Display coordinates
        display_x = stream_x * (self.info.display_resolution.width / self.info.stream_resolution.width)
        display_y = stream_y * (self.info.display_resolution.height / self.info.stream_resolution.height)

        return display_x, display_y

    def validate_coordinates(self, x: float, y: float, coord_system: str = "processing") -> bool:
        """Validate that coordinates are within bounds for the specified system."""
        if coord_system == "processing":
            return 0 <= x <= self.info.scaled_resolution.width and 0 <= y <= self.info.scaled_resolution.height
        elif coord_system == "display":
            return 0 <= x <= self.info.display_resolution.width and 0 <= y <= self.info.display_resolution.height
        elif coord_system == "stream":
            return 0 <= x <= self.info.stream_resolution.width and 0 <= y <= self.info.stream_resolution.height
        else:
            return False

    def get_debug_info(self) -> Dict:
        """Get debug information about coordinate transformations."""
        return {
            "transformation_chain": self.info.get_transform_chain(),
            "resolutions": {
                "raw": str(self.info.raw_resolution),
                "cropped": str(self.info.cropped_resolution),
                "scaled": str(self.info.scaled_resolution),
                "stream": str(self.info.stream_resolution),
                "display": str(self.info.display_resolution),
            },
            "transform_parameters": {
                "crop_top": self.info.crop_top,
                "crop_side": self.info.crop_side,
                "scale_factor": self.info.scale_factor,
            },
        }


class CoordinateCalibrator:
    """Helps calibrate coordinate transformations by providing reference points."""

    def __init__(self, transformer: CoordinateTransform):
        self.transformer = transformer
        self.calibration_points = []

    def add_calibration_point(self, display_x: float, display_y: float, expected_proc_x: float, expected_proc_y: float):
        """Add a calibration point to verify coordinate transformation accuracy."""
        actual_proc_x, actual_proc_y = self.transformer.display_to_processing(display_x, display_y)

        error_x = abs(actual_proc_x - expected_proc_x)
        error_y = abs(actual_proc_y - expected_proc_y)

        calibration_point = {
            "display": (display_x, display_y),
            "expected_processing": (expected_proc_x, expected_proc_y),
            "actual_processing": (actual_proc_x, actual_proc_y),
            "error": (error_x, error_y),
            "error_magnitude": (error_x**2 + error_y**2) ** 0.5,
        }

        self.calibration_points.append(calibration_point)

        logger.info(f"Calibration point added: {calibration_point}")
        return calibration_point

    def get_calibration_accuracy(self) -> Dict:
        """Get overall calibration accuracy statistics."""
        if not self.calibration_points:
            return {"error": "No calibration points available"}

        errors = [cp["error_magnitude"] for cp in self.calibration_points]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)

        return {
            "average_error": avg_error,
            "max_error": max_error,
            "num_points": len(self.calibration_points),
            "points": self.calibration_points,
        }
