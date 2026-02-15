# ruff: noqa: E501
"""Debug API routes — coordinate info, zone updates, toggle debug."""

import json
import logging

import redis
from fastapi import APIRouter
from fastapi import Request

from stopsign.config import Config
from stopsign.settings import FRAME_METADATA_KEY
from stopsign.settings import REDIS_URL

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/coordinate-info")
async def get_coordinate_info():
    try:
        config = Config("/app/config/config.yaml")
        redis_client = redis.from_url(REDIS_URL)

        try:
            latest_metadata = redis_client.get(FRAME_METADATA_KEY)
            if not latest_metadata:
                return {"error": "No video metadata available. Video analyzer may not be running."}

            metadata = json.loads(latest_metadata)
            if "raw_video_dimensions" not in metadata:
                return {"error": "Video dimensions not available in metadata."}

            raw_width = metadata["raw_video_dimensions"]["width"]
            raw_height = metadata["raw_video_dimensions"]["height"]

            if not raw_width or not raw_height:
                return {"error": "Invalid video dimensions in metadata."}

        except Exception as e:
            logger.error(f"Failed to get video dimensions from metadata: {e}")
            return {"error": f"Failed to get video dimensions: {str(e)}"}

        crop_side_pixels = int(raw_width * config.crop_side)
        crop_top_pixels = int(raw_height * config.crop_top)
        cropped_width = raw_width - (2 * crop_side_pixels)
        cropped_height = raw_height - crop_top_pixels
        processing_width = int(cropped_width * config.scale)
        processing_height = int(cropped_height * config.scale)

        return {
            "current_stop_zone": {
                "coordinates": [list(point) for point in config.stop_zone] if config.stop_zone else [],
                "coordinate_system": "raw_coordinates",
            },
            "coordinate_system_info": {
                "raw_resolution": f"{raw_width}x{raw_height}",
                "cropped_resolution": f"{cropped_width}x{cropped_height}",
                "processing_resolution": f"{processing_width}x{processing_height}",
                "config_values": {
                    "scale": config.scale,
                    "crop_top": config.crop_top,
                    "crop_side": config.crop_side,
                },
                "transformation_chain": f"{raw_width}x{raw_height} -> crop -> {cropped_width}x{cropped_height} -> scale({config.scale}) -> {processing_width}x{processing_height}",
            },
        }
    except Exception as e:
        logger.error(f"Error getting coordinate info: {str(e)}")
        return {"error": str(e)}


@router.post("/api/update-stop-zone-from-display")
async def update_stop_zone_from_display(request: Request):
    try:
        data = await request.json()
        display_points = data["display_points"]
        video_element_size = data["video_element_size"]

        config = Config("/app/config/config.yaml")
        redis_client = redis.from_url(REDIS_URL)

        try:
            latest_metadata = redis_client.get(FRAME_METADATA_KEY)
            if not latest_metadata:
                return {"error": "No video metadata available. Video analyzer must be running first."}

            metadata = json.loads(latest_metadata)
            if "raw_video_dimensions" not in metadata:
                return {"error": "Video dimensions not available in metadata."}

            raw_width = metadata["raw_video_dimensions"]["width"]
            raw_height = metadata["raw_video_dimensions"]["height"]

            if not raw_width or not raw_height:
                return {"error": "Invalid video dimensions in metadata."}

        except Exception as e:
            logger.error(f"Failed to get video dimensions: {e}")
            return {"error": f"Failed to get video dimensions: {str(e)}"}

        scale_x = raw_width / video_element_size["width"]
        scale_y = raw_height / video_element_size["height"]

        raw_points = []
        for p in display_points:
            raw_x = p["x"] * scale_x
            raw_y = p["y"] * scale_y
            if raw_x < 0 or raw_x > raw_width or raw_y < 0 or raw_y > raw_height:
                logger.warning(
                    f"Coordinate ({raw_x:.1f}, {raw_y:.1f}) is outside raw frame bounds ({raw_width}x{raw_height})"
                )
            raw_points.append({"x": raw_x, "y": raw_y})

        if len(raw_points) != 4:
            return {"error": "Stop zone requires exactly 4 points"}

        stop_zone = [
            (raw_points[0]["x"], raw_points[0]["y"]),
            (raw_points[1]["x"], raw_points[1]["y"]),
            (raw_points[2]["x"], raw_points[2]["y"]),
            (raw_points[3]["x"], raw_points[3]["y"]),
        ]

        try:
            logger.debug(f"Sending to config - stop_zone: {stop_zone}, type: {type(stop_zone)}")
            result = config.update_stop_zone({"stop_zone": stop_zone, "min_stop_duration": 2.0})
        except Exception as e:
            logger.error(f"Error in update_stop_zone: {str(e)}")
            return {"error": f"Config update failed: {str(e)}"}

        redis_client.set("config_updated", "1")

        return {
            "status": "success",
            "version": result["version"],
            "stop_zone": result.get("stop_zone", stop_zone),
            "display_coordinates": display_points,
            "raw_coordinates": raw_points,
            "video_element_size": video_element_size,
            "scaling_info": {
                "coordinate_system": "raw_frame_coordinates",
                "raw_resolution": f"{raw_width}x{raw_height}",
                "scale_factors": f"x={scale_x:.3f}, y={scale_y:.3f}",
                "browser_video_size": f"{video_element_size['width']}x{video_element_size['height']}",
                "note": "Stop zone defined by 4 corner points",
            },
        }
    except Exception as e:
        logger.error(f"Error updating stop zone from display: {str(e)}")
        return {"error": str(e)}


@router.post("/api/debug-coordinates")
async def debug_coordinates(request: Request):
    try:
        data = await request.json()
        display_points = data.get("display_points", [])
        video_element_size = data.get("video_element_size", {})

        config = Config("/app/config/config.yaml")
        redis_client = redis.from_url(REDIS_URL)

        try:
            latest_metadata = redis_client.get(FRAME_METADATA_KEY)
            if not latest_metadata:
                return {"error": "No video metadata available. Video analyzer must be running."}

            metadata = json.loads(latest_metadata)
            if "raw_video_dimensions" not in metadata:
                return {"error": "Video dimensions not available in metadata."}

            raw_width = metadata["raw_video_dimensions"]["width"]
            raw_height = metadata["raw_video_dimensions"]["height"]

            if not raw_width or not raw_height:
                return {"error": "Invalid video dimensions in metadata."}

        except Exception as e:
            logger.error(f"Failed to get video dimensions: {e}")
            return {"error": f"Failed to get video dimensions: {str(e)}"}

        if video_element_size:
            scale_x = raw_width / video_element_size["width"]
            scale_y = raw_height / video_element_size["height"]
        else:
            scale_x = scale_y = 1.0

        transformations = []
        for i, dp in enumerate(display_points):
            raw_x = dp["x"] * scale_x
            raw_y = dp["y"] * scale_y
            back_x = raw_x / scale_x
            back_y = raw_y / scale_y
            transformations.append(
                {
                    "point_index": i,
                    "display_input": {"x": dp["x"], "y": dp["y"]},
                    "raw_output": {"x": raw_x, "y": raw_y},
                    "display_roundtrip": {"x": back_x, "y": back_y},
                    "roundtrip_error": {"x": abs(dp["x"] - back_x), "y": abs(dp["y"] - back_y)},
                    "bounds_check": {
                        "display_valid": True,
                        "raw_valid": 0 <= raw_x <= raw_width and 0 <= raw_y <= raw_height,
                    },
                }
            )

        return {
            "coordinate_system": "raw_frame_coordinates",
            "current_stop_zone": {
                "coordinates": [list(point) for point in config.stop_zone] if config.stop_zone else [],
                "coordinate_system": "raw_coordinates",
            },
            "scaling_info": {
                "raw_resolution": f"{raw_width}x{raw_height}",
                "scale_factors": f"x={scale_x:.3f}, y={scale_y:.3f}",
                "browser_video_size": f"{video_element_size.get('width', 'unknown')}x{video_element_size.get('height', 'unknown')}",
            },
            "point_transformations": transformations,
        }
    except Exception as e:
        logger.error(f"Error in debug coordinates: {str(e)}")
        return {"error": str(e)}


@router.post("/api/toggle-debug-zones")
async def toggle_debug_zones(request: Request):
    try:
        data = await request.json()
        enabled = data.get("enabled", False)
        redis_client = redis.from_url(REDIS_URL)
        redis_client.set("debug_zones_enabled", "1" if enabled else "0")
        return {"status": "success", "debug_zones_enabled": enabled}
    except Exception as e:
        logger.error(f"Error toggling debug zones: {str(e)}")
        return {"error": str(e)}


@router.post("/api/update-zone-from-display")
async def update_zone_from_display(request: Request):
    try:
        data = await request.json()
        zone_type = data.get("zone_type")
        display_points = data["display_points"]
        video_element_size = data["video_element_size"]

        config = Config("/app/config/config.yaml")
        redis_client = redis.from_url(REDIS_URL)

        try:
            latest_metadata = redis_client.get(FRAME_METADATA_KEY)
            if not latest_metadata:
                return {"error": "No video metadata available. Video analyzer must be running first."}

            metadata = json.loads(latest_metadata)
            if "raw_video_dimensions" not in metadata:
                return {"error": "Video dimensions not available in metadata."}

            raw_width = metadata["raw_video_dimensions"]["width"]
            raw_height = metadata["raw_video_dimensions"]["height"]

            if not raw_width or not raw_height:
                return {"error": "Invalid video dimensions in metadata."}

        except Exception as e:
            logger.error(f"Failed to get video dimensions: {e}")
            return {"error": f"Failed to get video dimensions: {str(e)}"}

        scale_x = raw_width / video_element_size["width"]
        scale_y = raw_height / video_element_size["height"]

        if zone_type in ["pre-stop", "capture"]:
            if len(display_points) != 2:
                return {"error": f"{zone_type} zone requires exactly 2 points"}

            raw_line = []
            for p in display_points:
                raw_x = p["x"] * scale_x
                raw_y = p["y"] * scale_y
                raw_line.append((raw_x, raw_y))

            dx = abs(raw_line[1][0] - raw_line[0][0])
            dy = abs(raw_line[1][1] - raw_line[0][1])
            line_length = (dx * dx + dy * dy) ** 0.5

            if line_length < 10:
                return {"error": f"Line too short ({line_length:.1f} pixels). Draw a longer line."}

            if zone_type == "pre-stop":
                result = config.update_zone("pre_stop", raw_line)
            elif zone_type == "capture":
                result = config.update_zone("capture", raw_line)
        else:
            return {"error": f"Unsupported zone type '{zone_type}'"}

        try:
            redis_client.set("config_updated", "1")
        except Exception as e:
            logger.warning(f"Could not signal config update: {e}")

        return {
            "status": "success",
            "version": result.get("version", "unknown"),
            "zone_type": zone_type,
            "zone_data": result,
            "display_coordinates": display_points,
            "video_element_size": video_element_size,
            "message": f"{zone_type} zone updated successfully",
        }
    except Exception as e:
        logger.error(f"Error updating zone from display: {str(e)}")
        return {"error": str(e)}
