"""Image URL resolution for multiple storage backends."""

import os
from urllib.parse import quote

from stopsign.settings import LOCAL_IMAGE_DIR


def extract_image_object_name(image_path: str | None) -> str | None:
    """Extract the storage object name from a stored image path."""
    if not image_path or not isinstance(image_path, str):
        return None
    if image_path.startswith("local://"):
        return image_path.replace("local://", "")
    if image_path.startswith("bremen://"):
        return image_path.replace("bremen://", "")
    if image_path.startswith("minio://"):
        parts = image_path.split("/", 3)
        if len(parts) >= 4 and parts[3]:
            return parts[3]
    return None


def resolve_image_url(image_path: str | None) -> str:
    """Convert a stored image_path to a servable full-size URL."""
    object_name = extract_image_object_name(image_path)
    if not object_name:
        return "/static/placeholder.jpg"

    if image_path.startswith("local://"):
        local_file = os.path.join(LOCAL_IMAGE_DIR, object_name)
        if os.path.exists(local_file):
            return f"/vehicle-images/{quote(object_name, safe='/')}"

    return f"/vehicle-image/{quote(object_name, safe='/')}"


def resolve_card_thumbnail_url(image_path: str | None) -> str:
    """Return the optimized thumbnail URL for compact list cards."""
    object_name = extract_image_object_name(image_path)
    if not object_name:
        return resolve_image_url(image_path)
    return f"/vehicle-thumb/{quote(object_name, safe='/')}"
