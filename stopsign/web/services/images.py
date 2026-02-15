"""Image URL resolution for multiple storage backends."""

import os

from stopsign.settings import LOCAL_IMAGE_DIR


def resolve_image_url(image_path: str | None) -> str:
    """Convert a stored image_path to a servable URL."""
    if not image_path or not isinstance(image_path, str):
        return "/static/placeholder.jpg"
    if image_path.startswith("local://"):
        filename = image_path.replace("local://", "")
        local_file = os.path.join(LOCAL_IMAGE_DIR, filename)
        if os.path.exists(local_file):
            return f"/vehicle-images/{filename}"
        return f"/vehicle-image/{filename}"
    if image_path.startswith("bremen://"):
        filename = image_path.replace("bremen://", "")
        return f"/vehicle-image/{filename}"
    if image_path.startswith("minio://"):
        parts = image_path.split("/", 3)
        if len(parts) >= 4 and parts[3]:
            return f"/vehicle-image/{parts[3]}"
    return "/static/placeholder.jpg"
