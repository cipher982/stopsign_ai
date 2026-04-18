from __future__ import annotations

from io import BytesIO

from PIL import Image

from stopsign.web.routes.infrastructure import THUMBNAIL_SIZE
from stopsign.web.routes.infrastructure import _build_thumbnail_bytes
from stopsign.web.routes.infrastructure import _normalize_object_name
from stopsign.web.services import images


def test_extract_image_object_name_supports_all_storage_prefixes():
    assert images.extract_image_object_name("local://vehicle_123.jpg") == "vehicle_123.jpg"
    assert images.extract_image_object_name("bremen://archive/vehicle_456.jpg") == "archive/vehicle_456.jpg"
    assert (
        images.extract_image_object_name("minio://vehicle-images/archive/vehicle_789.jpg") == "archive/vehicle_789.jpg"
    )
    assert images.extract_image_object_name(None) is None


def test_resolve_image_and_thumbnail_urls_use_expected_public_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "LOCAL_IMAGE_DIR", str(tmp_path))

    local_name = "vehicle_local.jpg"
    (tmp_path / local_name).write_bytes(b"jpg-bytes")

    assert images.resolve_image_url(f"local://{local_name}") == f"/vehicle-images/{local_name}"
    assert images.resolve_card_thumbnail_url(f"local://{local_name}") == f"/vehicle-thumb/{local_name}"
    assert (
        images.resolve_card_thumbnail_url("minio://vehicle-images/archive/car 1.jpg")
        == "/vehicle-thumb/archive/car%201.jpg"
    )


def test_thumbnail_builder_returns_fixed_size_jpeg():
    source = Image.new("RGB", (320, 200), color=(10, 20, 30))
    buffer = BytesIO()
    source.save(buffer, format="JPEG")

    thumb_bytes = _build_thumbnail_bytes(buffer.getvalue())

    thumb = Image.open(BytesIO(thumb_bytes))
    assert thumb.format == "JPEG"
    assert thumb.size == THUMBNAIL_SIZE


def test_normalize_object_name_rejects_path_traversal():
    assert _normalize_object_name("archive/vehicle_123.jpg") == "archive/vehicle_123.jpg"

    try:
        _normalize_object_name("../secrets.txt")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected path traversal to be rejected")
