from __future__ import annotations

from stopsign.web.app import _compute_static_asset_hash
from stopsign.web.app import resolve_asset_version


def test_static_asset_hash_is_stable_for_identical_content(tmp_path, monkeypatch):
    monkeypatch.delenv("ASSET_VERSION", raising=False)

    static_dir = tmp_path / "static"
    (static_dir / "js").mkdir(parents=True)
    (static_dir / "base.css").write_text("body { color: black; }\n", encoding="utf-8")
    (static_dir / "js" / "home.js").write_text("console.log('ready');\n", encoding="utf-8")

    first = _compute_static_asset_hash(static_dir)
    second = _compute_static_asset_hash(static_dir)

    assert first == second


def test_static_asset_hash_changes_when_bytes_change(tmp_path, monkeypatch):
    monkeypatch.delenv("ASSET_VERSION", raising=False)

    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True)
    asset = static_dir / "base.css"
    asset.write_text("body { color: black; }\n", encoding="utf-8")

    first = _compute_static_asset_hash(static_dir)
    asset.write_text("body { color: white; }\n", encoding="utf-8")
    second = _compute_static_asset_hash(static_dir)

    assert first != second


def test_resolve_asset_version_prefers_explicit_override(tmp_path, monkeypatch):
    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True)
    (static_dir / "base.css").write_text("body { color: black; }\n", encoding="utf-8")

    monkeypatch.setenv("ASSET_VERSION", "manual-build-id")

    assert resolve_asset_version(static_dir) == "manual-build-id"
