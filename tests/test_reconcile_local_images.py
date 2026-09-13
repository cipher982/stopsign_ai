from __future__ import annotations

import io
from types import SimpleNamespace

import scripts.reconcile_local_images as reconcile
from scripts.reconcile_local_images import build_plan
from scripts.reconcile_local_images import delete_local


class _Response(io.BytesIO):
    def release_conn(self):
        pass


class _Archive:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.etag = "stable"

    def stat_object(self, _bucket, _name):
        return SimpleNamespace(size=len(self.payload), etag=self.etag)

    def get_object(self, _bucket, _name):
        return _Response(self.payload)


_NOW = 1_000_000.0


def _plan(*, pending_spool: set[str] | None):
    return build_plan(
        archived=set(),
        referenced=set(),
        names_on_local=set(),
        on_disk={"capture.jpg": _NOW - 2 * 86400},
        now=_NOW,
        settle=900,
        orphan_settle=86400,
        release_unreferenced=True,
        pending_spool=pending_spool,
    )


def test_pending_spool_reference_blocks_orphan_release():
    plan = _plan(pending_spool={"capture.jpg"})

    assert plan["orphan"] == []
    assert plan["keep_only_copy"] == ["capture.jpg"]
    assert plan["pending_spool_complete"] is True


def test_unreadable_spool_blocks_orphan_release():
    plan = _plan(pending_spool=None)

    assert plan["orphan"] == []
    assert plan["keep_only_copy"] == ["capture.jpg"]
    assert plan["pending_spool_complete"] is False


def test_delete_requires_remote_readback_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(reconcile, "LOCAL_IMAGE_DIR", str(tmp_path))
    (tmp_path / "capture.jpg").write_bytes(b"local")

    removed, failed = delete_local(_Archive(b"other"), ["capture.jpg"], apply=True)

    assert removed == 0
    assert failed == ["capture.jpg"]
    assert (tmp_path / "capture.jpg").exists()


def test_delete_removes_only_a_matching_remote_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(reconcile, "LOCAL_IMAGE_DIR", str(tmp_path))
    (tmp_path / "capture.jpg").write_bytes(b"local")

    removed, failed = delete_local(_Archive(b"local"), ["capture.jpg"], apply=True)

    assert removed == 1
    assert failed == []
    assert not (tmp_path / "capture.jpg").exists()
