#!/usr/bin/env python
"""Reconcile stored image paths and local files against the Bremen archive.

Until 2026-09-11 the local -> bremen database path flip could give up permanently
(the pass row is written at zone exit, the flip retried for 20 s and then never
again), so ~15.7k passes kept a ``local://`` path and ~8.4k files sat on disk that
the retention cap could never reclaim. The live path is fixed; this reconciles what
the bug already wrote. Background:
``docs/analysis/2026-09-11-ingest-starvation-postmortem.md``.

For every ``local://<name>`` pass row:

  object in the archive            -> flip the row to ``bremen://<name>``
  object missing, file on disk     -> re-upload it, then flip
  object missing, file gone        -> unrecoverable; reported, left alone

For every file on disk:

  object in the archive, settled   -> delete it. The archive serves it, and for a
                                      row still on ``local://`` the web falls back
                                      to ``/vehicle-image/<name>``, which streams
                                      from Bremen (``stopsign/web/services/images.py``).
  not in the archive               -> keep by default. It is the only copy, and no
                                      pass row may lose its image. With
                                      ``--release-unreferenced`` a file that no
                                      ``local://`` row references is released too:
                                      the row is written seconds after the capture,
                                      so an old file with no row is the
                                      capture-line image of a car that never
                                      completed a pass - nothing can ever serve it.

Nothing is written unless ``--apply`` is passed. Run it inside the analyzer
container, which has the database URL, the archive credentials and the files:

    docker cp scripts/reconcile_local_images.py <analyzer>:/tmp/reconcile.py
    docker exec <analyzer> python /tmp/reconcile.py            # dry run
    docker exec <analyzer> python /tmp/reconcile.py --apply
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import urllib3
from minio import Minio
from sqlalchemy import create_engine
from sqlalchemy import text

from stopsign.settings import BREMEN_MINIO_ACCESS_KEY
from stopsign.settings import BREMEN_MINIO_BUCKET
from stopsign.settings import BREMEN_MINIO_ENDPOINT
from stopsign.settings import BREMEN_MINIO_SECRET_KEY
from stopsign.settings import DB_URL
from stopsign.settings import LOCAL_IMAGE_DIR

LOCAL_PREFIX = "local://"
FLIP_CHUNK = 500
# Files younger than this may still be mid-flight in the live analyzer; leave them
# for the in-process sweep rather than racing it.
SETTLE_SECONDS = 900.0


def get_archive_client() -> Minio:
    return Minio(
        BREMEN_MINIO_ENDPOINT,
        access_key=BREMEN_MINIO_ACCESS_KEY,
        secret_key=BREMEN_MINIO_SECRET_KEY,
        secure=False,
        http_client=urllib3.PoolManager(timeout=60.0),
    )


def list_archived(client: Minio) -> set[str]:
    return {obj.object_name for obj in client.list_objects(BREMEN_MINIO_BUCKET, recursive=True)}


def local_row_names(engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("select image_path from vehicle_passes where image_path like :prefix"),
            {"prefix": f"{LOCAL_PREFIX}%"},
        )
        return {row[0][len(LOCAL_PREFIX) :] for row in rows}


def files_on_disk() -> dict[str, float]:
    found: dict[str, float] = {}
    for name in os.listdir(LOCAL_IMAGE_DIR):
        path = os.path.join(LOCAL_IMAGE_DIR, name)
        try:
            if not os.path.isfile(path):
                # e.g. the thumbnail cache directory that lives alongside the images
                continue
            found[name] = os.stat(path).st_mtime
        except OSError:
            continue
    return found


def flip_rows(engine, names: list[str], apply: bool) -> int:
    """Rewrite ``local://<name>`` to ``bremen://<name>`` (the archive already has it)."""
    if not names:
        return 0
    if not apply:
        return len(names)
    flipped = 0
    for start in range(0, len(names), FLIP_CHUNK):
        chunk = [f"{LOCAL_PREFIX}{n}" for n in names[start : start + FLIP_CHUNK]]
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    "update vehicle_passes "
                    "set image_path = 'bremen://' || substring(image_path from 9) "
                    "where image_path = any(:names)"
                ),
                {"names": chunk},
            )
            flipped += result.rowcount or 0
    return flipped


def reupload(client: Minio, names: list[str], apply: bool) -> list[str]:
    """Restore objects that never made it to the archive, from their local copy."""
    if not names:
        return []
    if not apply:
        return list(names)
    uploaded = []
    for name in names:
        path = os.path.join(LOCAL_IMAGE_DIR, name)
        try:
            client.fput_object(BREMEN_MINIO_BUCKET, name, path, content_type="image/jpeg")
            uploaded.append(name)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  re-upload failed for {name}: {exc}", file=sys.stderr)
    return uploaded


def delete_local(names: list[str], apply: bool) -> int:
    if not apply:
        return len(names)
    removed = 0
    for name in names:
        try:
            os.unlink(os.path.join(LOCAL_IMAGE_DIR, name))
            removed += 1
        except OSError as exc:
            print(f"  delete failed for {name}: {exc}", file=sys.stderr)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="perform the changes (default is a dry run)")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=SETTLE_SECONDS,
        help="leave files younger than this alone (default: %(default)s)",
    )
    parser.add_argument(
        "--release-unreferenced",
        action="store_true",
        help="also delete settled files that no local:// row references and the archive does not hold",
    )
    args = parser.parse_args()

    if not DB_URL or not BREMEN_MINIO_SECRET_KEY:
        print("DB_URL and BREMEN_MINIO_* must be set", file=sys.stderr)
        return 2

    engine = create_engine(DB_URL)
    client = get_archive_client()

    print("listing the archive ...", flush=True)
    archived = list_archived(client)
    names_on_local = local_row_names(engine)
    on_disk = files_on_disk()
    now = time.time()

    flippable = sorted(n for n in names_on_local if n in archived)
    recoverable = sorted(n for n in names_on_local - archived if n in on_disk)
    unrecoverable = sorted(names_on_local - archived - set(on_disk))
    settled = [n for n, mtime in on_disk.items() if now - mtime > args.settle_seconds]
    archived_settled = sorted(n for n in settled if n in archived)
    orphan_settled = sorted(n for n in settled if n not in archived and n not in names_on_local)

    print()
    print(f"archive objects            {len(archived)}")
    print(f"pass rows on local://      {len(names_on_local)}")
    print(f"files on disk              {len(on_disk)}   ({len(settled)} settled)")
    print()
    print(f"flip rows (object archived)        {len(flippable)}")
    print(f"re-upload then flip                {len(recoverable)}")
    print(f"unrecoverable (no object, no file) {len(unrecoverable)}")
    print(f"delete: redundant local copies     {len(archived_settled)}")
    print(
        f"keep:   only copy, row references  {len([n for n in settled if n not in archived and n in names_on_local])}"
    )
    print(
        f"orphan: no row, no archive copy    {len(orphan_settled)}"
        f"{'  (released)' if args.release_unreferenced else '  (kept; --release-unreferenced to drop)'}"
    )

    if not args.apply:
        print("\ndry run; pass --apply to perform this")
        return 0

    print()
    flipped = flip_rows(engine, flippable, True)
    print(f"flipped rows                {flipped}")
    uploaded = reupload(client, recoverable, True)
    print(f"re-uploaded objects         {len(uploaded)}")
    flipped += flip_rows(engine, uploaded, True)
    to_delete = set(archived_settled) | set(uploaded)
    if args.release_unreferenced:
        to_delete |= set(orphan_settled)
    removed = delete_local(sorted(to_delete), True)
    print(f"deleted local copies        {removed}")
    print(f"left on disk                {len(files_on_disk())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
