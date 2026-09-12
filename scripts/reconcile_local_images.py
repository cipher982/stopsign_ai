#!/usr/bin/env python
"""Reconcile stored image paths and local files against the Bremen archive.

Until 2026-09-11 the local -> bremen database path flip could give up permanently
(the pass row is written at zone exit, the flip retried for 20 s and then never
again), so ~15.7k passes kept a ``local://`` path and ~8.4k files sat on disk that
the retention cap could never reclaim. The live path is fixed; this reconciles what
the bug already wrote. Background:
``docs/analysis/2026-09-11-ingest-starvation-postmortem.md``.

For every ``local://<name>`` pass row:

  object in the archive     -> flip the row to ``bremen://<name>``
  object missing, file here -> re-upload the file, then flip
  object missing, file gone -> unrecoverable; reported, left alone

For every file on disk:

  object in the archive     -> delete it. The archive serves it, and for a row
                               still on ``local://`` the web falls back to
                               ``/vehicle-image/<name>``, which streams from
                               Bremen (``stopsign/web/routes/infrastructure.py``),
                               as does the thumbnail builder
                               (``_read_source_image_bytes``).
  not in the archive        -> keep it: it is the only copy. With
                               ``--release-unreferenced`` a file that no pass row
                               references *at all* is released too - the row is
                               written seconds after the capture, so a file this
                               old with no row is the capture-line image of a car
                               that never completed a pass, and nothing can serve
                               it. That is the one irreversible rule here, so it
                               is re-checked against a fresh archive listing and a
                               fresh reference query, and it uses its own, much
                               longer settle window.

Deleting the local copy of an image that is still referenced by a row on
``local://`` does not break the row, but a page rendered *before* the sweep may
still point at ``/vehicle-images/<name>`` and 404 until it is re-fetched. The live
pruner has the same effect; it is transient and needs no action.

Nothing is written unless ``--apply`` is passed, and re-running is safe: the plan is
recomputed from a fresh archive listing and a fresh query immediately before any
mutation. Run it inside the analyzer container, which has the database URL, the
archive credentials and the files:

    docker cp scripts/reconcile_local_images.py <analyzer>:/tmp/reconcile.py
    docker exec <analyzer> python /tmp/reconcile.py                # dry run
    docker exec <analyzer> python /tmp/reconcile.py --apply --release-unreferenced
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
BREMEN_PREFIX = "bremen://"
MINIO_PREFIX = "minio://"
FLIP_CHUNK = 500
# Files younger than this may still be mid-flight in the live analyzer: leave them
# to the in-process sweep rather than racing it.
SETTLE_SECONDS = 900.0
# The unreferenced-file rule destroys the only copy, so it waits far longer.
ORPHAN_SETTLE_SECONDS = 86400.0


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


def referenced_names(engine) -> set[str]:
    """Object names referenced by *any* row, under any prefix.

    A pass that stored ``minio://bucket/name`` or ``bremen://name`` still names the
    same image, so all three prefixes count as a reference.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("select image_path from vehicle_passes where image_path is not null and image_path <> ''")
        )
        names = set()
        for (path,) in rows:
            if path.startswith(LOCAL_PREFIX) or path.startswith(BREMEN_PREFIX):
                names.add(path.split("://", 1)[1])
            elif path.startswith(MINIO_PREFIX):
                parts = path.split("/", 3)
                if len(parts) >= 4 and parts[3]:
                    names.add(parts[3])
    return names


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


def build_plan(archived, referenced, names_on_local, on_disk, now, settle, orphan_settle, release_unreferenced):
    settled = {n for n, mtime in on_disk.items() if now - mtime > settle}
    flippable = sorted(n for n in names_on_local if n in archived)
    recoverable = sorted(n for n in (names_on_local - archived) & settled)
    unrecoverable = sorted(names_on_local - archived - set(on_disk))
    redundant = sorted(n for n in settled if n in archived)
    orphan = sorted(
        n for n, mtime in on_disk.items() if now - mtime > orphan_settle and n not in archived and n not in referenced
    )
    keep_only_copy = sorted(n for n in settled if n not in archived and n in referenced)
    return {
        "flippable": flippable,
        "recoverable": recoverable,
        "unrecoverable": unrecoverable,
        "redundant": redundant,
        "orphan": orphan,
        "orphan_released": orphan if release_unreferenced else [],
        "keep_only_copy": keep_only_copy,
        "archived_count": len(archived),
        "local_rows": len(names_on_local),
        "on_disk": len(on_disk),
        "settled": len(settled),
    }


def print_plan(plan, release_unreferenced) -> None:
    print()
    print(f"archive objects                    {plan['archived_count']}")
    print(f"pass rows on local://              {plan['local_rows']}")
    print(f"files on disk                      {plan['on_disk']}   ({plan['settled']} settled)")
    print()
    print(f"flip rows (object archived)        {len(plan['flippable'])}")
    print(f"re-upload then flip                {len(plan['recoverable'])}")
    print(f"unrecoverable (no object, no file) {len(plan['unrecoverable'])}")
    print(f"keep:   only copy, row references  {len(plan['keep_only_copy'])}")
    print(
        f"orphan: no row, no archive copy    {len(plan['orphan'])}"
        f"{'  (released)' if release_unreferenced else '  (kept; --release-unreferenced to drop)'}"
    )
    print()
    print(
        f"delete local copies                "
        f"{len(plan['redundant']) + len(plan['recoverable']) + len(plan['orphan_released'])}"
        f"   ({len(plan['redundant'])} already archived, {len(plan['recoverable'])} re-uploaded, "
        f"{len(plan['orphan_released'])} unreferenced)"
    )


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


def reupload(client: Minio, names: list[str], apply: bool) -> tuple[list[str], list[str]]:
    """Restore objects that never made it to the archive, from their local copy."""
    if not names:
        return [], []
    if not apply:
        return list(names), []
    uploaded: list[str] = []
    failed: list[str] = []
    for name in names:
        try:
            client.fput_object(
                BREMEN_MINIO_BUCKET, name, os.path.join(LOCAL_IMAGE_DIR, name), content_type="image/jpeg"
            )
            uploaded.append(name)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failed.append(name)
            print(f"  re-upload failed for {name}: {exc}", file=sys.stderr)
    return uploaded, failed


def delete_local(names: list[str], apply: bool) -> tuple[int, list[str]]:
    if not names:
        return 0, []
    if not apply:
        return len(names), []
    removed = 0
    failed: list[str] = []
    for name in names:
        try:
            os.unlink(os.path.join(LOCAL_IMAGE_DIR, name))
            removed += 1
        except OSError as exc:
            failed.append(name)
            print(f"  delete failed for {name}: {exc}", file=sys.stderr)
    return removed, failed


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
        "--orphan-settle-seconds",
        type=float,
        default=ORPHAN_SETTLE_SECONDS,
        help="age an unreferenced, unarchived file needs before it may be released (default: %(default)s)",
    )
    parser.add_argument(
        "--release-unreferenced",
        action="store_true",
        help="also delete unreferenced, unarchived files past the orphan settle window",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exit 0 even when rows are left unreachable (default: exit 1)",
    )
    args = parser.parse_args()

    if not DB_URL or not BREMEN_MINIO_SECRET_KEY:
        print("DB_URL and BREMEN_MINIO_* must be set", file=sys.stderr)
        return 2

    engine = create_engine(DB_URL)
    client = get_archive_client()

    print("listing the archive ...", flush=True)
    plan = build_plan(
        list_archived(client),
        referenced_names(engine),
        local_row_names(engine),
        files_on_disk(),
        time.time(),
        args.settle_seconds,
        args.orphan_settle_seconds,
        args.release_unreferenced,
    )
    print_plan(plan, args.release_unreferenced)

    if not args.apply:
        print("\ndry run; pass --apply to perform this")
        return 1 if plan["unrecoverable"] and not args.allow_incomplete else 0

    # Both sides are re-read immediately before mutating: the first read is a
    # snapshot, the analyzer is live, and the unreferenced rule destroys the only
    # copy of a file.
    print()
    print("re-reading the archive and the database before applying ...", flush=True)
    plan = build_plan(
        list_archived(client),
        referenced_names(engine),
        local_row_names(engine),
        files_on_disk(),
        time.time(),
        args.settle_seconds,
        args.orphan_settle_seconds,
        args.release_unreferenced,
    )

    flipped = flip_rows(engine, plan["flippable"], True)
    print(f"flipped rows                {flipped}")
    uploaded, upload_failed = reupload(client, plan["recoverable"], True)
    print(f"re-uploaded objects         {len(uploaded)}")
    flipped += flip_rows(engine, uploaded, True)
    print(f"flipped rows (after upload) {flipped}")

    removed, delete_failed = delete_local(
        sorted(set(plan["redundant"]) | set(uploaded) | set(plan["orphan_released"])), True
    )
    print(f"deleted local copies        {removed}")
    print(f"left on disk                {len(files_on_disk())}")

    remaining = len(plan["unrecoverable"])
    print()
    print(f"rows still on local://      {remaining} (image gone from both the file and the archive)")
    if upload_failed or delete_failed:
        print(f"failed operations           {len(upload_failed)} upload(s), {len(delete_failed)} delete(s)")
    if (remaining or upload_failed or delete_failed) and not args.allow_incomplete:
        print("incomplete; pass --allow-incomplete to exit 0 anyway")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
