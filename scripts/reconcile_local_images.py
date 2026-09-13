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
                               or durable pending-pass record references is
                               released too. An unreadable durable spool is
                               uncertainty, so every such candidate is kept.
                               The row is written seconds after the capture, so a
                               file this old with no row is the capture-line image
                               of a car that never completed a pass, and nothing
                               can serve it. That is the one irreversible rule
                               here: it needs ``--analyzer-stopped`` (stop the
                               container first), waits its own much longer settle
                               window, and re-checks each candidate against the
                               live database, archive, and durable pass spool
                               immediately before its own unlink.

Deleting the local copy of an image that is still referenced by a row on
``local://`` does not break the row, but a page rendered *before* the sweep may
still point at ``/vehicle-images/<name>`` and 404 until it is re-fetched. The live
pruner has the same effect; it is transient and needs no action.

Nothing is written unless ``--apply`` is passed, and re-running is safe: the plan is
recomputed from a fresh archive listing, fresh database query, and fresh durable
spool read immediately before any mutation.


Run the dry run inside the analyzer container, which has the database URL, the
archive credentials and the files:

    docker cp scripts/reconcile_local_images.py <analyzer>:/tmp/reconcile.py
    docker exec <analyzer> python /tmp/reconcile.py                # dry run

``--release-unreferenced`` additionally needs the analyzer to be *stopped*, so that
nothing can be mid-upload while a file's only copy is removed - and a stopped
container cannot be ``docker exec``'d into, so the apply runs in a throwaway
container from the same image, with the same volume and the analyzer's environment:

    docker stop <analyzer>
    docker run --rm --network host \
      -e DB_URL="$(docker inspect <analyzer> --format '{{range .Config.Env}}\
{{println .}}{{end}}' | sed -n 's/^DB_URL=//p')" \
      -e BREMEN_MINIO_ENDPOINT="$(...)" -e BREMEN_MINIO_ACCESS_KEY="$(...)" \
      -e BREMEN_MINIO_SECRET_KEY="$(...)" -e BREMEN_MINIO_BUCKET=vehicle-images \
      -v "$(docker inspect <analyzer> --format \
'{{range .Mounts}}{{if eq .Destination "/app/data"}}{{.Name}}{{end}}{{end}}'):/app/data" \
      -v "$PWD/scripts:/scripts:ro" \
      $(docker inspect <analyzer> --format '{{.Config.Image}}') \
      python /scripts/reconcile_local_images.py --apply --release-unreferenced \
      --analyzer-stopped
    docker start <analyzer>

Without ``--release-unreferenced`` the apply is not quiescence-sensitive (it only
touches rows and files the archive already serves) and can run in the live
container.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import urllib3
from minio import Minio
from minio.error import S3Error
from sqlalchemy import create_engine
from sqlalchemy import text

from stopsign.pass_spool import pending_pass_image_paths
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


def build_plan(
    archived,
    referenced,
    names_on_local,
    on_disk,
    now,
    settle,
    orphan_settle,
    release_unreferenced,
    pending_spool: set[str] | None = None,
):
    settled = {n for n, mtime in on_disk.items() if now - mtime > settle}
    flippable = sorted(n for n in names_on_local if n in archived)
    recoverable = sorted(n for n in (names_on_local - archived) & settled)
    unrecoverable = sorted(names_on_local - archived - set(on_disk))
    redundant = sorted(n for n in settled if n in archived)
    spool_complete = pending_spool is not None
    spool_names = pending_spool or set()
    orphan = sorted(
        n
        for n, mtime in on_disk.items()
        if now - mtime > orphan_settle
        and n not in archived
        and n not in referenced
        and spool_complete
        and n not in spool_names
    )
    keep_only_copy = sorted(
        n for n in settled if n not in archived and (n in referenced or not spool_complete or n in spool_names)
    )
    return {
        "flippable": flippable,
        "recoverable": recoverable,
        "unrecoverable": unrecoverable,
        "redundant": redundant,
        "orphan": orphan,
        "orphan_released": orphan if release_unreferenced else [],
        "keep_only_copy": keep_only_copy,
        "pending_spool_complete": spool_complete,
        "pending_spool_count": len(spool_names),
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
    spool_state = "complete" if plan["pending_spool_complete"] else "UNREADABLE"
    print(f"durable pending-pass spool         {plan['pending_spool_count']} ({spool_state})")
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


def _remote_object_digest(client: Minio, name: str) -> tuple[int, str, str | None] | None:
    """Read the object and prove it stayed the same while it was read."""
    try:
        before = client.stat_object(BREMEN_MINIO_BUCKET, name)
        response = client.get_object(BREMEN_MINIO_BUCKET, name)
        digest = hashlib.sha256()
        size = 0
        try:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        finally:
            response.close()
            response.release_conn()
        after = client.stat_object(BREMEN_MINIO_BUCKET, name)
        if size != before.size or after.size != before.size or after.etag != before.etag:
            return None
        return size, digest.hexdigest(), before.etag
    except Exception as exc:  # noqa: BLE001 - uncertainty must preserve the local copy
        print(f"  archive readback for {name} failed ({exc}); keeping the file", file=sys.stderr)
        return None


def archive_copy_matches_local(client: Minio, name: str) -> bool:
    """Require a remote readback with matching length and SHA-256 before unlinking."""
    path = os.path.join(LOCAL_IMAGE_DIR, name)
    try:
        local_size = os.stat(path).st_size
        local_digest = hashlib.sha256()
        with open(path, "rb") as local_file:
            while chunk := local_file.read(1024 * 1024):
                local_digest.update(chunk)
    except OSError as exc:
        print(f"  local readback for {name} failed ({exc}); keeping the file", file=sys.stderr)
        return False
    remote = _remote_object_digest(client, name)
    return remote is not None and remote[0] == local_size and remote[1] == local_digest.hexdigest()


def delete_local(client: Minio, names: list[str], apply: bool) -> tuple[int, list[str]]:
    """Remove local copies only after remote length and digest readback."""
    if not names:
        return 0, []
    if not apply:
        return len(names), []
    removed = 0
    failed: list[str] = []
    for name in names:
        if not archive_copy_matches_local(client, name):
            failed.append(name)
            continue
        try:
            os.unlink(os.path.join(LOCAL_IMAGE_DIR, name))
            removed += 1
        except OSError as exc:
            failed.append(name)
            print(f"  delete failed for {name}: {exc}", file=sys.stderr)
    return removed, failed


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


def object_referenced(engine, name: str) -> bool:
    """Is this object named by any pass row, under any prefix?"""
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    "select 1 from vehicle_passes "
                    "where image_path = :local or image_path = :bremen or image_path like :minio "
                    "limit 1"
                ),
                {"local": f"{LOCAL_PREFIX}{name}", "bremen": f"{BREMEN_PREFIX}{name}", "minio": f"%/{name}"},
            ).scalar()
        )


def archived_now(client: Minio, name: str) -> bool:
    """Is this object in the archive right now?

    Only an explicit not-found means no. Any other error (permissions, a 5xx, a
    bucket problem) is uncertainty, and uncertainty must keep the file: this answer
    decides whether the only local copy is removed.
    """
    try:
        client.stat_object(BREMEN_MINIO_BUCKET, name)
        return True
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket", "ResourceNotFound", "NotFound"}:
            return False
        print(f"  archive check for {name} failed ({exc.code}); keeping the file", file=sys.stderr)
        return True
    except Exception as exc:  # noqa: BLE001 - same reasoning: keep on doubt
        print(f"  archive check for {name} errored ({exc}); keeping the file", file=sys.stderr)
        return True


def release_unreferenced(
    client: Minio,
    engine,
    names: list[str],
    apply: bool,
    pending_spool_names: set[str] | None,
) -> tuple[int, list[str], list[str]]:
    """Delete only files absent from the database, archive, and durable spool."""
    if not names:
        return 0, [], []
    if not apply:
        return len(names), [], []
    removed = 0
    kept: list[str] = []
    failed: list[str] = []
    for name in names:
        if (
            pending_spool_names is None
            or name in pending_spool_names
            or object_referenced(engine, name)
            or archived_now(client, name)
        ):
            kept.append(name)
            continue
        try:
            os.unlink(os.path.join(LOCAL_IMAGE_DIR, name))
            removed += 1
        except OSError as exc:
            failed.append(name)
            print(f"  delete failed for {name}: {exc}", file=sys.stderr)
    return removed, kept, failed


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
        "--analyzer-stopped",
        action="store_true",
        help="acknowledge that the analyzer is stopped, required by --release-unreferenced",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exit 0 even when rows are left unreachable (default: exit 1)",
    )
    args = parser.parse_args()

    if args.release_unreferenced and not args.analyzer_stopped:
        print(
            "--release-unreferenced deletes the only copy of a file. Stop the analyzer\n"
            "container first (so nothing can be mid-upload) and pass --analyzer-stopped.",
            file=sys.stderr,
        )
        return 2

    if not DB_URL or not BREMEN_MINIO_SECRET_KEY:
        print("DB_URL and BREMEN_MINIO_* must be set", file=sys.stderr)
        return 2

    engine = create_engine(DB_URL)
    client = get_archive_client()

    pending_spool_names = pending_pass_image_paths()
    if pending_spool_names is None:
        print("durable pass spool is unreadable; orphan release candidates will be kept", file=sys.stderr)

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
        pending_spool_names,
    )
    print_plan(plan, args.release_unreferenced)

    if not args.apply:
        print("\ndry run; pass --apply to perform this")
        return 1 if plan["unrecoverable"] and not args.allow_incomplete else 0

    # Both sides are re-read immediately before mutating: the first read is a
    # snapshot, the analyzer is live, and the unreferenced rule destroys the only
    # copy of a file.
    print()
    print("re-reading the archive, database, and durable pass spool before applying ...", flush=True)
    pending_spool_names = pending_pass_image_paths()
    if args.release_unreferenced and pending_spool_names is None:
        print("refusing orphan release: durable pass spool is unreadable", file=sys.stderr)
        return 1
    plan = build_plan(
        list_archived(client),
        referenced_names(engine),
        local_row_names(engine),
        files_on_disk(),
        time.time(),
        args.settle_seconds,
        args.orphan_settle_seconds,
        args.release_unreferenced,
        pending_spool_names,
    )

    flipped = flip_rows(engine, plan["flippable"], True)
    print(f"flipped rows                {flipped}")
    uploaded, upload_failed = reupload(client, plan["recoverable"], True)
    print(f"re-uploaded objects         {len(uploaded)}")
    flipped += flip_rows(engine, uploaded, True)
    print(f"flipped rows (after upload) {flipped}")

    removed, delete_failed = delete_local(client, sorted(set(plan["redundant"]) | set(uploaded)), True)
    orphans_removed, orphans_kept, orphan_failed = release_unreferenced(
        client,
        engine,
        plan["orphan_released"],
        True,
        pending_spool_names,
    )
    removed += orphans_removed
    delete_failed += orphan_failed
    print(f"deleted local copies        {removed}")
    if orphans_kept:
        print(f"orphans kept (referenced, pending, or archived since the listing) {len(orphans_kept)}")
    print(f"left on disk                {len(files_on_disk())}")

    # Count what is actually left, not what the pre-mutation plan expected, and split
    # it: a row whose file is still here is simply waiting for the archive (a capture
    # from the last few minutes), which is not the same as an image that is gone.
    remaining = local_row_names(engine)
    on_disk = set(files_on_disk())
    still_here = sorted(name for name in remaining if name in on_disk)
    gone = sorted(name for name in remaining if name not in on_disk)
    print()
    print(f"rows still on local://      {len(remaining)}")
    print(f"  image still on disk       {len(still_here)} (pending the archive, served from here)")
    print(f"  gone from both            {len(gone)} (nothing can serve these)")
    if gone and not args.allow_incomplete:
        print("incomplete; pass --allow-incomplete to exit 0 anyway")
        return 1
    if upload_failed or delete_failed:
        print(f"failed operations           {len(upload_failed)} upload(s), {len(delete_failed)} delete(s)")
        if not args.allow_incomplete:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
