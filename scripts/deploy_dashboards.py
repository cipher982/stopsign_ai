"""Simple script to upload Grafana dashboard JSON files.

This replaces the brittle shell/`curl` logic that produced huge HTML dumps
in CI logs.  It expects two environment variables (GRAFANA_URL and
GRAFANA_API_KEY) and a single argument pointing to the directory that
contains the dashboard JSON files.

Usage (inside GitHub Actions or locally):

    python scripts/deploy_dashboards.py path/to/dashboards

The script walks the directory recursively, wraps each JSON file in the
payload structure Grafana expects, and POSTs it to `/api/dashboards/db`.

If any upload fails, the script exits non-zero so the workflow is marked as
failed, while still keeping log output concise (first ~500 chars of the
response).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Final

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover – handled in CI via pip install
    print("The 'requests' package is required. Run 'pip install requests'.", file=sys.stderr)
    sys.exit(1)


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Environment variable '{name}' is required", file=sys.stderr)
        sys.exit(1)
    return value


def upload_dashboards(directory: Path, grafana_url: str, api_key: str) -> None:
    headers: Final = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    failures = 0

    # Walk recursively to support sub-folders.
    json_files = sorted(p for p in directory.rglob("*.json") if p.is_file())

    if not json_files:
        print(f"No .json dashboards found under '{directory}'. Exiting early.")
        return

    print(f"Found {len(json_files)} dashboard(s) to deploy.\n")

    for path in json_files:
        print(f"Deploying: {path.relative_to(directory.parent)}")

        try:
            with path.open("r", encoding="utf-8") as fh:
                dashboard_def = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"  ✗ Failed to parse JSON – {exc}")
            failures += 1
            continue

        payload = {
            "dashboard": dashboard_def,
            # Folder 0 = General.  Adjust if you use a different folder.
            "folderId": 0,
            "overwrite": True,
        }

        url = grafana_url.rstrip("/") + "/api/dashboards/db"

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
        except requests.RequestException as exc:
            print(f"  ✗ HTTP request failed – {exc}")
            failures += 1
            continue

        if resp.status_code in (200, 202):
            uid = (
                resp.json().get("uid", "?")
                if resp.headers.get("content-type", "").startswith("application/json")
                else "?"
            )
            print(f"  ✓ Success (uid={uid})")
        else:
            snippet = resp.text.replace("\n", " ")[:500]
            print(f"  ✗ HTTP {resp.status_code} – {snippet}…")
            failures += 1

    if failures:
        print(f"\n{failures} dashboard(s) failed to deploy.")
        sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/deploy_dashboards.py <dashboards-dir>", file=sys.stderr)
        sys.exit(1)

    dashboards_dir = Path(sys.argv[1]).expanduser().resolve()
    if not dashboards_dir.exists():
        print(f"Provided dashboards directory does not exist: {dashboards_dir}", file=sys.stderr)
        sys.exit(1)

    grafana_url = _env("GRAFANA_URL")
    api_key = _env("GRAFANA_API_KEY")

    upload_dashboards(dashboards_dir, grafana_url, api_key)


if __name__ == "__main__":
    main()
