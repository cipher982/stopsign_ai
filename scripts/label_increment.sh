#!/usr/bin/env bash
# Daily incremental vehicle labeling.
#
# Labels every vehicle_passes image that does not yet have a vehicle_labels row.
# scripts/label_passes.py is idempotent and resumable, so this is safe to run on
# a schedule. Roughly 210 new passes/day, about $0.08/day at current pricing.
#
# Required env (loaded from files, never hardcoded):
#   - DB_URL, BREMEN_MINIO_*  -> main stack .env (STOPSIGN_ENV_FILE)
#   - OPENAI_API_KEY          -> secret file (STOPSIGN_LABEL_SECRET_FILE)
#
# Overridable via env: STOPSIGN_ENV_FILE, STOPSIGN_LABEL_SECRET_FILE,
# STOPSIGN_LABEL_LOG_DIR, STOPSIGN_LABEL_CONCURRENCY.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="${STOPSIGN_ENV_FILE:-/home/drose/manual-apps/stopsign_ai/docker/production/.env}"
SECRET_FILE="${STOPSIGN_LABEL_SECRET_FILE:-/home/drose/manual-apps/stopsign/label-secret.env}"
LOG_DIR="${STOPSIGN_LABEL_LOG_DIR:-/home/drose/manual-apps/stopsign/logs}"
CONCURRENCY="${STOPSIGN_LABEL_CONCURRENCY:-20}"

mkdir -p "$LOG_DIR"

# Load DB + MinIO + OpenAI env into the shell. python's repr quoting keeps
# special characters in DB_URL safe (a bare `source` would mangle them).
eval "$(python3 - "$ENV_FILE" "$SECRET_FILE" <<'PY'
import sys

KEYS = [
    "DB_URL",
    "BREMEN_MINIO_ACCESS_KEY",
    "BREMEN_MINIO_SECRET_KEY",
    "BREMEN_MINIO_ENDPOINT",
    "BREMEN_MINIO_BUCKET",
    "OPENAI_API_KEY",
]

vals = {}
for path in sys.argv[1:]:
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip("\"'")
    except FileNotFoundError:
        pass

missing = [k for k in KEYS if not vals.get(k)]
if missing:
    raise SystemExit(f"label_increment: missing required env keys: {missing}")

for k in KEYS:
    print(f"export {k}={vals[k]!r}")
PY
)"

exec >>"$LOG_DIR/label_increment.log" 2>&1
echo "=== label_increment $(date -u +%Y-%m-%dT%H:%M:%SZ) concurrency=$CONCURRENCY ==="
uv run --extra db --extra storage --extra labeling python scripts/label_passes.py --concurrency "$CONCURRENCY"
echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
