#!/usr/bin/env bash
# Daily incremental vehicle labeling.
#
# Labels every vehicle_passes image that does not yet have a vehicle_labels row.
# scripts/label_passes.py is idempotent and resumable. Roughly 210 new
# passes/day, about $0.08/day at current pricing.
#
# Secrets (DB_URL, BREMEN_MINIO_*, OPENAI_API_KEY) live in the `stopsign-ai`
# Infisical project and are injected at runtime via `infisical run`, using the
# same Universal Auth machine identity the web/analyzer containers use
# (INFISICAL_CLIENT_ID + INFISICAL_CLIENT_SECRET in the main stack .env).
# No secret files, no hardcoded creds; key rotation in Infisical is picked up
# on the next run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Cron runs with a minimal PATH; make the host's ~/.local/bin (uv, infisical) visible.
export PATH="$HOME/.local/bin:$PATH"

ENV_FILE="${STOPSIGN_ENV_FILE:-/home/drose/manual-apps/stopsign_ai/docker/production/.env}"
LOG_DIR="${STOPSIGN_LABEL_LOG_DIR:-/home/drose/manual-apps/stopsign/logs}"
CONCURRENCY="${STOPSIGN_LABEL_CONCURRENCY:-20}"

mkdir -p "$LOG_DIR"

# The only runtime-held secret: the machine identity creds the app already uses.
INFISICAL_CLIENT_ID="$(grep -E '^INFISICAL_CLIENT_ID=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
INFISICAL_CLIENT_SECRET="$(grep -E '^INFISICAL_CLIENT_SECRET=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
export INFISICAL_CLIENT_ID INFISICAL_CLIENT_SECRET

# Authenticate to the self-hosted Infisical and inject the project's secrets,
# exactly as the container entrypoint does.
export INFISICAL_TOKEN="$(curl -sf -X POST "https://secrets.drose.io/api/v1/auth/universal-auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"clientId\":\"${INFISICAL_CLIENT_ID}\",\"clientSecret\":\"${INFISICAL_CLIENT_SECRET}\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['accessToken'])")"

exec >>"$LOG_DIR/label_increment.log" 2>&1
echo "=== label_increment $(date -u +%Y-%m-%dT%H:%M:%SZ) concurrency=$CONCURRENCY ==="
infisical run \
  --projectId "9c373776-768f-454b-a7b3-d1cc40deb475" \
  --env prod \
  --domain "https://secrets.drose.io" \
  -- uv run --extra db --extra storage --extra labeling python scripts/label_passes.py --concurrency "$CONCURRENCY"
echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
