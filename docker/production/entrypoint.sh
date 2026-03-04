#!/bin/sh
set -e

# If Infisical machine identity creds are present, inject secrets at startup
if [ -n "$INFISICAL_CLIENT_ID" ] && [ -n "$INFISICAL_CLIENT_SECRET" ]; then
  INFISICAL_TOKEN=$(curl -sf -X POST "https://secrets.drose.io/api/v1/auth/universal-auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"clientId\":\"${INFISICAL_CLIENT_ID}\",\"clientSecret\":\"${INFISICAL_CLIENT_SECRET}\"}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['accessToken'])")
  export INFISICAL_TOKEN
  exec infisical run \
    --projectId "9c373776-768f-454b-a7b3-d1cc40deb475" \
    --env "${INFISICAL_ENV:-prod}" \
    --domain "https://secrets.drose.io" \
    -- "$@"
else
  # No Infisical creds — run directly (local dev, CI, etc.)
  exec "$@"
fi
