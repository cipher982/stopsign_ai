#!/usr/bin/env bash
# Post-deploy verification for raw tracking payloads.
# Run via: hatch "cd ~/git/stopsign_ai && bash scripts/post_deploy_check.sh"
# Or schedule: echo 'bash ~/git/stopsign_ai/scripts/post_deploy_check.sh' | at now + 2 hours
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Raw Payload Verification ($(date)) ==="

uv run python scripts/verify_raw_payloads.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Raw payloads are being captured correctly."
    echo ""
    # Grab a pass ID that has raw data and show its detail page URL
    PASS_ID=$(uv run python -c "
import psycopg2, os
conn = psycopg2.connect(os.environ.get('DB_URL', 'postgresql://postgres:***REMOVED***@clifford.coin-castor.ts.net:5432/stopsign'))
cur = conn.cursor()
cur.execute('SELECT vehicle_pass_id FROM vehicle_pass_raw ORDER BY created_at DESC LIMIT 1')
print(cur.fetchone()[0])
conn.close()
")
    echo "Verify detail page: https://crestwoodstopsign.com/passes/${PASS_ID}"
    echo "Expected: capture image, facts table, AND Raw Data section with sample count + model name"
elif [ $EXIT_CODE -eq 2 ]; then
    echo ""
    echo "NO DATA YET. The analyzer hasn't recorded a pass since deployment."
    echo "This is normal if it's nighttime or low traffic. Retry later."
elif [ $EXIT_CODE -eq 1 ]; then
    echo ""
    echo "FAILURES DETECTED. See above for details."
fi

exit $EXIT_CODE
