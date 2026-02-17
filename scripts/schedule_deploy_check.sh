#!/usr/bin/env bash
# Schedule post-deploy raw payload checks every 30 min for 24 hours.
# Self-removes after 24 hours or on first success.
set -euo pipefail

LOG_DIR=~/git/stopsign_ai/scripts/deploy_checks
mkdir -p "$LOG_DIR"

EXPIRE_AT=$(($(date +%s) + 86400))  # 24 hours from now
echo "$EXPIRE_AT" > "$LOG_DIR/.expire"

cat > "$LOG_DIR/run_check.sh" << 'INNER'
#!/usr/bin/env bash
set -euo pipefail
LOG_DIR=~/git/stopsign_ai/scripts/deploy_checks
EXPIRE_AT=$(cat "$LOG_DIR/.expire" 2>/dev/null || echo 0)

# Self-destruct after expiry
if [ "$(date +%s)" -gt "$EXPIRE_AT" ]; then
    echo "$(date): Expired, removing scheduled check" >> "$LOG_DIR/history.log"
    crontab -l 2>/dev/null | grep -v 'deploy_checks/run_check' | crontab -
    exit 0
fi

cd ~/git/stopsign_ai
OUTPUT=$(uv run python scripts/verify_raw_payloads.py 2>&1)
CODE=$?
echo "$(date): exit=$CODE" >> "$LOG_DIR/history.log"
echo "$OUTPUT" >> "$LOG_DIR/history.log"
echo "---" >> "$LOG_DIR/history.log"

# On success, remove cron and leave a final marker
if [ $CODE -eq 0 ]; then
    echo "$(date): SUCCESS - removing scheduled check" >> "$LOG_DIR/history.log"
    echo "$OUTPUT" > "$LOG_DIR/SUCCESS"
    crontab -l 2>/dev/null | grep -v 'deploy_checks/run_check' | crontab -
fi
INNER
chmod +x "$LOG_DIR/run_check.sh"

# Add to crontab (every 30 min)
(crontab -l 2>/dev/null | grep -v 'deploy_checks/run_check'; echo "*/30 * * * * $LOG_DIR/run_check.sh") | crontab -

echo "Scheduled: raw payload check every 30 min for 24 hours"
echo "Log: $LOG_DIR/history.log"
echo "Success marker: $LOG_DIR/SUCCESS (created on first pass)"
echo "Self-removes from crontab on success or after 24 hours."
