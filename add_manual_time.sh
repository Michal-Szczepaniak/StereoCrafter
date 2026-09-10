#!/usr/bin/env bash
# Track manually-spent time and add it to run_season.sh's cumulative counter
# ([output_root]/.season_elapsed_seconds) when stopped - for time spent
# babysitting/running things by hand instead of through run_season.sh.
#
# Usage:
#   ./add_manual_time.sh [output_root]
# Start it right before you begin the manual work, leave it running, then
# Ctrl+C (or `kill`) it when you're done - it adds the elapsed wall-clock
# time to the season total on exit and prints the new total. output_root
# defaults to ./outputs, matching run_season.sh's own default - pass the
# same value you use with run_season.sh (or its second argument) if you
# customized it.
#
# Sends the same optional Telegram notifications run_season.sh does (start
# and stop), if telegram.env exists - see the block below.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${1:-"$REPO_DIR/outputs"}"
mkdir -p "$OUTPUT_ROOT"

format_duration() {
    local total=$1
    printf '%dh %dm %ds' $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
}

# Same flat guess as run_season.sh - keep COST_PER_HOUR_PLN in sync if you
# override it there.
COST_PER_HOUR_PLN="${COST_PER_HOUR_PLN:-0.75}"
format_cost() {
    local total=$1
    awk -v s="$total" -v rate="$COST_PER_HOUR_PLN" 'BEGIN{printf "%.2f zł", s/3600*rate}'
}

# ---- optional Telegram notifications - completely silent no-op unless
# telegram.env exists (gitignored, never committed - see
# telegram.env.example for the format). Same block as run_season.sh's, so
# manual sessions show up in the same chat as scripted ones. Never
# blocks/fails the timer: capped retries, and a notification failure never
# exits the script.
if [[ -f "$REPO_DIR/telegram.env" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/telegram.env"
fi

notify_telegram() {
    [[ -z "${TELEGRAM_CHAT_ID:-}" || -z "${TELEGRAM_API_KEY:-}" ]] && return 0
    local message attempt
    message="$(printf '%s' "$1" | sed '1s/\(.*\)/*\1*/' | sed 's/_/\\_/g')"
    for attempt in 1 2 3 4 5; do
        if curl -s -G \
            --data-urlencode "parse_mode=Markdown" \
            --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
            --data-urlencode "text=${message}" \
            "https://api.telegram.org/bot${TELEGRAM_API_KEY}/sendMessage" >/dev/null 2>&1; then
            return 0
        fi
        sleep 10
    done
    echo "==> Telegram notification failed after 5 attempts - continuing anyway" >&2
    return 0
}

TIME_FILE="$OUTPUT_ROOT/.season_elapsed_seconds"
PRIOR_ELAPSED=0
[[ -f "$TIME_FILE" ]] && PRIOR_ELAPSED="$(cat "$TIME_FILE")"
START="$(date +%s)"

on_exit() {
    local end elapsed prior new_total
    end="$(date +%s)"
    elapsed=$((end - START))
    prior=0
    [[ -f "$TIME_FILE" ]] && prior="$(cat "$TIME_FILE")"
    new_total=$((prior + elapsed))
    echo "$new_total" > "$TIME_FILE"
    echo
    echo "==> Manual timer stopped after $(format_duration "$elapsed") (~$(format_cost "$elapsed"))"
    echo "==> Total season time so far: $(format_duration "$new_total") (~$(format_cost "$new_total"))"
    notify_telegram "Manual timer stopped.
This session: $(format_duration "$elapsed") (~$(format_cost "$elapsed"))
Total season time so far: $(format_duration "$new_total") (~$(format_cost "$new_total"))"
}
trap on_exit EXIT

echo "==> Manual timer started - press Ctrl+C (or kill this process) when done to record elapsed time"
notify_telegram "Manual timer started.
Total time already spent on this season: $(format_duration "$PRIOR_ELAPSED") (~$(format_cost "$PRIOR_ELAPSED"))"
while true; do sleep 3600; done
