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

TIME_FILE="$OUTPUT_ROOT/.season_elapsed_seconds"
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
}
trap on_exit EXIT

echo "==> Manual timer started - press Ctrl+C (or kill this process) when done to record elapsed time"
while true; do sleep 3600; done
