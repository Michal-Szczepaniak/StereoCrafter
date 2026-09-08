#!/usr/bin/env bash
# Add manually-spent time to run_season.sh's cumulative time counter
# ([output_root]/.season_elapsed_seconds) - for when you did something by
# hand (a manual run_stereo.sh invocation, babysitting a crash, re-running a
# stage yourself) instead of letting run_season.sh track it automatically.
#
# Usage:
#   ./add_manual_time.sh <duration> [output_root]
# duration accepts plain seconds (e.g. 5400) or a short Xh/Xm/Xs form (any
# subset/order, e.g. 2h15m, 45m, 1h30m10s). output_root defaults to
# ./outputs, matching run_season.sh's own default - pass the same value you
# use with run_season.sh (or its second argument) if you customized it.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <duration> [output_root]" >&2
    echo "  duration: plain seconds, or Xh/Xm/Xs (e.g. 2h15m, 45m, 90)" >&2
    exit 1
fi

DURATION_ARG="$1"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${2:-"$REPO_DIR/outputs"}"
mkdir -p "$OUTPUT_ROOT"

parse_duration() {
    local input="$1"
    if [[ "$input" =~ ^[0-9]+$ ]]; then
        echo "$input"
        return
    fi
    if ! [[ "$input" =~ ^([0-9]+h)?([0-9]+m)?([0-9]+s)?$ ]] || [[ -z "${BASH_REMATCH[0]}" ]]; then
        echo "Could not parse duration '$input' - use plain seconds or Xh/Xm/Xs (e.g. 2h15m)" >&2
        exit 1
    fi
    local h=0 m=0 s=0
    [[ "$input" =~ ([0-9]+)h ]] && h="${BASH_REMATCH[1]}"
    [[ "$input" =~ ([0-9]+)m ]] && m="${BASH_REMATCH[1]}"
    [[ "$input" =~ ([0-9]+)s ]] && s="${BASH_REMATCH[1]}"
    echo $((h * 3600 + m * 60 + s))
}

ADD_SECONDS="$(parse_duration "$DURATION_ARG")"

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
PRIOR_ELAPSED=0
[[ -f "$TIME_FILE" ]] && PRIOR_ELAPSED="$(cat "$TIME_FILE")"
NEW_TOTAL=$((PRIOR_ELAPSED + ADD_SECONDS))
echo "$NEW_TOTAL" > "$TIME_FILE"

echo "==> Added $(format_duration "$ADD_SECONDS") (~$(format_cost "$ADD_SECONDS")) of manual time"
echo "==> Total season time so far: $(format_duration "$NEW_TOTAL") (~$(format_cost "$NEW_TOTAL"))"
