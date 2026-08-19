#!/usr/bin/env bash
# Prints live GPU memory usage + running peak to the terminal. Nothing
# written to disk - just watch it and Ctrl+C when done.
#
# Usage:
#   ./watch_vram.sh [interval_seconds]
# Default: 0.1s (100ms) - each nvidia-smi call itself takes some tens of
# ms, so actual cadence runs a bit slower than the nominal interval.
set -euo pipefail

INTERVAL="${1:-0.1}"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }

TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
PEAK=0

while true; do
    USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    if [ "$USED" -gt "$PEAK" ]; then
        PEAK="$USED"
    fi
    PCT="$(awk -v u="$USED" -v t="$TOTAL_MIB" 'BEGIN{printf "%.1f", u/t*100}')"
    printf "\rused=%sMiB  peak=%sMiB  (%s%% of %sMiB)   " "$USED" "$PEAK" "$PCT" "$TOTAL_MIB"
    sleep "$INTERVAL"
done
