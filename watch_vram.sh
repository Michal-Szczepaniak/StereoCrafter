#!/usr/bin/env bash
# Logs GPU memory usage over time and tracks the running peak - for
# watching an unattended run (e.g. overnight) to see whether VRAM is
# creeping toward the card's ceiling, without babysitting nvidia-smi live.
# Appends timestamped samples to a log file so you can check back later
# (or just grep the peak) instead of needing to watch it in real time.
#
# Usage:
#   ./watch_vram.sh [interval_seconds] [logfile]
# Defaults: 0.1s (100ms) interval, ./vram_watch.log - note each nvidia-smi
# call itself takes some tens of ms, so actual cadence runs a bit slower
# than the nominal interval, and at this rate the CSV grows fast (~10
# lines/sec - hours of unattended logging means a genuinely large file,
# worth trimming/rotating or lowering the rate once you've confirmed VRAM
# is flat and just want a low-overhead background check).
#
# Run it detached so it survives you disconnecting (tmux/screen, or):
#   nohup ./watch_vram.sh 10 vram_watch.log > /dev/null 2>&1 &
# Check on it later:
#   tail -f vram_watch.log                        # live-ish view
#   sort -t, -k3 -n vram_watch.log | tail -1       # highest peak ever logged
#   grep WARNING vram_watch.log                     # did it ever cross the threshold
set -euo pipefail

INTERVAL="${1:-0.1}"
LOGFILE="${2:-vram_watch.log}"
WARN_PCT="${WARN_PCT:-95}"   # log a WARNING line once usage crosses this % of total

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }

TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
PEAK=0
WARNED=false

echo "Watching GPU memory every ${INTERVAL}s, logging to $LOGFILE (card total: ${TOTAL_MIB}MiB, warn at ${WARN_PCT}%). Ctrl+C to stop."
echo "timestamp,used_mib,peak_mib,total_mib,pct_of_total" > "$LOGFILE"

while true; do
    USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    if [ "$USED" -gt "$PEAK" ]; then
        PEAK="$USED"
    fi
    PCT="$(awk -v u="$USED" -v t="$TOTAL_MIB" 'BEGIN{printf "%.1f", u/t*100}')"
    TS="$(date -Iseconds)"
    echo "$TS,$USED,$PEAK,$TOTAL_MIB,$PCT" >> "$LOGFILE"

    OVER_THRESHOLD="$(awk -v p="$PCT" -v w="$WARN_PCT" 'BEGIN{print (p>=w)?1:0}')"
    if [ "$OVER_THRESHOLD" = "1" ] && [ "$WARNED" = "false" ]; then
        echo "$TS,WARNING: usage hit ${PCT}% of ${TOTAL_MIB}MiB (>= ${WARN_PCT}% threshold)" >> "$LOGFILE"
        WARNED=true
    elif [ "$OVER_THRESHOLD" = "0" ]; then
        WARNED=false   # re-arm - warn again if it climbs back over after dropping
    fi

    printf "\r%s  used=%sMiB  peak=%sMiB  (%s%% of %sMiB)   " "$TS" "$USED" "$PEAK" "$PCT" "$TOTAL_MIB"
    sleep "$INTERVAL"
done
