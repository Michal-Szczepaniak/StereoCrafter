#!/usr/bin/env bash
# Polls vast.ai's marketplace for offers matching your filters and fires a
# desktop notification + terminal bell when a new one shows up. By default
# does NOT book anything - vast.ai's own inventory changes fast enough
# (renters coming and going) that auto-booking on a script's first match is
# a good way to end up with a worse box than you'd have picked by eye; this
# just tells you to go look.
#
# Set AUTO_BOOK=true to have it book automatically instead. It books AT
# MOST ONE instance ever, per script run: the moment a create succeeds, the
# script stops watching entirely (no risk of looping back around and
# booking a second one on the next poll).
#
# On "buying without auto-starting": vast.ai's create-instance API has no
# flag to create directly in a stopped state (`--cancel-unavail` only
# affects what happens if scheduling itself fails, not normal success) - an
# instance starts running, and billing for active GPU rental starts, the
# moment create succeeds. The closest available thing is create then
# immediately stop (STOP_AFTER_BOOK=true, the default when AUTO_BOOK is
# on) - this limits the running/billed window to the few seconds between
# create and stop, then only storage billing continues while stopped
# (confirmed via vast.ai's own billing docs: active rental billing pauses
# on stop, storage billing continues until the instance is destroyed).
# CAVEAT, also from vast.ai's own docs: "Once stopped, starting an instance
# is subject to resource availability on the machine that the instance is
# located on" - stopping does NOT reserve the GPU for you. If someone else
# takes that machine's capacity before you resume, starting it back up can
# fail. Booking a cheap/popular offer and sitting on it stopped for a long
# time is a real risk, not just a theoretical one - if you want to actually
# start working immediately, set STOP_AFTER_BOOK=false instead.
#
# Requires:
#   - vastai CLI: pip install vastai && vastai set api-key <your key>
#   - jq (JSON parsing)
#   - notify-send (libnotify - standard on KDE Plasma)
#
# NOTE: the exact --raw JSON field names below (dph_total, reliability2,
# storage_cost, new_contract) are from vast.ai's documented offer/create
# schema, but this hasn't been run against a live account from this
# environment - if a field comes back empty/null, check the real names
# with:
#   vastai search offers --help
#   vastai create instance --help
# and adjust the jq paths below.
#
# Usage:
#   ./watch_vastai.sh                    # notify only
#   AUTO_BOOK=true ./watch_vastai.sh      # notify + book the first match, then stop it, then exit
# Tune via env vars, e.g.:
#   MAX_DPH=0.35 DISK_GB=400 AUTO_BOOK=true ./watch_vastai.sh
set -euo pipefail

GPU_NAME="${GPU_NAME:-RTX_4090}"
MAX_DPH="${MAX_DPH:-0.30}"            # max $/hr COMPUTE only (dph_total excludes storage)
MIN_RELIABILITY="${MIN_RELIABILITY:-0.98}"
DISK_GB="${DISK_GB:-120}"             # used both for the storage estimate AND as --disk on create
POLL_SECONDS="${POLL_SECONDS:-300}"   # 5 min - vast.ai inventory doesn't churn much faster than this
AUTO_BOOK="${AUTO_BOOK:-false}"
STOP_AFTER_BOOK="${STOP_AFTER_BOOK:-true}"   # see caveat above before disabling
IMAGE="${IMAGE:-pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel}"   # verify this matches what setup_rental_host.sh expects before relying on it
NET_COST="${NET_COST:-0.005}"

for bin in vastai jq notify-send; do
    command -v "$bin" >/dev/null || { echo "Missing required command: $bin" >&2; exit 1; }
done

SEEN_FILE="$(mktemp)"
trap 'rm -f "$SEEN_FILE"' EXIT

echo "Watching vast.ai: gpu=$GPU_NAME dph<=\$${MAX_DPH} reliability>=$MIN_RELIABILITY (poll every ${POLL_SECONDS}s, storage estimate at ${DISK_GB}GB)"
echo "Ctrl+C to stop."

while true; do
    offers_json="$(vastai search offers --storage ${DISK_GB} -o dph_total \
        "gpu_name=$GPU_NAME reliability>=$MIN_RELIABILITY dph_total<=$MAX_DPH rentable=true duration>=7 inet_down_cost<$NET_COST" \
        --raw 2>/dev/null || echo "[]")"

    while IFS= read -r offer; do
        [ -z "$offer" ] && continue

        id="$(jq -r '.id' <<<"$offer")"
        dph="$(jq -r '.dph_total' <<<"$offer")"
        storage_cost="$(jq -r '.storage_total_cost // 0' <<<"$offer")"
        reliability="$(jq -r '.reliability2 // .reliability // "?"' <<<"$offer")"
        country="$(jq -r '.geolocation' <<<"$offer")"


        if ! grep -qx "$id" "$SEEN_FILE" 2>/dev/null; then
            echo "$id" >> "$SEEN_FILE"
            msg="offer $id: \$${dph}/hr, reliability ${reliability}, storage ${storage_cost}/hr, country: ${country}"
            echo "==> $msg"

            if [ "$AUTO_BOOK" = "true" ]; then
                echo "==> AUTO_BOOK is on - booking offer $id..."
                create_json="$(vastai create instance "$id" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --raw 2>&1)" || {
                    echo "==> booking FAILED (offer likely gone - someone else grabbed it): $create_json" >&2
                    notify-send "vast.ai booking failed" "offer $id: $create_json" || true
                    continue
                }
                new_id="$(jq -r '.new_contract // .instance_id // empty' <<<"$create_json" 2>/dev/null)"
                echo "==> booked: $create_json"

                if [ "$STOP_AFTER_BOOK" = "true" ] && [ -n "$new_id" ]; then
                    echo "==> STOP_AFTER_BOOK is on - stopping instance $new_id to halt compute billing..."
                    sleep 5   # give the instance a moment to register as running before stopping it
                    vastai stop instance "$new_id" || echo "==> WARNING: stop failed - instance $new_id may still be running and billing, check manually" >&2
                fi

                notify-send "vast.ai: booked!" "$msg (instance ${new_id:-unknown})" || true
                printf '\a\a\a'
                echo "==> Booked one instance, exiting (AUTO_BOOK only ever books one per run)."
                exit 0
            else
                notify-send "vast.ai match found" "$msg" || true
                printf '\a'
            fi
        fi
    done < <(jq -c '.[]' <<<"$offers_json")

    sleep "$POLL_SECONDS"
done
