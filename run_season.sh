#!/usr/bin/env bash
# Convert a whole season to stereo 3D, one episode at a time. Safe to run
# intermittently: stop it (Ctrl+C, host reboot, whatever) and rerun this
# same command later - it picks up exactly where it left off.
#
# Usage:
#   ./run_season.sh [episodes_dir] [output_root]
# episodes_dir defaults to ./source_video (this repo's own convention) if
# omitted - override by passing a path explicitly for a different location.
#
# Any run_stereo.sh env var override (MAX_RES, CLASSICAL_ONLY, CHUNK_SIZE,
# etc.) works here too, since this just calls run_stereo.sh per episode:
#   CLASSICAL_ONLY=True ./run_season.sh source_video/mysea son
#
# Searches <episodes_dir> RECURSIVELY, so multiple seasons/shows can just be
# dumped in as subdirectories (e.g. source_video/toaru2/*.mkv,
# source_video/gintama_s2/*.mkv) and this finds all of them in one pass, in
# natural sort order over the full path - no need to tell it which season to
# work on, or to keep episode numbering unique across different shows (their
# subdirectory name disambiguates that below). Loose files directly in
# episodes_dir also still work fine.
#
# Each episode gets its own output dir under [output_root]/<path relative to
# episodes_dir, extension stripped> - e.g. source_video/toaru2/01.mkv ->
# outputs/toaru2/01/ - so results for different seasons land in their own
# subtree and can be fetched independently (output_root defaults to
# ./outputs).
#
# Resumability has two layers:
#   - Within an episode: run_stereo.sh's stages already checkpoint every
#     chunk to disk (see run_stereo.sh's own RESUME comments) - an episode
#     interrupted mid-run just re-invokes run_stereo.sh on the same input,
#     which resumes from the last completed chunk instead of restarting.
#   - Across episodes: a `.season_done` marker file is written into an
#     episode's output dir only after BOTH stages finish successfully.
#     Episodes with this marker are skipped entirely on the next run, so
#     a completed episode 1 is never redone just because you stopped
#     partway through episode 3.
#
# After an episode's marker is written, its stage-1 splat store (the large
# intermediate data - can be hundreds of GB for a full episode, see the
# disk-exhaustion incident in project history) is deleted before moving to
# the next episode, so disk usage doesn't accumulate across the season.
#
# A failure (after run_stereo.sh's own internal retries are exhausted)
# stops the whole season script rather than skipping ahead - rerun once
# you've fixed whatever broke; the failed episode's marker was never
# written, so it'll be retried (resuming from its own last checkpoint).
#
# Cumulative wall-clock time spent actually running this script (summed
# across every intermittent invocation) is tracked in
# [output_root]/.season_elapsed_seconds and printed at start/exit - this
# is total time spent, not an ETA to finish the season.
#
# Activates its own conda env and sources a VRAM preset on startup, so this
# can be invoked directly with no prior shell setup - e.g.
# `sudo -u someuser ./run_season.sh`. Override with CONDA_ENV_NAME (default
# "stereocrafter") and PRESET (default "24GB-7900xtx-rocm.env" - this box's
# own card; a filename under presets/) if this ever runs elsewhere.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPISODES_DIR="${1:-"$REPO_DIR/source_video"}"
if [[ ! -d "$EPISODES_DIR" ]]; then
    echo "Episodes dir not found: $EPISODES_DIR" >&2
    exit 1
fi
EPISODES_DIR="$(cd "$EPISODES_DIR" && pwd)"

OUTPUT_ROOT="${2:-"$REPO_DIR/outputs"}"
mkdir -p "$OUTPUT_ROOT"

# ---- conda env activation - a non-interactive `sudo -u user` shell never
# sources ~/.bashrc, so `conda` isn't even on PATH without this. Checks the
# common install locations directly rather than assuming `conda` is already
# callable. ----
CONDA_ENV_NAME="${CONDA_ENV_NAME:-stereocrafter}"
CONDA_SH=""
for candidate_base in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/conda"; do
    if [[ -f "$candidate_base/etc/profile.d/conda.sh" ]]; then
        CONDA_SH="$candidate_base/etc/profile.d/conda.sh"
        break
    fi
done
if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
    echo "Could not find conda (checked \$HOME/miniconda3, \$HOME/anaconda3, /opt/miniconda3, /opt/conda, and PATH) - activate the '$CONDA_ENV_NAME' env yourself first, or fix the search paths in run_season.sh." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_SH"
# conda's own activate.d hooks aren't written to be safe under `set -u`
# (reference backup vars like CONDA_BACKUP_CXX only set in some code paths) -
# same issue/fix as setup_rental_host.sh's identical note. Nounset stays off
# for the rest of the script.
set +u
conda activate "$CONDA_ENV_NAME"

# ---- VRAM preset - defaults to this box's own card (7900 XTX/ROCm);
# override with e.g. PRESET=24GB.env for a different machine/card.
PRESET="${PRESET:-24GB-7900xtx-rocm.env}"
if [[ ! -f "$REPO_DIR/presets/$PRESET" ]]; then
    echo "Preset not found: $REPO_DIR/presets/$PRESET" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$REPO_DIR/presets/$PRESET"

format_duration() {
    local total=$1
    printf '%dh %dm %ds' $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
}

# ---- optional Telegram notifications - completely silent no-op unless
# telegram.env exists (gitignored, never committed - see
# telegram.env.example for the format). Never blocks/fails the actual
# season run: capped retries, and a notification failure never exits the
# script (that would be a genuinely bad trade - losing real GPU work over
# a chat message not sending).
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

# Cumulative wall-clock time actually spent in this script, across every
# intermittent invocation (stop/resume, crashes, reboots) - a single
# counter for the whole season, not per-episode. Persisted as a plain
# integer-seconds file so a crash mid-episode still counts the time spent
# up to that point rather than losing it.
TIME_FILE="$OUTPUT_ROOT/.season_elapsed_seconds"
PRIOR_ELAPSED=0
[[ -f "$TIME_FILE" ]] && PRIOR_ELAPSED="$(cat "$TIME_FILE")"
RUN_START="$(date +%s)"

save_elapsed() {
    local this_run total
    this_run=$(($(date +%s) - RUN_START))
    total=$((PRIOR_ELAPSED + this_run))
    echo "$total" > "$TIME_FILE"
    echo
    echo "==> This session: $(format_duration "$this_run") | Total season time so far: $(format_duration "$total")"
    notify_telegram "Season script stopped.
This session: $(format_duration "$this_run")
Total season time so far: $(format_duration "$total")"
}
trap save_elapsed EXIT

echo "==> Total time already spent on this season: $(format_duration "$PRIOR_ELAPSED")"
notify_telegram "Season script started.
Total time already spent on this season: $(format_duration "$PRIOR_ELAPSED")"

mapfile -t EPISODES < <(find "$EPISODES_DIR" -type f \
    \( -iname '*.mkv' -o -iname '*.mp4' -o -iname '*.avi' -o -iname '*.webm' \) \
    | sort -V)

if [[ ${#EPISODES[@]} -eq 0 ]]; then
    echo "No video files found in $EPISODES_DIR" >&2
    exit 1
fi

echo "==> Found ${#EPISODES[@]} episode(s) in $EPISODES_DIR:"
printf '    %s\n' "${EPISODES[@]}"

DONE_COUNT=0
for EP in "${EPISODES[@]}"; do
    EP_REL="${EP#"$EPISODES_DIR"/}"  # keeps subdirectory structure (season name) intact
    EP_NAME="${EP_REL%.*}"
    EP_OUTPUT_DIR="$OUTPUT_ROOT/$EP_NAME"
    DONE_MARKER="$EP_OUTPUT_DIR/.season_done"

    if [[ -f "$DONE_MARKER" ]]; then
        echo "==> [$EP_NAME] already converted - skipping"
        DONE_COUNT=$((DONE_COUNT + 1))
        continue
    fi

    echo
    echo "###################################################################"
    echo "### Episode: $EP_NAME"
    echo "###################################################################"

    EP_START="$(date +%s)"
    "$REPO_DIR/run_stereo.sh" "$EP" "$EP_OUTPUT_DIR"
    EP_ELAPSED=$(($(date +%s) - EP_START))

    touch "$DONE_MARKER"
    echo "==> [$EP_NAME] stage 1+2 complete in $(format_duration "$EP_ELAPSED") - removing splat store to free disk"
    notify_telegram "Episode done: $EP_NAME
Took $(format_duration "$EP_ELAPSED")."
    rm -rf "$EP_OUTPUT_DIR/splat"
    DONE_COUNT=$((DONE_COUNT + 1))
done

echo
echo "==> Season complete: $DONE_COUNT/${#EPISODES[@]} episode(s) done."
notify_telegram "Season complete: $DONE_COUNT/${#EPISODES[@]} episode(s) done."
