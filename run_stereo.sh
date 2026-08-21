#!/usr/bin/env bash
# End-to-end 2D -> stereo SBS pipeline: depth splatting, then inpainting,
# then combine into a side-by-side 3D mp4 (tagged so players like Kodi
# auto-detect it as full SBS 3D).
#
# Usage:
#   ./run_stereo.sh <input_video> [output_dir]
#
# Defaults below are the settings validated to actually fit this machine's
# 12GB GPU (see conversation history: tile_num<4, cpu_offload=False, and
# decode_chunk_size>1 all OOM at 1080p). Override via environment variables,
# e.g.:
#   NUM_INFERENCE_STEPS=4 ./run_stereo.sh input.mp4
#
# For other card sizes, source a preset first (see presets/*.env for what
# each one assumes and how confident/tested it is - the 12GB and 24GB ones
# are measured, 16GB is an extrapolation):
#   source presets/24GB.env && ./run_stereo.sh input.mp4
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input_video> [output_dir]" >&2
    exit 1
fi

INPUT_VIDEO="$1"
if [[ ! -f "$INPUT_VIDEO" ]]; then
    echo "Input video not found: $INPUT_VIDEO" >&2
    exit 1
fi
INPUT_VIDEO="$(cd "$(dirname "$INPUT_VIDEO")" && pwd)/$(basename "$INPUT_VIDEO")"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VIDEO_NAME="$(basename "${INPUT_VIDEO%.*}")"
OUTPUT_DIR="${2:-"$REPO_DIR/outputs/$VIDEO_NAME"}"
SPLAT_DIR="$OUTPUT_DIR/splat"
mkdir -p "$OUTPUT_DIR"

# ---- weights ----
SVD_WEIGHTS="${SVD_WEIGHTS:-./weights/stable-video-diffusion-img2vid-xt-1-1}"
DEPTHCRAFTER_UNET="${DEPTHCRAFTER_UNET:-./weights/DepthCrafter}"
STEREOCRAFTER_UNET="${STEREOCRAFTER_UNET:-./weights/StereoCrafter}"

# ---- stage 1 (depth splatting) knobs ----
MAX_RES="${MAX_RES:-384}"
MAX_DISP="${MAX_DISP:-20.0}"
PROCESS_LENGTH="${PROCESS_LENGTH:--1}"  # -1 = full video; set lower for a quick smoke test
# CHUNK_OVERLAP/--chunk_overlap is accepted for CLI compatibility but ignored
# by the script itself now - cross-chunk continuity uses window_overlap for
# both the leading-context re-feed and a latent-space carry-over (see
# depth_splatting_inference.py's own docstrings). Kept only so old invocations
# don't break.
CHUNK_OVERLAP="${CHUNK_OVERLAP:-25}"
# How many frames per outer DepthCrafter call - bigger uses more VRAM (see
# presets/*.env) for marginally fewer chunk-boundary crossfades. Must stay
# greater than WINDOW_SIZE. WINDOW_SIZE/WINDOW_OVERLAP control DepthCrafter's
# own internal sliding-window inference - validated at 70/25 for the
# consistency fix, only change these if you know what you're doing.
CHUNK_SIZE="${CHUNK_SIZE:-110}"
WINDOW_SIZE="${WINDOW_SIZE:-70}"
WINDOW_OVERLAP="${WINDOW_OVERLAP:-25}"
# None = all model components resident on GPU (faster, more VRAM); "model" =
# cpu_offload trades speed for VRAM headroom. Safe to leave on None even on
# a tight card since stage 1 checkpoints every chunk (see RESUME below) - an
# OOM only loses the in-flight chunk.
CPU_OFFLOAD="${CPU_OFFLOAD:-None}"
# FFV1-compressed splat store instead of raw .npy - MEASURED ~6.9x smaller
# on a real 240-frame local store (1920x1080, anime source), verified
# bit-exact round-trip (see splat_store.py's module docstring for the full
# writeup and why plain H.264 was rejected). Decode cost is negligible next
# to stage 2's per-iteration diffusion cost. Set False to fall back to the
# original uncompressed format.
COMPRESS_STORE="${COMPRESS_STORE:-True}"

# ---- stage 2 (inpainting) knobs - these fit a 12GB card at 1080p; going
# below tile_num=4, or raising decode_latents_chunk_size above 1, OOM at
# full resolution (work_scale=1.0) on this card (verified). num_inference_steps
# is a real speed/quality tradeoff; work_scale is the big one - see
# inpainting_inference.py's own docstring for the full writeup of both.
TILE_NUM="${TILE_NUM:-4}"
FRAMES_CHUNK="${FRAMES_CHUNK:-5}"
OVERLAP="${OVERLAP:-3}"
# CFG (min/max_guidance_scale) killed by default below - see
# inpainting_inference.py's do_classifier_free_guidance fix. A guidance
# scale of 1.0 disables the second (unconditional) UNet forward entirely,
# ~2x by itself, for a correction term that was only ever weighted ~0.01
# against an unconditional branch built from zeroed inputs.
MIN_GUIDANCE_SCALE="${MIN_GUIDANCE_SCALE:-1.0}"
MAX_GUIDANCE_SCALE="${MAX_GUIDANCE_SCALE:-1.0}"
# Was unconditionally True regardless of card size - PROFILED this session
# (real cProfile run on the local 12GB card): the offload hooks' per-call
# CPU<->GPU reshuffling cost ~15%+ of stage-2 wall time (chained
# image_encoder->unet->vae, each stage onloaded/offloaded on every single
# tile pass, not once per run). CFG-off (above) roughly halves UNet
# activation memory, so this now defaults to False even on the 12GB card -
# override to True if a specific config still needs the VRAM headroom.
ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-False}"
DECODE_LATENTS_CHUNK_SIZE="${DECODE_LATENTS_CHUNK_SIZE:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
# work_scale: run the diffusion model at a fraction of the real output
# resolution, then upsample the result back before compositing against the
# full-resolution warp/mask - see inpainting_inference.py's docstring.
# 1.0 (unchanged/original behavior) here; presets/*.env override this once
# it's been benchmarked (--bench_iters) against that card's real footage.
WORK_SCALE="${WORK_SCALE:-1.0}"
DENOISE_STRENGTH="${DENOISE_STRENGTH:-1.0}"
MASK_SKIP_THRESHOLD="${MASK_SKIP_THRESHOLD:-}"
AGGRESSIVE_FREE="${AGGRESSIVE_FREE:-False}"
VAE_FORCE_UPCAST="${VAE_FORCE_UPCAST:-False}"
COMPILE_UNET="${COMPILE_UNET:-False}"
MAX_ITERS="${MAX_ITERS:-}"
# chunked_attention.py: memory-efficient attention fallback for GPUs with no
# working flash/efficient SDPA kernel (see chunked_attention.py's own module
# docstring - confirmed via this project's dev GPU, an AMD RX 6700 XT, that
# ROCm has no working kernel there at all). It always tries the real kernel
# first and only falls back to a slow manual chunked implementation if that
# genuinely fails, so this is safe to leave on by default - measured no
# meaningful slowdown at TILE_NUM=4 (the recommended default), only helps if
# you lower TILE_NUM on hardware that needs it. Set to False to force the
# old (crash-prone at low TILE_NUM on this hardware) behavior, e.g. to A/B
# it yourself.
CHUNKED_ATTENTION="${CHUNKED_ATTENTION:-True}"
ATTENTION_KV_CHUNK_SIZE="${ATTENTION_KV_CHUNK_SIZE:-1024}"
VAE_ENCODE_CHUNK_SIZE="${VAE_ENCODE_CHUNK_SIZE:-5}"
# chunked_attention's fast-kernel probe logs a UserWarning per unavailable
# backend every time it fails (constant on hardware with no working kernel)
# - suppressed by default since it spams the live progress line. Set False
# to see them again.
SUPPRESS_ATTENTION_KERNEL_WARNINGS="${SUPPRESS_ATTENTION_KERNEL_WARNINGS:-True}"

# ---- resumability: both stages checkpoint every chunk to disk (depth
# chunks + carry_latents / .mkv segments + generated tail, respectively -
# see each script's own comments), so a crash mid-run only loses the
# in-flight chunk. Stage 1 now defaults cpu_offload=None (faster, ~2.5-3GB
# more VRAM) specifically because this makes that safe to leave on - retry
# below just re-invokes the same command, which resumes automatically.
RESUME="${RESUME:-True}"
MAX_RETRIES="${MAX_RETRIES:-3}"

run_with_retries() {
    local desc="$1"; shift
    local attempt=1
    while true; do
        if "$@"; then
            return 0
        fi
        if [[ $attempt -ge $MAX_RETRIES ]]; then
            echo "==> $desc: failed after $attempt attempt(s), giving up." >&2
            return 1
        fi
        echo "==> $desc: attempt $attempt failed - retrying (resumes from the last checkpoint)..." >&2
        attempt=$((attempt + 1))
    done
}

stage1_run() {
    python depth_splatting_inference.py \
        --input_video_path "$INPUT_VIDEO" \
        --output_dir "$SPLAT_DIR" \
        --unet_path "$DEPTHCRAFTER_UNET" \
        --pre_trained_path "$SVD_WEIGHTS" \
        --max_res="$MAX_RES" \
        --max_disp="$MAX_DISP" \
        --process_length="$PROCESS_LENGTH" \
        --chunk_overlap="$CHUNK_OVERLAP" \
        --chunk_size="$CHUNK_SIZE" \
        --window_size="$WINDOW_SIZE" \
        --window_overlap="$WINDOW_OVERLAP" \
        --cpu_offload="$CPU_OFFLOAD" \
        --resume="$RESUME" \
        --compress_store="$COMPRESS_STORE"
}

echo "==================================================================="
echo "Stage 1/3: depth splatting -> $SPLAT_DIR"
echo "==================================================================="
run_with_retries "Stage 1" stage1_run

echo
echo "==================================================================="
echo "Stage 2/3: inpainting -> $OUTPUT_DIR"
echo "==================================================================="
STAGE2_LOG="$OUTPUT_DIR/stage2.log"
MAX_ITERS_ARG=()
if [[ -n "$MAX_ITERS" ]]; then
    MAX_ITERS_ARG=(--max_iters "$MAX_ITERS")
fi

stage2_run() {
    MASK_SKIP_ARG=()
    if [[ -n "$MASK_SKIP_THRESHOLD" ]]; then
        MASK_SKIP_ARG=(--mask_skip_threshold "$MASK_SKIP_THRESHOLD")
    fi
    python inpainting_inference.py \
        --pre_trained_path "$SVD_WEIGHTS" \
        --unet_path "$STEREOCRAFTER_UNET" \
        --splat_store_dir "$SPLAT_DIR" \
        --save_dir "$OUTPUT_DIR" \
        --tile_num "$TILE_NUM" \
        --frames_chunk "$FRAMES_CHUNK" \
        --overlap "$OVERLAP" \
        --decode_latents_chunk_size "$DECODE_LATENTS_CHUNK_SIZE" \
        --enable_model_cpu_offload="$ENABLE_MODEL_CPU_OFFLOAD" \
        --num_inference_steps "$NUM_INFERENCE_STEPS" \
        --min_guidance_scale="$MIN_GUIDANCE_SCALE" \
        --max_guidance_scale="$MAX_GUIDANCE_SCALE" \
        --work_scale="$WORK_SCALE" \
        --denoise_strength="$DENOISE_STRENGTH" \
        --aggressive_free="$AGGRESSIVE_FREE" \
        --vae_force_upcast="$VAE_FORCE_UPCAST" \
        --compile_unet="$COMPILE_UNET" \
        "${MASK_SKIP_ARG[@]}" \
        --chunked_attention="$CHUNKED_ATTENTION" \
        --attention_kv_chunk_size="$ATTENTION_KV_CHUNK_SIZE" \
        --suppress_attention_kernel_warnings="$SUPPRESS_ATTENTION_KERNEL_WARNINGS" \
        --vae_encode_chunk_size="$VAE_ENCODE_CHUNK_SIZE" \
        --resume="$RESUME" \
        "${MAX_ITERS_ARG[@]}" \
        2>&1 | tee "$STAGE2_LOG"
}
run_with_retries "Stage 2" stage2_run

echo
echo "==================================================================="
echo "Stage 3/3: combining into side-by-side 3D"
echo "==================================================================="
# Reuse the exact ffmpeg command inpainting_inference.py prints - it
# already has the fps-normalization fix (avoids the alternating-eye freeze
# from two streams with subtly different timebases) and the setsar=2/1 tag
# (so DAR reports doubled - e.g. 32:9 -> 64:9 - which is what makes players
# like Kodi auto-recognize full SBS 3D) baked in, driven by this run's own
# meta.json fps rather than recomputed here.
SBS_CMD="$(grep -A1 'To combine into side-by-side 3D' "$STAGE2_LOG" | tail -1 | sed 's/^[[:space:]]*//')"
if [[ -z "$SBS_CMD" ]]; then
    echo "Could not find the SBS ffmpeg command in $STAGE2_LOG - inpainting may have failed." >&2
    exit 1
fi
echo "$SBS_CMD"
eval "$SBS_CMD"

# The SBS output path is whatever inpainting_inference.py derived it to
# be (based on splat_store_dir's basename) - pull it from the command
# itself instead of guessing, so this stays correct if that naming ever
# changes.
SBS_OUT="$(grep -oE '"[^"]+"$' <<<"$SBS_CMD" | tr -d '"')"
echo
echo "==================================================================="
echo "Done: $SBS_OUT"
ffmpeg -i "$SBS_OUT" 2>&1 | grep -i "stream.*video" || true
echo "==================================================================="
