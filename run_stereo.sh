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
# Signed-disparity tolerance window for the z-buffer-gated splat
# (ForwardWarpStereo._zbuffer_splat): a contribution blends in only if its
# disp is within this many units of the local winning (nearest) disp at
# that destination pixel - anything farther is hard-excluded instead of
# blended, fixing the cross-eye conflict at silhouette edges. Unmeasured
# default - bench-verify before trusting for an unattended run (see
# depth_splatting_inference.py's ForwardWarpStereo docstring).
DISP_TOLERANCE="${DISP_TOLERANCE:-1.0}"
# DepthCrafter's own VAE encode/decode batch size (unrelated to stage 2's
# DECODE_LATENTS_CHUNK_SIZE) - MEASURED this session on a real rental 4090:
# VAE encode+decode was 43% of total depth-pass time (20.7s+45.6s of 153.4s
# for a 240-frame clip) at the library's own internal default of 8, entirely
# untouched by attention_slicing/TF32/cudnn.benchmark (none of which affect
# VAE conv layers). Raising to 16 measured a real (if modest) win: encode
# -21%, decode -13%, ~6.5% off total depth-pass wall time. 24 and 32 both
# OOM'd on this card's existing VRAM budget (MAX_RES=768) - 16 is the
# measured ceiling here, re-verify before raising on a different
# MAX_RES/card.
STAGE1_DECODE_CHUNK_SIZE="${STAGE1_DECODE_CHUNK_SIZE:-16}"
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
# Hardcoded True forever (see DepthCrafterDemo.__init__) - only needed on
# hardware with no working flash/efficient-attention kernel (this project's
# ROCm dev card). On real CUDA hardware, diffusers' fused SDPA kernel is
# available for free and enable_attention_slicing() swaps it out for a slow
# manual-loop SlicedAttnProcessor for nothing - same shape of bug already
# fixed for stage 2 via CHUNKED_ATTENTION. Default True here (safe on any
# hardware); override to False in a CUDA preset once VRAM headroom is
# reverified with slicing off (see presets/24GB.env).
ATTENTION_SLICING="${ATTENTION_SLICING:-True}"
# FFV1-compressed splat store instead of raw .npy - MEASURED ~6.9x smaller
# on a real 240-frame local store (1920x1080, anime source), verified
# bit-exact round-trip (see splat_store.py's module docstring for the full
# writeup and why plain H.264 was rejected). Decode cost is negligible next
# to stage 2's per-iteration diffusion cost. Set False to fall back to the
# original uncompressed format.
COMPRESS_STORE="${COMPRESS_STORE:-True}"
# EXPANSION pass on the LOW-RES depth, before upsampling: grows the
# foreground classification so it still covers the real silhouette after
# upsampling to full res. Needed because the depth-derived foreground is
# often slightly SMALLER than the real character (the low-res grid can't
# resolve the true silhouette), which leaves the disocclusion hole
# stopping short and a sliver of character warped by BACKGROUND disparity
# - character pixels in the wrong place, which have to be repainted, so
# the mask has to cover them. Measured to be a ~1px-per-iteration
# dilation, so one iteration = one LOW-RES pixel (~2.5 full-res px at
# MAX_RES=768 on 1080p). Costs a halo, so use the smallest value that
# actually covers the mismatch. It does NOT sharpen - the ramp width is
# unchanged at every iteration count, so it has no effect on splat
# tearing; see SHARPEN_MODE for that. Content-dependent, re-tune per
# episode. 0 = off.
EDGE_THRESHOLD_FRAC="${EDGE_THRESHOLD_FRAC:-0.10}"
EDGE_FILL_ITERS="${EDGE_FILL_ITERS:-3}"

# Optional re-hardening of the depth edge AFTER upsampling to full res.
# "none" (default) leaves it alone - that plus EDGE_FILL_ITERS above is
# the pipeline's original behavior (modulo bilinear vs nearest upsample).
# "stretch" = _position_preserving_sharpen: hardens about the LOCAL
# plateau midpoint, so the boundary's sub-pixel position is left where it
# is. This is the only knob that affects splat TEARING (the shredded
# 1px-on/1px-off holes along silhouettes) - it drives the number of
# intermediate disparity levels toward zero, and tearing is one gap per
# level. SHARPEN_* only apply to "stretch": radius just needs to be >= the
# ramp width (insensitive past that); gain=3 sharpens at zero positional
# cost, while a very high gain is what actually collapses the tearing.
SHARPEN_MODE="${SHARPEN_MODE:-none}"
SHARPEN_RADIUS="${SHARPEN_RADIUS:-6}"
SHARPEN_GAIN="${SHARPEN_GAIN:-3.0}"
# Contrast gate for "stretch": leave any transition weaker than this
# fraction of the depth range completely alone. Without it, a high
# SHARPEN_GAIN hard-steps every mild depth change in the frame (folds,
# curved surfaces) and each one then produces its own small hole where the
# warp previously covered it fine - measured as a large, purely additive
# increase in mask area. 0 disables the gate.
SHARPEN_MIN_CONTRAST_FRAC="${SHARPEN_MIN_CONTRAST_FRAC:-0.15}"


# EXPERIMENTAL, off by default (radius<=0 skips it entirely - see
# _guided_filter_batch's docstring in depth_splatting_inference.py). When
# enabled, refines depth against the actual full-res source RGB frame so
# its boundary can snap to the real silhouette edge instead of only
# reflecting the low-res DepthCrafter grid. NOT validated on real footage -
# a synthetic test found it can badly OVER-smooth (wide blur/halo) when the
# guide's local contrast at the true edge is weak. Re-tune per-episode and
# inspect real output before trusting it.
GUIDED_FILTER_RADIUS="${GUIDED_FILTER_RADIUS:-0}"
GUIDED_FILTER_EPS="${GUIDED_FILTER_EPS:-1e-3}"

# Keep the depth checkpoint around after a successful run instead of
# deleting it (both the per-chunk files as splatting consumes them, and the
# whole .depth_checkpoint dir at the end) - for when you actually need to
# inspect the raw depth data (e.g. checking whether DepthCrafter's own
# output has banding/terracing in a low-texture region), not just resume a
# failed run. Off by default since keeping it defeats the disk-usage point
# of streaming chunks in the first place.
KEEP_DEPTH_CHUNKS="${KEEP_DEPTH_CHUNKS:-False}"

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
# CLASSICAL_ONLY: skip the SVD diffusion model in stage 2 entirely (never
# loaded, no VRAM/time cost) - output is just cv2.inpaint(TELEA) over the
# splat holes. Only makes sense once stage 1's silhouette fix
# (EDGE_THRESHOLD_FRAC/EDGE_FILL_ITERS) has shrunk holes down to their small,
# mostly-background size; confirmed visually indistinguishable from full
# diffusion on isolated frame tests, not yet verified on a full episode.
CLASSICAL_ONLY="${CLASSICAL_ONLY:-True}"
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
# EXPERIMENTAL disparity-based source exclusion for stage 2's inpainting -
# see inpainting_inference.py's main() docstring (disp_exclude_margin/
# disp_bg_search_px) for the full reasoning. Auto-inactive on any splat
# store without disp data (i.e. one written before this existed) -
# DISP_BG_SEARCH_PX=0 also disables it explicitly on a store that has it.
DISP_EXCLUDE_MARGIN="${DISP_EXCLUDE_MARGIN:-1.0}"
DISP_BG_SEARCH_PX="${DISP_BG_SEARCH_PX:-25}"
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

STAGE1_LOG="$OUTPUT_DIR/stage1.log"

stage1_run() {
    python -u depth_splatting_inference.py \
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
        --attention_slicing="$ATTENTION_SLICING" \
        --resume="$RESUME" \
        --compress_store="$COMPRESS_STORE" \
        --disp_tolerance="$DISP_TOLERANCE" \
        --decode_chunk_size="$STAGE1_DECODE_CHUNK_SIZE" \
        --edge_threshold_frac="$EDGE_THRESHOLD_FRAC" \
        --edge_fill_iters="$EDGE_FILL_ITERS" \
        --sharpen_mode="$SHARPEN_MODE" \
        --sharpen_radius="$SHARPEN_RADIUS" \
        --sharpen_gain="$SHARPEN_GAIN" \
        --sharpen_min_contrast_frac="$SHARPEN_MIN_CONTRAST_FRAC" \
        --guided_filter_radius="$GUIDED_FILTER_RADIUS" \
        --guided_filter_eps="$GUIDED_FILTER_EPS" \
        --keep_depth_chunks="$KEEP_DEPTH_CHUNKS" \
        2>&1 | tee "$STAGE1_LOG"
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
    python -u inpainting_inference.py \
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
        --classical_only="$CLASSICAL_ONLY" \
        "${MASK_SKIP_ARG[@]}" \
        --chunked_attention="$CHUNKED_ATTENTION" \
        --attention_kv_chunk_size="$ATTENTION_KV_CHUNK_SIZE" \
        --suppress_attention_kernel_warnings="$SUPPRESS_ATTENTION_KERNEL_WARNINGS" \
        --vae_encode_chunk_size="$VAE_ENCODE_CHUNK_SIZE" \
        --disp_exclude_margin="$DISP_EXCLUDE_MARGIN" \
        --disp_bg_search_px="$DISP_BG_SEARCH_PX" \
        --resume="$RESUME" \
        "${MAX_ITERS_ARG[@]}" \
        2>&1 | tee "$STAGE2_LOG"
}
run_with_retries "Stage 2" stage2_run

echo
echo "==================================================================="
echo "Stage 3/3: side-by-side 3D combine - NOT run automatically"
echo "==================================================================="
# Deliberately not eval'd here (used to be) - the VAAPI hwaccel command
# inpainting_inference.py prints is meant to be run wherever a VAAPI render
# node actually lives (e.g. locally, not on a rented GPU box billed by the
# hour with no VAAPI device), and burning rental time on an encode step
# that doesn't need the rented GPU at all is just wasted money. Copy the
# command below and run it yourself, on whichever box has /dev/dri/renderD128.
SBS_CMD="$(grep -A1 'To combine into side-by-side 3D' "$STAGE2_LOG" | tail -1 | sed 's/^[[:space:]]*//')"
if [[ -z "$SBS_CMD" ]]; then
    echo "Could not find the SBS ffmpeg command in $STAGE2_LOG - inpainting may have failed." >&2
    exit 1
fi
echo "$SBS_CMD"
