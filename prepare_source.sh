#!/usr/bin/env bash
# Prepares a source video for run_stereo.sh: downscales to a target
# resolution (default 1920x1080, so the final SBS combine comes out
# 3840x1080) and, if the source is HDR (PQ/HLG), tonemaps it down to SDR
# BT.709 first.
#
# Why this exists: depth_splatting_inference.py has NO resolution or
# color-space handling of its own - it reads the input at whatever
# resolution/color space it's tagged with and writes the warp/mask/right-eye
# output at that same native resolution (see its own docstring). Every
# validated run_stereo.sh setting (TILE_NUM, WORK_SCALE, VRAM budgets) was
# tuned at 1080p SDR. Feeding it a 4K HDR remux directly either blows past
# every VRAM/time budget ever measured here, or hands the depth/diffusion
# models raw PQ code values they were never trained on (or both) - this is
# what "failed miserably" on The Hunt for Red October traced back to.
#
# Usage:
#   ./prepare_source.sh <input_video> [output_path]
#
# Output is always .mkv (needed for TrueHD/DTS audio and PGS subtitles,
# which get stream-copied through untouched - only the video is
# re-encoded). Skips the whole job if output_path already exists, since a
# 4K source at PREPARE_CRF=12/slow is an hours-long job you do not want to
# accidentally restart - pass FORCE=True to redo it anyway.
#
# Override any knob via environment variable, e.g.:
#   PREPARE_CRF=16 PREPARE_PRESET=medium ./prepare_source.sh input.mkv
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input_video> [output_path]" >&2
    exit 1
fi

INPUT_VIDEO="$1"
if [[ ! -f "$INPUT_VIDEO" ]]; then
    echo "Input video not found: $INPUT_VIDEO" >&2
    exit 1
fi
INPUT_VIDEO="$(cd "$(dirname "$INPUT_VIDEO")" && pwd)/$(basename "$INPUT_VIDEO")"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_NAME="$(basename "${INPUT_VIDEO%.*}")"
OUTPUT_PATH="${2:-"$REPO_DIR/source_video/${VIDEO_NAME}_prepared.mkv"}"
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Match inpainting_inference.py's ENCODE_PRESET/ENCODE_CRF defaults - this
# intermediate becomes the pipeline's "left eye" and quality ceiling for
# everything downstream, so there's no reason to encode it any worse than
# the right-eye output it'll sit next to in the final SBS.
PREPARE_PRESET="${PREPARE_PRESET:-slow}"
PREPARE_CRF="${PREPARE_CRF:-12}"

TARGET_WIDTH="${TARGET_WIDTH:-1920}"
TARGET_HEIGHT="${TARGET_HEIGHT:-1080}"
# Downscaling only by default - an SD/720p source getting upscaled here
# would just hand depth/inpainting soft, interpolated detail to work with
# instead of real pixels. Set FORCE_UPSCALE=True to override.
FORCE_UPSCALE="${FORCE_UPSCALE:-False}"

# Tonemap knobs, only used if the source is actually detected as HDR (see
# below). hable/npl=100 is a generic, no-metadata-required starting point,
# not tuned against this specific title - re-check a frame or two
# (ffmpeg -ss <t> -i out.mkv -frames:v 1 check.png) before trusting it for
# an unattended multi-hour run on new source material.
TONEMAP_OPERATOR="${TONEMAP_OPERATOR:-hable}"
TONEMAP_NPL="${TONEMAP_NPL:-100}"
TONEMAP_DESAT="${TONEMAP_DESAT:-0}"

VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD128}"
FORCE="${FORCE:-False}"

if [[ -f "$OUTPUT_PATH" && "$FORCE" != "True" ]]; then
    echo "==> $OUTPUT_PATH already exists - skipping (set FORCE=True to redo)." >&2
    exit 0
fi

echo "==> Probing $INPUT_VIDEO ..."
probe() {
    ffprobe -v error -select_streams v:0 -show_entries stream="$1" \
        -of default=noprint_wrappers=1:nokey=1 "$INPUT_VIDEO"
}
SRC_WIDTH="$(probe width)"
SRC_HEIGHT="$(probe height)"
SRC_CODEC="$(probe codec_name)"
SRC_TRANSFER="$(probe color_transfer)"
SRC_PIX_FMT="$(probe pix_fmt)"
echo "    ${SRC_WIDTH}x${SRC_HEIGHT} $SRC_CODEC, pix_fmt=$SRC_PIX_FMT, color_transfer=$SRC_TRANSFER"

# VAAPI decode surface format - must match the source's real bit depth
# (p010le for 10-bit Main10 content, nv12 for 8-bit) or hwdownload fails.
case "$SRC_PIX_FMT" in
    *10*|*12*) HWDOWNLOAD_FORMAT="p010le" ;;
    *) HWDOWNLOAD_FORMAT="nv12" ;;
esac

# HDR detection: PQ (smpte2084, most UHD BluRay HDR10/DV) or HLG
# (arib-std-b67, most broadcast HDR). Anything else (bt709, unknown, etc.)
# is treated as already-SDR and skips the tonemap filter chain entirely -
# running zscale's linear-light round trip on genuinely SDR content is
# needless cost and a needless place for it to get the color subtly wrong.
IS_HDR=False
if [[ "$SRC_TRANSFER" == "smpte2084" || "$SRC_TRANSFER" == "arib-std-b67" ]]; then
    IS_HDR=True
fi

# Never upscale by default - only add a scale filter if the source is
# actually larger than the target (or FORCE_UPSCALE says otherwise).
SCALE_FILTER=""
if [[ "$SRC_WIDTH" != "$TARGET_WIDTH" || "$SRC_HEIGHT" != "$TARGET_HEIGHT" ]]; then
    if [[ "$SRC_WIDTH" -gt "$TARGET_WIDTH" || "$SRC_HEIGHT" -gt "$TARGET_HEIGHT" || "$FORCE_UPSCALE" == "True" ]]; then
        SCALE_FILTER="scale=${TARGET_WIDTH}:${TARGET_HEIGHT}:flags=lanczos,"
    else
        echo "==> WARNING: source (${SRC_WIDTH}x${SRC_HEIGHT}) is smaller than target (${TARGET_WIDTH}x${TARGET_HEIGHT}) - leaving resolution untouched (set FORCE_UPSCALE=True to scale up anyway)." >&2
    fi
fi

if [[ "$IS_HDR" == "True" ]]; then
    echo "==> HDR source detected ($SRC_TRANSFER) - tonemapping to SDR BT.709 ($TONEMAP_OPERATOR, npl=$TONEMAP_NPL)."
    VIDEO_FILTER="zscale=t=linear:npl=${TONEMAP_NPL},format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=${TONEMAP_OPERATOR}:desat=${TONEMAP_DESAT},zscale=t=bt709:m=bt709,${SCALE_FILTER}format=yuv420p"
else
    echo "==> SDR source - no tonemap needed."
    VIDEO_FILTER="${SCALE_FILTER}format=yuv420p"
fi

# Every audio and subtitle track, stream-copied untouched - this file is
# what run_stereo.sh's printed combine commands read `source_video_path`
# from (see inpainting_inference.py), so whatever ends up in the final SBS
# deliverable's audio/subs comes from here, not the original 4K remux.
MAP_ARGS=(-map 0:v:0 -map 0:a? -map 0:s?)
ENCODE_ARGS=(-c:v libx264 -preset "$PREPARE_PRESET" -crf "$PREPARE_CRF" -c:a copy -c:s copy -y "$OUTPUT_PATH")

echo "==> Encoding -> $OUTPUT_PATH (preset=$PREPARE_PRESET crf=$PREPARE_CRF) ..."
if [[ -e "$VAAPI_DEVICE" ]]; then
    echo "    trying VAAPI decode ($VAAPI_DEVICE) first ..."
    if ffmpeg -hwaccel vaapi -hwaccel_device "$VAAPI_DEVICE" -hwaccel_output_format vaapi \
        -i "$INPUT_VIDEO" "${MAP_ARGS[@]}" \
        -vf "hwdownload,format=${HWDOWNLOAD_FORMAT},${VIDEO_FILTER}" \
        "${ENCODE_ARGS[@]}"; then
        echo "==> Done: $OUTPUT_PATH"
        exit 0
    fi
    echo "==> VAAPI decode failed - falling back to software decode." >&2
fi

ffmpeg -i "$INPUT_VIDEO" "${MAP_ARGS[@]}" -vf "$VIDEO_FILTER" "${ENCODE_ARGS[@]}"
echo "==> Done: $OUTPUT_PATH"
