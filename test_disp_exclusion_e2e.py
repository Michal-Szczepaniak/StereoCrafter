"""End-to-end synthetic test for disparity-based inpainting source-exclusion
(experimental/disparity-source-exclusion branch) - runs the REAL pipeline
(depth_splatting_inference.DepthSplatting + inpainting_inference.main),
not a hand-rolled unit test, so it exercises the actual splat_store
round-trip, quantization, and compositing code paths.

Scenario: a left-to-right gradient image ("original") with a solid circle
("character") drawn on top of it ("source" - what the pipeline actually
sees). The depth map is: circle = MAX disparity, everything else = the
disparity that maps to a dead-zero shift (background never moves at all -
see BG_DEPTHNORM below) - chosen deliberately so the background is
perfectly static under the warp, which makes the ground truth trivial: the
correct content at ANY destination pixel is exactly the "original"
gradient image at that same pixel position, no warp-position bookkeeping
needed. The circle's disparity is chosen so it shifts by exactly
SHIFT_PX pixels - bigger than the circle's diameter, so the vacated
footprint becomes one clean, non-overlapping circular hole, sitting
exactly where the circle used to be in "original".

The question this answers: after depth_splatting + inpainting run for
real, does the generated content inside that hole match "original" at
those exact pixels? That's the bar the user set: "I want this hole to
look EXACTLY like it looks on the original image."

Run with the project's venv, e.g.:
    /mnt/transmission/stereocrafter/bin/python3 test_disp_exclusion_e2e.py
"""
import os
import shutil

import cv2
import numpy as np

from depth_splatting_inference import DepthSplatting
from splat_store import ffv1_encode, DEPTH_QUANT_LEVELS
import inpainting_inference

# Real 1080p, matching the actual content this pipeline runs on - resolution
# changes real behavior here (e.g. work_scale/64-padding, the comb/notch
# fix's own docstring notes the halo's low-res-pixel width differs between
# max_res=384 and 768), so a synthetic canvas smaller than real footage
# risks validating something that doesn't transfer.
H, W = 1080, 1920
NUM_FRAMES = 4
BG_DEPTHNORM = 0.5    # depthnorm=0.5 -> disp=(0.5*2-1)*max_disp=0 -> background never moves
CIRCLE_DEPTHNORM = 1.0  # depthnorm=1.0 -> disp=+max_disp -> circle moves left by max_disp

# Circle diameter = 70% of H (for spotting artifacts more easily), centered
# in frame - this is the ORIGINAL position, which is what matters: the hole
# (and the ground-truth check below) is exactly this footprint. The shift
# distance just needs to exceed the diameter so the shifted copy lands
# fully clear of the original (no overlap, or the hole degenerates into a
# partial crescent instead of a full clean disk) - it does NOT need to
# stay fully on-canvas itself; the shifted circle is free to clip off the
# left edge (harmless - we don't check anything there), which is what
# happens here since a 70%-of-H circle plus its required shift is wider
# than a single 1080p frame can fit twice. GAP just keeps a visibly empty
# band between the two footprints so it's obvious in the dumped PNGs which
# is which.
CIRCLE_R = round(0.35 * H)
GAP = max(10, CIRCLE_R // 2)
SHIFT_PX = 2 * CIRCLE_R + GAP
MAX_DISP = SHIFT_PX  # circle disp == max_disp -> exactly SHIFT_PX of movement
CIRCLE_CENTER = (H // 2, W // 2)

assert CIRCLE_CENTER[1] - CIRCLE_R >= 0 and CIRCLE_CENTER[1] + CIRCLE_R <= W, (
    "circle doesn't fit centered in W - shrink CIRCLE_R's fraction of H or widen W"
)

OUT_ROOT = "outputs/synthetic_disp_e2e"


def make_original() -> np.ndarray:
    """(H,W,3) uint8 - pure left-to-right gradient, no circle. This is the
    ground truth: whatever the pipeline generates in the hole should match
    this array at those pixel positions."""
    xs = np.arange(W, dtype=np.float32)
    row = (xs / (W - 1) * 255).astype(np.uint8)
    gradient = np.tile(row, (H, 1))
    return np.stack([gradient] * 3, axis=-1)


def make_source(original: np.ndarray) -> np.ndarray:
    """original + a solid, visually distinct circle drawn on top - this is
    what the pipeline actually sees as the input frame."""
    source = original.copy()
    cv2.circle(source, (CIRCLE_CENTER[1], CIRCLE_CENTER[0]), CIRCLE_R, (0, 0, 255), -1)  # solid red (BGR)
    return source


def make_depthnorm() -> np.ndarray:
    """(H,W) float32 in [0,1] - DepthCrafter's own storage convention (see
    depth_splatting_inference.py's DepthSplatting: dequantized chunk value,
    then affine-mapped to disp via (depthnorm*2-1)*max_disp)."""
    depthnorm = np.full((H, W), BG_DEPTHNORM, dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    circle_mask = (xx - CIRCLE_CENTER[1]) ** 2 + (yy - CIRCLE_CENTER[0]) ** 2 <= CIRCLE_R ** 2
    depthnorm[circle_mask] = CIRCLE_DEPTHNORM
    return depthnorm


def main():
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    os.makedirs(OUT_ROOT, exist_ok=True)

    original = make_original()
    source = make_source(original)
    depthnorm = make_depthnorm()

    cv2.imwrite(os.path.join(OUT_ROOT, "original.png"), original)
    cv2.imwrite(os.path.join(OUT_ROOT, "source.png"), source)

    # --- write synthetic "source video" (what depth_splatting_inference.py's
    # VideoReader reads as input_video_path) - lossless FFV1/rgb24, same
    # encoder this project already uses for its own stores, so no lossy
    # compression corrupts the ground truth before it even reaches the warp.
    video_path = os.path.join(OUT_ROOT, "source.mkv")
    source_frames = np.stack([cv2.cvtColor(source, cv2.COLOR_BGR2RGB)] * NUM_FRAMES, axis=0)
    ffv1_encode(source_frames, video_path, "rgb24", W, H)

    # --- write synthetic depth checkpoint chunk (DepthCrafter's own
    # gray16le/uint16 quantization scheme, min=0/max=1 passthrough so the
    # dequantized value IS depthnorm, up to quantization rounding).
    depth_chunk_path = os.path.join(OUT_ROOT, "depth_chunk_000.mkv")
    quantized = np.stack(
        [(depthnorm * DEPTH_QUANT_LEVELS).round().astype(np.uint16)] * NUM_FRAMES, axis=0
    )
    ffv1_encode(quantized, depth_chunk_path, "gray16le", W, H)
    chunk_meta = [{"min": 0.0, "max": 1.0, "frames": NUM_FRAMES}]

    # --- stage 1: real DepthSplatting (real ForwardWarpStereo, real
    # splat_store, real disp persistence) - device="cpu" since this is a
    # tiny synthetic clip, no need for GPU.
    splat_dir = os.path.join(OUT_ROOT, "splat")
    DepthSplatting(
        input_video_path=video_path,
        store_dir=splat_dir,
        chunk_files=[depth_chunk_path],
        chunk_meta=chunk_meta,
        global_min=0.0,
        global_max=1.0,
        max_disp=MAX_DISP,
        batch_size=2,
        target_fps=24,
        stride=1,
        original_height=H,
        original_width=W,
        store_params={"max_disp": MAX_DISP},
        compress_store=False,
        disp_tolerance=1.0,
        device="cpu",
        keep_depth_chunks=True,
        max_frames=None,
    )
    print(f"==> stage 1 (real DepthSplatting) done -> {splat_dir}")

    # --- stage 2: real inpainting_inference.main(), classical_only=True
    # (fast - never loads the SVD pipeline, pre_trained_path/unet_path are
    # unused in that mode). Exercises the real disp-exclusion mask, real
    # TELEA prefill, real compositing gate - the actual code path, not a
    # hand-rolled reimplementation of it.
    save_dir = os.path.join(OUT_ROOT, "result")
    inpainting_inference.main(
        pre_trained_path="unused",
        unet_path="unused",
        splat_store_dir=splat_dir,
        save_dir=save_dir,
        classical_only=True,
        resume=False,
        overlap=0,
    )

    video_name = os.path.basename(os.path.normpath(splat_dir)) + "_inpainting_results"
    right_path = os.path.join(save_dir, f"{video_name}_right.mkv")
    print(f"==> stage 2 (real inpainting_inference.main) done -> {right_path}")

    # --- verify: does the hole (the circle's original footprint, since
    # SHIFT_PX > 2*CIRCLE_R guarantees no overlap with the shifted circle)
    # match "original" at those exact pixels?
    cap = cv2.VideoCapture(right_path)
    ok, result_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame back from {right_path}")
    cv2.imwrite(os.path.join(OUT_ROOT, "result_frame0.png"), result_bgr)

    yy, xx = np.mgrid[0:H, 0:W]
    hole_mask = (xx - CIRCLE_CENTER[1]) ** 2 + (yy - CIRCLE_CENTER[0]) ** 2 <= CIRCLE_R ** 2

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    true_vals = original[hole_mask].astype(np.float32)
    got_vals = result_rgb[hole_mask].astype(np.float32)
    mae = np.abs(true_vals - got_vals).mean()

    diff_vis = np.zeros((H, W), dtype=np.uint8)
    diff_vis[hole_mask] = np.clip(
        np.abs(original[..., 0].astype(np.int16) - result_rgb[..., 0].astype(np.int16)), 0, 255
    )[hole_mask].astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_ROOT, "hole_diff.png"), diff_vis)

    print()
    print(f"hole pixels: {hole_mask.sum()}")
    print(f"hole fill MAE vs. original (0-255 scale): {mae:.2f}")
    print(f"see {OUT_ROOT}/ for original.png, source.png, result_frame0.png, hole_diff.png")


if __name__ == "__main__":
    main()
