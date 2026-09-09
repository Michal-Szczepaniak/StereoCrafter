"""End-to-end synthetic test for disparity-based inpainting source-exclusion
(experimental/disparity-source-exclusion branch) - runs the REAL pipeline
(depth_splatting_inference.DepthSplatting + inpainting_inference.main),
not a hand-rolled unit test, so it exercises the actual splat_store
round-trip, quantization, and compositing code paths.

Scenario: a left-to-right VISUAL gradient image with a solid circle
("character") drawn on top of it. The background's DEPTH is ALSO a
gradient (not flat) - this is deliberate: a flat background disparity
can't reproduce the "painting white over wood" bug, which came from a
continuous depth gradient on the background itself (see conversation
history). The circle's disparity is set far above the background's whole
range, and large enough that it shifts by more than its own diameter, so
its vacated footprint becomes one clean, non-overlapping circular hole.

Ground truth complication: since the background now has real (if gentle)
disparity variation, it moves slightly under the warp too - so "the hole
should match the pre-warp image at the same pixel position" is no longer
exactly true (that shortcut only worked when the background was
perfectly static). Fix: run the REAL splat twice - once with the circle
(produces the actual hole) and once without it at all (same background
gradient, no character) - and use the no-circle run's own warped output
as ground truth. That's exactly "what would be there if the character
never existed," computed by the same real warp math instead of guessed by
hand, so it's correct regardless of how the background moves.

Run with the project's venv, e.g.:
    /mnt/transmission/stereocrafter/bin/python3 test_disp_exclusion_e2e.py
"""
import os
import shutil

import cv2
import numpy as np

from depth_splatting_inference import DepthSplatting
from splat_store import ffv1_encode, open_store, DEPTH_QUANT_LEVELS
import inpainting_inference

# Real 1080p, matching the actual content this pipeline runs on - resolution
# changes real behavior here (e.g. work_scale/64-padding, the comb/notch
# fix's own docstring notes the halo's low-res-pixel width differs between
# max_res=384 and 768), so a synthetic canvas smaller than real footage
# risks validating something that doesn't transfer.
H, W = 1080, 1920
NUM_FRAMES = 4

# Circle diameter = 70% of H (for spotting artifacts more easily), centered
# in frame - this is the ORIGINAL position, which is what matters: the hole
# is exactly this footprint regardless of what the background does (the
# character's shift depends only on its OWN disparity). The shift distance
# just needs to exceed the diameter so the shifted copy lands fully clear
# of the original (no overlap, or the hole degenerates into a partial
# crescent) - it does NOT need to stay fully on-canvas itself; the shifted
# circle is free to clip off the left edge (harmless, unchecked). GAP keeps
# a visibly empty band between the two footprints in the dumped PNGs.
CIRCLE_R = round(0.35 * H)
GAP = max(10, CIRCLE_R // 2)
SHIFT_PX = 2 * CIRCLE_R + GAP
CHAR_MAX_DISP = float(SHIFT_PX)  # circle disp == this -> exactly SHIFT_PX of movement
CIRCLE_CENTER = (H // 2, W // 2)

assert CIRCLE_CENTER[1] - CIRCLE_R >= 0 and CIRCLE_CENTER[1] + CIRCLE_R <= W, (
    "circle doesn't fit centered in W - shrink CIRCLE_R's fraction of H or widen W"
)

# Background disparity gradient range - realistic scale (matches this
# project's own real max_disp default of 20), deliberately tiny relative to
# CHAR_MAX_DISP so the character stays unambiguously "much closer than any
# background" while the background still has genuine, non-flat depth
# variation across the frame (the wood bug's actual precondition).
BG_DISP_LO, BG_DISP_HI = -20.0, 20.0

OUT_ROOT = "outputs/synthetic_disp_e2e"


def make_visual_gradient() -> np.ndarray:
    """(H,W,3) uint8 BGR - pure left-to-right 0-255 gradient, no circle.
    Purely cosmetic/for-eyeballing - NOT tied to the depth gradient below
    (real footage's color and depth aren't correlated either); ground
    truth for the hole comes from the no-circle warp, not from this array
    directly."""
    xs = np.arange(W, dtype=np.float32)
    row = (xs / (W - 1) * 255).astype(np.uint8)
    gradient = np.tile(row, (H, 1))
    return np.stack([gradient] * 3, axis=-1)


def make_source(base: np.ndarray, with_circle: bool) -> np.ndarray:
    if not with_circle:
        return base.copy()
    source = base.copy()
    cv2.circle(source, (CIRCLE_CENTER[1], CIRCLE_CENTER[0]), CIRCLE_R, (0, 0, 255), -1)  # solid red (BGR)
    return source


def make_depthnorm(with_circle: bool) -> np.ndarray:
    """(H,W) float32 in [0,1] - DepthCrafter's own storage convention (see
    depth_splatting_inference.py's DepthSplatting: dequantized chunk value,
    then affine-mapped to disp via (depthnorm*2-1)*max_disp, using
    CHAR_MAX_DISP as `max_disp` for both variants so the SAME depthnorm
    value always means the SAME real disp value in both runs)."""
    xs = np.arange(W, dtype=np.float32)
    bg_disp_row = BG_DISP_LO + (xs / (W - 1)) * (BG_DISP_HI - BG_DISP_LO)
    bg_depthnorm_row = bg_disp_row / (2.0 * CHAR_MAX_DISP) + 0.5
    depthnorm = np.tile(bg_depthnorm_row, (H, 1)).astype(np.float32)

    if with_circle:
        yy, xx = np.mgrid[0:H, 0:W]
        circle_mask = (xx - CIRCLE_CENTER[1]) ** 2 + (yy - CIRCLE_CENTER[0]) ** 2 <= CIRCLE_R ** 2
        depthnorm[circle_mask] = 1.0  # disp = +CHAR_MAX_DISP -> shifts left by SHIFT_PX
    return depthnorm


def run_splat(name: str, with_circle: bool) -> str:
    """Builds the synthetic video + depth checkpoint for one variant and
    runs the REAL DepthSplatting on it. Returns the splat store dir."""
    variant_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(variant_dir, exist_ok=True)

    base = make_visual_gradient()
    source = make_source(base, with_circle)
    depthnorm = make_depthnorm(with_circle)
    cv2.imwrite(os.path.join(variant_dir, "source.png"), source)

    video_path = os.path.join(variant_dir, "source.mkv")
    source_frames = np.stack([cv2.cvtColor(source, cv2.COLOR_BGR2RGB)] * NUM_FRAMES, axis=0)
    ffv1_encode(source_frames, video_path, "rgb24", W, H)

    depth_chunk_path = os.path.join(variant_dir, "depth_chunk_000.mkv")
    quantized = np.stack(
        [(depthnorm * DEPTH_QUANT_LEVELS).round().astype(np.uint16)] * NUM_FRAMES, axis=0
    )
    ffv1_encode(quantized, depth_chunk_path, "gray16le", W, H)
    chunk_meta = [{"min": 0.0, "max": 1.0, "frames": NUM_FRAMES}]

    splat_dir = os.path.join(variant_dir, "splat")
    DepthSplatting(
        input_video_path=video_path,
        store_dir=splat_dir,
        chunk_files=[depth_chunk_path],
        chunk_meta=chunk_meta,
        global_min=0.0,
        global_max=1.0,
        max_disp=CHAR_MAX_DISP,
        batch_size=2,
        target_fps=24,
        stride=1,
        original_height=H,
        original_width=W,
        store_params={"max_disp": CHAR_MAX_DISP},
        compress_store=False,
        disp_tolerance=1.0,
        device="cpu",
        keep_depth_chunks=True,
        max_frames=None,
    )
    return splat_dir


def main():
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    os.makedirs(OUT_ROOT, exist_ok=True)

    splat_with = run_splat("with_circle", with_circle=True)
    print(f"==> stage 1 (with circle) done -> {splat_with}")
    splat_without = run_splat("no_circle", with_circle=False)
    print(f"==> stage 1 (no circle, ground-truth reference) done -> {splat_without}")

    # Sanity check: the no-circle run should have essentially no holes of
    # its own (a gentle background gradient shouldn't self-occlude) - if it
    # does, the ground truth below isn't trustworthy and the background
    # gradient needs to be gentler.
    warp_ref, mask_ref, _disp_ref, _meta_ref = open_store(splat_without, mode="r")
    ref_hole_frac = (np.array(mask_ref[0:1])[0] > 127).mean()
    print(f"==> no-circle run hole fraction: {ref_hole_frac:.4%} (expect ~0)")

    # --- stage 2: real inpainting_inference.main(), classical_only=True
    # (fast - never loads the SVD pipeline, pre_trained_path/unet_path are
    # unused in that mode). Exercises the real disp-exclusion mask, real
    # TELEA prefill, real compositing gate - the actual code path, not a
    # hand-rolled reimplementation of it. Only the WITH-circle store needs
    # this - the no-circle store has no real holes to fill.
    save_dir = os.path.join(OUT_ROOT, "result")
    inpainting_inference.main(
        pre_trained_path="unused",
        unet_path="unused",
        splat_store_dir=splat_with,
        save_dir=save_dir,
        classical_only=True,
        resume=False,
        overlap=0,
    )

    video_name = os.path.basename(os.path.normpath(splat_with)) + "_inpainting_results"
    right_path = os.path.join(save_dir, f"{video_name}_right.mkv")
    print(f"==> stage 2 (real inpainting_inference.main) done -> {right_path}")

    cap = cv2.VideoCapture(right_path)
    ok, result_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame back from {right_path}")
    cv2.imwrite(os.path.join(OUT_ROOT, "result_frame0.png"), result_bgr)

    # Ground truth: the no-circle store's OWN warped output - real warp
    # math, not a hand-derived formula, so it's correct regardless of how
    # much the background moved under its own gradient.
    ground_truth_bgr = np.array(warp_ref[0:1])[0]  # (H,W,3) RGB, uint8
    ground_truth_bgr = cv2.cvtColor(ground_truth_bgr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT_ROOT, "ground_truth.png"), ground_truth_bgr)

    # Hole mask: the circle's ORIGINAL footprint - exact regardless of
    # background motion, since the character's own shift only depends on
    # its own disparity.
    yy, xx = np.mgrid[0:H, 0:W]
    hole_mask = (xx - CIRCLE_CENTER[1]) ** 2 + (yy - CIRCLE_CENTER[0]) ** 2 <= CIRCLE_R ** 2

    true_vals = ground_truth_bgr[hole_mask].astype(np.float32)
    got_vals = result_bgr[hole_mask].astype(np.float32)
    mae = np.abs(true_vals - got_vals).mean()

    diff_vis = np.zeros((H, W), dtype=np.uint8)
    diff_vis[hole_mask] = np.clip(
        np.abs(ground_truth_bgr[..., 0].astype(np.int16) - result_bgr[..., 0].astype(np.int16)), 0, 255
    )[hole_mask].astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_ROOT, "hole_diff.png"), diff_vis)

    print()
    print(f"hole pixels: {hole_mask.sum()}")
    print(f"hole fill MAE vs. real no-circle warp (0-255 scale): {mae:.2f}")
    print(f"see {OUT_ROOT}/ for with_circle/source.png, no_circle/source.png,")
    print("    ground_truth.png, result_frame0.png, hole_diff.png")


if __name__ == "__main__":
    main()
