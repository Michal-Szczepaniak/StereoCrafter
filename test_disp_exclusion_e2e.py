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
history). The circle's disparity is set to this project's own real
max_disp default (20) - NOT scaled up to exceed the circle's own diameter
- since no real footage can ever shift anything farther than max_disp
regardless of how large/close the object is on screen. That means the
hole is NOT a full clean disk (a 756px-diameter circle moving only 20px
barely uncovers anything) - it's a thin ~20px crescent sliver along the
circle's trailing edge, which is what a real disocclusion next to a large
nearby object's silhouette actually looks like.

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
from inpainting_inference import _disp_exclusion_mask

# Real 1080p, matching the actual content this pipeline runs on - resolution
# changes real behavior here (e.g. work_scale/64-padding, the comb/notch
# fix's own docstring notes the halo's low-res-pixel width differs between
# max_res=384 and 768), so a synthetic canvas smaller than real footage
# risks validating something that doesn't transfer.
H, W = 1080, 1920
NUM_FRAMES = 4

# Circle diameter = 70% of H (for spotting artifacts more easily), centered
# in frame. CHAR_MAX_DISP is this project's own real max_disp default
# (see run_stereo.sh's MAX_DISP) - the circle's disparity is set to exactly
# this, so it shifts by SHIFT_PX=CHAR_MAX_DISP pixels, same as any real
# object at maximum practical closeness would. Since that's far smaller
# than the circle's own diameter, the shifted copy overlaps almost all of
# the original - the hole is only the crescent-shaped sliver of the
# original footprint the shifted copy doesn't cover.
CIRCLE_R = round(0.35 * H)
CHAR_MAX_DISP = 20.0  # matches run_stereo.sh's MAX_DISP default
SHIFT_PX = round(CHAR_MAX_DISP)
CIRCLE_CENTER = (H // 2, W // 2)
CIRCLE_CENTER_SHIFTED = (H // 2, W // 2 - SHIFT_PX)  # where the circle actually lands after the warp

assert CIRCLE_CENTER[1] - CIRCLE_R >= 0 and CIRCLE_CENTER[1] + CIRCLE_R <= W, (
    "circle doesn't fit centered in W - shrink CIRCLE_R's fraction of H or widen W"
)

# Background disparity gradient. NOT the full +-CHAR_MAX_DISP range -
# empirically (this session, real hardware) that pegs one CPU core for a
# very long time in _forward_legacy_cuda_splat's CPU fallback. Cause:
# ForwardWarpStereo's near-bias weighting is `1.414 ** (disp - disp.min())`
# - unbounded upward, not clamped like a softmax's "subtract max" trick.
# With background reaching -CHAR_MAX_DISP in the SAME batch as the circle
# at +CHAR_MAX_DISP, that exponent hits 2*CHAR_MAX_DISP=40, i.e. weight
# ratios around 1.4e6 - not an overflow, but exactly the kind of extreme
# dynamic range that produces denormalized floats for the smallest-weighted
# contributions, and denormal float arithmetic is a well-known single-
# thread slowdown (10-100x) on CPU. +-10 keeps exponent span to 30 (ratio
# ~2.9e4) - clearly visible in depth_chunk_000.mkv (spans the middle 50%
# of the full storage range) without the cliff. NOTE: since this is
# purely a property of the batch's total disp SPAN, not of the device,
# real footage that puts a near-max_disp foreground object in the same
# frame as background reaching toward -max_disp could hit the same
# slowdown on the real pipeline, not just this synthetic test - worth
# keeping in mind if a real run ever mysteriously stalls at the splat
# step specifically.
BG_DISP_LO, BG_DISP_HI = -10.0, 10.0

OUT_ROOT = "outputs/synthetic_disp_e2e"


def _bg_disp_row() -> np.ndarray:
    """(W,) float32 - the background's per-column real disparity value.
    HIGH at x=0 (left), LOW at x=W-1 (right) - reversed from a plain
    left-to-right ramp, per request. Single source of truth shared by both
    the visual color gradient and the depth gradient below, so they always
    correspond (high disp = bright, low disp = dark) instead of being two
    independent ramps."""
    xs = np.arange(W, dtype=np.float32)
    return BG_DISP_HI - (xs / (W - 1)) * (BG_DISP_HI - BG_DISP_LO)


def make_visual_gradient() -> np.ndarray:
    """(H,W,3) uint8 BGR - grayscale gradient derived directly from
    _bg_disp_row(), so color always corresponds to depth (bright = high
    disparity/near, dark = low disparity/far), not an independent ramp."""
    bg_disp_row = _bg_disp_row()
    row = ((bg_disp_row - BG_DISP_LO) / (BG_DISP_HI - BG_DISP_LO) * 255).astype(np.uint8)
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
    value always means the SAME real disp value in both runs). Derived
    from the SAME _bg_disp_row() the visual gradient uses, so they stay
    correlated."""
    bg_disp_row = _bg_disp_row()
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
        compress_store=True,
        disp_tolerance=1.0,
        # NOT "cpu": production's own main() never passes `device` to
        # DepthSplatting() at all, relying on its default (device="cuda"),
        # so real runs always use the compiled CUDA/HIP splat kernel, never
        # the pure-Python CPU fallback. Omitting it here too, rather than
        # hardcoding "cpu", so this test exercises the exact same code path
        # production does instead of one that's never actually used for
        # real - a real discrepancy this had, not just cosmetic.
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

    # Dump the RAW stage-1 splat output (before any inpainting at all) so
    # the circle's shift and the real hole shape can be inspected directly,
    # independent of stage 2 - a 20px shift on a 756px-diameter circle is
    # only ~2.6% of its size, easy to miss by eye against the whole circle,
    # but the mask makes the actual (thin crescent) hole shape unambiguous.
    warp_with, mask_with, disp_with, meta_with = open_store(splat_with, mode="r")
    splat_frame = cv2.cvtColor(np.array(warp_with[0:1])[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT_ROOT, "splat_with_circle.png"), splat_frame)
    mask_frame = np.array(mask_with[0:1])[0]
    cv2.imwrite(os.path.join(OUT_ROOT, "splat_mask.png"), mask_frame)

    # Dump the ACTUAL generation mask (hole | exclusion) using the real
    # persisted disp data and inpainting_inference.main()'s own defaults
    # (disp_exclude_margin=1.0, disp_bg_search_px=25) - this is exactly
    # what _prefill_occlusion/the diffusion conditioning saw as "needs
    # filling", so it directly shows whether the exclusion is eating into
    # real background near the circle (pushing TELEA's usable source
    # farther away than necessary) rather than just the circle itself.
    disp_max_with = meta_with["params"]["max_disp"]
    disp_u16_with = np.array(disp_with[0:1])[0]
    disp_np_with = disp_u16_with.astype(np.float32) / DEPTH_QUANT_LEVELS * (2.0 * disp_max_with) - disp_max_with
    hole_bool_with = mask_frame[None] > 127
    exclude_bool = _disp_exclusion_mask(disp_np_with[None], hole_bool_with, 1.0, 25)[0]
    gen_mask_vis = np.where(hole_bool_with[0] | exclude_bool, 255, 0).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_ROOT, "gen_mask.png"), gen_mask_vis)

    # white = real hole, gray = excluded-as-source but NOT a real hole (the
    # circle itself, plus any real background caught by the search radius)
    excl_only_vis = np.zeros_like(gen_mask_vis)
    excl_only_vis[exclude_bool & ~hole_bool_with[0]] = 128
    excl_only_vis[hole_bool_with[0]] = 255
    cv2.imwrite(os.path.join(OUT_ROOT, "gen_mask_excl_highlighted.png"), excl_only_vis)
    print(f"==> real hole px: {hole_bool_with.sum()}, exclusion-only px (beyond the real hole): {(exclude_bool & ~hole_bool_with[0]).sum()}")

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

    # Hole mask: the crescent sliver of the circle's ORIGINAL footprint that
    # the SHIFTED copy doesn't cover - exact regardless of background
    # motion, since the character's own shift only depends on its own
    # disparity, not on the background at all.
    yy, xx = np.mgrid[0:H, 0:W]
    orig_circle = (xx - CIRCLE_CENTER[1]) ** 2 + (yy - CIRCLE_CENTER[0]) ** 2 <= CIRCLE_R ** 2
    shifted_circle = (
        (xx - CIRCLE_CENTER_SHIFTED[1]) ** 2 + (yy - CIRCLE_CENTER_SHIFTED[0]) ** 2 <= CIRCLE_R ** 2
    )
    hole_mask = orig_circle & (~shifted_circle)

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
