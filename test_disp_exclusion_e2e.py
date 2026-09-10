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
import torch
import torch.nn.functional as F

from depth_splatting_inference import DepthSplatting, _edge_threshold_fill
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

# Real production defaults (depth_splatting_inference.py's DepthSplatting/
# main() signatures) - the low-res-compute -> edge-threshold-fill ->
# nearest-upsample pipeline this test now replicates uses these exact
# values, not placeholders.
MAX_RES = 768
EDGE_THRESHOLD_FRAC = 0.10
EDGE_FILL_ITERS = 3

# Guided-filter refinement (prototype). Tested on real footage and judged
# not worth its complexity - it can't distinguish a real depth boundary
# from a pure ink line, and even band-gated it didn't visibly improve the
# silhouette. Left in place but OFF, so it can't confound the
# position-preserving-sharpen test below.
GUIDED_FILTER_ENABLED = False
GUIDED_FILTER_RADIUS = 8
GUIDED_FILTER_EPS = 1e-3

# Matches production: EDGE_FILL_ITERS is a LOW-RES expansion pass before
# upsampling, and SHARPEN_MODE is the optional full-res re-hardening after
# it. "stretch" = _position_preserving_sharpen, "none" = leave it alone.
# main() splats BOTH and dumps a mask for each, so they can be A/B'd in
# one run.
SHARPEN_MODE = "stretch"
# Window must reach the flat plateau on BOTH sides of the transition: the
# ramp is ~(1-2 low-res px) x (upsample ratio) ~= 2.5-6 full-res px here,
# so radius needs margin past that. Swept offline (numpy/scipy sim of this
# exact scenario): results are IDENTICAL for radius 6, 8 and 12, and only
# slightly worse at 4 - i.e. anything >= the ramp width works and this
# knob barely needs tuning. Too large only risks spanning multiple
# separate structures (thin hair strands etc.), so stay near the minimum.
SHARPEN_RADIUS = 6
# How hard to collapse the ramp. gain -> infinity is the classic
# morphological "toggle contrast" operator (snap to whichever plateau is
# closer). Swept offline: gain=3 narrows the edge from ~6.3px to ~1.8px
# with LITERALLY ZERO positional cost (RMS 0.777 -> 0.777, jitter 0.1590
# -> 0.1591 vs. not sharpening at all) - a free lunch. Past that you pay:
# gain=6 gives a ~1.1px edge but +35% jitter, gain=12 +130%, toggle/inf
# +265%. Since the comb IS a jitter artifact, 3 is the safe default;
# raise it only if a ~1.8px soft edge proves too soft for the splat.
SHARPEN_GAIN = 3.0
# Only hard-step transitions steep enough to actually TEAR in the warp,
# as a fraction of the depth range. Without this, a high gain hard-steps
# every mild depth change too, and each one then produces its own small
# hole where the warp previously covered it fine. 0 disables the gate.
SHARPEN_MIN_CONTRAST_FRAC = 0.15


def _production_low_res(height: int, width: int, max_res: int) -> tuple[int, int]:
    """Mirrors get_video_info's own resize math exactly (round-to-64,
    only downscale if over max_res) - this is what DepthCrafter's real
    low-res inference resolution actually is for a given source size."""
    low_h = round(height / 64) * 64
    low_w = round(width / 64) * 64
    if max(low_h, low_w) > max_res:
        scale = max_res / max(height, width)
        low_h = round(height * scale / 64) * 64
        low_w = round(width * scale / 64) * 64
    return low_h, low_w


def _position_preserving_sharpen(depth: np.ndarray, radius: int, gain: float,
                                 min_contrast: float = 0.0) -> np.ndarray:
    """Re-hardens a soft depth transition WITHOUT moving where it sits.

    The whole point (see the comb/aliasing investigation): a low-res depth
    pixel that straddles a real silhouette gets a coverage-weighted blend
    of the foreground and background depth - so its exact value encodes
    the SUB-PIXEL position of the boundary inside that pixel, exactly the
    way anti-aliasing works. _edge_threshold_fill discards that: it snaps
    every low-res pixel to pure-FG or pure-BG, which forces the boundary
    onto integer low-res pixel edges and MANUFACTURES the staircase whose
    splat is the comb artifact.

    This instead works on the full-res, bilinear-upsampled ramp (which
    still carries that coverage information) and stretches contrast about
    the LOCAL midpoint between the two plateaus:

      local_lo/local_hi = grayscale erode/dilate = the background and
        foreground plateau values on either side of the transition
      level = their midpoint = the value corresponding to 50% coverage
      out = clip((depth - level) * gain + level, local_lo, local_hi)

    Properties that make this the right operator here:
      - At the 50% crossing, depth == level, so the output is unchanged -
        the boundary's position is preserved EXACTLY, at sub-low-res-pixel
        precision, instead of being quantized to the grid.
      - Over a linear gradient, a symmetric window's min and max average
        to the center value, so depth - level == 0 and smooth background
        depth ramps pass through completely untouched (no terracing).
      - In flat regions local_lo == local_hi == depth, so it self-disables
        rather than amplifying noise.
      - Clamping to [local_lo, local_hi] prevents overshoot/ringing past
        the true plateau values (which a plain unsharp mask would cause).

    gain -> infinity is the classic morphological toggle-contrast operator
    (snap to the nearer plateau); finite gain keeps a ~1-2px soft edge.

    Validated offline against _edge_threshold_fill on this test's own
    scenario (gradient background + circle, exact coverage downsample,
    numpy/scipy sim - see the commit message). Boundary position RMS error
    in full-res px, and mean outward shift ("halo"):

        no sharpening (soft ramp)        RMS 1.30   shift +1.18
        edge_fill iters=1                RMS 2.34   shift +2.25
        edge_fill iters=3                RMS 4.43   shift +4.34
        stretch (this, gain=3)           RMS 0.96   shift +0.87

    The edge_fill numbers show what it actually IS on this content: a
    plain 1px-per-iteration DILATION, not a sharpener. Its threshold is
    global (min + frac*range), and with 85% of the frame sitting above
    that threshold, essentially every pixel takes the max-of-neighbors
    branch - including both sides of a real boundary, so it grows the
    brighter side instead of snapping to the transition. That is exactly
    the "aura/halo around characters" complaint that kicked off this whole
    investigation, and why EDGE_FILL_ITERS had to be lowered to 1 to make
    the mask hug characters at all.
    """
    k = 2 * radius + 1
    # MORPH_RECT, not ELLIPSE: a square window is what the offline
    # verification measured, and what production's torch implementation
    # (F.max_pool2d) gives - keep all three identical so the numbers above
    # actually describe what runs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    local_lo = cv2.erode(depth, kernel)
    local_hi = cv2.dilate(depth, kernel)
    level = 0.5 * (local_lo + local_hi)
    stretched = np.clip((depth - level) * gain + level, local_lo, local_hi)
    if min_contrast <= 0:
        return stretched
    # Gate: leave transitions too gentle to tear completely alone (see the
    # production copy in depth_splatting_inference.py for the measurements).
    # Ramped, not switched, so the gate boundary isn't itself an edge.
    half = 0.5 * min_contrast
    weight = np.clip((local_hi - local_lo - half) / half, 0.0, 1.0)
    return depth + weight * (stretched - depth)


def _depth_boundary_band(depth_2d: np.ndarray, band_radius: int, rel_thresh: float = 0.02) -> np.ndarray:
    """Boolean mask, True within band_radius pixels of a place where
    depth_2d itself (NOT the RGB guide) changes by more than rel_thresh of
    its own [min,max] range between adjacent pixels - i.e. a REAL existing
    depth transition. Used to restrict guided-filter refinement to only
    the vicinity of an already-known depth boundary, so it can never touch
    flat-depth regions no matter how much RGB linework/texture sits on top
    of them - real anime footage showed guided filtering with no such gate
    treats every clothing-fold/hair-strand line as if it were a depth
    edge, since it can't otherwise distinguish "real depth boundary" from
    "strong-contrast ink line with zero depth information," which is
    common in cel-shaded art but rare in photos (what guided filtering was
    originally designed for)."""
    dmin, dmax = float(depth_2d.min()), float(depth_2d.max())
    span = max(dmax - dmin, 1e-6)
    thresh = rel_thresh * span
    dx = np.abs(np.diff(depth_2d, axis=1, append=depth_2d[:, -1:]))
    dy = np.abs(np.diff(depth_2d, axis=0, append=depth_2d[-1:, :]))
    edge = (dx > thresh) | (dy > thresh)
    k = 2 * band_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(edge.astype(np.uint8), kernel) > 0


def _guided_filter(guide_gray: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He/Sun/Tang 2010 guided image filter (single-channel guide), built
    from plain cv2.boxFilter only - NOT cv2.ximgproc (that module needs
    opencv-contrib-python; this repo's requirements.txt pins plain
    opencv-python, so ximgproc.jointBilateralFilter/guidedFilter aren't
    available without a dependency swap).

    Refines `src` (here: depth, already bilinear-upsampled + edge-filled to
    full res) so it locally tracks real structure in `guide_gray` (here:
    the full-res RGB source frame, grayscale) instead of only reflecting
    what a coarse low-res depth grid could represent. Mechanism: in each
    local window, fits depth as an affine function of the guide's
    intensity (depth = a*guide + b, solved by local least squares via box-
    filtered first/second moments - a plain box filter computes an exact
    per-pixel local mean in O(1) per pixel via an integral image, which is
    what makes this cheap regardless of radius). Where the guide has a
    strong edge (e.g. the real character silhouette), `a` swings sharply
    to follow it; where the guide is flat, `a` collapses to ~0 and the
    filter just locally averages src, same as a plain box blur.
    """
    guide = guide_gray.astype(np.float32)
    p = src.astype(np.float32)
    k = 2 * radius + 1

    mean_I = cv2.boxFilter(guide, cv2.CV_32F, (k, k))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (k, k))
    corr_I = cv2.boxFilter(guide * guide, cv2.CV_32F, (k, k))
    corr_Ip = cv2.boxFilter(guide * p, cv2.CV_32F, (k, k))

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (k, k))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (k, k))

    filtered = mean_a * guide + mean_b
    band = _depth_boundary_band(p, radius)
    return np.where(band, filtered, p)


def _apply_production_depth_pipeline(depthnorm_full: np.ndarray, mode: str = SHARPEN_MODE) -> np.ndarray:
    """Replicates the transform chain real depth goes through in
    depth_splatting_inference.py BEFORE it ever reaches DepthSplatting's
    splat step: (1) the depth model only ever sees/produces depth at a
    downscaled low-res resolution (get_video_info's own math, see
    _production_low_res), never at full source resolution; (2) that raw
    low-res depth is upsampled to full resolution via BILINEAR, which
    preserves the soft transition ramp (and with it the sub-low-res-pixel
    boundary position encoded in it) instead of snapping to a blocky
    low-res-pixel step the way nearest does; (3) the softened edge is
    re-hardened, by whichever of the two operators `mode` selects.

    EDGE_FILL_ITERS runs _edge_threshold_fill on the LOW-RES depth first
    (an expansion pass - grows the foreground so it still covers the real
    silhouette after upsampling), then mode selects the full-res step:
    mode="stretch" (_position_preserving_sharpen) re-hardens WITHOUT
    moving the boundary; mode="none" leaves the upsampled depth alone.

    This test hand-authors a depth map directly (no real DepthCrafter
    inference), so without the resolution round-trip here, the synthetic
    circle's boundary was analytically smooth at full 1080p, which no real
    depth map ever is. Downscaling with INTER_AREA (not NEAREST/LINEAR) so
    the low-res circle boundary starts genuinely soft/anti-aliased - and
    note INTER_AREA is an exact coverage average, so in this test the soft
    values ARE true sub-pixel coverage, which is precisely the information
    "stretch" is supposed to preserve and "edge_fill" is supposed to
    destroy. That makes this a decisive test of the hypothesis, not just a
    qualitative look.
    """
    low_h, low_w = _production_low_res(H, W, MAX_RES)
    low_res = cv2.resize(depthnorm_full, (low_w, low_h), interpolation=cv2.INTER_AREA)

    low_res_t = torch.from_numpy(low_res).unsqueeze(0).unsqueeze(0).float().cuda()
    if EDGE_FILL_ITERS > 0:
        threshold = float(low_res.min()) + EDGE_THRESHOLD_FRAC * (
            float(low_res.max()) - float(low_res.min())
        )
        low_res_t = _edge_threshold_fill(low_res_t, threshold, EDGE_FILL_ITERS)
    upsampled_t = F.interpolate(low_res_t, size=(H, W), mode="bilinear", align_corners=False)

    if mode == "stretch":
        upsampled = upsampled_t[0, 0].cpu().numpy()
        min_contrast = SHARPEN_MIN_CONTRAST_FRAC * (
            float(low_res.max()) - float(low_res.min())
        )
        return _position_preserving_sharpen(
            upsampled, SHARPEN_RADIUS, SHARPEN_GAIN, min_contrast
        )

    if mode == "none":
        return upsampled_t[0, 0].cpu().numpy()

    raise ValueError(f"unknown sharpen mode {mode!r} - expected 'stretch' or 'none'")


def _apply_guided_refinement(depthnorm_full: np.ndarray, guide_bgr: np.ndarray) -> np.ndarray:
    """Prototype-only extra step (not yet in production): guided-filter the
    bilinear+edge-fill result from _apply_production_depth_pipeline against
    the actual full-res RGB source frame, so the depth boundary snaps to
    the REAL silhouette edge (visible in the RGB pixels at full 1080p, cost-
    free) instead of only reflecting what the 768-wide depth grid could
    represent. Does not touch DepthCrafter's own cost/resolution at all -
    this runs after it, on its output."""
    guide_gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY)
    return _guided_filter(guide_gray, depthnorm_full, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPS)


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


def run_splat(name: str, with_circle: bool, mode: str = SHARPEN_MODE) -> str:
    """Builds the synthetic video + depth checkpoint for one variant and
    runs the REAL DepthSplatting on it. Returns the splat store dir.
    `mode` selects the re-hardening operator - see
    _apply_production_depth_pipeline."""
    variant_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(variant_dir, exist_ok=True)

    base = make_visual_gradient()
    source = make_source(base, with_circle)
    depthnorm_analytic = make_depthnorm(with_circle)
    depthnorm = _apply_production_depth_pipeline(depthnorm_analytic, mode)
    cv2.imwrite(os.path.join(variant_dir, "source.png"), source)
    cv2.imwrite(
        os.path.join(variant_dir, "depthnorm_pipeline.png"),
        (np.clip(depthnorm, 0.0, 1.0) * 255).astype(np.uint8),
    )

    if GUIDED_FILTER_ENABLED:
        depthnorm = _apply_guided_refinement(depthnorm, source)
        cv2.imwrite(
            os.path.join(variant_dir, "depthnorm_guided.png"),
            (np.clip(depthnorm, 0.0, 1.0) * 255).astype(np.uint8),
        )

    video_path = os.path.join(variant_dir, "source.mkv")
    source_frames = np.stack([cv2.cvtColor(source, cv2.COLOR_BGR2RGB)] * NUM_FRAMES, axis=0)
    ffv1_encode(source_frames, video_path, "rgb24", W, H)

    depth_chunk_path = os.path.join(variant_dir, "depth_chunk_000.mkv")
    # Clip before quantizing to uint16 - a negative float would silently
    # wrap to a huge uint16 rather than erroring. Both sharpen modes are
    # already bounded (stretch clamps to [local_lo, local_hi]), but the
    # guided filter is not (local linear-regression extrapolation can ring
    # past its input's range, like unsharp masking), so this stays.
    depthnorm_clipped = np.clip(depthnorm, 0.0, 1.0)
    quantized = np.stack(
        [(depthnorm_clipped * DEPTH_QUANT_LEVELS).round().astype(np.uint16)] * NUM_FRAMES, axis=0
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
    print(f"==> stage 1 (with circle, sharpen={SHARPEN_MODE}) done -> {splat_with}")
    splat_without = run_splat("no_circle", with_circle=False)
    print(f"==> stage 1 (no circle, ground-truth reference) done -> {splat_without}")

    # A/B the two re-hardening operators on the SAME scene, so the comb's
    # presence/absence is attributable to that one variable and nothing
    # else. Only the mask matters for this comparison (the comb is a
    # stage-1 artifact, fully visible before any inpainting), so this
    # extra run only needs its mask dumped, not a full stage-2 pass.
    ab_mode = "none" if SHARPEN_MODE == "stretch" else "stretch"
    splat_ab = run_splat(f"with_circle_{ab_mode}", with_circle=True, mode=ab_mode)
    _warp_ab, mask_ab, _disp_ab, _meta_ab = open_store(splat_ab, mode="r")
    cv2.imwrite(
        os.path.join(OUT_ROOT, f"splat_mask_{ab_mode}.png"), np.array(mask_ab[0:1])[0]
    )
    print(f"==> stage 1 A/B comparison (sharpen={ab_mode}) done -> {splat_ab}")

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
    print()
    print("THE COMPARISON THAT MATTERS (comb is a stage-1 artifact, visible")
    print("in the mask before any inpainting):")
    print(f"    {OUT_ROOT}/splat_mask.png              <- sharpen={SHARPEN_MODE}")
    print(f"    {OUT_ROOT}/splat_mask_{ab_mode}.png    <- sharpen={ab_mode}")
    print("Smooth crescent edges = position preserved. Staircase/comb edges")
    print("= boundary position quantized to the low-res grid.")
    print()
    print(f"also in {OUT_ROOT}/: with_circle/source.png, with_circle/depthnorm_pipeline.png,")
    print("    ground_truth.png, result_frame0.png, hole_diff.png")


if __name__ == "__main__":
    main()
