"""Synthetic parameter sweep for disparity-based inpainting source-exclusion
(experimental/disparity-source-exclusion branch) - no GPU/real footage
needed. Ground truth is known analytically, so this measures actual fill
ACCURACY, not just whether exclusion fired.

Scenario: a background disparity field that's a smooth horizontal gradient
(NOT flat - this is what exposed the "painting white over wood" bug: a
continuous depth gradient, not just a flat background), with a circular
"character" of much higher disparity placed in the middle. RGB content
mirrors the disparity value 1:1 (grayscale), so the correct fill at any
pixel is knowable exactly - it's just that pixel's true gradient value.
A thin hole ring sits just outside the circle (simulating the disocclusion
a real character causes at its silhouette).

For each (disp_exclude_margin, disp_bg_search_px) combination, this:
  1. Builds the gen_mask (hole | exclusion) via the real
     _disp_exclusion_mask from inpainting_inference.py.
  2. Runs the real cv2.inpaint(TELEA) call (same one _prefill_occlusion
     uses) over gen_mask.
  3. Measures mean absolute error between the filled hole pixels and their
     true (known) gradient value - this is the number that actually
     matters, not just "did exclusion trigger".
  4. Separately checks a control region on the far side of the gradient,
     away from the circle entirely, for false-positive exclusion - this is
     the wood failure mode in isolation (pure smooth gradient, no real
     foreground/background boundary anywhere nearby).

Run with the project's venv, e.g.:
    /mnt/transmission/stereocrafter/bin/python3 tune_disp_exclusion.py
"""
import cv2
import numpy as np

from inpainting_inference import _disp_exclusion_mask

H, W = 120, 240
CIRCLE_CENTER = (60, 120)  # (row, col)
CIRCLE_R = 18
HOLE_BAND = 6  # ring thickness just outside the circle
GRAD_MAX = 6.0  # background disparity range: 0..GRAD_MAX across the frame
CHAR_DISP = 20.0  # circle's disparity - much higher than any background value


def make_synthetic():
    xs = np.arange(W, dtype=np.float32)
    grad_row = xs / (W - 1) * GRAD_MAX
    disp = np.tile(grad_row, (H, 1))  # pure background gradient, no circle yet

    yy, xx = np.mgrid[0:H, 0:W]
    dist2 = (xx - CIRCLE_CENTER[1]) ** 2 + (yy - CIRCLE_CENTER[0]) ** 2
    circle_mask = dist2 <= CIRCLE_R ** 2
    hole_mask = (dist2 <= (CIRCLE_R + HOLE_BAND) ** 2) & (~circle_mask)

    true_content = np.clip(disp / GRAD_MAX * 255, 0, 255).astype(np.uint8)  # ground truth, BEFORE circle/hole

    disp_with_circle = disp.copy()
    disp_with_circle[circle_mask] = CHAR_DISP

    content_with_circle = true_content.copy()
    content_with_circle[circle_mask] = 255  # circle rendered as a distinct color too

    return disp_with_circle, content_with_circle, true_content, circle_mask, hole_mask


def run_one(disp, content_rgb_2d, hole_mask, margin, search_px):
    hole_bool = hole_mask[None]  # (1,H,W) - _disp_exclusion_mask expects a T dim
    disp_batched = disp[None]
    exclude = _disp_exclusion_mask(disp_batched, hole_bool, margin, search_px)[0]

    gen_mask = ((hole_mask | exclude).astype(np.uint8)) * 255
    content_bgr = cv2.cvtColor(content_rgb_2d, cv2.COLOR_GRAY2BGR)
    filled = cv2.inpaint(content_bgr, gen_mask, 5, cv2.INPAINT_TELEA)
    filled_gray = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)
    return filled_gray, exclude, gen_mask


def main():
    disp, content_with_circle, true_content, circle_mask, hole_mask = make_synthetic()

    print(f"hole pixels: {hole_mask.sum()}, circle pixels: {circle_mask.sum()}")
    print()
    print(f"{'margin':>7} {'search_px':>9} | {'hole MAE':>9} | {'false-excl (far control)':>24}")
    print("-" * 60)

    # Control region: a wide strip on the far side of the frame from the
    # circle, well outside its influence - pure gradient, no real
    # foreground/background boundary at all. Any exclusion firing here is
    # a false positive caused purely by the gradient itself (the wood bug).
    control_col_lo, control_col_hi = W - 40, W - 5
    control_region = np.zeros((H, W), dtype=bool)
    control_region[:, control_col_lo:control_col_hi] = True

    for margin in [0.25, 0.5, 1.0, 2.0]:
        for search_px in [5, 10, 15, 25]:
            filled_gray, exclude, gen_mask = run_one(disp, content_with_circle, hole_mask, margin, search_px)

            true_hole_vals = true_content[hole_mask].astype(np.float32)
            filled_hole_vals = filled_gray[hole_mask].astype(np.float32)
            mae = np.abs(true_hole_vals - filled_hole_vals).mean()

            false_excl_count = int((exclude & control_region).sum())
            flag = "  <-- FALSE EXCLUSION" if false_excl_count > 0 else ""
            print(f"{margin:7.2f} {search_px:9d} | {mae:9.2f} | {false_excl_count:24d}{flag}")

    print()
    print("MAE is on a 0-255 grayscale scale (true background gradient span is 0-255).")
    print("Lower MAE = better fill accuracy. false-excl > 0 = the wood bug is reproduced")
    print("at that setting (pure gradient region getting excluded with nothing nearby to justify it).")


if __name__ == "__main__":
    main()
