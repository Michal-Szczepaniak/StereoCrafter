"""Standalone smoke test for the disparity-based inpainting source-exclusion
change (experimental/disparity-source-exclusion branch) - no real footage
needed, all synthetic tensors/arrays. Run with the project's own venv, e.g.:

    /mnt/transmission/stereocrafter/bin/python3 test_disp_exclusion.py

Covers:
  1. ForwardWarpStereo(extra=...) warps a second channel (disp) through the
     exact same splat as the image, and is backward-compatible (no `extra`
     still returns the original 2-tuple with identical res/occlu_map).
  2. The uint16 quantize/dequantize round trip used to persist disp in
     splat_store matches within a tight error bound.
  3. _disp_exclusion_mask correctly flags character-adjacent (high-disp)
     valid pixels as excluded, leaves far background (low-disp) pixels
     alone, and never flags hole pixels themselves.
"""
import numpy as np
import torch

from depth_splatting_inference import ForwardWarpStereo
from inpainting_inference import _disp_exclusion_mask
from splat_store import DEPTH_QUANT_LEVELS

# --- 1. ForwardWarpStereo extra-channel warp ---
torch.manual_seed(0)
B, C, H, W = 1, 3, 8, 16
im = torch.rand(B, C, H, W)
disp = torch.zeros(B, 1, H, W)
disp[:, :, :, :8] = 5.0   # "character" - high disp, shifts right
disp[:, :, :, 8:] = 0.0   # "background" - no shift

stereo = ForwardWarpStereo(occlu_map=True, use_zbuffer_splat=False)
right, occl, right_disp = stereo(im, disp, extra=disp)
assert right.shape == im.shape
assert occl.shape == (B, 1, H, W)
assert right_disp.shape == (B, 1, H, W)
print("ForwardWarpStereo extra-channel warp: shapes OK")

# Backward-compat: no `extra` still returns a 2-tuple.
right2, occl2 = stereo(im, disp)
assert torch.allclose(right2, right)
assert torch.allclose(occl2, occl)
print("ForwardWarpStereo backward-compat (no extra): OK, identical to extra-channel path's res/occl")

# --- 2. Quantize/dequantize round trip (matches depth_splatting_inference.py's scheme) ---
max_disp = 20.0
real = right_disp.clamp(-max_disp, max_disp)
norm = (real + max_disp) / (2 * max_disp) * DEPTH_QUANT_LEVELS
u16 = norm.round().clamp(0, DEPTH_QUANT_LEVELS).to(torch.int32).numpy().astype(np.uint16)
back = u16.astype(np.float32) / DEPTH_QUANT_LEVELS * (2.0 * max_disp) - max_disp
err = np.abs(back - real.numpy())
print(f"quantize round-trip max abs error: {err.max():.6f} (expect << 1)")
assert err.max() < 1e-2

# --- 3. _disp_exclusion_mask: synthetic hole with character on one side ---
T, Hh, Ww = 1, 20, 40
disp_np = np.zeros((T, Hh, Ww), dtype=np.float32)
disp_np[:, :, :15] = 10.0   # character (high disp)
disp_np[:, :, 15:] = 1.0    # background (low disp)
hole_np = np.zeros((T, Hh, Ww), dtype=bool)
hole_np[:, :, 15:20] = True  # the hole sits just past the character, in "background"

excl = _disp_exclusion_mask(disp_np, hole_np, margin=1.0, search_px=10)
# character pixels (valid, high disp, near the hole) should be excluded
assert excl[:, :, 5:15].all(), "expected character-adjacent pixels to be excluded"
# far background pixels (valid, low disp) should NOT be excluded
assert not excl[:, :, 25:].any(), "expected far background pixels to stay un-excluded"
# hole pixels themselves are never marked (they're not valid sources anyway)
assert not excl[hole_np].any()
print("_disp_exclusion_mask: character excluded, background kept, holes untouched - OK")

print("ALL CHECKS PASSED")
