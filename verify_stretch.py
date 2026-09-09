"""Offline check of the position-preserving-sharpen hypothesis, using only
numpy/scipy (no cv2/torch needed). Integer 4x ratio so the coarse-grid
block mean is an EXACT coverage average (what INTER_AREA computes).

Measures: for each row, where does the field cross 0.5 on the circle's
right edge, versus where the true analytic circle boundary is? A staircase
shows up as crossings snapping to multiples of the coarse pixel size.
"""
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, zoom

N = 1024
RATIO = 4           # coarse grid is 256x256
R = 350.0
CY = CX = N / 2.0
SS = 4              # supersampling for an exact coverage reference


def coverage_lowres():
    """Exact per-coarse-cell coverage of the circle = what INTER_AREA of a
    hard full-res mask gives (and what a depth model's soft band encodes)."""
    n = N * SS
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    yy = (yy + 0.5) / SS
    xx = (xx + 0.5) / SS
    hard = ((xx - CX) ** 2 + (yy - CY) ** 2 <= R * R).astype(np.float32)
    block = RATIO * SS
    return hard.reshape(n // block, block, n // block, block).mean(axis=(1, 3))


def stretch(depth, radius, gain):
    size = 2 * radius + 1
    lo = minimum_filter(depth, size=size, mode="nearest")
    hi = maximum_filter(depth, size=size, mode="nearest")
    level = 0.5 * (lo + hi)
    return np.clip((depth - level) * gain + level, lo, hi)


def crossing_x(field, rows):
    """Sub-pixel x where field crosses 0.5, scanning right from center."""
    out = []
    for r in rows:
        row = field[r]
        seg = row[int(CX):]
        idx = np.argmax(seg < 0.5)
        if idx == 0:
            out.append(np.nan)
            continue
        a, b = seg[idx - 1], seg[idx]
        frac = (a - 0.5) / (a - b) if a != b else 0.0
        out.append(CX + idx - 1 + frac)
    return np.array(out)


def true_x(rows):
    dy = np.array(rows, dtype=np.float64) + 0.5 - CY
    return CX + np.sqrt(np.maximum(R * R - dy * dy, 0.0))


low = coverage_lowres()
up = zoom(low, RATIO, order=1)                      # bilinear upsample
sharp = stretch(up, radius=RATIO + 2, gain=6.0)

# "harden on the coarse grid, then upsample" - the essential property of
# _edge_threshold_fill (snap to whole coarse pixels), both upsample modes.
hard_low = (low >= 0.5).astype(np.float32)
hard_nearest = zoom(hard_low, RATIO, order=0)
hard_bilinear = zoom(hard_low, RATIO, order=1)

rows = range(int(CY) - 300, int(CY) + 300)
tx = true_x(rows)

print(f"coarse pixel = {RATIO} full-res px; circle R={R:.0f}\n")
print(f"{'variant':<34} {'RMS err':>9} {'max err':>9} {'row-to-row jitter':>18}")
print("-" * 74)
for name, field in [
    ("hardened on coarse grid, nearest", hard_nearest),
    ("hardened on coarse grid, bilinear", hard_bilinear),
    ("bilinear upsample, no sharpening", up),
    ("bilinear upsample + stretch", sharp),
]:
    cx = crossing_x(field, rows)
    ok = np.isfinite(cx)
    err = cx[ok] - tx[ok]
    # jitter = how much the crossing deviates from a locally smooth curve;
    # a staircase has large second differences, a smooth arc has tiny ones.
    jitter = np.abs(np.diff(err[ok[ok]], n=2)).mean() if ok.sum() > 3 else np.nan
    print(f"{name:<34} {np.sqrt((err**2).mean()):9.3f} {np.abs(err).max():9.3f} {jitter:18.4f}")
