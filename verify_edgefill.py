"""Does _edge_threshold_fill actually SHARPEN a boundary, or does it DILATE
the foreground? Faithful numpy port of its exact rule, run on the harness's
own scenario: gradient background (depthnorm 0.25..0.75) + circle at 1.0,
with the same GLOBAL threshold production computes (min + 0.10*range).

The suspicion: the rule only sharpens symmetrically when the background is
BELOW the global threshold. When both sides of a transition are above it,
both sides take max-of-neighbors, which just grows the brighter side -
a dilation, not a sharpen, and one that MOVES the boundary.
"""
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, zoom

N, RATIO, R, SS = 1024, 4, 350.0, 4
CY = CX = N / 2.0
BG_LO, BG_HI, FG = 0.25, 0.75, 1.0
FRAC = 0.10


def scene_lowres():
    """Gradient background + circle, area-averaged onto the coarse grid."""
    n = N * SS
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    yy = (yy + 0.5) / SS
    xx = (xx + 0.5) / SS
    bg = BG_HI - (xx / (N - 1)) * (BG_HI - BG_LO)
    field = np.where((xx - CX) ** 2 + (yy - CY) ** 2 <= R * R, FG, bg).astype(np.float32)
    b = RATIO * SS
    return field.reshape(n // b, b, n // b, b).mean(axis=(1, 3))


def edge_threshold_fill(depth, threshold, n_iters):
    cur = depth.copy()
    for _ in range(n_iters):
        p = np.pad(cur, 1, mode="edge")
        nb = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        is_bg_nb = nb <= threshold
        cand = np.where(is_bg_nb, np.inf, nb)
        nmin = cand.min(axis=0)
        nmin = np.where(is_bg_nb.all(axis=0), cur, nmin)
        cur = np.where(cur <= threshold, nmin, nb.max(axis=0))
    return cur


def stretch(depth, radius, gain):
    s = 2 * radius + 1
    lo = minimum_filter(depth, size=s, mode="nearest")
    hi = maximum_filter(depth, size=s, mode="nearest")
    level = 0.5 * (lo + hi)
    return np.clip((depth - level) * gain + level, lo, hi)


low = scene_lowres()
up = zoom(low, RATIO, order=1)
threshold = float(low.min()) + FRAC * (float(low.max()) - float(low.min()))
print(f"global threshold = {threshold:.4f}")
print(f"background spans {low.min():.3f}..{BG_HI:.3f}, circle = {FG}")
print(f"fraction of the frame ABOVE that global threshold: "
      f"{(up > threshold).mean():.1%}\n")

rows = list(range(int(CY) - 300, int(CY) + 300))
dy = np.array(rows, dtype=np.float64) + 0.5 - CY
tx = CX + np.sqrt(np.maximum(R * R - dy * dy, 0.0))


def crossing(field, rows):
    """Where the field crosses the LOCAL midpoint between the circle and
    the background just outside it, scanning right from center."""
    out = []
    for r in rows:
        seg = field[r][int(CX):]
        bgv = seg[-50:].mean()
        half = 0.5 * (FG + bgv)
        idx = np.argmax(seg < half)
        if idx == 0:
            out.append(np.nan)
            continue
        a, b = seg[idx - 1], seg[idx]
        out.append(CX + idx - 1 + ((a - half) / (a - b) if a != b else 0.0))
    return np.array(out)


print(f"{'variant':<34} {'RMS err':>8} {'mean shift':>11} {'jitter':>9}")
print("-" * 66)
for name, field in [
    ("no sharpening (soft)", up),
    ("edge_fill iters=1 (global thresh)", edge_threshold_fill(up, threshold, 1)),
    ("edge_fill iters=3 (global thresh)", edge_threshold_fill(up, threshold, 3)),
    ("stretch r=6 gain=3", stretch(up, 6, 3.0)),
    ("stretch r=6 gain=6", stretch(up, 6, 6.0)),
]:
    cx = crossing(field, rows)
    ok = np.isfinite(cx)
    err = cx[ok] - tx[ok]
    print(f"{name:<34} {np.sqrt((err**2).mean()):8.3f} {err.mean():+11.3f} "
          f"{np.abs(np.diff(err, n=2)).mean():9.4f}")

print("\nmean shift > 0 means the circle's boundary moved OUTWARD (dilated).")
