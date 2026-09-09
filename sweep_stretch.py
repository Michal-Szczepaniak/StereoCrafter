"""Gain/radius sweep for _position_preserving_sharpen, plus a check of how
HARD the resulting edge actually is (the reason for sharpening at all -
a soft depth ramp warps to a smeared band of in-between disparities).
"""
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, zoom

N, RATIO, R, SS = 1024, 4, 350.0, 4
CY = CX = N / 2.0


def coverage_lowres():
    n = N * SS
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    yy = (yy + 0.5) / SS
    xx = (xx + 0.5) / SS
    hard = ((xx - CX) ** 2 + (yy - CY) ** 2 <= R * R).astype(np.float32)
    b = RATIO * SS
    return hard.reshape(n // b, b, n // b, b).mean(axis=(1, 3))


def stretch(depth, radius, gain):
    size = 2 * radius + 1
    lo = minimum_filter(depth, size=size, mode="nearest")
    hi = maximum_filter(depth, size=size, mode="nearest")
    level = 0.5 * (lo + hi)
    return np.clip((depth - level) * gain + level, lo, hi)


def crossing_x(field, rows):
    out = []
    for r in rows:
        seg = field[r][int(CX):]
        idx = np.argmax(seg < 0.5)
        if idx == 0:
            out.append(np.nan)
            continue
        a, b = seg[idx - 1], seg[idx]
        out.append(CX + idx - 1 + ((a - 0.5) / (a - b) if a != b else 0.0))
    return np.array(out)


def ramp_width(field, rows):
    """Mean number of full-res px between the 10% and 90% levels at the
    edge - i.e. how soft the transition still is after sharpening."""
    widths = []
    for r in rows:
        seg = field[r][int(CX):]
        hi_i = np.argmax(seg < 0.9)
        lo_i = np.argmax(seg < 0.1)
        if hi_i and lo_i > hi_i:
            widths.append(lo_i - hi_i)
    return float(np.mean(widths)) if widths else float("nan")


low = coverage_lowres()
up = zoom(low, RATIO, order=1)
rows = list(range(int(CY) - 300, int(CY) + 300))
dy = np.array(rows, dtype=np.float64) + 0.5 - CY
tx = CX + np.sqrt(np.maximum(R * R - dy * dy, 0.0))


def report(name, field):
    cx = crossing_x(field, rows)
    ok = np.isfinite(cx)
    err = cx[ok] - tx[ok]
    jitter = np.abs(np.diff(err, n=2)).mean()
    print(f"{name:<30} {np.sqrt((err**2).mean()):8.3f} {np.abs(err).max():8.3f} "
          f"{jitter:9.4f} {ramp_width(field, rows):9.2f}")


print(f"{'variant':<30} {'RMS err':>8} {'max err':>8} {'jitter':>9} {'edge width':>10}")
print("-" * 70)
report("no sharpening (soft)", up)
print()
for radius in [RATIO, RATIO + 2, 2 * RATIO, 3 * RATIO]:
    for gain in [3.0, 6.0, 12.0, 1e6]:
        g = "toggle(inf)" if gain > 1e5 else f"{gain:g}"
        report(f"r={radius:<3} gain={g:<11}", stretch(up, radius, gain))
    print()
