"""Shared pieces for Extended Data Figure 4, panels c and d.

Both panels draw the same per-group, per-category home-cage occupancy
densities; panel c overlays the five categories in one axes, panel d gives
each its own. This module holds what they share: the category colors, the
smoothing that turns a coarse histogram into the displayed image, and the
arena outline.

The panels read the densities from state_density_by_group.npz, fifteen
100 x 100 histograms named <group>_<category>. They are a reduction of
xy_control_data.mat, which is 6.2 GB: every valid home-cage tracking sample
of each group is bucketed into one of 80 fine states, those are pooled into
the five categories below, and the result is histogrammed over the arena.
Shipping the 270 KB reduction rather than the source is the same choice the
repository makes elsewhere, for instance in the state-usage arrays.

The reduction uses the full valid population. The original implementation
drew a 10,000,000-sample random resample with replacement first; that
resample approximates the population it is drawn from, and the result here
is only ever a coarse smoothed image with no statistic read off it, so the
two agree up to sampling noise rather than exactly.
"""

import numpy as np

# Category order and color, left to right as the published panel d runs.
# The names are the published titles; the colors are the originals.
COLOR_LAYERS = [
    ("Climbing",      np.array([102,  98, 215]) / 255.0),
    ("Active",        np.array([168,  95, 186]) / 255.0),
    ("Alone",         np.array([240,  92, 155]) / 255.0),
    ("Social active", np.array([251, 141, 123]) / 255.0),
    ("Social resting", np.array([254, 193,  93]) / 255.0),
]
CACHE_KEYS = ["blue", "purple", "red", "orange", "yellow"]

GROUPS = [("wt", "WT"), ("sh", "SH"), ("gt", "GT")]
ARENA_XY = ([50, 750, 750, 50, 50], [50, 50, 750, 750, 50])
EXTENT = (0, 800, 0, 800)


def smoothed_alpha(hist_coarse, zoom=8):
    """Turn a 100 x 100 histogram into the 800 x 800 transparency image.

    The coarse histogram is resampled up with a cubic spline, then scaled so
    that mean + 3 standard deviations maps to fully opaque and anything above
    it clips. That is the original's normalization, and it is what gives
    every category a comparable dynamic range regardless of how much time
    the group spent in it.
    """
    from scipy.ndimage import zoom as _zoom

    fine = _zoom(np.asarray(hist_coarse, dtype=float), zoom, order=3)
    fine[fine < 0] = 0.0
    fine = fine / 1000.0
    denom = np.nanmean(fine) + 3.0 * (np.nanstd(fine, ddof=1) if fine.size > 1 else 0.0)
    if denom <= 0:
        return np.zeros_like(fine)
    return np.clip(fine / denom, 0.0, 1.0)


def layer_rgba(hist_coarse, rgb):
    """A constant-color image whose transparency carries the density."""
    alpha = smoothed_alpha(hist_coarse)
    rgba = np.zeros(alpha.shape + (4,), dtype=float)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
    rgba[..., 3] = alpha
    return rgba


def draw_arena(ax):
    """The arena boundary, and the axes settings both panels share."""
    ax.plot(ARENA_XY[0], ARENA_XY[1], "-", color="k", linewidth=0.5, zorder=3)
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 800)
    ax.set_aspect("equal")
    ax.axis("off")
