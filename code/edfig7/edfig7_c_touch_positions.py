"""Extended Data Figure 7, panel c: touch-position density maps for wild type,
Shank3 and gene therapy.

Data: `xy_type1.mat`, `xy_type2.mat` and `xy_type3.mat`, variables
`x_touches_*` and `y_touches_*`. The script pools touches over a band of
animal rows per group and concatenates them across session types 1, 2 and 3.
The bands are rows 201-500 for wild type (300 rows) and rows 101-120 for
Shank3 and for gene therapy (20 rows each), 1-based.

Derived coordinates: X = abs(0.5 - x_touches), the distance from the horizontal
center of the screen, and Y = 1 - y_touches, the vertical touch position with
the axis flipped. X and Y are NaN-filtered independently; their missing-value
patterns coincide, so the two vectors stay paired and equal in length.

Density: a 2-D histogram of (Y, X) over 50 x 50 bins of width BIN_WIDTH = 0.02
spanning [0, 1] x [0, 1], smoothed with a Gaussian kernel of
sigma = max(0.5, SMOOTH_SIGMA / BIN_WIDTH) = 2.5 bins and support
2 * ceil(2 * sigma) + 1 = 11 bins, using zero padding at the edges. The result
is normalized to its own peak and reoriented for display, and the outermost
ring of bins is dropped to remove the smoothing edge artefact, leaving 48 x 48
bins.

Display: each group uses its own white to mid-color to black colormap built by
`_white_mid_black`, a per-channel linear RGB interpolation. The script
overlays seven evenly spaced density contours, drawn darker at higher levels.
To the right of each map sits a bar histogram of the raw counts summed along
each displayed Y bin. All four spines are shown on both the map and its side
histogram, so the heatmap keeps its frame. A single colorbar to the right of
the figure shows the colormap of the last panel drawn (gene therapy).

The panel is descriptive; no statistical test is applied.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import convolve2d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_e_touch_positions")

# Animal-row bands per group, converted from 1-based to 0-based slices
WT_SEL = slice(201 - 1, 500)
SH_SEL = slice(101 - 1, 120)
GT_SEL = slice(101 - 1, 120)
TYPES = (1, 2, 3)

BIN_WIDTH = 0.02
# Touch position is stored normalized to [0, 1]. The published panels plot
# it on a 0 to 10 screen scale, ticked 0, 2, 4, 6, 8, so the y axis alone is
# multiplied by this factor at draw time. The density, its bins and every
# statistic stay in normalized units.
Y_SCALE = 10.0
SMOOTH_SIGMA = 0.05
OVERLAY_POINTS = False
N_LEVELS = 7
CMAP_RES = 256
CMAP_GAMMA = 1.0
LOG_ALPHA = 0.0
MID_LIGHTEN = 0.30
MID_SH = np.array([230, 35, 45]) / 255.0
MID_GT = np.array([60, 50, 222]) / 255.0


def _load_group_xy(data_dir, row_slice, xkey, ykey):
    Xp, Yp = [], []
    for t in TYPES:
        S = load_struct(os.path.join(data_dir, f"xy_type{t}.mat"))
        x = np.asarray(S[xkey], dtype=float)[row_slice]
        y = np.asarray(S[ykey], dtype=float)[row_slice]
        Xp.append(np.abs(0.5 - x))
        Yp.append(1 - y)
    X = np.concatenate(Xp, axis=1)
    Y = np.concatenate(Yp, axis=1)
    X = X[~np.isnan(X)]
    Y = Y[~np.isnan(Y)]
    if X.size != Y.size:
        raise ValueError("X/Y NaN patterns diverge; the two vectors are no longer paired")
    return X, Y


def _fspecial_gaussian(sz, sigma):
    """Normalized 2-D Gaussian kernel of side `sz` and width `sigma`, summing
    to 1."""
    ax = np.arange(sz) - (sz - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    h[h < np.finfo(float).eps * h.max()] = 0
    s = h.sum()
    if s != 0:
        h = h / s
    return h


def _binned_smooth_2d(X, Y, edges, sigma_px):
    """2-D histogram over `edges` plus Gaussian smoothing of width `sigma_px`
    bins, with zero padding at the edges. Returns (raw, smoothed)."""
    Nraw, _, _ = np.histogram2d(X, Y, bins=[edges, edges])
    sz = 2 * int(np.ceil(2 * sigma_px)) + 1
    g = _fspecial_gaussian(sz, sigma_px)
    Nsm = convolve2d(Nraw, g, mode="same", boundary="fill", fillvalue=0)
    return Nraw, Nsm


def _log_tone_map(X, alpha):
    X = np.clip(X, 0, 1)
    if alpha <= 0:
        return X
    return np.log1p(alpha * X) / np.log1p(alpha)


def _white_mid_black(mid_color, n, gamma_val, lighten, gray_ref):
    """Colormap running white to `mid_color` to black by per-channel linear RGB
    interpolation. `lighten` blends the mid color toward white, `gamma_val`
    warps the position of the mid color, and `n` is the number of levels."""
    if isinstance(mid_color, str):
        if mid_color.lower() in ("gray", "grey"):
            target = np.asarray(gray_ref, dtype=float)
        else:
            raise ValueError("Unsupported color name. Use 'gray' or supply an RGB triplet.")
    else:
        target = np.asarray(mid_color, dtype=float).reshape(3)
        target = (1 - lighten) * target + lighten * np.ones(3)
        target = np.clip(target, 0, 1)

    t = np.array([0.0, 0.5, 1.0])
    Rk = np.array([1.0, target[0], 0.0])
    Gk = np.array([1.0, target[1], 0.0])
    Bk = np.array([1.0, target[2], 0.0])

    x = np.linspace(0, 1, n)
    xg = x ** gamma_val

    R = np.interp(xg, t, Rk)
    G = np.interp(xg, t, Gk)
    B = np.interp(xg, t, Bk)

    return np.clip(np.column_stack([R, G, B]), 0, 1)


def _draw_panel(ax_main, ax_hist, Nraw, Nsm, title_str, centers, rel_levels,
                 cmap_rgb, log_alpha, STY):
    maxVal = np.max(Nsm)
    Nsm_norm = Nsm / max(maxVal, np.finfo(float).eps)
    Nsm_tone = _log_tone_map(Nsm_norm, log_alpha)

    Zdisp = np.flipud(np.rot90(Nsm_tone.T, 1))
    CrawDisp = np.flipud(np.rot90(Nraw.T, 1))

    if centers.size >= 3:
        keep = slice(1, -1)
        centers_plot = centers[keep]
        Zdisp_plot = Zdisp[keep, keep]
        CrawDisp_plot = CrawDisp[keep, keep]
    else:
        centers_plot = centers
        Zdisp_plot = Zdisp
        CrawDisp_plot = CrawDisp

    cmap = mcolors.ListedColormap(cmap_rgb)
    lo, hi = centers_plot[0], centers_plot[-1]
    ax_main.imshow(Zdisp_plot, origin="lower",
                   extent=[lo, hi, lo * Y_SCALE, hi * Y_SCALE],
                   cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")

    for lvl in rel_levels:
        grayVal = 1 - lvl
        lineCol = (grayVal, grayVal, grayVal)
        ax_main.contour(centers_plot, centers_plot * Y_SCALE, Zdisp_plot, levels=[lvl],
                        colors=[lineCol], linewidths=0.9)

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, Y_SCALE)
    # The published panel leaves this axis unlabeled: "Touch position" on
    # y and the "Center" bracket carry the reading, and "Density" belongs
    # to the color bar.
    ax_main.set_ylabel("Touch position", fontname=STY.font.name, fontsize=STY.font.lab)
    ax_main.set_title(title_str, fontname=STY.font.name, fontweight="bold",
                      fontsize=STY.font.title)
    gt_axes(ax_main, STY)
    ax_main.spines["top"].set_visible(True)      # the heatmap keeps its full frame
    ax_main.spines["right"].set_visible(True)
    for side in ("top", "right"):
        ax_main.spines[side].set_linewidth(STY.lw.axis)
        ax_main.spines[side].set_color(tuple(STY.clr.axis))

    # ---- side histogram: one bar per displayed Y bin, edge bins excluded --
    rowCounts = CrawDisp_plot.sum(axis=1)
    yCentersH = centers_plot
    collect = rowCounts

    histColor = tuple(cmap_rgb[round(cmap_rgb.shape[0] / 2) - 1])
    ax_hist.barh(yCentersH[:collect.size] * Y_SCALE, collect,
                 height=BIN_WIDTH * Y_SCALE, color=histColor, edgecolor="none")
    gt_axes(ax_hist, STY)
    ax_hist.spines["top"].set_visible(True)
    ax_hist.spines["right"].set_visible(True)
    for side in ("top", "right"):
        ax_hist.spines[side].set_linewidth(STY.lw.axis)
        ax_hist.spines[side].set_color(tuple(STY.clr.axis))
    ax_hist.set_ylim(0, Y_SCALE)
    ax_hist.set_yticks([])
    mx = collect.max() if collect.size else 0
    if mx <= 0:
        mx = 1
    ax_hist.set_xlim(0, mx * 1.05)


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    Xw, Yw = _load_group_xy(data_dir, WT_SEL, "x_touches_wt", "y_touches_wt")
    Xs, Ys = _load_group_xy(data_dir, SH_SEL, "x_touches_sh", "y_touches_sh")
    Xg, Yg = _load_group_xy(data_dir, GT_SEL, "x_touches_gt", "y_touches_gt")
    print(f"wt n_touches={Xw.size}  sh n_touches={Xs.size}  gt n_touches={Xg.size}")

    edges = np.arange(0, 1 + 1e-9, BIN_WIDTH)
    centers = edges[:-1] + BIN_WIDTH / 2.0
    sigma_px = max(0.5, SMOOTH_SIGMA / BIN_WIDTH)

    # Y occupies the first histogram axis and X the second
    Nw_raw, Nw_sm = _binned_smooth_2d(Yw, Xw, edges, sigma_px)
    Ns_raw, Ns_sm = _binned_smooth_2d(Ys, Xs, edges, sigma_px)
    Ng_raw, Ng_sm = _binned_smooth_2d(Yg, Xg, edges, sigma_px)

    rel_levels = np.linspace(0, 1, N_LEVELS)

    cmap_wt = _white_mid_black("gray", CMAP_RES, CMAP_GAMMA, 0.0, STY.clr.gray_ref)
    cmap_sh = _white_mid_black(MID_SH, CMAP_RES, CMAP_GAMMA, MID_LIGHTEN, STY.clr.gray_ref)
    cmap_gt = _white_mid_black(MID_GT, CMAP_RES, CMAP_GAMMA, MID_LIGHTEN, STY.clr.gray_ref)

    # ---- figure: three tiles plus a shared colorbar -----------------------
    fig = plt.figure(figsize=(9.6, 3.4), facecolor="w")
    left, right_cbar = 0.045, 0.10
    bottom, top = 0.16, 0.84
    inter_gap = 0.022
    n_tiles = 3
    avail_w = 1 - left - right_cbar
    tile_w = (avail_w - (n_tiles - 1) * inter_gap) / n_tiles
    tile_h = top - bottom

    panels = [
        (Nw_raw, Nw_sm, "WT", cmap_wt),
        (Ns_raw, Ns_sm, "Shank3", cmap_sh),
        (Ng_raw, Ng_sm, "GT", cmap_gt),
    ]

    ax_main_last, cmap_last = None, None
    for i, (Nraw, Nsm, title_str, cmap_rgb) in enumerate(panels):
        tile_left = left + i * (tile_w + inter_gap)
        gap = 0.02 * tile_w
        histW = 0.24 * tile_w
        mainW = tile_w - histW - gap
        ax_main = fig.add_axes([tile_left, bottom, mainW, tile_h])
        ax_hist = fig.add_axes([tile_left + mainW + gap, bottom, histW, tile_h])
        _draw_panel(ax_main, ax_hist, Nraw, Nsm, title_str, centers, rel_levels,
                    cmap_rgb, LOG_ALPHA, STY)
        ax_main_last, cmap_last = ax_main, cmap_rgb

    # ---- shared colorbar on the right, showing the last panel's colormap
    # (gene therapy); each panel has its own colormap.
    cax = fig.add_axes([1 - right_cbar + 0.02, bottom, 0.018, tile_h])
    sm = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap(cmap_last),
                                norm=mcolors.Normalize(vmin=0, vmax=1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Density", fontname=STY.font.name, fontsize=STY.font.tick)
    cb.ax.tick_params(labelsize=STY.font.tick)
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontname(STY.font.name)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig7_c_touch_positions", out_dir)

    return fig


if __name__ == "__main__":
    main()
