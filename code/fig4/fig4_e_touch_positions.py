"""
fig4_e_touch_positions.py

Figure 4, panel e: touch-position density maps for one example animal per
group (WT, SH, GT).

Data. xy_type1.mat through xy_type5.mat, fields x_touches_* and y_touches_*.
The example animals are rows 269 (WT), 112 (SH) and 110 (GT), the same
animals shown in panel g; all five session types are concatenated. The
script derives two coordinates per touch: X = |0.5 - x_touches| (horizontal
distance from the screen center) and Y = 1 - y_touches (vertical touch
position, flipped so that larger values are higher on the screen).

Display. Per group, a 2-D histogram of (Y, X) on a 50 x 50 grid of
0.02-wide bins covering the unit square, smoothed with a Gaussian filter
(sigma = 0.05 in data units, i.e. 2.5 bins; radius 5 bins; edge values
replicated), normalized to its own maximum and shown with the outer ring of
bins removed, since those bins are dominated by the smoothing edge. Seven
evenly spaced contour lines run from dark (high density) to light (low
density). Each group uses its own white to group-color to black colormap,
interpolated linearly in RGB. The narrow panel on the right of each map is
the marginal distribution of touch position: row sums of the raw, unsmoothed
counts, drawn in the mid color of that group's colormap. Its count axis is
left unlabeled because the tick labels would overlap the neighboring map.

The script draws a single colorbar for the figure, using the GT colormap;
the color scale is common (0-1, normalized density) but the hue is per group.

The panel is descriptive: no statistical test is applied.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_e_touch_positions")

WT_SEL, SH_SEL, GT_SEL = 269 - 1, 112 - 1, 110 - 1

BIN_WIDTH = 0.02
# Touch position is stored normalized to [0, 1]. The published panels plot
# it on a 0 to 10 screen scale, ticked 0, 2, 4, 6, 8, so the y axis alone is
# multiplied by this factor at draw time. The density, its bins and every
# statistic stay in normalized units.
Y_SCALE = 10.0
SMOOTH_SIGMA = 0.05
N_LEVELS = 7
MID_LIGHTEN = 0.30


def _load_group_xy(data_dir, row, xkey, ykey):
    X, Y = [], []
    for t in range(1, 6):
        S = load_struct(os.path.join(data_dir, f"xy_type{t}.mat"))
        x = np.asarray(S[xkey], dtype=float)[row]
        y = np.asarray(S[ykey], dtype=float)[row]
        X.append(np.abs(0.5 - x))
        Y.append(1 - y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X[~np.isnan(X)], Y[~np.isnan(Y)]


def _binned_smooth_2d(X, Y, edges, sigma_px):
    """2-D histogram of the two coordinates over `edges`, then a Gaussian
    filter of width `sigma_px` bins. Returns (raw counts, smoothed counts).

    Callers pass (Y, X) in that order, so the first argument is the
    touch-position coordinate and the second is the distance-from-center
    coordinate; this matches the axis convention of the rendered map.
    """
    Nraw, _, _ = np.histogram2d(X, Y, bins=[edges, edges])
    radius = int(np.ceil(2 * sigma_px))
    Nsm = gaussian_filter(Nraw, sigma=sigma_px, radius=radius, mode="nearest")
    return Nraw, Nsm


def _log_tone_map(X, alpha=0.0):
    X = np.clip(X, 0, 1)
    if alpha <= 0:
        return X
    return np.log1p(alpha * X) / np.log1p(alpha)


def _white_mid_black(mid_color, n=256, gamma_val=1.0, lighten=0.0, STY=None):
    """Colormap running white -> mid_color -> black, linearly interpolated in
    RGB with control points at t = 0, 0.5 and 1."""
    if isinstance(mid_color, str):
        if mid_color.lower() in ("gray", "grey"):
            target = np.asarray(STY.clr.gray_ref, dtype=float)
        else:
            raise ValueError("Unsupported color name; use 'gray' or an RGB triplet.")
    else:
        target = np.asarray(mid_color, dtype=float).reshape(3)
        target = (1 - lighten) * target + lighten * np.ones(3)
        target = np.clip(target, 0, 1)

    t = np.array([0.0, 0.5, 1.0])
    anchors = np.vstack([np.ones(3), target, np.zeros(3)])   # white, mid, black

    x = np.linspace(0, 1, n)
    xg = x ** gamma_val

    cmap = np.column_stack([
        np.interp(xg, t, anchors[:, 0]),
        np.interp(xg, t, anchors[:, 1]),
        np.interp(xg, t, anchors[:, 2]),
    ])
    return np.clip(cmap, 0, 1)


def _draw_panel(fig, rect, Nraw, Nsm, title_str, centers, cmap, STY):
    """Draw one group's density map, contours and marginal histogram within
    `rect`, a [left, bottom, width, height] tile in figure-fraction units."""
    left, bottom, width, height = rect
    gap = 0.02 * width
    hist_w = 0.24 * width
    main_w = width - hist_w - gap

    ax_main = fig.add_axes([left, bottom, main_w, height])
    ax_hist = fig.add_axes([left + main_w + gap, bottom, hist_w, height])

    max_val = np.max(Nsm) if Nsm.size else 0.0
    Nsm_norm = Nsm / max(max_val, np.finfo(float).eps)
    Nsm_tone = _log_tone_map(Nsm_norm, 0.0)

    Zdisp = np.flipud(np.rot90(Nsm_tone.T, 1))
    CrawDisp = np.flipud(np.rot90(Nraw.T, 1))

    if centers.size >= 3:
        keep = slice(1, -1)
        centers_plot = centers[keep]
        Zdisp_plot = Zdisp[keep, keep]
        CrawDisp_plot = CrawDisp[keep, keep]
    else:
        centers_plot, Zdisp_plot, CrawDisp_plot = centers, Zdisp, CrawDisp

    lcmap = ListedColormap(cmap)
    extent = [centers_plot[0], centers_plot[-1],
              centers_plot[0] * Y_SCALE, centers_plot[-1] * Y_SCALE]
    ax_main.imshow(Zdisp_plot, extent=extent, origin="lower", cmap=lcmap,
                    vmin=0, vmax=1, aspect="auto", interpolation="nearest")

    rel_levels = np.linspace(0, 1, N_LEVELS)
    for lvl in rel_levels:
        gray_val = 1 - lvl
        line_col = (gray_val, gray_val, gray_val)
        try:
            ax_main.contour(centers_plot, centers_plot * Y_SCALE, Zdisp_plot, levels=[lvl],
                            colors=[line_col], linewidths=0.9)
        except Exception:
            pass

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, Y_SCALE)
    # The published panel leaves this axis unlabeled; "Density" is the
    # color bar, as in Extended Data Fig. 7c.
    ax_main.set_ylabel("Touch position", fontname=STY.font.name, fontsize=STY.font.lab)
    ax_main.set_title(title_str, fontname=STY.font.name, fontsize=STY.font.title, fontweight="bold")
    gt_axes(ax_main, STY)
    for spine in ax_main.spines.values():
        spine.set_visible(True)   # full frame around the density map

    row_counts = CrawDisp_plot.sum(axis=1)
    y_centers_h = centers_plot
    hist_color = cmap[round(cmap.shape[0] / 2)]

    dy = np.diff(centers_plot).mean() if centers_plot.size > 1 else BIN_WIDTH
    ax_hist.barh(y_centers_h * Y_SCALE, row_counts, height=dy * Y_SCALE,
                 color=hist_color, edgecolor="none")
    ax_hist.set_ylim(0, Y_SCALE)
    ax_hist.set_yticks([])
    mx = row_counts.max() if row_counts.size and row_counts.max() > 0 else 1
    ax_hist.set_xlim(0, mx * 1.05)
    # The marginal is shown as a distribution shape only; its count axis is
    # left unticked because the labels would run into the adjacent map.
    ax_hist.set_xticks([])
    gt_axes(ax_hist, STY, matlab_ticks=False)
    for spine in ax_hist.spines.values():
        spine.set_visible(True)

    return ax_main, lcmap


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    Xw, Yw = _load_group_xy(data_dir, WT_SEL, "x_touches_wt", "y_touches_wt")
    Xs, Ys = _load_group_xy(data_dir, SH_SEL, "x_touches_sh", "y_touches_sh")
    Xg, Yg = _load_group_xy(data_dir, GT_SEL, "x_touches_gt", "y_touches_gt")

    edges = np.arange(0, 1 + 1e-9, BIN_WIDTH)
    centers = edges[:-1] + BIN_WIDTH / 2
    sigma_px = max(0.5, SMOOTH_SIGMA / BIN_WIDTH)

    # Arguments are (touch position, distance from center); see _binned_smooth_2d.
    Nw_raw, Nw_sm = _binned_smooth_2d(Yw, Xw, edges, sigma_px)
    Ns_raw, Ns_sm = _binned_smooth_2d(Ys, Xs, edges, sigma_px)
    Ng_raw, Ng_sm = _binned_smooth_2d(Yg, Xg, edges, sigma_px)

    cmap_wt = _white_mid_black("gray", 256, 1.0, 0.0, STY)
    cmap_sh = _white_mid_black(np.array([230, 35, 45]) / 255.0, 256, 1.0, MID_LIGHTEN, STY)
    cmap_gt = _white_mid_black(np.array([60, 50, 222]) / 255.0, 256, 1.0, MID_LIGHTEN, STY)

    fig = plt.figure(figsize=(6.070, 1.790), facecolor="w")

    outer_left, outer_right = 0.075, 0.90   # wider left so "Touch position" y-label clears the edge
    tile_gap = 0.018
    tile_w = (outer_right - outer_left - 2 * tile_gap) / 3
    bottom, height = 0.20, 0.62

    panels = [
        (Nw_raw, Nw_sm, "WT", cmap_wt),
        (Ns_raw, Ns_sm, "Shank3", cmap_sh),
        (Ng_raw, Ng_sm, "Gene therapy", cmap_gt),
    ]

    last_cmap = None
    for i, (Nraw, Nsm, title, cmap) in enumerate(panels):
        rect = [outer_left + i * (tile_w + tile_gap), bottom, tile_w, height]
        _draw_panel(fig, rect, Nraw, Nsm, title, centers, cmap, STY)
        last_cmap = cmap

    # One colorbar for the figure, drawn with the GT colormap. The scale
    # (0-1 normalized density) is shared; only the hue is per group.
    cax = fig.add_axes([outer_right + 0.03, bottom, 0.025, height])
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    mpl.colorbar.ColorbarBase(cax, cmap=ListedColormap(last_cmap), norm=norm)
    cax.set_ylabel("Density", fontname=STY.font.name, fontsize=STY.font.lab)
    cax.tick_params(labelsize=STY.font.tick)
    for lbl in cax.get_yticklabels():
        lbl.set_fontname(STY.font.name)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig4_e_touch_positions", out_dir)

    print(f"n touches: WT={Xw.size}, SH={Xs.size}, GT={Xg.size}")
    return fig


if __name__ == "__main__":
    main()
