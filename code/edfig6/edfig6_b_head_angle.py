"""Extended Data Figure 6, panel b: head angle relative to the window.

This script produces four figures.

  p1-p3  One polar histogram per group (WT, SH, GT), from
         `hist_example_wt.mat`, `_sh.mat` and `_gt.mat`. Each file holds
         `all_theta_abs_cell` (absolute head angles in radians on [0, pi]),
         `edges` (19 edges, giving 18 bins) and `bin_colors` (18 x 3, one
         color per bin). Each bin is drawn as one bar at its own count in
         its own color. Zero is at the top, angle increases
         counter-clockwise, the angular axis spans 0 to 180 degrees, and the
         radial axis is fixed to [0, 8000] counts. Because the radial axis is
         a raw count rather than a normalized density, the three groups are
         individually legible but not directly comparable bin for bin.

  p4     `theta.mat`, in a 1 x 3 layout: for each of WT, SH and GT in the
         "init" condition, a paired jittered scatter of `toward_*_init`
         against `away_*_init` (1000 slots each, dropped pairwise where
         either is not finite), with a line joining each pair. At most
         PLOT_N_MAX = 250 pairs are subsampled for the lines and points; the
         median and quartile bars are computed from the full, unsubsampled
         vectors. The y axis is fixed to FIXED_YLIM = (0, 0.66).

MEDIAN_MARKER_SIZE is a matplotlib `plot` marker diameter in points, a
different unit from the area-based `s` used by `scatter`.

Both the 250-pair subsample and the point jitter are unseeded, so they differ
between runs.

The panel is descriptive; no statistical test is applied.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig6", "edfig6_b_head_angle")

PLOT_N_MAX = 250
JITTER_AMOUNT = 0.05
LINE_ALPHA = 0.15
POINT_ALPHA = 0.65
MEDIAN_MARKER_SIZE = 7   # ax.plot() markersize: diameter in points, not the
                          # area-based `s` used by scatter
BAR_LINE_WIDTH = 1.8
FIXED_YLIM = (0, 0.66)


def _concat_cell(x):
    """Concatenate cell-array contents into one flat array.

    The loader returns a bare ndarray for a single-element cell and a list of
    ndarrays for a multi-element cell; both are handled.
    """
    if isinstance(x, list):
        return np.concatenate([np.asarray(v, dtype=float).ravel() for v in x])
    return np.asarray(x, dtype=float).ravel()


def _polar_hist_fig(data_dir, fname, STY, group_color_unused=None):
    S = load_struct(os.path.join(data_dir, fname))
    theta = _concat_cell(S.all_theta_abs_cell)
    edges = np.asarray(S.edges, dtype=float).ravel()
    bin_colors = np.asarray(S.bin_colors, dtype=float)
    n_bins = edges.size - 1

    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    # Clockwise from North, so the semicircle opens to the right, matching
    # the published panel.
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    for b in range(n_bins):
        th1, th2 = edges[b], edges[b + 1]
        if b < n_bins - 1:
            in_bin = (theta >= th1) & (theta < th2)
        else:
            in_bin = (theta >= th1) & (theta <= th2)
        count = np.sum(in_bin)
        width = th2 - th1
        ax.bar((th1 + th2) / 2.0, count, width=width, color=bin_colors[b],
               edgecolor="none", linewidth=0)

    ax.set_rlim(0, 8000)
    # The published polar panels carry no title.
    ax.set_title("",
                  fontname="Arial", fontsize=STY.font.title)
    return fig, theta.size


def _med_quarts(x):
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


def _get_pair(S, tow_name, aw_name):
    tow = np.asarray(S[tow_name], dtype=float).ravel()
    aw = np.asarray(S[aw_name], dtype=float).ravel()
    ok = np.isfinite(tow) & np.isfinite(aw)
    return tow[ok], aw[ok]


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()   # unseeded: subsample and jitter vary per run

    figs = []

    # ---- p1-p3: polar histograms, one example animal set per group -------
    for name, fname in (("wt", "hist_example_wt.mat"),
                         ("sh", "hist_example_sh.mat"),
                         ("gt", "hist_example_gt.mat")):
        fig, n = _polar_hist_fig(data_dir, fname, STY)
        gt_figStyle(fig, STY)
        gt_saveFig(fig, f"edfig6_b_head_angle_p{len(figs)+1}_polar_{name}", out_dir)
        print(f"  polar {name}: n_points = {n}")
        figs.append(fig)

    # ---- p4: paired toward/away scatter, all animals, "init" condition ----
    S = load_struct(os.path.join(data_dir, "theta.mat"))
    conds = [
        ("WT init", "toward_wt_init", "away_wt_init", STY.cmean[0]),
        ("SH init", "toward_sh_init", "away_sh_init", STY.cmean[1]),
        ("GT init", "toward_gt_init", "away_gt_init", STY.cmean[2]),
    ]
    tow_color = STY.clr.toward
    aw_color = STY.clr.away
    pair_line_color = STY.clr.pairline

    fig4, axes4 = plt.subplots(1, 3, facecolor="w")
    for i, (name, tow_key, aw_key, _grp_color) in enumerate(conds):
        ax = axes4[i]
        tow, aw = _get_pair(S, tow_key, aw_key)
        N = tow.size

        if N > PLOT_N_MAX:
            idx = rng.choice(N, size=PLOT_N_MAX, replace=False)   # unseeded subsample
            tow_p, aw_p = tow[idx], aw[idx]
        else:
            tow_p, aw_p = tow, aw

        # paired lines
        for k in range(tow_p.size):
            ax.plot([1, 2], [tow_p[k], aw_p[k]], "-", linewidth=STY.lw.pair,
                     color=pair_line_color, alpha=LINE_ALPHA)

        # jittered points
        x1 = 1 + (rng.random(tow_p.size) - 0.5) * 2 * JITTER_AMOUNT
        x2 = 2 + (rng.random(aw_p.size) - 0.5) * 2 * JITTER_AMOUNT
        ax.scatter(x1, tow_p, s=STY.mk.pt, c=[tow_color], alpha=POINT_ALPHA,
                   edgecolors="none")
        ax.scatter(x2, aw_p, s=STY.mk.pt, c=[aw_color], alpha=POINT_ALPHA,
                   edgecolors="none")

        # median and quartiles, computed on the full unsubsampled vectors
        med_t, q25_t, q75_t = _med_quarts(tow)
        med_a, q25_a, q75_a = _med_quarts(aw)
        for x, med, q25, q75 in ((1, med_t, q25_t, q75_t), (2, med_a, q25_a, q75_a)):
            ax.plot(x, med, "o", color="k", markerfacecolor="k",
                     markersize=MEDIAN_MARKER_SIZE)
            ax.plot([x, x], [q25, q75], "-", color="k", linewidth=BAR_LINE_WIDTH)

        ax.set_ylim(*FIXED_YLIM)
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Tow.", "Aw."])
        gt_axes(ax, STY)
        # The published panel titles each group with its name alone; the
        # condition and n stay in the console summary below.
        ax.set_title(name.split()[0], fontsize=STY.font.title,
                     fontname=STY.font.name)

        print(f"  {name}: n={N}, median toward={med_t:.4f}, median away={med_a:.4f}")

    # The published figure carries no overall title.
    gt_figStyle(fig4, STY)
    gt_saveFig(fig4, "edfig6_b_head_angle_p4_paired_toward_away", out_dir)
    figs.append(fig4)

    return figs


if __name__ == "__main__":
    main()
