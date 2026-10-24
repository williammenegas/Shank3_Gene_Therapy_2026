"""
Figure 2, panel h: state usage of the two animals in a cagemate pair, one
example session per genotype.

The state-usage vectors give the fraction of frames each of the two animals
in a pair spent in each of the 80 states. "Red" and "yellow" are cagemate
color codes from the tracking pipeline, not genotype labels. This script
shows one example session per genotype (session 21 for WT, 58 for SH, 93 for
GT, 1-based). For each, it draws two plots:

  1. A mirrored bar plot, one bar per state: the first animal's usage upward
     and the second animal's usage downward, on y limits [-0.1, 0.1].

  2. A log-log scatter of the first animal's usage against the second
     animal's for the same 80 states, with the y = x line and a dashed
     perpendicular segment from each point to that line, on square axes
     spanning [-10.5, -2] in natural log units.

States with zero usage in a session give -inf under the logarithm; those
points fall outside the axis limits and are clipped. The associated
divide-by-zero warning is expected and the script suppresses it.

No statistical test is applied; the panel is descriptive.

Data: Figure_2h_data.mat
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_h_cagemate_scatter")

# (genotype label, session index converted from 1-based to 0-based, color row)
EXAMPLES = [("WT", 21 - 1, 0), ("SH", 58 - 1, 1), ("GT", 93 - 1, 2)]


def _mirror_bar_panel(temp1, temp4, title, STY, outname, out_dir):
    fig = plt.figure(figsize=(1.26, 0.75), facecolor="w")
    ax = fig.add_axes([0.20, 0.22, 0.76, 0.60])
    x = np.arange(1, temp1.size + 1)
    ax.bar(x, temp1, color=(0, 0.447, 0.741), edgecolor="none")
    ax.bar(x, -1 * temp4, color=(0.851, 0.325, 0.098), edgecolor="none")
    ax.set_title(title, fontname=STY.font.name, fontsize=STY.font.title)
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.1, 0.1)
    gt_axes(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, outname, out_dir)
    return fig


def _loglog_panel(temp1, temp4, title, col, STY, outname, out_dir):
    fig = plt.figure(figsize=(1.26, 0.75), facecolor="w")
    ax = fig.add_axes([0.20, 0.20, 0.62, 0.62])
    ax.set_aspect("equal")

    ax.plot([-10.5, -2], [-10.5, -2], "-", color="k", lw=STY.lw.hair, zorder=1)

    with np.errstate(divide="ignore"):
        x = np.log(temp1)
        y = np.log(temp4)
    ax.scatter(x, y, s=STY.mk.pt, c=[col], edgecolors="none", zorder=2)

    # Perpendicular projection onto y = x, one dashed segment per point.
    finite = np.isfinite(x) & np.isfinite(y)
    xp = (x[finite] + y[finite]) / 2.0
    for xi, yi, xpi in zip(x[finite], y[finite], xp):
        ax.plot([xi, xpi], [yi, xpi], "--", color="k", lw=STY.lw.hair, zorder=1)

    ax.set_xlim(-10.5, -2)
    ax.set_ylim(-10.5, -2)
    ax.set_title(title, fontname=STY.font.name, fontsize=STY.font.title)
    gt_axes(ax, STY)
    # Three evenly spaced x ticks instead of the seven-bin house locator,
    # whose five labels merge visually at this panel width. matlab_ticks is
    # False on the second call so the locator is not re-applied. The y axis
    # keeps the ticks gt_axes set above.
    ax.set_xticks([-10, -6, -2])
    gt_axes(ax, STY, matlab_ticks=False)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, outname, out_dir)
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    S = load_struct(os.path.join(data_dir, "Figure_2h_data.mat"))

    red_running = S.Bmat_red_running
    yellow_running = S.Bmat_yellow_running

    figs = []
    for i, (name, f_idx, cidx) in enumerate(EXAMPLES):
        temp1 = np.asarray(red_running[f_idx], dtype=float).ravel()
        temp4 = np.asarray(yellow_running[f_idx], dtype=float).ravel()

        p_bar = f"fig2_h_cagemate_scatter_p{2*i+1}"
        p_scatter = f"fig2_h_cagemate_scatter_p{2*i+2}"

        figs.append(_mirror_bar_panel(temp1, temp4, name, STY, p_bar, out_dir))
        figs.append(_loglog_panel(temp1, temp4, name, STY.cmean[cidx], STY,
                                   p_scatter, out_dir))

        print(f"{name} (f={f_idx+1}): red usage sum={temp1.sum():.4f}, "
              f"yellow usage sum={temp4.sum():.4f}")

    return figs


if __name__ == "__main__":
    main()
