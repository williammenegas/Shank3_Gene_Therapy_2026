"""
fig4_j_state_occupancy.py

Figure 4, panel j: per-animal occupancy of the center-biased GLM-HMM state,
in per cent (WT, SH, GT).

Data. Fields r1 (WT) / r2 (SH) / r3 (GT) of state3.mat, stored as a
percentage on 0-100. The panel plots the complement, 100 - r, so higher
values mean more time in the center-biased state.

Display. One box per group (median, quartiles, whiskers at 0.75*IQR) with
jittered individual animals, drawn by the shared gt_colorboxplot helper.

Statistics. The shared three-group test (gt_boxsig): Levene-gated one-way
ANOVA (Welch when variances are unequal) with Tukey-Kramer post-hoc.
Pairwise brackets are drawn only when the omnibus test is significant.

Point jitter is unseeded, so horizontal offsets of individual animals differ
slightly between runs.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_j_state_occupancy")


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    S = load_struct(os.path.join(data_dir, "state3.mat"))
    r1 = 100.0 - np.asarray(S.r1, dtype=float).ravel()
    r2 = 100.0 - np.asarray(S.r2, dtype=float).ravel()
    r3 = 100.0 - np.asarray(S.r3, dtype=float).ravel()

    fig, ax = gt_boxscatter_figure(STY)
    gt_colorboxplot(ax, [r1, r2, r3], STY, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.0, 100.0)
    gt_boxsig(ax, STY)          # house 3-group test, drawn after ylim
    ax.set_ylabel("Center-biased state usage rate", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig4_j_state_occupancy", out_dir)

    print(f"n = {np.sum(~np.isnan(r1))}/{np.sum(~np.isnan(r2))}/{np.sum(~np.isnan(r3))}, "
          f"means = {np.nanmean(r1):.4f}/{np.nanmean(r2):.4f}/{np.nanmean(r3):.4f}")

    return fig


if __name__ == "__main__":
    main()
