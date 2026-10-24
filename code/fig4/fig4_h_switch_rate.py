"""
fig4_h_switch_rate.py

Figure 4, panel h: probability of staying in the center state between
consecutive touches (WT, SH, GT).

Data. Fields wt_trans_1 / sh_trans_1 / gt_trans_1 of
transition_type1_data.mat, transition_type4_data.mat and
transition_type5_data.mat (1000-slot NaN-padded per-animal vectors, one file
per session type). The script averages the three session types within animal.

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
DATA = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_h_switch_rate")

TYPE_FILES = ["transition_type1_data.mat", "transition_type4_data.mat", "transition_type5_data.mat"]


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    wt_cols, sh_cols, gt_cols = [], [], []
    for fn in TYPE_FILES:
        S = load_struct(os.path.join(data_dir, fn))
        wt_cols.append(np.asarray(S.wt_trans_1, dtype=float).ravel())
        sh_cols.append(np.asarray(S.sh_trans_1, dtype=float).ravel())
        gt_cols.append(np.asarray(S.gt_trans_1, dtype=float).ravel())

    with np.errstate(invalid="ignore"):
        r1 = np.nanmean(np.column_stack(wt_cols), axis=1)
        r2 = np.nanmean(np.column_stack(sh_cols), axis=1)
        r3 = np.nanmean(np.column_stack(gt_cols), axis=1)

    fig, ax = gt_boxscatter_figure(STY)
    gt_colorboxplot(ax, [r1, r2, r3], STY, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.0, 0.8)
    gt_boxsig(ax, STY)          # house 3-group test, drawn after ylim
    ax.set_ylabel("Stay rate in center", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig4_h_switch_rate", out_dir)

    print(f"n = {np.sum(~np.isnan(r1))}/{np.sum(~np.isnan(r2))}/{np.sum(~np.isnan(r3))}, "
          f"means = {np.nanmean(r1):.4f}/{np.nanmean(r2):.4f}/{np.nanmean(r3):.4f}")

    return fig


if __name__ == "__main__":
    main()
