"""
fig4_d_task_efficiency.py

Figure 4, panel d: reward rate on touchscreen tasks 4 and 5 combined
(WT, SH, GT).

Data. t_efficiency_type4_data.mat and t_efficiency_type5_data.mat, fields
eff_wt / eff_sh / eff_gt (1000-slot NaN-padded per-animal vectors). The
script centers each task by subtracting that task's own WT mean, then
averages the two centered tasks within animal, so the plotted quantity is
reward rate relative to WT, pooled over tasks 4 and 5.

Display. The shared gt_colorboxplot helper draws one box per group (median,
quartiles, whiskers at 0.75*IQR) with jittered individual animals.

Statistics. The shared three-group test (gt_boxsig): Levene-gated one-way
ANOVA (Welch when variances are unequal) with Tukey-Kramer post-hoc.
Pairwise brackets are drawn only when the omnibus test is significant. The
script applies the test after setting the y limits, so the brackets sit
inside the axis.

Point jitter is unseeded, so horizontal offsets of individual animals differ
slightly between runs.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_c_task_progress")


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    S4 = load_struct(os.path.join(data_dir, "t_efficiency_type4_data.mat"))
    S5 = load_struct(os.path.join(data_dir, "t_efficiency_type5_data.mat"))

    def centered(S):
        wt = np.asarray(S.eff_wt, dtype=float).ravel()
        sh = np.asarray(S.eff_sh, dtype=float).ravel()
        gt = np.asarray(S.eff_gt, dtype=float).ravel()
        wt_mean = np.nanmean(wt)
        return wt - wt_mean, sh - wt_mean, gt - wt_mean

    wt4, sh4, gt4 = centered(S4)
    wt5, sh5, gt5 = centered(S5)

    with np.errstate(invalid="ignore"):
        r1 = np.nanmean(np.column_stack([wt4, wt5]), axis=1)
        r2 = np.nanmean(np.column_stack([sh4, sh5]), axis=1)
        r3 = np.nanmean(np.column_stack([gt4, gt5]), axis=1)

    fig, ax = gt_boxscatter_figure(STY)
    gt_colorboxplot(ax, [r1, r2, r3], STY, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-0.4, 0.4)
    gt_boxsig(ax, STY)          # house 3-group test, drawn after ylim
    ax.set_ylabel("Reward rate (task 4+5)", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig4_d_task_efficiency", out_dir)

    print(f"n = {np.sum(~np.isnan(r1))}/{np.sum(~np.isnan(r2))}/{np.sum(~np.isnan(r3))}, "
          f"means = {np.nanmean(r1):.4f}/{np.nanmean(r2):.4f}/{np.nanmean(r3):.4f}")

    return fig


if __name__ == "__main__":
    main()
