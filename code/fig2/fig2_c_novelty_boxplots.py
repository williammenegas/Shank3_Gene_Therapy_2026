"""
Figure 2, panel c: time spent near a stimulus, and the novel-minus-control
contrast. Two box-and-scatter plots, groups WT / SH / GT.

  1. Time near control (%): the fraction of session time each animal spent
     near the control stimulus. y limits [0, 0.8].

  2. % time (novel - control): for each of two stimulus pairs
     (object minus treat, face minus box) the script subtracts the matched
     control value from the novel value per animal, then averages the two
     differences per animal. y limits [-0.3, 0.1].

Each plot shows the box (median, quartiles and whiskers) with a jittered
scatter of individual animals over it.

Statistics: Levene's test for equal variance, then one-way ANOVA with
Tukey-Kramer post-hoc, or Welch's ANOVA with Games-Howell where equal
variance is rejected (gt_boxsig). Pairwise brackets are drawn only when the
omnibus test passes; the call is made after the y limits are set, since
bracket heights are placed relative to the final limits.

Data: time_near_control.mat, time_near_treat.mat, time_near_object.mat,
time_near_box.mat, time_near_face.mat
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_c_novelty_boxplots")


def _load_means(data_dir, fname):
    S = load_struct(os.path.join(data_dir, fname))
    wt = np.asarray(S.wt_2_mean, dtype=float).ravel()
    sh = np.asarray(S.sh_2_mean, dtype=float).ravel()
    gt = np.asarray(S.gt_2_mean, dtype=float).ravel()
    return wt, sh, gt


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    # Plot 1: time near the control stimulus.
    wt_control, sh_control, gt_control = _load_means(data_dir, "time_near_control.mat")

    # Plot 2: novel minus control, averaged over the object/treat and
    # face/box stimulus pairs.
    wt_treat, sh_treat, gt_treat = _load_means(data_dir, "time_near_treat.mat")
    wt_obj, sh_obj, gt_obj = _load_means(data_dir, "time_near_object.mat")
    wt_box, sh_box, gt_box = _load_means(data_dir, "time_near_box.mat")
    wt_face, sh_face, gt_face = _load_means(data_dir, "time_near_face.mat")

    wt_f2 = wt_obj - wt_treat
    sh_f2 = sh_obj - sh_treat
    gt_f2 = gt_obj - gt_treat

    wt_f3 = wt_face - wt_box
    sh_f3 = sh_face - sh_box
    gt_f3 = gt_face - gt_box

    wt_collect = np.nanmean(np.column_stack([wt_f2, wt_f3]), axis=1)
    sh_collect = np.nanmean(np.column_stack([sh_f2, sh_f3]), axis=1)
    gt_collect = np.nanmean(np.column_stack([gt_f2, gt_f3]), axis=1)

    figs = []

    fig1, ax1 = gt_boxscatter_figure(STY)
    gt_colorboxplot(ax1, [wt_control, sh_control, gt_control], STY, rng=rng)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(["WT", "SH", "GT"])
    ax1.set_xlim(0.5, 3.5)
    ax1.set_ylim(0, 0.8)
    ax1.set_ylabel("Time near control (%)", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_boxsig(ax1, STY)          # three-group test; must follow set_ylim
    gt_axes(ax1, STY)
    gt_color_xticklabels(ax1, STY)
    gt_figStyle(fig1, STY)
    gt_saveFig(fig1, "fig2_c_novelty_boxplots_p1", out_dir)
    figs.append(fig1)

    fig2, ax2 = gt_boxscatter_figure(STY)
    gt_colorboxplot(ax2, [wt_collect, sh_collect, gt_collect], STY, rng=rng)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["WT", "SH", "GT"])
    ax2.set_xlim(0.5, 3.5)
    ax2.set_ylim(-0.3, 0.1)
    ax2.set_ylabel(r"$\Delta$ Time (Novel - Control)", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_boxsig(ax2, STY)          # three-group test; must follow set_ylim
    gt_axes(ax2, STY)
    gt_color_xticklabels(ax2, STY)
    gt_figStyle(fig2, STY)
    gt_saveFig(fig2, "fig2_c_novelty_boxplots_p2", out_dir)
    figs.append(fig2)

    print(f"time near control:  n = {wt_control.size}/{sh_control.size}/{gt_control.size}, "
          f"means = {np.nanmean(wt_control):.4f}/{np.nanmean(sh_control):.4f}/{np.nanmean(gt_control):.4f}")
    print(f"novel-control:      n = {wt_collect.size}/{sh_collect.size}/{gt_collect.size}, "
          f"means = {np.nanmean(wt_collect):.4f}/{np.nanmean(sh_collect):.4f}/{np.nanmean(gt_collect):.4f}")

    return figs


if __name__ == "__main__":
    main()
