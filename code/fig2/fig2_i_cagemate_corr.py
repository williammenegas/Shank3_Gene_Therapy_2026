"""
Figure 2, panel i: per-animal correlation with the cagemate, control versus
novel. Two box-and-scatter plots, groups WT / SH / GT.

The correlation is available for the same four stimulus conditions as
panel f. The control plot averages the treat and box correlations per animal;
the novel plot averages the object and face correlations per animal. Both
plots carry the y label "Correlation with cagemate" and y limits [0.5, 0.9],
and show the box (median, quartiles and whiskers) with a jittered scatter of
individual animals over it.

Statistics: Levene's test for equal variance, then one-way ANOVA with
Tukey-Kramer post-hoc, or Welch's ANOVA with Games-Howell where equal
variance is rejected (gt_boxsig). Pairwise brackets are drawn only when the
omnibus test passes; the script makes the call after setting the y limits,
since bracket heights are placed relative to the final limits.

Data: mCM_corr_treat_data.mat, mCM_corr_object_data.mat,
mCM_corr_box_data.mat, mCM_corr_face_data.mat
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_i_cagemate_corr")


def _load_collect(data_dir, fname):
    S = load_struct(os.path.join(data_dir, fname))
    wt = np.asarray(S.collect_wt, dtype=float).ravel()
    sh = np.asarray(S.collect_sh, dtype=float).ravel()
    gt = np.asarray(S.collect_gt, dtype=float).ravel()
    return wt, sh, gt


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    wt_treat, sh_treat, gt_treat = _load_collect(data_dir, "mCM_corr_treat_data.mat")
    wt_obj, sh_obj, gt_obj = _load_collect(data_dir, "mCM_corr_object_data.mat")
    wt_box, sh_box, gt_box = _load_collect(data_dir, "mCM_corr_box_data.mat")
    wt_face, sh_face, gt_face = _load_collect(data_dir, "mCM_corr_face_data.mat")

    wt_control = np.nanmean(np.column_stack([wt_treat, wt_box]), axis=1)
    sh_control = np.nanmean(np.column_stack([sh_treat, sh_box]), axis=1)
    gt_control = np.nanmean(np.column_stack([gt_treat, gt_box]), axis=1)

    wt_collect = np.nanmean(np.column_stack([wt_obj, wt_face]), axis=1)
    sh_collect = np.nanmean(np.column_stack([sh_obj, sh_face]), axis=1)
    gt_collect = np.nanmean(np.column_stack([gt_obj, gt_face]), axis=1)

    figs = []
    for suffix, G in (("p1", (wt_control, sh_control, gt_control)),
                       ("p2", (wt_collect, sh_collect, gt_collect))):
        fig, ax = gt_boxscatter_figure(STY)
        gt_colorboxplot(ax, list(G), STY, rng=rng)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["WT", "SH", "GT"])
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0.5, 0.9)
        gt_boxsig(ax, STY)          # three-group test; must follow set_ylim
        ax.set_ylabel("Correlation with cagemate",
                       fontname=STY.font.name, fontsize=STY.font.lab)
        gt_axes(ax, STY)
        gt_color_xticklabels(ax, STY)
        gt_figStyle(fig, STY)
        gt_saveFig(fig, f"fig2_i_cagemate_corr_{suffix}", out_dir)
        figs.append(fig)

    print(f"control: n={wt_control.size}/{sh_control.size}/{gt_control.size}, "
          f"means={np.nanmean(wt_control):.4f}/{np.nanmean(sh_control):.4f}/{np.nanmean(gt_control):.4f}")
    print(f"novel:   n={wt_collect.size}/{sh_collect.size}/{gt_collect.size}, "
          f"means={np.nanmean(wt_collect):.4f}/{np.nanmean(sh_collect):.4f}/{np.nanmean(gt_collect):.4f}")

    return figs


if __name__ == "__main__":
    main()
