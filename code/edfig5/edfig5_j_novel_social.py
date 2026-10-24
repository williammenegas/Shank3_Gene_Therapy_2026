"""Extended Data Figure 5, panel j: stimulus-approach correlation, WT/SH/GT,
box and face conditions.

Data: `stim_corr_box_data.mat` and `stim_corr_face_data.mat`, each holding
`collect_B_wt_inds2`, `collect_B_sh_inds2` and `collect_B_gt_inds2`,
per-animal vectors padded to 1000 slots with NaN. The companion
`collect_B_*_inds1` variables are not used by this panel and are not read.
The script produces one figure per source file.

Plot: box and density-jittered scatter, one group per x position (WT, SH, GT),
figure size 1.821 x 2.000 in. Boxes show the median and quartiles, and
whiskers extend to the most extreme value within 0.75 x IQR of the box edge.
The y axis is fixed to [-0.6, 0.9].

Statistics: the three-group comparison drawn by gt_boxsig, i.e. a one-way
ANOVA (Welch's ANOVA when Levene's test rejects equal variance) with
Tukey-Kramer (or Games-Howell) post-hoc contrasts. Pairwise brackets appear
only when the omnibus test is significant; otherwise the panel is annotated
'ANOVA n/s'. The script prints counts and group medians at run time.

The scatter jitter is unseeded, so the horizontal offsets of individual points
differ between runs.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_f_stimulus_corr")

FIG_W, FIG_H = 1.821, 2.000   # figure size in inches


def _plot_one(S, label, out_stem, out_dir, rng, STY):
    r1 = np.asarray(S.collect_B_wt_inds2, dtype=float).ravel()
    r2 = np.asarray(S.collect_B_sh_inds2, dtype=float).ravel()
    r3 = np.asarray(S.collect_B_gt_inds2, dtype=float).ravel()

    fig, ax = gt_boxscatter_figure(STY, fig_w=FIG_W, fig_h=FIG_H)
    gt_colorboxplot(ax, [r1, r2, r3], STY, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-0.6, 0.9)
    gt_boxsig(ax, STY)          # three-group comparison, drawn after set_ylim
    ax.set_ylabel("Correlation", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, out_stem, out_dir)

    n = [np.sum(~np.isnan(v)) for v in (r1, r2, r3)]
    print(f"  {label}: n = {n[0]}/{n[1]}/{n[2]}, "
          f"medians = {np.nanmedian(r1):.4f}/{np.nanmedian(r2):.4f}/{np.nanmedian(r3):.4f}")
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()
    print("edfig5_j_novel_social -- stimulus-approach correlation, WT/SH/GT box+scatter")

    S1 = load_struct(os.path.join(data_dir, "stim_corr_box_data.mat"))
    fig1 = _plot_one(S1, "box", "edfig5_j_novel_social_box", out_dir, rng, STY)

    S2 = load_struct(os.path.join(data_dir, "stim_corr_face_data.mat"))
    fig2 = _plot_one(S2, "face", "edfig5_j_novel_social_face", out_dir, rng, STY)

    return [fig1, fig2]


if __name__ == "__main__":
    main()
