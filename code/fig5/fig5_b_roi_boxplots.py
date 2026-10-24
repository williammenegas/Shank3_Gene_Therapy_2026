"""
fig5_b_roi_boxplots.py

Figure 5, panel b: normalized global functional connectivity in two regions
of interest (WT, SH, GT).

Data. This script produces two figures, one per region of interest:
  - Visual cortex: fields wt_control / sh_control / gt_control of
    horizontal_data_visual.mat.
  - Frontostriatal: fields wt_collect / sh_collect / gt_collect of
    horizontal_data_frontostriatal.mat.
Each value is one animal's normalized global connectivity in that region.

Display. One box per group (median, quartiles, whiskers at 0.75*IQR) with
jittered individual animals, drawn by the shared gt_colorboxplot helper. Both
regions use the same y limits and y label; only the title differs.

Statistics. The shared three-group test in nonparametric mode: a
Kruskal-Wallis omnibus test gating pairwise Mann-Whitney rank-sum tests. The
nonparametric mode is used for the Figure 5 fMRI panels, whose connectivity
values are not normally distributed.

Point jitter is unseeded, so horizontal offsets of individual animals differ
slightly between runs.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig5", "fig5_b_roi_boxplots")

ROIS = [
    ("visual", "horizontal_data_visual.mat", "wt_control", "sh_control", "gt_control",
     "Visual cortex ROI"),
    ("frontostriatal", "horizontal_data_frontostriatal.mat", "wt_collect", "sh_collect", "gt_collect",
     "Frontostriatal ROI"),
]


def _one_roi(name, fname, kwt, ksh, kgt, title, data_dir, out_dir, STY):
    rng = np.random.default_rng()
    S = load_struct(os.path.join(data_dir, fname))
    # Stored at 1/100 of the published scale.
    r1 = 100.0 * np.asarray(S[kwt], dtype=float).ravel()
    r2 = 100.0 * np.asarray(S[ksh], dtype=float).ravel()
    r3 = 100.0 * np.asarray(S[kgt], dtype=float).ravel()

    fig, ax = gt_boxscatter_figure(STY, fig_w=1.45)
    gt_colorboxplot(ax, [r1, r2, r3], STY, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-0.1, 0.3)
    gt_boxsig(ax, STY, test="nonparametric")   # Kruskal-Wallis + rank-sum post-hoc
    ax.set_ylabel("Global connectivity (normalized)", fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_title(title, fontname=STY.font.name, fontsize=STY.font.title)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, f"fig5_b_roi_boxplots_{name}", out_dir)

    print(f"{title}: n = {r1.size}/{r2.size}/{r3.size}, "
          f"means = {np.nanmean(r1):.6g}/{np.nanmean(r2):.6g}/{np.nanmean(r3):.6g}")
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    figs = []
    for name, fname, kwt, ksh, kgt, title in ROIS:
        figs.append(_one_roi(name, fname, kwt, ksh, kgt, title, data_dir, out_dir, STY))
    return figs


if __name__ == "__main__":
    main()
