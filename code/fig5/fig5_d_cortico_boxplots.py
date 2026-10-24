"""
fig5_d_cortico_boxplots.py

Figure 5, panel d: cortico-striatal and cortico-cortical connectivity
contrasts, averaged across six cortical regions (WT, SH, GT).

Data. This script produces two figures, each assembled from six region files
(prefrontal, premotor, parietal, temporal, somatosensory, visual). Within
each file the plotted quantity is the per-animal difference
collect_*_r1 - collect_*_r2, averaged across the six regions:
  1. Cortico-striatal: *_str_paired_midbrain_str_data.mat, the
     cortico-striatal minus midbrain-striatal contrast.
  2. Cortico-cortical: *_self_paired_other_data.mat, the within-region
     versus other-region cortical contrast.
Each title names the pairing that its panel loads.

Display. One box per group (median, quartiles, whiskers at 0.75*IQR) with
jittered individual animals, drawn by the shared gt_colorboxplot helper.

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
DATA = os.path.join(HERE, "..", "..", "data", "fig5", "fig5_d_cortico_boxplots")

REGIONS = ["prefrontal", "premotor", "parietal", "temporal", "somatosensory", "visual"]

# (output stem, file suffix, title naming the pairing the block loads)
BLOCKS = [
    ("str_paired_midbrain_str", "_str_paired_midbrain_str_data.mat", "Cortico-striatal"),
    ("self_paired_other", "_self_paired_other_data.mat", "Cortico-cortical"),
]


def _one_block(stem, suffix, title, data_dir, out_dir, STY):
    rng = np.random.default_rng()

    wt_cols, sh_cols, gt_cols = [], [], []
    for region in REGIONS:
        S = load_struct(os.path.join(data_dir, f"{region}{suffix}"))
        wt_cols.append(np.asarray(S.collect_wt_r1, dtype=float).ravel()
                        - np.asarray(S.collect_wt_r2, dtype=float).ravel())
        sh_cols.append(np.asarray(S.collect_sh_r1, dtype=float).ravel()
                        - np.asarray(S.collect_sh_r2, dtype=float).ravel())
        gt_cols.append(np.asarray(S.collect_gt_r1, dtype=float).ravel()
                        - np.asarray(S.collect_gt_r2, dtype=float).ravel())

    with np.errstate(invalid="ignore"):
        r1 = np.nanmean(np.column_stack(wt_cols), axis=1)
        r2 = np.nanmean(np.column_stack(sh_cols), axis=1)
        r3 = np.nanmean(np.column_stack(gt_cols), axis=1)

    fig, ax = gt_boxscatter_figure(STY, fig_w=1.45)
    gt_colorboxplot(ax, [r1, r2, r3], STY, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-0.1, 0.45)
    gt_boxsig(ax, STY, test="nonparametric")   # Kruskal-Wallis + rank-sum post-hoc
    ax.set_ylabel("Correlation (normalized)", fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_title(title, fontname=STY.font.name, fontsize=STY.font.title)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, f"fig5_d_cortico_boxplots_{stem}", out_dir)

    print(f"[{title}] n = {np.sum(~np.isnan(r1))}/{np.sum(~np.isnan(r2))}/{np.sum(~np.isnan(r3))}, "
          f"means = {np.nanmean(r1):.4f}/{np.nanmean(r2):.4f}/{np.nanmean(r3):.4f}")
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    figs = []
    for stem, suffix, title in BLOCKS:
        figs.append(_one_block(stem, suffix, title, data_dir, out_dir, STY))
    return figs


if __name__ == "__main__":
    main()
