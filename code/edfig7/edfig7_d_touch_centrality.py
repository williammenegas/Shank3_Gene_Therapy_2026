"""Extended Data Figure 7, panel d: touch centrality, task types 2 and 3
averaged, WT/SH/GT.

Data: `xy_type2_data.mat` and `xy_type3_data.mat`, variables `r1`, `r2` and
`r3` for WT, SH and GT respectively, per-animal vectors padded to 1000 slots
with NaN. These `_data` files are distinct from the per-touch `xy_type2.mat`
and `xy_type3.mat` files used by panels c and e. The script averages each
animal's two task-type values, ignoring NaN.

Plot: box and density-jittered scatter, one group per x position (WT, SH, GT),
figure size 1.821 x 2.000 in. Boxes show the median and quartiles, and
whiskers extend to the most extreme value within 0.75 x IQR of the box edge.
The y axis is fixed to [0.2, 1].

Statistics: the three-group comparison drawn by gt_boxsig, i.e. a one-way
ANOVA (Welch's ANOVA when Levene's test rejects equal variance) with
Tukey-Kramer (or Games-Howell) post-hoc contrasts. Pairwise brackets appear
only when the omnibus test is significant; otherwise the panel is annotated
'ANOVA n/s'. The script prints counts and group means at run time.

The scatter jitter is unseeded, so the horizontal offsets of individual points
differ between runs.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig7", "edfig7_d_touch_centrality")


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    S2 = load_struct(os.path.join(data_dir, "xy_type2_data.mat"))
    S3 = load_struct(os.path.join(data_dir, "xy_type3_data.mat"))

    wt_control = np.column_stack([
        np.asarray(S2.r1, dtype=float).ravel(),
        np.asarray(S3.r1, dtype=float).ravel(),
    ])
    sh_control = np.column_stack([
        np.asarray(S2.r2, dtype=float).ravel(),
        np.asarray(S3.r2, dtype=float).ravel(),
    ])
    gt_control = np.column_stack([
        np.asarray(S2.r3, dtype=float).ravel(),
        np.asarray(S3.r3, dtype=float).ravel(),
    ])

    with np.errstate(invalid="ignore"):
        r1 = np.nanmean(wt_control, axis=1)
        r2 = np.nanmean(sh_control, axis=1)
        r3 = np.nanmean(gt_control, axis=1)

    fig, ax = gt_boxscatter_figure(STY, fig_w=1.821, fig_h=2.000)
    gt_colorboxplot(ax, [r1, r2, r3], STY, whisker=0.75, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.2, 1)
    gt_boxsig(ax, STY)          # three-group comparison, drawn after set_ylim
    ax.set_ylabel("Fraction in center", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig7_d_touch_centrality", out_dir)

    print(f"n = {np.sum(~np.isnan(r1))}/{np.sum(~np.isnan(r2))}/{np.sum(~np.isnan(r3))}, "
          f"means = {np.nanmean(r1):.4f}/{np.nanmean(r2):.4f}/{np.nanmean(r3):.4f}")

    return fig


if __name__ == "__main__":
    main()
