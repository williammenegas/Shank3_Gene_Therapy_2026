"""Extended Data Figure 7, panel h: state 3 occupancy, WT/SH/GT.

Data: `state3_control.mat`, variables `r1_control`, `r2_control` and
`r3_control` for WT, SH and GT respectively, per-animal vectors padded to 1000
slots with NaN. The stored values are percentages; the plotted quantity is
their complement expressed as a fraction, (100 - value) / 100, so that the
panel carries the 0 to 1 scale the published figure uses.

Plot: box and density-jittered scatter, one group per x position (WT, SH, GT),
figure size 1.821 x 2.000 in. Boxes show the median and quartiles, and
whiskers extend to the most extreme value within 0.75 x IQR of the box edge.
The script fixes the y axis to [0, 1], matching the published panel. The
matching main-figure panel, fig4_j, plots the same kind of rate on a 0 to 100
axis because that is the scale its own published panel uses.

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
DATA = os.path.join(HERE, "..", "..", "data", "edfig7", "edfig7_h_state_occupancy")


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    S = load_struct(os.path.join(data_dir, "state3_control.mat"))
    r1 = (100 - np.asarray(S.r1_control, dtype=float).ravel()) / 100.0
    r2 = (100 - np.asarray(S.r2_control, dtype=float).ravel()) / 100.0
    r3 = (100 - np.asarray(S.r3_control, dtype=float).ravel()) / 100.0

    fig, ax = gt_boxscatter_figure(STY, fig_w=1.821, fig_h=2.000)
    gt_colorboxplot(ax, [r1, r2, r3], STY, whisker=0.75, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0, 1)
    gt_boxsig(ax, STY)          # three-group comparison, drawn after set_ylim
    ax.set_ylabel("Center-biased state usage rate", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig7_h_state_occupancy", out_dir)

    print(f"n = {np.sum(~np.isnan(r1))}/{np.sum(~np.isnan(r2))}/{np.sum(~np.isnan(r3))}, "
          f"means = {np.nanmean(r1):.2f}/{np.nanmean(r2):.2f}/{np.nanmean(r3):.2f}")

    return fig


if __name__ == "__main__":
    main()
