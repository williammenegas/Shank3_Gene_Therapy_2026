"""Extended Data Figure 6, panel i: call probability, WT/SH/GT.

Data: `state_usage_data.mat`, variables `wt12`, `sh12` and `gt12`, per-animal
vectors of 1000 slots. The plotted quantity is the complement of each value,
1 - usage.

Plot: box and density-jittered scatter, one group per x position (WT, SH, GT),
figure size 1.821 x 2.000 in. Boxes show the median and quartiles, and
whiskers extend to the most extreme value within 0.75 x IQR of the box edge.
The y axis is fixed to [0.9, 1].

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
DATA = os.path.join(HERE, "..", "..", "data", "edfig6", "edfig6_i_call_probability")


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    S = load_struct(os.path.join(data_dir, "state_usage_data.mat"))
    r1 = 100.0 * (1 - np.asarray(S.wt12, dtype=float).ravel())
    r2 = 100.0 * (1 - np.asarray(S.sh12, dtype=float).ravel())
    r3 = 100.0 * (1 - np.asarray(S.gt12, dtype=float).ravel())

    fig, ax = gt_boxscatter_figure(STY, fig_w=1.821, fig_h=2.000)
    gt_colorboxplot(ax, [r1, r2, r3], STY, whisker=0.75, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(90, 100)
    gt_boxsig(ax, STY)          # three-group comparison, drawn after set_ylim
    ax.set_ylabel("Behavioral state usage (%)", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig6_i_call_probability", out_dir)

    print(f"n = {r1.size}/{r2.size}/{r3.size}, "
          f"means = {np.nanmean(r1):.4f}/{np.nanmean(r2):.4f}/{np.nanmean(r3):.4f}")

    return fig


if __name__ == "__main__":
    main()
