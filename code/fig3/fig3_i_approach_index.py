"""
Figure 3, panel i: neutral behavioral-state usage.

One box-and-scatter plot, groups WT / SH / GT. The script plots one minus each
animal's fraction of session time in the approach and avoid states, i.e. the
fraction of time in every other state, as a box (median, quartiles and
whiskers) with a jittered scatter of individual animals over it. y limits
[0.9, 1.0].

Statistics: Levene's test for equal variance, then one-way ANOVA with
Tukey-Kramer post-hoc, or Welch's ANOVA with Games-Howell where equal
variance is rejected (gt_boxsig). Pairwise brackets are drawn only when the
omnibus test passes; the script makes the call after setting the y limits,
since bracket heights are placed relative to the final limits.

Data: state_usage_data.mat
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig3", "fig3_i_approach_index")


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    S = load_struct(os.path.join(data_dir, "state_usage_data.mat"))
    # Stored as fractions; the published axis is per cent.
    r1 = 100.0 * (1 - np.asarray(S.wt12, dtype=float).ravel())
    r2 = 100.0 * (1 - np.asarray(S.sh12, dtype=float).ravel())
    r3 = 100.0 * (1 - np.asarray(S.gt12, dtype=float).ravel())

    fig, ax = gt_boxscatter_figure(STY)
    gt_colorboxplot(ax, [r1, r2, r3], STY, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(90, 100)
    gt_boxsig(ax, STY)          # three-group test; must follow set_ylim
    ax.set_ylabel("Neutral behavioral state usage (%)", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig3_i_approach_index", out_dir)

    print(f"n = {r1.size}/{r2.size}/{r3.size}, "
          f"means = {np.nanmean(r1):.4f}/{np.nanmean(r2):.4f}/{np.nanmean(r3):.4f}")

    return fig


if __name__ == "__main__":
    main()
