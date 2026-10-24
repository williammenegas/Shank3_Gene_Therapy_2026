"""
fig6_f_correlations.py

Figure 6, panel f, correlation matrices: the correlation structure among
endpoints within each domain, test condition.

Data. marm_beh_all.mat, marm_soc_all.mat, marm_cog_all.mat and
marm_fmri_all.mat, test-condition fields (<domain>_<group>_t). Values are
z-scored to WT separately within each condition.

Quantity. Pearson correlation between each pair of the four endpoints of a
domain, pooled across the three groups and computed pairwise-complete, with a
correlation formed only where more than three animals have both endpoints.
The lower triangle and the diagonal are drawn; the largest absolute
off-diagonal correlation is printed in each title.

These matrices set the context for the link score in panels c and d: because
endpoints within a domain are correlated, a plain average over endpoints
would weight most heavily whatever is measured most redundantly, whereas the
ridge penalty in the link-score model makes correlated endpoints share
weight.

Display. All four domains share the -1 to 1 color scale shown in the bar
below the matrices. Group sizes are small, so individual animals can move a
correlation appreciably.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403
from _f6common import DOMS, DOMLAB, local_corrmat, local_corrpanel, local_corrbar  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    STY.fs_tick = 7
    STY.fs_lab = 7
    STY.fs_title = 8
    STY.fs_cell = 6

    PAN = 1.20
    MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.42, 0.10, 0.58, 0.34
    PAN_GAP = 0.22
    FIG_W = MARGIN_L + 4 * PAN + 3 * PAN_GAP + MARGIN_R
    FIG_H = MARGIN_B + PAN + MARGIN_T

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="w")

    print("Figure 6f: endpoint correlation structure (test condition)")
    mx = np.full(len(DOMS), np.nan)
    for d, dom in enumerate(DOMS):
        x = MARGIN_L + d * (PAN + PAN_GAP)
        ax = fig.add_axes([x / FIG_W, MARGIN_B / FIG_H, PAN / FIG_W, PAN / FIG_H])
        C = local_corrmat(data_dir, dom, "t")
        local_corrpanel(ax, C, dom, DOMLAB[d], STY)
        if d == 0:
            gt_panelLetter(ax, "f", STY)
        off = C[~np.eye(4, dtype=bool)]
        off = off[~np.isnan(off)]
        mx[d] = np.max(np.abs(off))
        print(f"  {DOMLAB[d]:<10} max |r| = {mx[d]:.2f}")

    axcb = fig.add_axes([MARGIN_L / FIG_W, (MARGIN_B - 0.30) / FIG_H,
                          (4 * PAN + 3 * PAN_GAP) / FIG_W, 0.10 / FIG_H])
    local_corrbar(axcb, STY)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig6_f_correlations", out_dir)
    return fig


if __name__ == "__main__":
    main()
