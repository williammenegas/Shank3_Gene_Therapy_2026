"""
fig6_a1_radar_control.py

Figure 6, panel a, control condition: 16-endpoint radar of group means.

Data. marm_beh_all.mat, marm_soc_all.mat, marm_cog_all.mat and
marm_fmri_all.mat, control-condition fields (<domain>_<group>_c). Figure 6's
endpoints are stored by domain across these four shared files. Each domain
contributes four endpoints, giving the 16 spokes; the domain name is drawn
once at the centroid of its arc on the rim.

Display. Each spoke is one endpoint. Radius is the group mean z-score, with
larger radius meaning more WT-like, and the shaded band spans +/- one
standard error of that mean. Values are z-scored to WT separately within each
condition, so WT sits at 0 +/- 1 by construction in both the control and the
test radar.

Radial distance is a display convention rather than a magnitude, and the
enclosed area is not a summary statistic, since it depends on the order of
the spokes. The quantitative per-endpoint comparisons are in panel e.

The panel is descriptive: no statistical test is applied, and group sizes
vary by domain.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403
from _f6common import DOMS, DOMLAB, local_build, local_radar  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    STY.fs_tick = 7
    STY.fs_title = 8
    CMEAN = STY.cmean

    S = local_build(data_dir, DOMS, "c")

    FIG_W, FIG_H = 3.30, 3.10
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="w")
    ax = fig.add_axes([0.03, 0.14, 0.94, 0.80])

    handles = local_radar(ax, S, "Control endpoints", CMEAN, STY)
    gt_panelLetter(ax, "a", STY)

    fig.legend(handles, ["WT", "SH", "GT"], loc="lower center", ncol=3,
               frameon=False, fontsize=7, handlelength=1.4,
               bbox_to_anchor=(0.5, 0.0))

    print("Figure 6a: Control radar")
    print("  16 endpoints, group mean z-score +/- SE")
    for ep in S:
        wt = ep.wt[~np.isnan(ep.wt)]
        sh = ep.sh[~np.isnan(ep.sh)]
        gt = ep.gt[~np.isnan(ep.gt)]
        print(f"  {ep.name:<6}  WT {wt.mean():+5.2f}   SH {sh.mean():+5.2f}   GT {gt.mean():+5.2f}")

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig6_a1_radar_control", out_dir)
    return fig


if __name__ == "__main__":
    main()
