"""Figure 6, panel g: AAV transgene expression across brain regions.

WPRE in-situ quantification in six brain regions, expressed as the percentage
of cells positive for the AAV transgene transcript, in AAV-injected animals
versus uninjected controls.

Groups. AAV-injected (n = 4) and uninjected (n = 2). Regions run top to
bottom in the plotted order: frontal cortex, visual cortex, striatum,
thalamus, hindbrain, cerebellum.

Data. data/fig6/fig6_g_virus_density/summary_by_animal.csv, the
per-animal rollup of the in-situ analysis, one row per tissue, condition and
animal. This panel reads the six brain regions; Extended Data Figure 14e
shows the same values as vertical bars and 14f reads the peripheral rows of
the same file, both pointing at this folder.

Display. The panel is horizontal: regions on the y axis, percentage on the x
axis. Per region the uninjected group is a single dark marker at its group
mean, since both control values sit near zero at this scale, and the injected
group is a line spanning the mean plus and minus one standard error with a
filled marker at the mean, over the four individual animal values as pale
dots on the same rule; animals with near-equal values overlap, as they do in
the published panel. A dashed rule marks zero and dashed gray rules mark
the remaining x ticks, behind the data.

The x axis runs to just past 60 so that the largest single animal value,
63.30 (hindbrain), sits inside the axis.

The standard error is the sample standard deviation over the square root of
the group size, so it is defined only for the injected group; the uninjected
group carries no error mark, matching the published panel. No statistical
test is applied. The script prints per-region group means and animal counts
to the console.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import gt_init, gt_axes, gt_figStyle, gt_saveFig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_g_virus_density")
SUMMARY_CSV = "summary_by_animal.csv"

REGIONS = ["frontal_cortex", "visual_cortex", "striatum",
           "thalamus", "hindbrain", "cerebellum"]
REGION_LABELS = ["Frontal cortex", "Visual cortex", "Striatum",
                 "Thalamus", "Hindbrain", "Cerebellum"]

# ---- panel colors, sampled from the published figure ---------------------
C_AAV = (0.133, 0.600, 0.278)      # injected mean marker and error line
C_AAV_PT = (0.522, 0.796, 0.584)   # injected animal dots
C_CTRL = (0.125, 0.125, 0.125)     # uninjected mean marker
C_GRID = (0.650, 0.650, 0.650)     # dashed x rules

X_MAX = 66.0
X_TICKS = range(0, 61, 10)
PT_S = 26          # animal dot area, points squared
MEAN_S = 44        # group mean marker area, points squared


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    df = pd.read_csv(os.path.join(data_dir, SUMMARY_CSV))

    nR = len(REGIONS)
    AX_W, AX_H = 2.10, 1.45
    ML, MR, MB, MT = 0.90, 0.20, 0.46, 0.30
    FW, FH = ML + AX_W + MR, MB + AX_H + MT
    fig = plt.figure(figsize=(FW, FH), facecolor="w")
    ax = fig.add_axes([ML / FW, MB / FH, AX_W / FW, AX_H / FH])

    print("AAV+ cells by brain region (%)")
    print(f"  {'region':<14} {'uninjected':>14}   {'AAV-injected, mean +- SEM':>28}")
    for j, region in enumerate(REGIONS):
        sub = df[df.tissue == region]
        ctrl = sub[sub.condition == "control"]["pct_positive"].to_numpy(float)
        aav = sub[sub.condition == "gt"]["pct_positive"].to_numpy(float)
        m = aav.mean()
        se = aav.std(ddof=1) / np.sqrt(aav.size)
        y = j

        # Injected: individual animals, then the mean +- SEM over them.
        ax.scatter(aav, np.full(aav.size, y), s=PT_S,
                   c=[C_AAV_PT], edgecolors="none", zorder=2)
        ax.plot([m - se, m + se], [y, y], "-", color=C_AAV,
                lw=STY.lw.mean, zorder=3, solid_capstyle="butt")
        ax.scatter([m], [y], s=MEAN_S, c=[C_AAV], edgecolors="none", zorder=4)
        # Uninjected: a single marker at the group mean.
        ax.scatter([ctrl.mean()], [y], s=MEAN_S, c=[C_CTRL],
                   edgecolors="none", zorder=4)

        print(f"  {region:<14} {ctrl.mean():8.2f} (n={ctrl.size})   "
              f"{m:8.2f} +- {se:5.2f} (n={aav.size})")

    ax.set_xlim(-2.0, X_MAX)
    ax.set_ylim(nR - 0.5, -0.5)              # first region at the top
    ax.set_xticks(X_TICKS)
    ax.set_yticks(range(nR))
    ax.set_yticklabels(REGION_LABELS, fontname=STY.font.name, fontsize=STY.font.tick)
    ax.set_xticklabels([str(t) for t in X_TICKS],
                       fontname=STY.font.name, fontsize=STY.font.tick)
    ax.set_xlabel("AAV+ cells (%)", fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_title("Fraction of cells", fontname=STY.font.name, fontsize=STY.font.title)
    gt_axes(ax, STY, matlab_ticks=False)

    # gt_axes clears the grid, so the dashed x rules are added after it.
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=C_GRID, linestyle=(0, (4, 3)), linewidth=0.6)
    ax.yaxis.grid(False)
    ax.axvline(0.0, color=C_CTRL, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)

    for dy, txt, col in ((0.00, "Uninjected", C_CTRL), (0.115, "AAV-injected", C_AAV)):
        ax.text(0.98, 0.97 - dy, txt, transform=ax.transAxes, ha="right", va="top",
                color=col, fontname=STY.font.name, fontsize=STY.font.leg)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig6_g_virus_density", out_dir)
    return fig


if __name__ == "__main__":
    main()
