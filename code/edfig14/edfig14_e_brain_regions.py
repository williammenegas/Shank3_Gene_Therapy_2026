"""Extended Data Figure 14, panel e: AAV transgene expression by brain region.

WPRE in-situ quantification in six brain regions, expressed as the percentage
of cells positive for the AAV transgene transcript, in AAV-injected animals
versus uninjected controls.

This panel and Figure 6g report the same quantity from the same source, so
the two carry identical values by construction. The difference is
presentation: Figure 6g draws the regions horizontally with a mean marker and
the range across animals; this panel draws the injected group as a bar at its
mean, matching panel f of this figure.

Groups. AAV-injected (n = 4) and uninjected (n = 2). Regions in plotted
order: frontal cortex, visual cortex, striatum, thalamus, hindbrain,
cerebellum.

Data. data/fig6/fig6_g_virus_density/summary_by_animal.csv, the
per-animal rollup of the in-situ analysis, one row per tissue, condition and
animal. This panel reads the six brain regions; panel f reads the peripheral
rows of the same file. Group values are the mean across animals of each
animal's pct_positive, which is the level the rows are already at.

Display. Regions sit at integer x, the uninjected group offset to x - 0.20
and the injected group to x + 0.20. The injected group is a solid bar at its
mean; the uninjected group, whose values are near zero at this scale, is a
short rule at its mean rather than a bar that would not be visible. Both
carry their individual animal values on top as pale dots, beeswarmed so that
coincident values stay separable. Dashed rules mark the y ticks behind the
data. The colors are sampled from the published panel and are local to the
Extended Data Figure 14 pair, since the shared palette carries no AAV green.

The y axis reproduces the published panel: ticks every 10 up to 60, the axis
itself ending at 66.6, and panel f carries the same range so that peripheral
expression is read against the brain.

The panel is descriptive and applies no statistical test. The script prints
per-region group means and animal counts to the console.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import gt_init, gt_axes, gt_figStyle, gt_saveFig  # noqa: E402
from gtstyle.plotting import gt_beeswarm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_g_virus_density")
SUMMARY_CSV = "summary_by_animal.csv"

REGIONS = ["frontal_cortex", "visual_cortex", "striatum",
           "thalamus", "hindbrain", "cerebellum"]
REGION_LABELS = ["Frontal cortex", "Visual cortex", "Striatum",
                 "Thalamus", "Hindbrain", "Cerebellum"]

# ---- panel colors, sampled from the published figure ---------------------
C_AAV = (0.133, 0.600, 0.278)      # injected bar
C_AAV_PT = (0.522, 0.796, 0.584)   # injected animal dots
C_CTRL = (0.416, 0.416, 0.416)     # uninjected mean rule
C_CTRL_PT = (0.824, 0.824, 0.831)  # uninjected animal dots
C_GRID = (0.645, 0.645, 0.645)     # dashed y rules

DX = 0.20          # group offset from the region center
BAR_W = 0.38       # bar width, and the length of the uninjected mean rule
PT_S = 30          # dot area, points squared
Y_MAX = 66.6       # published axis ceiling, shared with panel f
Y_TICKS = range(0, 61, 10)


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    df = pd.read_csv(os.path.join(data_dir, SUMMARY_CSV))

    nR = len(REGIONS)
    AX_W, AX_H = 3.90, 1.45
    ML, MR, MB, MT = 0.62, 0.22, 0.46, 0.30
    FW, FH = ML + AX_W + MR, MB + AX_H + MT
    fig = plt.figure(figsize=(FW, FH), facecolor="w")
    ax = fig.add_axes([ML / FW, MB / FH, AX_W / FW, AX_H / FH])

    print("AAV+ cells by brain region (%)")
    print(f"  {'region':<14} {'uninjected':>18}   {'AAV-injected':>18}")
    for j, region in enumerate(REGIONS):
        sub = df[df.tissue == region]
        ctrl = sub[sub.condition == "control"]["pct_positive"].to_numpy(float)
        aav = sub[sub.condition == "gt"]["pct_positive"].to_numpy(float)

        xc, xa = j - DX, j + DX
        ax.plot([xc - BAR_W / 2, xc + BAR_W / 2], [ctrl.mean()] * 2, "-",
                color=C_CTRL, lw=STY.lw.mean, zorder=2, solid_capstyle="butt")
        ax.bar(xa, aav.mean(), width=BAR_W, color=C_AAV, edgecolor="none", zorder=2)

        for x, v, c in ((xc, ctrl, C_CTRL_PT), (xa, aav, C_AAV_PT)):
            ax.scatter(gt_beeswarm(v, x, 0.085), v, s=PT_S, c=[c],
                       edgecolors="none", zorder=3)

        print(f"  {region:<14} {ctrl.mean():10.2f} (n={ctrl.size})   "
              f"{aav.mean():10.2f} (n={aav.size})")

    ax.set_xlim(-0.55, nR - 0.45)
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks(range(nR))
    ax.set_xticklabels(REGION_LABELS, fontname=STY.font.name, fontsize=STY.font.tick)
    ax.set_yticks(Y_TICKS)
    ax.set_ylabel("AAV+ (% of total cells)",
                  fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_title("Brain regions", fontname=STY.font.name, fontsize=STY.font.title)
    gt_axes(ax, STY, matlab_ticks=False)

    # gt_axes clears the grid, so the dashed y rules are added after it.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_GRID, linestyle=(0, (4, 3)), linewidth=0.6)
    ax.xaxis.grid(False)

    for dy, txt, col in ((0.00, "Uninjected", C_CTRL), (0.065, "AAV-injected", C_AAV)):
        ax.text(0.02, 0.885 - dy, txt, transform=ax.transAxes, ha="left", va="top",
                color=col, fontname=STY.font.name, fontsize=STY.font.leg)
    ax.text(0.98, 0.95, "n = 4", transform=ax.transAxes, ha="right", va="top",
            color=C_AAV, fontname=STY.font.name, fontsize=STY.font.leg)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig14_e_brain_regions", out_dir)
    return fig


if __name__ == "__main__":
    main()
