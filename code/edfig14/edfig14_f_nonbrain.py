"""Extended Data Figure 14, panel f: AAV transgene expression outside the brain.

WPRE in-situ quantification in three peripheral organs (heart, kidney and
liver), expressed as the percentage of cells positive for the AAV transgene
transcript, in AAV-injected animals versus uninjected controls.

Groups. AAV-injected and uninjected, one value per animal. This is a
different cohort from the brain panels (one injected and one control animal
appear in both) because the peripheral organs were collected from a separate
set of animals. The number of animals also differs by tissue, since not every
organ was collected from every animal: heart 3 injected / 2 control, kidney
3 / 2, liver 2 / 1. Each printed n is the number of animals actually
contributing, not a nominal group size. The panel itself carries no n, since
no single number is right for every tissue.

Data. data/fig6/fig6_g_virus_density/summary_by_animal.csv, the
per-animal rollup of the mixed-model in-situ analysis. One row per tissue,
condition and animal, with n_cells, n_positive and pct_positive; this panel
reads pct_positive and aggregates to the animal level, which is the level the
rows are already at. Where an animal contributed more than one section the
rollup has already pooled them, recorded in its n_sections column. The file
also carries the brain rows that panel e reads.

Display. Tissues sit at integer x, the uninjected group offset to x - 0.20
and the injected group to x + 0.20, drawn exactly as in panel e: the injected
group a solid bar at its mean, the uninjected group a short rule at its mean,
both carrying their individual animal values as pale beeswarmed dots over
dashed y rules.

The y axis reproduces the published panel, which is the same axis as panel e:
ticks every 10 up to 60, the axis itself ending at 66.6. Every value here is
below 1%, so the panel reads as flat; the shared axis is for reading
peripheral expression against the brain. Passing y_max lowers the ceiling to
inspect these values on their own scale, which changes nothing but the axis.

The panel is descriptive and applies no statistical test, matching the
published legend, which reports group means only. The script prints
per-tissue group means and animal counts to the console.
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

TISSUES = ["heart", "kidney", "liver"]
TISSUE_LABELS = ["Heart", "Kidney", "Liver"]

# ---- panel colors, sampled from the published figure ---------------------
C_AAV = (0.133, 0.600, 0.278)      # injected bar
C_AAV_PT = (0.522, 0.796, 0.584)   # injected animal dots
C_CTRL = (0.416, 0.416, 0.416)     # uninjected mean rule
C_CTRL_PT = (0.824, 0.824, 0.831)  # uninjected animal dots
C_GRID = (0.645, 0.645, 0.645)     # dashed y rules

DX = 0.20          # group offset from the tissue center
BAR_W = 0.38       # bar width, and the length of the uninjected mean rule
PT_S = 30          # dot area, points squared
Y_MAX = 66.6       # published axis ceiling, shared with panel e
Y_TICKS = range(0, 61, 10)


def main(data_dir=DATA, out_dir=HERE, y_max=Y_MAX):
    STY = gt_init()
    df = pd.read_csv(os.path.join(data_dir, SUMMARY_CSV))

    nT = len(TISSUES)
    AX_W, AX_H = 2.00, 1.45
    ML, MR, MB, MT = 0.62, 0.22, 0.46, 0.30
    FW, FH = ML + AX_W + MR, MB + AX_H + MT
    fig = plt.figure(figsize=(FW, FH), facecolor="w")
    ax = fig.add_axes([ML / FW, MB / FH, AX_W / FW, AX_H / FH])

    print("AAV+ cells outside the brain (%)")
    print(f"  {'tissue':<8} {'uninjected':>18}   {'AAV-injected':>18}")
    for j, tissue in enumerate(TISSUES):
        sub = df[df.tissue == tissue]
        ctrl = sub[sub.condition == "control"]["pct_positive"].to_numpy(float)
        aav = sub[sub.condition == "gt"]["pct_positive"].to_numpy(float)

        xc, xa = j - DX, j + DX
        ax.plot([xc - BAR_W / 2, xc + BAR_W / 2], [ctrl.mean()] * 2, "-",
                color=C_CTRL, lw=STY.lw.mean, zorder=2, solid_capstyle="butt")
        ax.bar(xa, aav.mean(), width=BAR_W, color=C_AAV, edgecolor="none", zorder=2)

        for x, v, c in ((xc, ctrl, C_CTRL_PT), (xa, aav, C_AAV_PT)):
            ax.scatter(gt_beeswarm(v, x, 0.085), v, s=PT_S, c=[c],
                       edgecolors="none", zorder=3)

        print(f"  {tissue:<8} {ctrl.mean():10.3f} (n={ctrl.size})   "
              f"{aav.mean():10.3f} (n={aav.size})")

    ax.set_xlim(-0.55, nT - 0.45)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(nT))
    ax.set_xticklabels(TISSUE_LABELS, fontname=STY.font.name, fontsize=STY.font.tick)
    if y_max == Y_MAX:
        ax.set_yticks(Y_TICKS)
    ax.set_ylabel("AAV+ (% of total cells)",
                  fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_title("Non-brain tissue", fontname=STY.font.name, fontsize=STY.font.title)
    gt_axes(ax, STY, matlab_ticks=False)

    # gt_axes clears the grid, so the dashed y rules are added after it.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_GRID, linestyle=(0, (4, 3)), linewidth=0.6)
    ax.xaxis.grid(False)

    for dy, txt, col in ((0.00, "Uninjected", C_CTRL), (0.065, "AAV-injected", C_AAV)):
        ax.text(0.03, 0.885 - dy, txt, transform=ax.transAxes, ha="left", va="top",
                color=col, fontname=STY.font.name, fontsize=STY.font.leg)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig14_f_nonbrain", out_dir)
    return fig


if __name__ == "__main__":
    main()
