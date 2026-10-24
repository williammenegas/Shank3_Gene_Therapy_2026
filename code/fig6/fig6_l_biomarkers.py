"""Figure 6, panel l: liver and kidney biomarkers over the first five years.

Serum ALT and creatinine in treated and untreated animals, binned by year of
age. This script shows two of the nine analytes in the safety panel, one for
each organ, as a check that systemic AAV exposure left no mark on liver or
kidney chemistry over the life of the study.

Groups. The data file labels the two groups Control and Treatment; the
published figure labels them SH and GT, and this panel follows the figure.
Animal counts vary from bin to bin because not every animal was bled in every
year of its life; each printed n is the number of animals contributing to
that box, not a nominal group size.

Data. data/edfig11/bloodwork_data.csv, one row per blood draw with the
animal, its group, its age in days at the draw and the nine analytes. The
file is owned by the Extended Data Fig. 11 panel, which reads the whole set;
this panel reads the ALT_SGPT and CREATININE columns with Age_days and Group.

Each animal is averaged within each year bin before the quartiles are taken,
so a box summarizes one value per animal rather than one per draw and an
animal bled more often does not weigh more. That is the same per-animal,
per-bin reduction Extended Data Fig. 11 applies, over the same bin edges, so
a box here and the matching point in that figure's timecourse are the same
number.

Display. Two figures, one per published sub-panel. The five age bins sit at
integer x with the untreated group offset to x - 0.18 and the treated group
to x + 0.18. Each box runs from the first to the third quartile with a
near-black edge and median rule, and carries its animal values on top as
jittered dots; the published panel draws no whiskers, so the spread is
carried by the dots alone. Dashed rules mark the y ticks behind the data. The
colors are those of Extended Data Fig. 11, so the two figures read together.

The y axes reproduce the published panels: ALT from 0 to 40 ticked every 20,
and creatinine from 0 to 0.6 ticked every 0.2.

The panel is descriptive and no statistical test is applied, matching the
published figure, which carries no significance marks. Per-bin quartiles and
animal counts are printed to the console.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import gt_init, gt_axes, gt_figStyle, gt_saveFig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig11")
BLOODWORK_CSV = "bloodwork_data.csv"

# Year bins, the edges Extended Data Fig. 11 uses.
BIN_EDGES = [0, 365, 730, 1095, 1461, 1826]
BIN_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-5"]

# One entry per published sub-panel: stem, column, title, y label, y max,
# y tick step.
PANELS = [
    ("liver", "ALT_SGPT", "Liver biomarker", "ALT [U/L]", 40.0, 20.0),
    ("kidney", "CREATININE", "Kidney biomarker", "Creatinine [mg/dL]", 0.6, 0.2),
]

# ---- panel colors, as in Extended Data Fig. 11 ---------------------------
C_SH = (0.840, 0.450, 0.490)       # untreated animal dots, salmon
C_GT = (0.520, 0.500, 0.780)       # treated animal dots, lavender
C_SH_L = (0.905, 0.745, 0.755)     # untreated box fill
C_GT_L = (0.825, 0.810, 0.915)     # treated box fill
C_EDGE = (0.100, 0.100, 0.130)     # box edge and median rule
C_GRID = (0.650, 0.660, 0.670)     # dashed y rules

DX = 0.18          # group offset from the bin center
HW = 0.12          # half box width, in bin units
JIT = 0.15         # full width of the dot jitter, in bin units


def _box(ax, v, x0, fill):
    """One group's box at x0: a tinted body from the first to the third
    quartile with a near-black edge and median rule. The published panel
    draws no whiskers, so the spread is carried by the dots alone.
    Returns the quartiles for the console summary."""
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    ax.add_patch(Rectangle((x0 - HW, q1), 2 * HW, max(q3 - q1, 1e-9),
                           facecolor=fill, edgecolor=C_EDGE, lw=0.6, zorder=3))
    ax.plot([x0 - HW, x0 + HW], [q2, q2], "-", color=C_EDGE, lw=1.0, zorder=4)
    return q1, q2, q3


def main(data_dir=DATA, out_dir=HERE, rng_seed=7):
    STY = gt_init()
    rng = np.random.default_rng(rng_seed)

    T = pd.read_csv(os.path.join(data_dir, BLOODWORK_CSV))
    T["TimeBin"] = pd.cut(T["Age_days"], bins=BIN_EDGES, labels=False,
                          right=True) + 1
    columns = [p[1] for p in PANELS]
    binned = T.groupby(["Unique_ID", "Group", "TimeBin"],
                       as_index=False)[columns].mean()

    figs = []
    for stem, column, title, ylab, y_max, y_step in PANELS:
        AX_W, AX_H = 1.30, 0.80
        ML, MR, MB, MT = 0.60, 0.10, 0.46, 0.26
        FW, FH = ML + AX_W + MR, MB + AX_H + MT
        fig = plt.figure(figsize=(FW, FH), facecolor="w")
        ax = fig.add_axes([ML / FW, MB / FH, AX_W / FW, AX_H / FH])

        print(f"{title}, {ylab}")
        print(f"  {'bin':<5} {'untreated (SH)':>28}   {'treated (GT)':>28}")
        for j, label in enumerate(BIN_LABELS):
            cells = []
            for dx, group, dot, fill in ((-DX, "Control", C_SH, C_SH_L),
                                         (+DX, "Treatment", C_GT, C_GT_L)):
                sub = binned[(binned.TimeBin == j + 1) & (binned.Group == group)]
                v = sub[column].dropna().to_numpy(float)
                x0 = j + dx
                if v.size >= 2:
                    q1, q2, q3 = _box(ax, v, x0, fill)
                    cells.append(f"{q1:6.2f} {q2:6.2f} {q3:6.2f} (n={v.size})")
                else:
                    cells.append(f"{'':<20} (n={v.size})")
                ax.scatter(x0 + JIT * (rng.random(v.size) - 0.5), v, s=STY.mk.pt,
                           c=[dot], alpha=0.8, edgecolors="none", zorder=5)
            print(f"  {label:<5} {cells[0]:>28}   {cells[1]:>28}")

        ax.set_xlim(-0.5, len(BIN_LABELS) - 0.5)
        ax.set_ylim(0, y_max)
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS, fontname=STY.font.name,
                           fontsize=STY.font.tick)
        ax.set_yticks(np.arange(0, y_max + y_step / 2, y_step))
        ax.set_xlabel("Age (years)", fontname=STY.font.name, fontsize=STY.font.lab)
        ax.set_ylabel(ylab, fontname=STY.font.name, fontsize=STY.font.lab)
        ax.set_title(title, fontname=STY.font.name, fontsize=STY.font.title)
        gt_axes(ax, STY, matlab_ticks=False)

        # gt_axes clears the grid, so the dashed y rules are added after it.
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=C_GRID, linestyle=(0, (2, 2)), linewidth=0.6)
        ax.xaxis.grid(False)

        gt_figStyle(fig, STY)
        gt_saveFig(fig, f"fig6_l_biomarkers_{stem}", out_dir)
        figs.append(fig)

    return tuple(figs)


if __name__ == "__main__":
    main()
