"""
Figure 3, panel g: t-SNE embedding of behavioral states for one example
animal.

The coordinate file gives a two-dimensional t-SNE embedding of every frame of
one example session; the label file gives that frame's behavioral-state
label (1-80), remapped through the LIST_5 lookup shared with Figures 2d, 2g
and 3g. Two views of the same embedding:

  1. A per-frame scatter, colored by eight coarse behavior groups
     (climbing alone, large movement, movement to or from the other animal,
     near-other climbing, alone, alone looking at other, near other,
     near-other resting), using the same groupings and colors as Figure 2d
     and 2g.

  2. A per-state bubble map: the median embedding position of each of the 80
     states, with marker area proportional to that state's usage fraction
     (scaled and clipped to [0.2, 0.8] before sizing) and colored with the
     same 17-anchor CIELAB-interpolated palette as Figure 2a. Embedding
     coordinates are rescaled to a 0-400 box for this view.

The per-frame point cloud is rasterized inside the otherwise-vector EPS and
PDF files; axes, ticks and text remain vector. This affects the file
encoding only.

No statistical test is applied; the panel is descriptive.

Data: hcoord_Red_sparky_1_control_2_12months.csv,
hlabel_Red_sparky_1_control_2_12months.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig3", "fig3_g_approach_avoid")
DATA_HLABEL = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_d_state_maps")

LIST_5 = np.array([
    1, 6, 29, 13, 36, 44, 2, 37, 60, 43, 62, 63, 18, 38, 14, 22, 3, 67, 7, 45,
    68, 8, 40, 69, 30, 9, 19, 70, 53, 25, 16, 71, 23, 72, 28, 73, 74, 32, 54,
    46, 4, 64, 26, 47, 65, 55, 27, 10, 48, 20, 41, 75, 33, 24, 56, 42, 39, 5,
    17, 34, 66, 49, 50, 76, 21, 61, 11, 58, 77, 51, 78, 15, 52, 35, 31, 79,
    12, 59, 57, 80,
])

INDS_0 = [62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80]
INDS_1 = [30, 31, 32, 33, 35, 40, 41, 42]
INDS_2 = [36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59]
INDS_3 = [15, 19, 21, 22, 23, 24, 27, 29, 34]
INDS_4 = [6, 7, 8, 9, 10, 11, 12, 28]
INDS_5 = [1, 2, 3, 4, 5, 13, 14, 16, 17, 18, 20, 25, 26]
INDS_6 = [62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80, 60, 61, 63, 64, 65]
INDS_7 = [43, 53, 54, 55, 56, 57]

# (state indices, color); later layers draw over earlier ones.
LAYERS = [
    (INDS_5, np.array([102, 98, 215]) / 255.0),   # climbing (blue)
    (INDS_4, np.array([168, 95, 186]) / 255.0),    # large movement (purple)
    (INDS_3, np.array([168, 95, 186]) / 255.0),    # move to/from other (purple)
    (INDS_1, np.array([168, 95, 186]) / 255.0),    # near other climbing (purple)
    (INDS_0, np.array([240, 92, 155]) / 255.0),    # alone (red)
    (INDS_6, np.array([240, 92, 155]) / 255.0),    # alone (look at other) (red)
    (INDS_2, np.array([251, 141, 123]) / 255.0),   # near other (orange)
    (INDS_7, np.array([254, 193, 93]) / 255.0),    # near other resting (yellow)
]

# 17 RGB anchors (0-255) from which the 80-state palette is interpolated.
CM_INPUT = np.array([
    [101, 98, 214], [118, 97, 209], [155, 95, 192], [166, 95, 187],
    [184, 101, 176], [202, 110, 163], [230, 127, 141], [245, 137, 128],
    [251, 147, 119], [252, 164, 111], [253, 177, 103], [253, 189, 95],
    [250, 151, 124], [247, 131, 136], [244, 115, 144], [241, 98, 152],
    [240, 91, 154],
], dtype=float) / 255.0


def _load_hlabel(path, list5=LIST_5):
    raw = pd.read_csv(path, header=None).to_numpy().astype(int).ravel()
    return list5[raw - 1]      # labels are 1-based; index the lookup 0-based


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    rT = pd.read_csv(os.path.join(data_dir, "hcoord_Red_sparky_1_control_2_12months.csv"),
                      header=None).to_numpy()
    hlabel_r = _load_hlabel(os.path.join(DATA_HLABEL, "hlabel_Red_sparky_1_control_2_12months.csv"))

    # Per-frame scatter, colored by behavior group. The point cloud is
    # rasterized inside the vector file; axes, ticks and text stay vector.
    fig1 = plt.figure(figsize=(1.45, 1.55), facecolor="w")
    ax1 = fig1.add_axes([0.16, 0.14, 0.80, 0.78])
    for inds, col in LAYERS:
        mask = np.isin(hlabel_r, inds)
        ax1.scatter(rT[mask, 0], rT[mask, 1], s=1, c=[col], edgecolors="none",
                    rasterized=True)
    ax1.set_aspect("equal")
    gt_axes(ax1, STY)
    gt_figStyle(fig1, STY)
    gt_saveFig(fig1, "fig3_g_approach_avoid_p1", out_dir)

    # Per-state bubble map: median embedding position, sized by state usage.
    cmap = gt_lab_colormap(CM_INPUT, 80)
    print(f"colormap[0]={cmap[0]}  (anchor0={CM_INPUT[0]})")
    print(f"colormap[-1]={cmap[-1]}  (anchor-1={CM_INPUT[-1]})")

    x = 2 * rT[:, 0] + 200
    y = 2 * rT[:, 1] + 200

    density_per_label = np.array([(hlabel_r == f).sum() for f in range(1, 81)], dtype=float)
    density_per_label /= density_per_label.sum()
    alpha_values = np.clip(density_per_label * 20, 0.2, 0.8)

    fig2 = plt.figure(figsize=(1.85, 1.55), facecolor="w")   # wide enough for the two-line title
    ax2 = fig2.add_axes([0.14, 0.14, 0.82, 0.72])
    for f in range(1, 81):
        mask = hlabel_r == f
        if not np.any(mask):
            continue
        mx, my = np.nanmedian(x[mask]), np.nanmedian(y[mask])
        ax2.scatter(mx, my, s=250 * alpha_values[f - 1], c=[cmap[f - 1]],
                    edgecolors="k", linewidths=STY.mk.ew, zorder=2)
    ax2.set_xlim(0, 400)
    ax2.set_ylim(0, 400)
    ax2.set_aspect("equal")
    ax2.set_title("Regions Corresponding to Labels\n(median position, sized by usage)",
                   fontname=STY.font.name, fontsize=STY.font.title)
    ax2.set_xlabel("tSNE-1", fontname=STY.font.name, fontsize=STY.font.lab)
    ax2.set_ylabel("tSNE-2", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax2, STY)
    gt_figStyle(fig2, STY)
    gt_saveFig(fig2, "fig3_g_approach_avoid_p2", out_dir)

    print(f"n frames={hlabel_r.size}, states present={np.unique(hlabel_r).size}/80")

    return fig1, fig2


if __name__ == "__main__":
    main()
