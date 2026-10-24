"""Extended Data Figure 6, panel g: approach and avoid behavioral-state map
for one example animal.

Data: `hlabel_Red_sparky_1_control_2_12months.csv`, one raw state label per
frame (codes 1-80), and `hcoord_Red_sparky_1_control_2_12months.csv`, the
corresponding t-SNE (x, y) coordinates, roughly 660,000 frames. Raw state
labels are renumbered through the fixed 80-entry permutation LIST_5.

This script produces two figures.

  1. All frames scattered at their t-SNE coordinate, colored by which of
     eight semantic clusters the frame's state belongs to: climbing alone,
     large movements, move to or from other, near other climbing, alone, alone
     looking at other, near other, near other resting. The clusters are drawn
     in that order, so later groups overpaint earlier ones where coordinates
     coincide.

     These scatter calls are rasterized. Written as vector paths the point
     cloud produces a very large EPS; rasterizing embeds it as a single raster
     image inside an otherwise vector figure, leaving axes, ticks and text as
     vector. Coordinates, colors and data are unchanged.

  2. Per-state summary: for each of the 80 state codes, one marker at the
     median (x, y) of that state's points, rescaled to 2 * value + 200, with
     area 250 * alpha where alpha = clip(density * 20, 0.2, 0.8) and density is
     the state's share of all frames. Colors come from an 80-entry map built
     by interpolating the 17 CM_INPUT anchor colors through CIELAB
     (`gt_lab_colormap`).

The panel is descriptive; no statistical test is applied.
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

HLABEL_FILE = "hlabel_Red_sparky_1_control_2_12months.csv"
HCOORD_FILE = "hcoord_Red_sparky_1_control_2_12months.csv"

LIST_5 = np.array([
    1, 6, 29, 13, 36, 44, 2, 37, 60, 43, 62, 63, 18, 38, 14, 22, 3, 67, 7, 45, 68, 8, 40, 69,
    30, 9, 19, 70, 53, 25, 16, 71, 23, 72, 28, 73, 74, 32, 54, 46, 4, 64, 26, 47, 65, 55, 27,
    10, 48, 20, 41, 75, 33, 24, 56, 42, 39, 5, 17, 34, 66, 49, 50, 76, 21, 61, 11, 58, 77, 51,
    78, 15, 52, 35, 31, 79, 12, 59, 57, 80,
])

INDS_0 = np.array([62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80])   # alone
INDS_1 = np.array([30, 31, 32, 33, 35, 40, 41, 42])                                    # near other climbing
INDS_2 = np.array([36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59])         # near other moving/facing
INDS_3 = np.array([15, 19, 21, 22, 23, 24, 27, 29, 34])                                # move to/from other
INDS_4 = np.array([6, 7, 8, 9, 10, 11, 12, 28])                                        # large movements
INDS_5 = np.array([1, 2, 3, 4, 5, 13, 14, 16, 17, 18, 20, 25, 26])                     # climbing alone
INDS_6 = np.array([62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80,
                    60, 61, 63, 64, 65])                                               # looking at other alone
INDS_7 = np.array([43, 53, 54, 55, 56, 57])                                            # near other resting

CM_INPUT = np.array([
    [101, 98, 214], [118, 97, 209], [155, 95, 192], [166, 95, 187], [184, 101, 176],
    [202, 110, 163], [230, 127, 141], [245, 137, 128], [251, 147, 119], [252, 164, 111],
    [253, 177, 103], [253, 189, 95], [250, 151, 124], [247, 131, 136], [244, 115, 144],
    [241, 98, 152], [240, 91, 154],
], dtype=float) / 255.0


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    # pandas.read_csv is used instead of np.loadtxt for speed on these
    # 660,000-row files; the parsed values are identical.
    rB = pd.read_csv(os.path.join(DATA_HLABEL, HLABEL_FILE), header=None).to_numpy().ravel()
    rT = pd.read_csv(os.path.join(data_dir, HCOORD_FILE), header=None).to_numpy()

    # renumber the raw state codes through the fixed permutation
    hlabel_r = LIST_5[rB.astype(int) - 1]

    figs = []

    # ---- figure 1: all frames, colored by eight-cluster grouping ---------
    fig1, ax1 = plt.subplots(facecolor="w")
    groups = [
        (INDS_5, np.array([102, 98, 215]) / 255.0),    # climbing (blue)
        (INDS_4, np.array([168, 95, 186]) / 255.0),    # large movement (purple)
        (INDS_3, np.array([168, 95, 186]) / 255.0),    # move to/from other (purple)
        (INDS_1, np.array([168, 95, 186]) / 255.0),    # near other climbing (purple)
        (INDS_0, np.array([240, 92, 155]) / 255.0),    # alone (red)
        (INDS_6, np.array([240, 92, 155]) / 255.0),    # alone (look at other) (red)
        (INDS_2, np.array([251, 141, 123]) / 255.0),   # near other (orange)
        (INDS_7, np.array([254, 193, 93]) / 255.0),    # near other resting (yellow)
    ]
    for inds, color in groups:
        mask = np.isin(hlabel_r, inds)
        # rasterized=True embeds this dense point cloud as a raster image
        # inside the otherwise vector EPS/PDF; axes, ticks and text remain
        # vector. Affects file size and rendering time only.
        ax1.scatter(rT[mask, 0], rT[mask, 1], s=1, c=[color], marker="o",
                    edgecolors="none", rasterized=True)
    ax1.set_aspect("equal", adjustable="box")
    gt_axes(ax1, STY)
    gt_figStyle(fig1, STY)
    gt_saveFig(fig1, "edfig6_g_approach_avoid_p1_tsne_states", out_dir)
    figs.append(fig1)

    # ---- figure 2: per-state median summary -------------------------------
    # Marker size is driven by density_per_label, the share of all frames
    # carrying that state code.
    x2 = 2 * rT[:, 0] + 200
    y2 = 2 * rT[:, 1] + 200

    density_per_label = np.array([np.sum(hlabel_r == f) for f in range(1, 81)], dtype=float)
    total = density_per_label.sum()
    density_per_label = density_per_label / total if total > 0 else density_per_label
    alpha_values = np.clip(density_per_label * 20.0, 0.2, 0.8)

    cmap = gt_lab_colormap(CM_INPUT, 80)

    fig2, ax2 = plt.subplots(facecolor="w")
    for f in range(1, 81):
        m = hlabel_r == f
        if not np.any(m):
            continue
        mx = np.nanmedian(x2[m])
        my = np.nanmedian(y2[m])
        ax2.scatter(mx, my, s=250 * alpha_values[f - 1], c=[cmap[f - 1]],
                    edgecolors="k", linewidths=0.5)
    ax2.set_xlim(0, 400)
    ax2.set_ylim(0, 400)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Regions Corresponding to Labels\n(median position, sized by usage)",
                    fontname=STY.font.name, fontsize=STY.font.title)
    ax2.set_xlabel("tSNE-1", fontname=STY.font.name, fontsize=STY.font.lab)
    ax2.set_ylabel("tSNE-2", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax2, STY)
    gt_figStyle(fig2, STY)
    gt_saveFig(fig2, "edfig6_g_approach_avoid_p2_tsne_watershed_medians", out_dir)
    figs.append(fig2)

    print(f"n_frames={rB.size}, n_states_present={(density_per_label>0).sum()}/80")

    return figs


if __name__ == "__main__":
    main()
