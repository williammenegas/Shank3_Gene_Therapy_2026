"""
Figure 2, panel d: cage position of one example animal, colored by
behavior group.

The track file gives the animal's tracked (x, y) position for every frame;
the label file gives that frame's behavioral-state label (1-80), remapped
through the LIST_5 lookup shared with Figures 2a, 2g and 3g. Each frame's
position is drawn as a point colored by which of eight coarse behavior
groups its state belongs to (climbing alone, large movements, near other,
alone, near-other resting, and so on). The script produces six figures: all
groups overlaid, then each color group alone.

The script trims the first 1200 rows from the track array but not from the
label array, so the two differ in length by that amount. `_hlabel_select`
truncates to the shorter of the two, so only the leading frames covered by
both are drawn.

The point clouds are rasterized inside the otherwise-vector EPS and PDF
files; axes, ticks and text remain vector. This affects the file encoding
only.

No statistical test is applied; the panel is descriptive.

Data: mRed_tracks_sparky_1_control_2_12months.csv,
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
DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_d_state_maps")

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

# (state indices, color) for the combined plot; later layers draw over earlier ones.
LAYERS_COMBINED = [
    (INDS_5, np.array([102, 98, 215]) / 255.0),
    (INDS_4, np.array([168, 95, 186]) / 255.0),
    (INDS_3, np.array([168, 95, 186]) / 255.0),
    (INDS_1, np.array([168, 95, 186]) / 255.0),
    (INDS_0, np.array([240, 92, 155]) / 255.0),
    (INDS_6, np.array([240, 92, 155]) / 255.0),
    (INDS_2, np.array([251, 141, 123]) / 255.0),
    (INDS_7, np.array([254, 193, 93]) / 255.0),
]

# One entry per single-group plot: (state index lists sharing a color, color).
SEPARATE_PANELS = [
    ([INDS_5], np.array([102, 98, 215]) / 255.0),
    ([INDS_4, INDS_3, INDS_1], np.array([168, 95, 186]) / 255.0),
    ([INDS_0, INDS_6], np.array([240, 92, 155]) / 255.0),
    ([INDS_2], np.array([251, 141, 123]) / 255.0),
    ([INDS_7], np.array([254, 193, 93]) / 255.0),
]

AXIS_LIM = (200, 1100, -100, 800)


def _hlabel_select(arr_col, mask):
    """Select frames by state mask, truncating to the shorter of the track
    array and the label array (see module docstring)."""
    n = min(arr_col.shape[0], mask.size)
    return arr_col[:n][mask[:n]]


def _draw_layer(ax, X, Y, hlabel_r, inds_list, col):
    mask = np.zeros(hlabel_r.size, dtype=bool)
    for inds in inds_list:
        mask |= np.isin(hlabel_r, inds)
    xs = _hlabel_select(X, mask)
    ys = _hlabel_select(Y, mask)
    ax.scatter(xs, ys, s=1, c=[col], edgecolors="none", rasterized=True)


def _new_axes(STY):
    fig = plt.figure(figsize=(0.81, 0.78), facecolor="w")
    ax = fig.add_axes([0.20, 0.20, 0.76, 0.72])
    return fig, ax


def _style_axes(ax, fig, STY):
    ax.set_aspect("equal")
    ax.set_xlim(AXIS_LIM[0], AXIS_LIM[1])
    ax.set_ylim(AXIS_LIM[2], AXIS_LIM[3])
    # Three evenly spaced ticks per axis instead of the seven-bin house
    # locator, whose five labels merge visually at this panel size. Limits
    # and data are unchanged.
    ax.set_xticks([200, 600, 1000])
    ax.set_yticks([0, 400, 800])
    gt_axes(ax, STY, matlab_ticks=False)
    gt_figStyle(fig, STY)


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    tracks = pd.read_csv(os.path.join(data_dir, "mRed_tracks_sparky_1_control_2_12months.csv"),
                          header=None).to_numpy()
    tracks = tracks[1200:, :]           # leading 1200 frames trimmed
    X, Y = tracks[:, 0], tracks[:, 1]

    raw = pd.read_csv(os.path.join(data_dir, "hlabel_Red_sparky_1_control_2_12months.csv"),
                       header=None).to_numpy().astype(int).ravel()
    hlabel_r = LIST_5[raw - 1]          # labels are not trimmed

    print(f"tracks rows (post-trim)={tracks.shape[0]}, hlabel rows={hlabel_r.size} "
          f"(MATLAB indexes only the first {min(tracks.shape[0], hlabel_r.size)})")

    figs = []

    # All eight behavior groups overlaid.
    fig1, ax1 = _new_axes(STY)
    for inds, col in LAYERS_COMBINED:
        _draw_layer(ax1, X, Y, hlabel_r, [inds], col)
    _style_axes(ax1, fig1, STY)
    gt_saveFig(fig1, "fig2_d_state_maps_p1", out_dir)
    figs.append(fig1)

    # Each color group alone.
    for i, (inds_list, col) in enumerate(SEPARATE_PANELS):
        fig, ax = _new_axes(STY)
        _draw_layer(ax, X, Y, hlabel_r, inds_list, col)
        _style_axes(ax, fig, STY)
        gt_saveFig(fig, f"fig2_d_state_maps_p{i+2}", out_dir)
        figs.append(fig)

    return figs


if __name__ == "__main__":
    main()
