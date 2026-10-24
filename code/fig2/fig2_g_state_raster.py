"""
Figure 2, panel g: behavior-group raster over time, two example cagemates.

For each of two example animals from one session, the 80-state ethogram
(remapped through the LIST_5 lookup shared with panel d and Figure 3g) is
collapsed into five coarse behavior-group indicator traces: climbing alone;
movement near or around the other animal; alone; near other; near-other
resting. A sixth trace marks frames in which the animal is within 100 units
of its cagemate. Each trace is drawn as a black-on-white raster strip over
the same window, WINDOW (frames 351000 to 639000, about 80 min at 60 fps).

The script trims 1200 leading samples from the distance traces but not from
the label traces, so the two differ in length. Both are long enough for
WINDOW and are sliced independently, so the difference does not affect the
panel.

No statistical test is applied; the panel is descriptive.

Data: hlabel_Red_sparky_1_control_2_12months.csv,
hlabel_Yellow_sparky_1_control_2_12months.csv,
mRed_dist_sparky_1_control_2_12months.csv,
mYellow_dist_sparky_1_control_2_12months.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_g_state_raster")
DATA_HLABEL_RED = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_d_state_maps")

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

WINDOW = slice(350999, 639000)   # frames 351000-639000, 1-based, as 0-based indices


def _load_hlabel(path):
    # dtype is pinned to avoid pandas' per-column type inference, which
    # dominates read time on a single-column file of this length.
    raw = pd.read_csv(path, header=None, dtype=np.int32).to_numpy().ravel()
    return LIST_5[raw - 1]


def _load_dist(path):
    # These files are a single CSV row about 660,000 columns wide. Reading
    # the raw text and parsing with np.fromstring is far faster than pandas'
    # row-wise parser on that shape.
    with open(path) as f:
        line = f.readline()
    return np.fromstring(line, sep=",")[1200:]


def _group_traces(hlabel):
    n = hlabel.size
    r1 = np.isin(hlabel, INDS_5).astype(float)
    r2 = (np.isin(hlabel, INDS_1) | np.isin(hlabel, INDS_3) | np.isin(hlabel, INDS_4)).astype(float)
    r3 = (np.isin(hlabel, INDS_0) | np.isin(hlabel, INDS_6)).astype(float)
    r4 = np.isin(hlabel, INDS_2).astype(float)
    r5 = np.isin(hlabel, INDS_7).astype(float)
    return r1, r2, r3, r4, r5


def _raster_figure(traces, STY, outname, out_dir):
    fig, axes = plt.subplots(6, 1, figsize=(2.44, 0.73), facecolor="w")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, hspace=0.6)
    for ax, tr in zip(axes, traces):
        ax.imshow(1 - tr[WINDOW][np.newaxis, :], cmap="gray", vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction="out", width=STY.lw.axis)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, outname, out_dir)
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    hlabel_r = _load_hlabel(os.path.join(DATA_HLABEL_RED, "hlabel_Red_sparky_1_control_2_12months.csv"))
    hlabel_y = _load_hlabel(os.path.join(data_dir, "hlabel_Yellow_sparky_1_control_2_12months.csv"))
    dist_red = _load_dist(os.path.join(data_dir, "mRed_dist_sparky_1_control_2_12months.csv"))
    dist_yellow = _load_dist(os.path.join(data_dir, "mYellow_dist_sparky_1_control_2_12months.csv"))

    r1, r2, r3, r4, r5 = _group_traces(hlabel_r)
    rN = (dist_red < 100).astype(float)
    y1, y2, y3, y4, y5 = _group_traces(hlabel_y)
    yN = (dist_yellow < 100).astype(float)

    print(f"Red window occupancy:    climb={r1[WINDOW].mean():.3f} move={r2[WINDOW].mean():.3f} "
          f"alone={r3[WINDOW].mean():.3f} near={r4[WINDOW].mean():.3f} rest={r5[WINDOW].mean():.3f} "
          f"close={rN[WINDOW].mean():.3f}")
    print(f"Yellow window occupancy: climb={y1[WINDOW].mean():.3f} move={y2[WINDOW].mean():.3f} "
          f"alone={y3[WINDOW].mean():.3f} near={y4[WINDOW].mean():.3f} rest={y5[WINDOW].mean():.3f} "
          f"close={yN[WINDOW].mean():.3f}")

    fig1 = _raster_figure([r1, r2, r3, r4, r5, rN], STY, "fig2_g_state_raster_p1", out_dir)
    fig2 = _raster_figure([y1, y2, y3, y4, y5, yN], STY, "fig2_g_state_raster_p2", out_dir)

    return fig1, fig2


if __name__ == "__main__":
    main()
