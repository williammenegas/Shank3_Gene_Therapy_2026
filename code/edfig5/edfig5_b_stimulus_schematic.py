"""Extended Data Figure 5, panel b: time near the stimulus against time in
each behavior category, one example animal.

Data: `hlabel_Red_sparky_1_control_2_12months.csv`, a 60 fps frame-by-frame
ethogram state code (1-80), and
`stim_dist_red_sparky_1_control_2_12months.csv`, the per-frame distance to
the enrichment stimulus (the bed). This script plots only this one animal
(Red) and does not read the other tracked animals.

Data selection: the first 1201 samples of the distance trace are discarded
before use. Raw per-frame state codes are renumbered through the fixed
80-entry permutation LIST_5 before category membership is tested.

Binning: frames are grouped into non-overlapping 10-minute windows
(60 fps x 600 s = 36000 frames; any trailing partial window is dropped). Within
each window, `chunk_frac_logical` returns the fraction of frames for which a
0/1 indicator is true.

Plot: five sub-panels, one per behavior category (Climbing, Active, Alone,
Social active, Social resting), each built from the state-code sets INDS_0 to
INDS_7 listed below. For each window, x is the fraction of frames in that
category and y the fraction of frames near the bed (distance < BED_THRESH =
100). A dashed gray ordinary-least-squares line is drawn through each
category's points for display; slope and intercept are printed at run time.
Axes are fixed to x in [0, 0.6] and y in [0, 1].

The panel is descriptive; no statistical test is applied.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig5", "edfig5_b_stimulus_schematic")
DATA_HLABEL = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_d_state_maps")

# ---- parameters ------------------------------------------------------------
FPS = 60
CHUNK_MIN = 10
CHUNK_FR = FPS * 60 * CHUNK_MIN     # 36000 frames per window
BED_THRESH = 100                    # distance below which a frame counts as near

# 80-entry, 1-based state-code renumbering permutation.
LIST_5 = np.array([
    1, 6, 29, 13, 36, 44, 2, 37, 60, 43, 62, 63, 18, 38, 14, 22, 3, 67, 7, 45,
    68, 8, 40, 69, 30, 9, 19, 70, 53, 25, 16, 71, 23, 72, 28, 73, 74, 32, 54,
    46, 4, 64, 26, 47, 65, 55, 27, 10, 48, 20, 41, 75, 33, 24, 56, 42, 39, 5,
    17, 34, 66, 49, 50, 76, 21, 61, 11, 58, 77, 51, 78, 15, 52, 35, 31, 79,
    12, 59, 57, 80,
])

# Eight base state-code groups (1-based, matching the renumbered labels).
INDS_0 = np.array([62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80])          # alone
INDS_1 = np.array([30, 31, 32, 33, 35, 40, 41, 42])                                          # near other climbing
INDS_2 = np.array([36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59])               # near other moving
INDS_3 = np.array([15, 19, 21, 22, 23, 24, 27, 29, 34])                                       # move to/away from other
INDS_4 = np.array([6, 7, 8, 9, 10, 11, 12, 28])                                               # large movements
INDS_5 = np.array([1, 2, 3, 4, 5, 13, 14, 16, 17, 18, 20, 25, 26])                            # climbing alone
INDS_6 = np.array([62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80,
                    60, 61, 63, 64, 65])                                                      # looking at other alone
INDS_7 = np.array([43, 53, 54, 55, 56, 57])                                                   # near other resting

# Five display categories, one sub-panel each.
CATS = [
    dict(name="Climbing", inds=INDS_5, color=np.array([102, 98, 215]) / 255),
    dict(name="Active", inds=np.unique(np.concatenate([INDS_4, INDS_3, INDS_1])),
         color=np.array([168, 95, 186]) / 255),
    dict(name="Alone", inds=np.unique(np.concatenate([INDS_0, INDS_6])),
         color=np.array([240, 92, 155]) / 255),
    dict(name="Social active", inds=INDS_2, color=np.array([251, 141, 123]) / 255),
    dict(name="Social resting", inds=INDS_7, color=np.array([254, 193, 93]) / 255),
]

HLABEL_RED = "hlabel_Red_sparky_1_control_2_12months.csv"
DIST_RED = "stim_dist_red_sparky_1_control_2_12months.csv"


def chunk_frac_logical(v, chunk_len):
    """Fraction of True values per non-overlapping window of `chunk_len`.

    v: a 0/1 or boolean vector. Consecutive samples are grouped into windows
    of `chunk_len` and averaged; a trailing partial window is dropped.
    """
    v = np.asarray(v, dtype=float).ravel()
    n_chunks = int(np.floor(v.size / chunk_len))
    if n_chunks < 1:
        return np.array([])
    v = v[: n_chunks * chunk_len]
    V = v.reshape(n_chunks, chunk_len)
    return np.nanmean(V, axis=1)


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    rB = np.loadtxt(os.path.join(DATA_HLABEL, HLABEL_RED))
    dist_red = np.loadtxt(os.path.join(data_dir, DIST_RED), delimiter=",")
    dist_red = dist_red[1201:]          # discard the first 1201 samples

    hlabel_r = LIST_5[rB.astype(int) - 1]

    labels = hlabel_r
    dist = dist_red
    n = min(labels.size, dist.size)
    labels = labels[:n]
    dist = dist[:n]

    near_bed = dist < BED_THRESH
    y_full = chunk_frac_logical(near_bed, CHUNK_FR)

    fig, axes = plt.subplots(1, len(CATS), figsize=(7.4, 1.6), facecolor="w")
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.30, top=0.82, wspace=0.55)

    print("edfig5_b_stimulus_schematic -- fraction near bed vs fraction in category (animal: Red)")
    for ax, cat in zip(axes, CATS):
        in_cat = np.isin(labels, cat["inds"])
        x = chunk_frac_logical(in_cat, CHUNK_FR)
        m = min(x.size, y_full.size)
        # Stored as fractions; the published axes are per cent.
        xu = 100.0 * x[:m]
        yu = 100.0 * y_full[:m]

        ax.scatter(xu, yu, s=STY.mk.pt, c=[cat["color"]], edgecolors="none")
        if xu.size >= 2 and np.ptp(xu) > 0:
            slope, intercept = np.polyfit(xu, yu, 1)
            xf = np.array([xu.min(), xu.max()])
            # dashed gray ordinary-least-squares fit line
            ax.plot(xf, slope * xf + intercept, "--", color=(0.40, 0.40, 0.40),
                    lw=STY.lw.fit, dashes=(4, 2))
            print(f"  {cat['name']:<15} n={m:2d}  slope={slope:+.3f}  intercept={intercept:+.3f}")
        else:
            print(f"  {cat['name']:<15} n={m:2d}  (too few/degenerate x for a fit line)")

        ax.set_title(cat["name"])
        ax.set_xlabel("Time in category (%)")
        ax.set_ylabel("Time near bed (%)")
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 100)
        gt_axes(ax, STY)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig5_b_stimulus_schematic", out_dir)
    return fig


if __name__ == "__main__":
    main()
