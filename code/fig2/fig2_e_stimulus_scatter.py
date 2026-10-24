"""
Figure 2, panel e: social-state occupancy against proximity, one
example animal per genotype.

For one example WT, SH and GT animal (IDs 238, 119 and 115 in the
condition 1, 2 and 3 arrays), the frame-by-frame 80-state ethogram label is
collapsed into two binary indicators: INDS_1, the climbing and alone states,
and INDS_2, the social and active states (near other, moving, facing other).
A third indicator marks frames in which the Dist array is below 150. The
three indicators are averaged over non-overlapping windows of WINDOW frames
(1 min at 60 fps), and the panel plots the first N_WINDOWS window means of
the social/active fraction against the close-proximity fraction, with a
dashed gray ordinary-least-squares fit line. Axes are square and are set per
animal in EXAMPLES.

The fit line and the Pearson correlation are display and reporting values;
no hypothesis test is applied.

The script reads only the ethogram, distance and ID arrays from
the .mat file, which is a large workspace dump.

Data: xy_object.mat
"""

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_e_stimulus_scatter")

INDS_1 = [1, 2, 3, 4, 5, 13, 14, 16, 17, 18, 20, 25, 26,
          6, 7, 9, 10, 12,
          30, 31, 32, 33, 35,
          40, 41, 42,
          62, 67, 69, 71, 77,
          66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80]

INDS_2 = [36, 37, 38, 39,
          8, 11, 28,
          15, 19, 21, 22, 23, 24, 27,
          29, 34,
          43, 53, 54, 55, 56, 57,
          44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59,
          60, 61, 63, 64, 65]

WINDOW = 3600        # frames per averaging window (1 min at 60 fps)
N_WINDOWS = 60       # windows plotted per animal

# (condition index, animal ID, genotype label, axis limits)
EXAMPLES = [
    (1, 238, "WT", (0, 100, 0, 100)),
    (2, 119, "SH", (0, 100, 0, 100)),
    (3, 115, "GT", (0, 80, 0, 80)),
]


def _load_example(mat_path, id_type, ind_to_try):
    S = sio.loadmat(mat_path, variable_names=[
        f"Bmat_condition_{id_type}", f"Dist_condition_{id_type}", f"ID_condition_{id_type}"])
    B = S[f"Bmat_condition_{id_type}"]
    D = S[f"Dist_condition_{id_type}"]
    ID = S[f"ID_condition_{id_type}"].ravel()

    rows = np.flatnonzero(ID == ind_to_try)
    if rows.size == 0:
        raise ValueError(f"No rows where ID_condition_{id_type} == {ind_to_try}")

    rB = B[rows, :500000].ravel()               # row-major flatten to one frame series
    dist_total = D[rows, :].ravel()

    ok = dist_total > 0
    n_sampled = min(int(ok.sum()), 500000)
    rB = rB[ok][:n_sampled]
    dist_total = dist_total[ok][:n_sampled]
    return rB, dist_total


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    mat_path = os.path.join(data_dir, "xy_object.mat")

    figs = []
    for id_type, ind_to_try, name, axlim in EXAMPLES:
        rB, dist_total = _load_example(mat_path, id_type, ind_to_try)

        a = np.isin(rB, INDS_1).astype(float)
        b = np.isin(rB, INDS_2).astype(float)
        d = (dist_total < 150).astype(float)

        n = d.size
        a1, b1, d1 = [], [], []
        f = 0
        while f < n - WINDOW - 1:
            a1.append(np.nanmean(a[f:f + WINDOW + 1]))
            b1.append(np.nanmean(b[f:f + WINDOW + 1]))
            d1.append(np.nanmean(d[f:f + WINDOW + 1]))
            f += WINDOW
        a1, b1, d1 = np.array(a1), np.array(b1), np.array(d1)

        n_use = min(N_WINDOWS, b1.size)
        # Stored as fractions; the published axes are per cent.
        bx, dy = 100.0 * b1[:n_use], 100.0 * d1[:n_use]

        fig = plt.figure(figsize=(1.26, 1.48), facecolor="w")
        ax = fig.add_axes([0.20, 0.16, 0.72, 0.70])
        ax.scatter(bx, dy, s=STY.mk.pt, c="k")
        if n_use >= 2:
            slope, intercept = np.polyfit(bx, dy, 1)
            xf = np.array([bx.min(), bx.max()])
            # Ordinary-least-squares fit line, drawn as a dashed gray guide.
            ax.plot(xf, slope * xf + intercept, "--", color=(0.40, 0.40, 0.40),
                    lw=STY.lw.fit, dashes=(4, 2))
            r = np.corrcoef(bx, dy)[0, 1]
        else:
            slope, intercept, r = np.nan, np.nan, np.nan
        ax.set_aspect("equal")
        ax.set_xlabel("Time social / active (%)")
        ax.set_ylabel("Time near stimulus (%)")
        ax.set_xlim(axlim[0], axlim[1])
        ax.set_ylim(axlim[2], axlim[3])
        gt_axes(ax, STY)
        gt_figStyle(fig, STY)
        gt_saveFig(fig, f"fig2_e_stimulus_scatter_{name}", out_dir)
        figs.append(fig)

        print(f"{name} (ID={ind_to_try}): n_windows={n_use}, slope={slope:.4f}, "
              f"intercept={intercept:.4f}, r={r:.4f}")

    return figs


if __name__ == "__main__":
    main()
