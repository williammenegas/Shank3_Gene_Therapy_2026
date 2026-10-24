"""Extended Data Figure 5, panel i: per-state usage, WT against SH, in the
box and face conditions.

Data: `usage_box.mat` and `usage_face.mat`, each holding `Bmat_WT` and
`Bmat_SH` (rows of 80 per-state usage fractions, one row per trial) together
with `ID_WT` and `ID_SH` giving the animal each row belongs to. The script
produces one figure per source file.

Aggregation, data selection and plotting follow panel d exactly, on the box and
face stimulus conditions instead of treat and object:

  * For each of 1000 animal-ID slots the matching rows are averaged across
    trials, ignoring NaN, into one 80-state usage vector. An aggregated row is
    retained only when all 80 states are non-NaN.
  * In the SH ID vector, ID 117 is set to NaN and ID 272 is relabeled to 111.
  * Each of the 80 states is drawn as a median point in log(WT) against
    log(SH) with gray bars spanning the x and y quartiles, colored by which of
    five behavior groups it belongs to (INDS_1 to INDS_5). A y = x reference
    line from (-9, -9) to (-2, -2) is drawn underneath, ticks are placed at
    log(0.0001), log(0.001), log(0.01) and log(0.1), and the aspect ratio is 1.
  * `_house_style` draws all four spines rather than the open L-shaped axes.
  * INDS_3 lists its first 16 codes twice; its only use is a color-table
    assignment, which repeated indices leave unchanged.

The panel is descriptive; no statistical test is applied.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig5", "edfig5_i_time_near_social")

N_STATES = 80
N_SLOTS = 1000

INDS_1 = np.array([1, 2, 3, 4, 5, 13, 14, 16, 17, 18, 20, 25, 26])                      # climbing
INDS_2 = np.array([30, 31, 32, 33, 35, 40, 41, 42, 15, 19, 21, 22, 23, 24, 27, 29, 34,
                    6, 7, 8, 9, 10, 11, 12, 28])                                        # active
INDS_3 = np.array([62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80,
                    62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80,
                    60, 61, 63, 64, 65])                                                # alone (first 16 codes repeat)
INDS_4 = np.array([36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59])          # social active
INDS_5 = np.array([43, 53, 54, 55, 56, 57])                                             # social resting


def _colors_to_use():
    c = np.zeros((N_STATES, 3))
    c[INDS_1 - 1] = np.array([102, 98, 215]) / 255
    c[INDS_2 - 1] = np.array([168, 95, 186]) / 255
    c[INDS_3 - 1] = np.array([240, 92, 155]) / 255
    c[INDS_4 - 1] = np.array([251, 141, 123]) / 255
    c[INDS_5 - 1] = np.array([254, 193, 93]) / 255
    return c


COLORS_TO_USE = _colors_to_use()


def _collect_animal_state_usage(Bmat, ID, n_states=N_STATES, n_slots=N_SLOTS):
    """Aggregate per-trial state usage into one vector per animal-ID slot.

    For each 1-based slot f = 1..n_slots, the matching rows of Bmat are
    averaged (ignoring NaN) into one n_states-long usage vector; slots with no
    matching rows stay all-NaN. A slot is written only when every state in its
    aggregated vector is non-NaN.
    """
    Bmat = np.asarray(Bmat, dtype=float)
    ID = np.asarray(ID, dtype=float).ravel()
    collect = np.full((n_slots, n_states), np.nan)
    for f in range(1, n_slots + 1):
        rows = Bmat[ID == f]
        if rows.shape[0] > 1:
            t1 = np.nanmean(rows, axis=0)
        elif rows.shape[0] == 1:
            t1 = rows[0]
        else:
            t1 = rows.reshape(0, n_states) if rows.ndim == 2 else np.array([])
        if t1.size > 0 and np.all(~np.isnan(t1)):
            collect[f - 1, :] = t1
    return collect


def _house_style(ax, STY):
    """Axis styling for this panel: Arial text and line weights taken from STY,
    ticks pointing out, grid off, and all four spines visible (unlike gt_axes,
    which hides the top and right spines)."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(STY.lw.axis)
    ax.tick_params(direction="out", width=STY.lw.axis, labelsize=STY.font.tick)
    ax.grid(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname("Arial")
        lbl.set_fontsize(STY.font.tick)
    ax.xaxis.label.set_fontname("Arial")
    ax.xaxis.label.set_fontsize(STY.font.lab)
    ax.yaxis.label.set_fontname("Arial")
    ax.yaxis.label.set_fontsize(STY.font.lab)
    ax.title.set_fontname("Arial")
    ax.title.set_fontsize(STY.font.title)


def _plot_with_std(ax, collect_1a, collect_1b, color, STY):
    """Draw one state as a median point with gray x- and y-quartile bars."""
    x_median = np.nanmedian(collect_1a)
    y_median = np.nanmedian(collect_1b)
    x_q1 = np.nanpercentile(collect_1a, 25)
    x_q3 = np.nanpercentile(collect_1a, 75)
    y_q1 = np.nanpercentile(collect_1b, 25)
    y_q3 = np.nanpercentile(collect_1b, 75)

    # One point summarizing the whole state, drawn at the median-marker size.
    ax.scatter([x_median], [y_median], s=STY.mk.mean, c=[color], edgecolors="none")
    ax.plot([x_q1, x_q3], [y_median, y_median], color=(0.5, 0.5, 0.5), lw=STY.lw.hair)
    ax.plot([x_median, x_median], [y_q1, y_q3], color=(0.5, 0.5, 0.5), lw=STY.lw.hair)


def _fix_sh_ids(ID_SH):
    ID_SH = np.asarray(ID_SH, dtype=float).ravel().copy()
    ID_SH[ID_SH == 117] = np.nan
    ID_SH[ID_SH == 272] = 111
    return ID_SH


def _plot_dataset(ax, mat_path, label, STY):
    S = load_struct(mat_path)
    ID_SH = _fix_sh_ids(S.ID_SH)
    collect_wt = _collect_animal_state_usage(S.Bmat_WT, S.ID_WT)
    collect_sh = _collect_animal_state_usage(S.Bmat_SH, ID_SH)

    # y = x reference line, drawn first so it sits underneath the points.
    ax.plot([-9, -2], [-9, -2], "k-", lw=STY.lw.hair)
    n_finite = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        for f in range(N_STATES):
            a = np.log(collect_wt[:, f])
            b = np.log(collect_sh[:, f])
            if np.any(np.isfinite(a)) and np.any(np.isfinite(b)):
                n_finite += 1
            _plot_with_std(ax, a, b, COLORS_TO_USE[f], STY)

    ax.set_title("state usage WT vs SH")
    xt = np.log([0.0001, 0.001, 0.01, 0.1])
    ax.set_xticks(xt)
    ax.set_yticks(xt)
    ax.set_xticklabels([".0001", ".001", ".01", ".1"])
    ax.set_yticklabels([".0001", ".001", ".01", ".1"])
    ax.set_aspect(1)
    _house_style(ax, STY)
    print(f"  {label}: {n_finite}/{N_STATES} states with finite WT & SH medians "
          f"(WT n_animals={np.sum(~np.isnan(collect_wt).all(axis=1))}, "
          f"SH n_animals={np.sum(~np.isnan(collect_sh).all(axis=1))})")


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    print("edfig5_i_time_near_social -- per-state usage WT vs SH, log-log median+IQR cross")

    fig1, ax1 = plt.subplots(figsize=(3.2, 3.0), facecolor="w")
    _plot_dataset(ax1, os.path.join(data_dir, "usage_box.mat"), "box", STY)
    gt_saveFig(fig1, "edfig5_i_time_near_social_box", out_dir)

    fig2, ax2 = plt.subplots(figsize=(3.2, 3.0), facecolor="w")
    _plot_dataset(ax2, os.path.join(data_dir, "usage_face.mat"), "face", STY)
    gt_saveFig(fig2, "edfig5_i_time_near_social_face", out_dir)

    return [fig1, fig2]


if __name__ == "__main__":
    main()
