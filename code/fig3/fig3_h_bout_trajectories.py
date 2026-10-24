"""
Figure 3, panel h: movement-bout trajectories for one example pair per
genotype, and approach/avoid state usage on day 3 versus day 5 of pairing.

  1. Bout trajectories, one figure per example pair (WT, SH, GT). The
     subject's 80-state ethogram label is remapped through the LIST_5 lookup
     and padded with PAD_LABELS leading NaNs to align with the tracked
     frames. Bouts are contiguous runs of approach states (APPROACH_INDS),
     avoid states (AVOID_INDS) or neutral movement states (MOVE_INDS) within
     the first MINUTES_PER_VIDEO minutes. Each bout's path, re-centered on the
     cage center taken from the session's cage-boundary file, is drawn as a
     line with start and end markers, in three tiles (approach, avoid,
     neutral) on the shared axis box AXIS_BOX.

  2. Approach/avoid state usage on day 3 versus day 5, two figures. The
     script plots each animal's fraction of session time in the approach-like
     state group and, separately, the avoid-like state group, on the two days.
     Each animal's two days are joined by a faint line drawn through its own
     jittered points, with a black median and interquartile-range marker at
     each un-jittered column center. The three genotypes are separated
     horizontally, with the two days 2 x-units apart within a genotype and a
     wider gap between genotypes. y limits [0, 0.075].

Statistics: the bout-trajectory tiles carry no test. On the day 3 versus
day 5 figures, day is a within-animal contrast, so each genotype carries a
paired Student t-test drawn as a bracket over its two columns and labeled on
the house 0.05 / 0.01 / 0.001 scale. Each bracket sits just above its own
genotype's tallest point, clamped under the axis top.

Bouts are drawn in start-time order; this affects draw order only, not which
bouts are shown. Point jitter is drawn from an unseeded random number
generator and varies between runs; the medians and quartiles do not.

The script reads only the subject and cagemate track columns from the .mat
files, which also hold two unused animals. Cage-boundary files exist in
two .mat formats across sessions, and the gtstyle loader dispatches on the
file magic to handle both.

Data: tracks_<session>.mat, <session>_cage_boundaries.mat,
hlabel_Blue_<session>.csv, state_data_day3.mat, state_data_day5.mat
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig3", "fig3_h_bout_trajectories")

LIST_5 = np.array([
    1, 6, 29, 13, 36, 44, 2, 37, 60, 43, 62, 63, 18, 38, 14, 22, 3, 67, 7, 45,
    68, 8, 40, 69, 30, 9, 19, 70, 53, 25, 16, 71, 23, 72, 28, 73, 74, 32, 54,
    46, 4, 64, 26, 47, 65, 55, 27, 10, 48, 20, 41, 75, 33, 24, 56, 42, 39, 5,
    17, 34, 66, 49, 50, 76, 21, 61, 11, 58, 77, 51, 78, 15, 52, 35, 31, 79,
    12, 59, 57, 80,
])

KP_FRONT, KP_HEADCTR = 0, 3     # keypoint indices: front of head, head center
APPROACH_INDS = [29, 34]        # remapped states counted as approach
AVOID_INDS = [21, 22, 23, 24]   # remapped states counted as avoid
MOVE_INDS = list(range(6, 13))  # remapped states counted as neutral movement

PAD_LABELS = 1200               # leading NaN labels aligning labels to tracks
INPUT_FPS = 60                  # tracking frame rate
MINUTES_PER_VIDEO = 30          # session duration analyzed
AXIS_BOX = (-200, 600, -300, 500)   # cage-centered axis limits, in pixels

# (genotype label, session base name)
EXAMPLES = [
    ("WT", "pairtrial_sir_FA_3"),
    ("SH", "pairtrial_taquito_FA_3"),
    ("GT", "pairtrial_helios_FA_3"),
]


def _keypoints_xy(XY, idx_ctr, idx_front):
    xi = lambda k: 2 * k
    yi = lambda k: 2 * k + 1
    C = np.column_stack([XY[:, xi(idx_ctr)], XY[:, yi(idx_ctr)]])
    F = np.column_stack([XY[:, xi(idx_front)], XY[:, yi(idx_front)]])
    return C, F


def _find_bouts_noMin(lab, inds):
    mask = np.zeros(lab.shape, dtype=bool)
    for v in inds:
        mask |= (lab == v)
    d = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    bouts = np.column_stack([starts, ends]) if starts.size else np.zeros((0, 2), int)
    if bouts.shape[0]:
        bouts = bouts[bouts[:, 1] >= bouts[:, 0]]
    return bouts


def _remap_via_list5(lab):
    lab = np.asarray(lab, dtype=float).ravel()
    out = lab.copy()
    valid = np.isfinite(lab) & (lab >= 1) & (lab <= LIST_5.size)
    idx = np.round(lab[valid]).astype(int) - 1
    out[valid] = LIST_5[idx]
    return out


def _plot_bouts_basic(ax, bouts, Sctr, cage_ctr, col, STY):
    n_done = 0
    for s, e in bouts:
        XY = Sctr[s:e + 1, :] - cage_ctr
        ax.plot(XY[:, 0], XY[:, 1], "-", color=col, lw=STY.lw.trace, alpha=1)
        ax.scatter(XY[0, 0], XY[0, 1], s=STY.mk.pt, c=[col], edgecolors="none")
        ax.scatter(XY[-1, 0], XY[-1, 1], s=STY.mk.pt, c=[col], edgecolors="none")
        n_done += 1
    return n_done


def movement_bouts_single(avi_base, data_dir, STY, fig, tiles):
    ax1A, ax1V, ax1N = tiles

    XYs = sio.loadmat(os.path.join(data_dir, f"tracks_{avi_base}.mat"),
                       variable_names=["Blue"])["Blue"]
    # Cage-boundary files are a mix of .mat v7 and v7.3 (HDF5) across
    # sessions; gtstyle.loadmat dispatches on the file magic and reads both.
    boundaries = np.atleast_2d(loadmat(os.path.join(data_dir, f"{avi_base}_cage_boundaries.mat"))["boundaries"])
    cage_ctr = np.array([np.nanmean(boundaries[:, 0]), np.nanmean(boundaries[:, 1])])

    lab_path = os.path.join(data_dir, f"hlabel_Blue_{avi_base}.csv")
    lab_raw = pd.read_csv(lab_path, header=None, dtype=np.float64).to_numpy().ravel()
    lab = np.concatenate([np.full(PAD_LABELS, np.nan), lab_raw])
    lab = _remap_via_list5(lab)

    N = min(XYs.shape[0], lab.size, round(INPUT_FPS * 60 * MINUTES_PER_VIDEO))
    XYs = XYs[:N]
    lab = lab[:N]

    Sctr, _ = _keypoints_xy(XYs, KP_HEADCTR, KP_FRONT)

    boutsA = _find_bouts_noMin(lab, APPROACH_INDS)
    boutsV = _find_bouts_noMin(lab, AVOID_INDS)
    boutsM = _find_bouts_noMin(lab, MOVE_INDS)

    nA = _plot_bouts_basic(ax1A, boutsA, Sctr, cage_ctr, STY.clr.approach, STY)
    nV = _plot_bouts_basic(ax1V, boutsV, Sctr, cage_ctr, STY.clr.avoid, STY)
    nN = _plot_bouts_basic(ax1N, boutsM, Sctr, cage_ctr, STY.clr.neutral, STY)

    print(f"  {avi_base}: approach bouts={nA}, avoid bouts={nV}, neutral bouts={nN}")

    for ax, title in ((ax1A, "Approach"), (ax1V, "Avoid"), (ax1N, "Neutral")):
        ax.set_aspect("equal")
        ax.set_xlim(AXIS_BOX[0], AXIS_BOX[1])
        ax.set_ylim(AXIS_BOX[2], AXIS_BOX[3])
        gt_axes(ax, STY)


def _bout_trajectory_figure(name, avi_base, data_dir, out_dir, STY, idx):
    fig, axes = plt.subplots(1, 3, figsize=(3.6, 1.35), facecolor="w")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.82, wspace=0.5)
    movement_bouts_single(avi_base, data_dir, STY, fig, axes)
    for ax, title in zip(axes, ("Approach", "Avoid", "Neutral")):
        ax.set_title(title, fontname=STY.font.name, fontsize=STY.font.title, fontweight="bold")
    gt_figStyle(fig, STY)
    gt_saveFig(fig, f"fig3_h_bout_trajectories_p{idx}", out_dir)
    return fig


def _paired_state_panel(day3_path, day5_path, field2, field1_lookup, STY, out_dir, outname):
    S3 = load_struct(day3_path)
    S5 = load_struct(day5_path)

    d1wt, d1sh, d1gt = (np.asarray(S3[f"{g}{field2}"], float).ravel() for g in ("wt", "sh", "gt"))
    d2wt, d2sh, d2gt = (np.asarray(S5[f"{g}{field2}"], float).ravel() for g in ("wt", "sh", "gt"))

    fig = plt.figure(figsize=(1.6, 1.85), facecolor="w")
    ax = fig.add_axes([0.24, 0.14, 0.72, 0.72])

    # The two days are 2 x-units apart within a genotype, so the paired
    # connectors read as clear diagonals; genotypes are separated by a wider
    # gap.
    xs = {"wt": (1, 3), "sh": (6, 8), "gt": (11, 13)}
    data = {"wt": (d1wt, d2wt), "sh": (d1sh, d2sh), "gt": (d1gt, d2gt)}
    cidx = {"wt": 0, "sh": 1, "gt": 2}

    # Each day's points are jittered horizontally so overlapping animals are
    # separable, and each animal's connector is drawn through its own
    # jittered end points. JITTER stays well inside the day gap, so the two
    # days never overlap.
    rng = np.random.default_rng()
    JITTER = 0.12
    LINE_ALPHA, POINT_ALPHA = 0.30, 0.75
    for g in ("wt", "sh", "gt"):
        x1, x2 = xs[g]
        v1, v2 = data[g]
        # Stored as fractions; the published axis is per cent.
        v1, v2 = 100.0 * v1, 100.0 * v2
        col = STY.cmean[cidx[g]]
        j1 = (rng.random(v1.size) - 0.5) * 2 * JITTER
        j2 = (rng.random(v2.size) - 0.5) * 2 * JITTER
        n = min(v1.size, v2.size)
        for f in range(n):
            ax.plot([x1 + j1[f], x2 + j2[f]], [v1[f], v2[f]], "-",
                    color=col, lw=STY.lw.pair, alpha=LINE_ALPHA)
        ax.scatter(x1 + j1, v1, s=STY.mk.pt, c=[col], alpha=POINT_ALPHA,
                   edgecolors="none")
        ax.scatter(x2 + j2, v2, s=STY.mk.pt, c=[col], alpha=POINT_ALPHA,
                   edgecolors="none")

        # Black median and interquartile-range marker per day column, drawn
        # at the un-jittered column center and on top of the points.
        for xc, v in ((x1, v1), (x2, v2)):
            vf = v[np.isfinite(v)]
            if vf.size == 0:
                continue
            med, q25, q75 = np.median(vf), np.percentile(vf, 25), np.percentile(vf, 75)
            ax.plot([xc, xc], [q25, q75], "-", color="k", lw=STY.lw.se, zorder=5)
            ax.scatter([xc], [med], s=STY.mk.mean, c="k", zorder=6)

    ax.set_ylim(0, 7.5)

    # Day 3 versus day 5 is a within-animal contrast, so each genotype
    # carries a paired Student t-test, labeled on the house
    # 0.05 / 0.01 / 0.001 scale. Each bracket sits just above that
    # genotype's tallest visible point, clamped under the axis top.
    yl = ax.get_ylim()
    rng = yl[1] - yl[0]
    h = 0.012 * rng
    for g in ("wt", "sh", "gt"):
        x1, x2 = xs[g]
        v1, v2 = data[g]
        # Stored as fractions; the published axis is per cent.
        v1, v2 = 100.0 * v1, 100.0 * v2
        t_stat, p_paired, n_pair = gt_pairedt(v1, v2)
        vis_max = np.nanmax(np.concatenate([v1, v2])) if v1.size and v2.size else yl[0]
        base = min(vis_max + 0.06 * rng, yl[1] - 0.05 * rng)
        gt_drawSig(ax, x1, x2, base, h, gt_pstar(p_paired), STY)
        print(f"    {outname} {g.upper()}: n={n_pair} paired t={t_stat:+.3f} "
              f"p={p_paired:.4g} -> {gt_pstar(p_paired)}")

    gt_axes(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, outname, out_dir)

    print(f"  {outname}: WT day3/5={np.nanmean(d1wt):.4f}/{np.nanmean(d2wt):.4f}  "
          f"SH day3/5={np.nanmean(d1sh):.4f}/{np.nanmean(d2sh):.4f}  "
          f"GT day3/5={np.nanmean(d1gt):.4f}/{np.nanmean(d2gt):.4f}")
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    figs = []

    print("Bout trajectories:")
    for i, (name, avi_base) in enumerate(EXAMPLES):
        figs.append(_bout_trajectory_figure(name, avi_base, data_dir, out_dir, STY, i + 1))

    print("Approach/avoid usage, day3 vs day5:")
    figs.append(_paired_state_panel(
        os.path.join(data_dir, "state_data_day3.mat"),
        os.path.join(data_dir, "state_data_day5.mat"),
        "2", None, STY, out_dir, "fig3_h_bout_trajectories_p4"))
    figs.append(_paired_state_panel(
        os.path.join(data_dir, "state_data_day3.mat"),
        os.path.join(data_dir, "state_data_day5.mat"),
        "1", None, STY, out_dir, "fig3_h_bout_trajectories_p5"))

    return figs


if __name__ == "__main__":
    main()
