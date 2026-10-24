"""Extended Data Figure 6, panel h: movement-bout trajectories and day 3
against day 5 approach and avoid usage.

This script produces five figures.

  p1-p3  Cage-centered movement-bout paths for one example pair per group:
         `pairtrial_mandarin_FA_6` (WT), `pairtrial_nathanial_FA_6` (SH) and
         `pairtrial_lars_FA_6` (GT). The subject is Blue and the cagemate Red.
         Each trial loads `tracks_<trial>.mat`, `<trial>_cage_boundaries.mat`
         (the cage center is the mean of the first two boundary columns) and
         `hlabel_Blue_<trial>.csv` (one state label per frame). Labels are
         front-padded with PAD_LABELS = 1200 NaN frames and then renumbered
         through the fixed 80-entry permutation LIST_5. Analysis is capped at
         the first MINUTES_PER_VIDEO = 30 minutes at INPUT_FPS = 60.

         A bout is a run of consecutive frames labeled approach
         (APPROACH_INDS = 29, 34), avoid (AVOID_INDS = 21-24) or neutral
         (MOVE_INDS = 6-12). For every bout the script draws the subject's
         head-center path, translated so that the cage center is the
         origin, as a line with filled start and end markers, into an
         approach / avoid / neutral triplet of axes spanning x in
         [-200, 600] and y in [-300, 500], with y increasing downward.

  p4/p5  Day 3 against day 5 paired usage, from `state_data_day3.mat` and
         `state_data_day5.mat`. The approach figure uses variables wt2, sh2 and
         gt2; the avoid figure uses wt1, sh1 and gt1. Both files carry wt1-wt4,
         sh1-sh4 and gt1-gt4. For each of up to 1000 animals a thin line joins
         the day 3 value (x = 1, 5, 9 for WT, SH, GT) to the day 5 value
         (x = 2, 6, 10), and all day 3 and day 5 values are scattered on top in
         the group color. The y axis is fixed to [0, 0.075]. These two figures
         carry no axis labels or title.

Bout draw order is randomised without a seed. MAX_BOUTS_PER_TYPE is far larger
than the bout counts here, so the order affects only which paths overlap on
top of which.

The panel is descriptive; no statistical test is applied.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig6", "edfig6_h_bout_trajectories")
DATA_STATE = os.path.join(HERE, "..", "..", "data", "fig3", "fig3_h_bout_trajectories")

LIST_5 = np.array([
    1, 6, 29, 13, 36, 44, 2, 37, 60, 43, 62, 63, 18, 38, 14, 22, 3, 67, 7, 45, 68, 8, 40, 69,
    30, 9, 19, 70, 53, 25, 16, 71, 23, 72, 28, 73, 74, 32, 54, 46, 4, 64, 26, 47, 65, 55, 27,
    10, 48, 20, 41, 75, 33, 24, 56, 42, 39, 5, 17, 34, 66, 49, 50, 76, 21, 61, 11, 58, 77, 51,
    78, 15, 52, 35, 31, 79, 12, 59, 57, 80,
])

APPROACH_INDS = np.array([29, 34])
AVOID_INDS = np.arange(21, 25)      # 21:24
MOVE_INDS = np.arange(6, 13)        # 6:12  ("neutral")

KP_FRONT = 1
KP_HEADCTR = 4

PAD_LABELS = 1200
INPUT_FPS = 60
MINUTES_PER_VIDEO = 30
MAX_BOUTS_PER_TYPE = 50000
LINE_ALPHA = 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _keypoints_xy(XY, idx_ctr, idx_front):
    xi = lambda k: 2 * (k - 1)
    yi = lambda k: 2 * (k - 1) + 1
    C = np.column_stack([XY[:, xi(idx_ctr)], XY[:, yi(idx_ctr)]])
    F = np.column_stack([XY[:, xi(idx_front)], XY[:, yi(idx_front)]])
    return C, F


def _remap_via_list(lab, list5):
    lab2 = lab.copy()
    valid = ~np.isnan(lab2) & np.isfinite(lab2) & (lab2 >= 1) & (lab2 <= list5.size)
    idx = np.round(lab2[valid]).astype(int)
    lab2[valid] = list5[idx - 1]
    return lab2


def _find_bouts(lab, inds):
    mask = np.isin(lab, inds)
    ext = np.concatenate(([False], mask, [False])).astype(int)
    d = np.diff(ext)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    if starts.size == 0:
        return np.zeros((0, 2), dtype=int)
    bouts = np.column_stack([starts, ends])
    return bouts[bouts[:, 1] >= bouts[:, 0]]


def _plot_bouts_basic(ax, bouts, Sctr, cage_ctr, col, rng, STY, endpoint_size=None,
                       line_alpha=LINE_ALPHA, line_width=None, n_cap=MAX_BOUTS_PER_TYPE):
    if endpoint_size is None:
        endpoint_size = STY.mk.pt
    if line_width is None:
        line_width = STY.lw.trace   # individual bout path -> trace role
    if bouts.shape[0] == 0:
        return 0
    order = rng.permutation(bouts.shape[0])   # unseeded: draw order varies per run
    n_done = 0
    for ii in order:
        if n_done >= n_cap:
            break
        s, e = bouts[ii]
        XYr = Sctr[s:e + 1, :] - cage_ctr    # cage-centered: translate to the cage center
        ax.plot(XYr[:, 0], XYr[:, 1], "-", color=col, alpha=line_alpha, linewidth=line_width)
        ax.scatter([XYr[0, 0]], [XYr[0, 1]], s=endpoint_size, c=[col], edgecolors="none")
        ax.scatter([XYr[-1, 0]], [XYr[-1, 1]], s=endpoint_size, c=[col], edgecolors="none")
        n_done += 1
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-200, 600)
    ax.set_ylim(-300, 500)
    ax.invert_yaxis()   # image-style axes: y increases downward
    return n_done


def _movement_bouts_single(data_dir, out_dir, STY, rng, avi_base, label,
                            subject_color="Blue", cagemate_color="Red"):
    P_tracks = os.path.join(data_dir, f"tracks_{avi_base}.mat")
    P_bounds = os.path.join(data_dir, f"{avi_base}_cage_boundaries.mat")
    P_hlab = os.path.join(data_dir, f"hlabel_{subject_color}_{avi_base}.csv")

    S = load_struct(P_tracks)
    B = load_struct(P_bounds)
    boundaries = np.asarray(B.boundaries, dtype=float)
    cage_ctr = np.array([np.nanmean(boundaries[:, 0]), np.nanmean(boundaries[:, 1])])

    XYs = np.asarray(S[subject_color], dtype=float)
    XYo = np.asarray(S[cagemate_color], dtype=float)

    # pandas.read_csv is used instead of np.loadtxt for speed on these
    # single-column files of roughly a million rows; the values are identical.
    lab_raw = pd.read_csv(P_hlab, header=None).to_numpy().ravel().astype(float)
    lab = np.concatenate([np.full(PAD_LABELS, np.nan), lab_raw])
    lab = _remap_via_list(lab, LIST_5)

    cap_frames_src = max(1, round(INPUT_FPS * 60 * MINUTES_PER_VIDEO))
    N = min(XYs.shape[0], XYo.shape[0], lab.size, cap_frames_src)
    XYs = XYs[:N, :]
    XYo = XYo[:N, :]
    lab = lab[:N]
    # The position-smoothing window is 1, so no smoothing is applied.

    Sctr, _ = _keypoints_xy(XYs, KP_HEADCTR, KP_FRONT)
    Octr, _ = _keypoints_xy(XYo, KP_HEADCTR, KP_FRONT)   # not used in cage-centered coordinates

    boutsA = _find_bouts(lab, APPROACH_INDS)
    boutsV = _find_bouts(lab, AVOID_INDS)
    boutsM = _find_bouts(lab, MOVE_INDS)

    STY_ = STY
    COL_APPROACH = STY_.clr.approach
    COL_AVOID = STY_.clr.avoid
    COL_NEUTRAL = STY_.clr.neutral

    # Wider canvas plus tight_layout so the three axis titles do not overlap.
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), facecolor="w")
    nA = _plot_bouts_basic(axes[0], boutsA, Sctr, cage_ctr, COL_APPROACH, rng, STY)
    nV = _plot_bouts_basic(axes[1], boutsV, Sctr, cage_ctr, COL_AVOID, rng, STY)
    nM = _plot_bouts_basic(axes[2], boutsM, Sctr, cage_ctr, COL_NEUTRAL, rng, STY)

    for ax, title in zip(axes, ("Approach", "Avoid", "Neutral")):
        ax.set_xlabel("X (rotated/translated)", fontname=STY.font.name, fontweight="bold")
        ax.set_ylabel("Y (rotated/translated)", fontname=STY.font.name, fontweight="bold")
        ax.set_title(f"{title} - Movement bouts (Cage-centered)",
                      fontname=STY.font.name, fontweight="bold", fontsize=STY.font.title)
        gt_axes(ax, STY)

    fig.tight_layout()
    gt_figStyle(fig, STY)
    gt_saveFig(fig, f"edfig6_h_bout_trajectories_p{['wt','sh','gt'].index(label)+1}_{label}_bouts", out_dir)

    print(f"  {label} ({avi_base}): approach bouts={boutsA.shape[0]} (plotted {nA}), "
          f"avoid bouts={boutsV.shape[0]} (plotted {nV}), "
          f"neutral bouts={boutsM.shape[0]} (plotted {nM})")
    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()   # unseeded: bout draw order varies per run

    figs = []

    # ---- p1-p3: example movement-bout paths, one pair per group -----------
    for avi_base, label in (
        ("pairtrial_mandarin_FA_6", "wt"),
        ("pairtrial_nathanial_FA_6", "sh"),
        ("pairtrial_lars_FA_6", "gt"),
    ):
        fig = _movement_bouts_single(data_dir, out_dir, STY, rng, avi_base, label)
        figs.append(fig)

    # ---- p4/p5: day3 vs day5 paired approach/avoid usage -------------------
    S3 = load_struct(os.path.join(DATA_STATE, "state_data_day3.mat"))
    S5 = load_struct(os.path.join(DATA_STATE, "state_data_day5.mat"))

    def _paired_fig(key, suffix, ylim):
        d1wt = np.asarray(S3[f"wt{key}"], dtype=float).ravel()
        d1sh = np.asarray(S3[f"sh{key}"], dtype=float).ravel()
        d1gt = np.asarray(S3[f"gt{key}"], dtype=float).ravel()
        d2wt = np.asarray(S5[f"wt{key}"], dtype=float).ravel()
        d2sh = np.asarray(S5[f"sh{key}"], dtype=float).ravel()
        d2gt = np.asarray(S5[f"gt{key}"], dtype=float).ravel()

        # Stored as fractions; the published axis is per cent.
        d1wt, d1sh, d1gt = 100.0 * d1wt, 100.0 * d1sh, 100.0 * d1gt
        d2wt, d2sh, d2gt = 100.0 * d2wt, 100.0 * d2sh, 100.0 * d2gt

        fig, ax = plt.subplots(facecolor="w")
        n = min(1000, d1wt.size, d1sh.size, d1gt.size, d2wt.size, d2sh.size, d2gt.size)

        # Up to 3000 paired segments (1000 per group) are batched into one
        # LineCollection per group rather than drawn one Line2D at a time;
        # the segments, colors and coordinates are the same either way.
        segs_wt = np.stack([np.column_stack([[1, 2], [d1wt[f], d2wt[f]]]) for f in range(n)])
        segs_sh = np.stack([np.column_stack([[5, 6], [d1sh[f], d2sh[f]]]) for f in range(n)])
        segs_gt = np.stack([np.column_stack([[9, 10], [d1gt[f], d2gt[f]]]) for f in range(n)])
        # One thin line per animal, joining its day 3 and day 5 values.
        for segs, color in ((segs_wt, STY.cmean[0]), (segs_sh, STY.cmean[1]), (segs_gt, STY.cmean[2])):
            ax.add_collection(LineCollection(segs, colors=[color], linewidths=STY.lw.pair))
        ax.autoscale_view()

        ax.scatter(np.full(n, 1), d1wt[:n], s=STY.mk.pt, c=[STY.cmean[0]])
        ax.scatter(np.full(n, 5), d1sh[:n], s=STY.mk.pt, c=[STY.cmean[1]])
        ax.scatter(np.full(n, 9), d1gt[:n], s=STY.mk.pt, c=[STY.cmean[2]])
        ax.scatter(np.full(n, 2), d2wt[:n], s=STY.mk.pt, c=[STY.cmean[0]])
        ax.scatter(np.full(n, 6), d2sh[:n], s=STY.mk.pt, c=[STY.cmean[1]])
        ax.scatter(np.full(n, 10), d2gt[:n], s=STY.mk.pt, c=[STY.cmean[2]])

        ax.set_ylim(*ylim)
        gt_axes(ax, STY)
        gt_figStyle(fig, STY)
        gt_saveFig(fig, f"edfig6_h_bout_trajectories_p{len(figs)+1}_{suffix}", out_dir)
        print(f"  {suffix}: n={n}, day3 means={np.nanmean(d1wt):.4f}/{np.nanmean(d1sh):.4f}/"
              f"{np.nanmean(d1gt):.4f}, day5 means={np.nanmean(d2wt):.4f}/{np.nanmean(d2sh):.4f}/"
              f"{np.nanmean(d2gt):.4f}")
        return fig

    figs.append(_paired_fig("2", "approach_day3v5", (0, 7.5)))
    figs.append(_paired_fig("1", "avoid_day3v5", (0, 7.5)))

    return figs


if __name__ == "__main__":
    main()
