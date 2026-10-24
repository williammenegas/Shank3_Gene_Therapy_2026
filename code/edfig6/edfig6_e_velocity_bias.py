"""Extended Data Figure 6, panel e: bout-wise relative velocity, one example
pair per group.

This script draws three example trials, one figure each:
`unicorn_pairtrial_horatio_FA_6` (WT-WT pair),
`yertle_pairtrial_poseidon_FA_6` (SH-WT pair) and
`lemon_pairtrial_filo_FA_6` (GT-WT pair).

Data: `tracks_<trial>.mat`, holding keypoint tracks for the four colored
animals. The subject is Blue and the cagemate Red. Head-center velocity is
differenced frame to frame and scaled by FPS, then projected onto the unit
vector along the subject-cagemate axis to give each animal's relative velocity
in pixels per second. Coordinates are cage-centered, i.e. a pure translation.

Bout extraction: the first MINUTES_TO_ANALYZE = 30 minutes at FPS = 60 are
used. A bout is a run of consecutive frames in which an animal's own speed
exceeds THRESH_PX_PER_FRAME = 5 px/frame (300 px/s), lasting at least
MIN_BOUT_LEN_FR = 6 frames. Each bout contributes the mean relative velocity
of both animals over that bout.

Sign convention: the underlying computation gives positive for avoiding
(increasing separation) and negative for approaching. The plotted values are
multiplied by -1 for display, so on the figure positive is approaching
("Toward", green, right) and negative is avoiding ("Away", purple, left). Only
the plotted sign is changed.

Plot, per trial:
  left   Mirrored histogram of per-bout mean relative velocity, subject bouts
         above the axis and cagemate bouts below, on FIXED_BINS (50 px/s bins
         spanning FIXED_LIM = [-1000, 1000] px/s), expressed as a percentage of
         each side's bout count.
  right  Paired subject and cagemate swarm of the same per-bout means, with a
         black mean and SD marker and the bout count printed beneath.

`_local_swarmjit` lays points out deterministically: values are binned by
magnitude, and within each bin they are sorted and spread with a linspace, so
the horizontal offset increases with value inside each bin. It uses no random
number generator.

The panel is descriptive; no statistical test is applied.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig6", "edfig6_e_velocity_bias")

KP_FRONT = 1     # 1-based keypoint index
KP_HEADCTR = 4

FPS = 60
MINUTES_TO_ANALYZE = 30
THRESH_PX_PER_FRAME = 5
MIN_BOUT_LEN_FR = 6

FIXED_LIM = (-1000.0, 1000.0)
FIXED_BINS = np.arange(-1000, 1000 + 1, 50)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _keypoints_xy(XY, idx_ctr, idx_front):
    xi = lambda k: 2 * (k - 1)       # 0-based column holding x for keypoint k
    yi = lambda k: 2 * (k - 1) + 1
    C = np.column_stack([XY[:, xi(idx_ctr)], XY[:, yi(idx_ctr)]])
    F = np.column_stack([XY[:, xi(idx_front)], XY[:, yi(idx_front)]])
    return C, F


def _diff_pad(XY):
    D = np.diff(XY, axis=0)
    if D.shape[0] == 0:
        return np.zeros_like(XY)
    return np.vstack([D[0:1, :], D])


def _unit_rows(X):
    r = np.sqrt(np.sum(X ** 2, axis=1))
    u = np.zeros_like(X)
    ok = r > 0
    u[ok, :] = X[ok, :] / r[ok, None]
    return u


def _rowdot(A, B):
    return np.nansum(A * B, axis=1)


def _rowmag(A):
    return np.sqrt(np.sum(A ** 2, axis=1))


def _find_bouts(mask, min_len):
    mask = np.asarray(mask, dtype=bool).ravel()
    ext = np.concatenate(([False], mask, [False])).astype(int)
    d = np.diff(ext)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    if starts.size == 0:
        return np.zeros((0, 2), dtype=int)
    bouts = np.column_stack([starts, ends])
    lens = bouts[:, 1] - bouts[:, 0] + 1
    return bouts[lens >= max(1, min_len)]


def _bout_means_pair(x, y, bouts):
    if bouts.shape[0] == 0:
        return np.array([]), np.array([])
    xmean = np.full(bouts.shape[0], np.nan)
    ymean = np.full(bouts.shape[0], np.nan)
    for b in range(bouts.shape[0]):
        s, e = bouts[b]
        if s <= e:
            xmean[b] = np.nanmean(x[s:e + 1])
            ymean[b] = np.nanmean(y[s:e + 1])
    ok = np.isfinite(xmean) & np.isfinite(ymean)
    return xmean[ok], ymean[ok]


def _local_swarmjit(v, halfw=0.18):
    """Deterministic horizontal offsets for a swarm plot.

    Values are binned by magnitude; within each bin they are sorted and spread
    with a linspace over +/- `halfw` scaled by the bin's relative occupancy, so
    the offset increases monotonically with value inside each bin. No random
    number generator is used.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    xj = np.zeros(n)
    if n == 0:
        return xj
    nb = max(6, int(round(2.2 * np.sqrt(n))))
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return xj
    edges = np.linspace(lo, hi + np.spacing(1.0), nb + 1)
    b = np.digitize(v, edges) - 1
    b[b < 0] = 0
    b[b > nb - 1] = nb - 1
    counts = np.bincount(b, minlength=nb)
    kmax = counts.max() if counts.max() > 0 else 1
    for bi in range(nb):
        idx = np.flatnonzero(b == bi)
        k = idx.size
        if k == 0:
            continue
        order = np.argsort(v[idx])
        idx_sorted = idx[order]
        spread = halfw * np.sqrt(k / kmax)
        off = np.zeros(1) if k == 1 else np.linspace(-spread, spread, k)
        xj[idx_sorted] = off
    return xj


# ---------------------------------------------------------------------------
# main per-example plot
# ---------------------------------------------------------------------------
def _rel_vel(data_dir, out_dir, STY, avi_base, subject_color="Blue", cagemate_color="Red"):
    P_tracks = os.path.join(data_dir, f"tracks_{avi_base}.mat")
    S = load_struct(P_tracks)

    XYs = np.asarray(S[subject_color], dtype=float)
    XYo = np.asarray(S[cagemate_color], dtype=float)
    Sctr, _ = _keypoints_xy(XYs, KP_HEADCTR, KP_FRONT)
    Octr, _ = _keypoints_xy(XYo, KP_HEADCTR, KP_FRONT)

    N = min(Sctr.shape[0], Octr.shape[0])
    cap_frames = min(N, round(FPS * 60 * MINUTES_TO_ANALYZE))
    Sctr = Sctr[:cap_frames, :]
    Octr = Octr[:cap_frames, :]

    Vs = _diff_pad(Sctr) * FPS
    Vo = _diff_pad(Octr) * FPS

    Rpos = Sctr - Octr
    u = _unit_rows(Rpos)

    subj_rel_ps = _rowdot(Vs, u)
    cmat_rel_ps = _rowdot(Vo, -u)
    speed_subj_ps = _rowmag(Vs)
    speed_cmat_ps = _rowmag(Vo)

    vth_ps = THRESH_PX_PER_FRAME * FPS

    print(avi_base)

    mask_subj = speed_subj_ps > vth_ps
    mask_cmat = speed_cmat_ps > vth_ps
    bouts_subj = _find_bouts(mask_subj, MIN_BOUT_LEN_FR)
    bouts_cmat = _find_bouts(mask_cmat, MIN_BOUT_LEN_FR)

    print(f"[Gate] thr = {vth_ps:.1f} px/s | subj bouts: {bouts_subj.shape[0]} | "
          f"cagemate bouts: {bouts_cmat.shape[0]} | min={MIN_BOUT_LEN_FR} fr")

    subj_on_subjB, cmat_on_subjB = _bout_means_pair(subj_rel_ps, cmat_rel_ps, bouts_subj)
    subj_on_cmatB, cmat_on_cmatB = _bout_means_pair(subj_rel_ps, cmat_rel_ps, bouts_cmat)

    # Display sign flip: positive becomes approaching. Values above are unchanged.
    subj_on_subjB = -subj_on_subjB
    cmat_on_subjB = -cmat_on_subjB
    subj_on_cmatB = -subj_on_cmatB
    cmat_on_cmatB = -cmat_on_cmatB

    COL_NEG = STY.clr.neg    # purple, Away (negative after the flip)
    COL_POS = STY.clr.pos    # green,  Toward (positive after the flip)
    COL_PT = STY.clr.pt_gray

    edges = FIXED_BINS
    centers = edges[:-1] + np.diff(edges) / 2

    top_counts, _ = np.histogram(subj_on_subjB, bins=edges)
    bot_counts, _ = np.histogram(cmat_on_cmatB, bins=edges)
    top_pct = 100 * top_counts / max(top_counts.sum(), 1)
    bot_pct = 100 * bot_counts / max(bot_counts.sum(), 1)

    ymax = max(np.max(top_pct) if top_pct.size else 0,
               np.max(bot_pct) if bot_pct.size else 0, np.finfo(float).eps)
    ymax = max(5.0, np.ceil(ymax / 5.0) * 5.0)

    AX_H, HIST_W, SW_W = 1.30, 1.40, 0.85
    ML, MR, MB, MT, GAP = 0.62, 0.10, 0.70, 0.34, 0.55
    FIG_W = ML + HIST_W + GAP + SW_W + MR
    FIG_H = MB + AX_H + MT

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="w")

    axH = fig.add_axes([ML / FIG_W, MB / FIG_H, HIST_W / FIG_W, AX_H / FIG_H])
    bw = np.diff(edges)
    for k in range(centers.size):
        cc = COL_POS if centers[k] >= 0 else COL_NEG
        axH.add_patch(matplotlib.patches.Rectangle(
            (edges[k], 0), bw[k], top_pct[k], facecolor=cc, edgecolor="none"))
        axH.add_patch(matplotlib.patches.Rectangle(
            (edges[k], -bot_pct[k]), bw[k], bot_pct[k], facecolor=cc, edgecolor="none"))
    axH.plot([0, 0], [-ymax, ymax], "--", color=(0.2, 0.2, 0.2), linewidth=STY.lw.hair)
    axH.plot(list(FIXED_LIM), [0, 0], "-", color="w", linewidth=STY.lw.hair)
    axH.set_xlim(*FIXED_LIM)
    axH.set_ylim(-ymax, ymax)
    axH.set_xticks([-1000, -500, 0, 500, 1000])
    axH.set_yticks([-ymax, -ymax / 2, 0, ymax / 2, ymax])
    axH.set_yticklabels([f"{ymax:g}", f"{ymax/2:g}", "0", f"{ymax/2:g}", f"{ymax:g}"])
    gt_axes(axH, STY)
    axH.set_ylabel("Bouts (%)", fontname=STY.font.name, fontsize=STY.font.lab)
    axH.set_xlabel("Relative velocity [px/s]", fontname=STY.font.name, fontsize=STY.font.lab)
    axH.text(-0.30 * FIXED_LIM[1], 0.86 * ymax, "Away", color=COL_NEG, ha="center",
              fontname=STY.font.name, fontsize=STY.font.lab, fontweight="bold")
    axH.text(0.30 * FIXED_LIM[1], 0.86 * ymax, "Toward", color=COL_POS, ha="center",
              fontname=STY.font.name, fontsize=STY.font.lab, fontweight="bold")

    axS = fig.add_axes([(ML + HIST_W + GAP) / FIG_W, MB / FIG_H, SW_W / FIG_W, AX_H / FIG_H])
    A = subj_on_subjB.ravel()
    B = cmat_on_cmatB.ravel()
    axS.plot([0.4, 2.6], [0, 0], "--", color=(0.4, 0.4, 0.4), linewidth=STY.lw.hair)
    xA = 1 + _local_swarmjit(A, 0.18)
    xB = 2 + _local_swarmjit(B, 0.18)
    axS.scatter(xA, A, s=STY.mk.pt, c=[COL_PT], edgecolors="none", alpha=0.8)
    axS.scatter(xB, B, s=STY.mk.pt, c=[COL_PT], edgecolors="none", alpha=0.8)
    if A.size:
        mA, sA = np.mean(A), np.std(A, ddof=1) if A.size > 1 else 0.0
        axS.plot([1, 1], [mA - sA, mA + sA], "-", color="k", linewidth=1.2)
        axS.scatter([1], [mA], s=STY.mk.mean, c="k")
    if B.size:
        mB, sB = np.mean(B), np.std(B, ddof=1) if B.size > 1 else 0.0
        axS.plot([2, 2], [mB - sB, mB + sB], "-", color="k", linewidth=1.2)
        axS.scatter([2], [mB], s=STY.mk.mean, c="k")
    axS.set_xlim(0.4, 2.6)
    axS.set_ylim(*FIXED_LIM)
    axS.set_xticks([1, 2])
    axS.set_xticklabels(["Subj.", "Cm."])
    axS.set_yticks([-1000, -500, 0, 500, 1000])
    gt_axes(axS, STY)
    axS.set_ylabel("Relative velocity per bout [px/s]", fontname=STY.font.name,
                     fontsize=STY.font.lab)
    yl = axS.get_ylim()
    ny = yl[0] - 0.20 * (yl[1] - yl[0])
    # Bout counts, set at the label text size.
    axS.text(1, ny, f"{A.size}", ha="center", fontname=STY.font.name,
              fontsize=STY.font.lab, color=(0.35, 0.35, 0.35))
    axS.text(2, ny, f"{B.size}", ha="center", fontname=STY.font.name,
              fontsize=STY.font.lab, color=(0.35, 0.35, 0.35))
    axS.text(0.55, ny, "n =", ha="right", fontname=STY.font.name,
              fontsize=STY.font.lab, color=(0.35, 0.35, 0.35))

    gt_figStyle(fig, STY)
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    figs = []
    for suffix, avi_base in (
        ("wt", "unicorn_pairtrial_horatio_FA_6"),
        ("sh", "yertle_pairtrial_poseidon_FA_6"),
        ("gt", "lemon_pairtrial_filo_FA_6"),
    ):
        fig = _rel_vel(data_dir, out_dir, STY, avi_base)
        gt_saveFig(fig, f"edfig6_e_velocity_bias_p{len(figs)+1}_{suffix}", out_dir)
        figs.append(fig)
    return figs


if __name__ == "__main__":
    main()
