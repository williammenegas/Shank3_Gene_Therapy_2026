"""
Figure 3, panel e: relative velocity during movement bouts, one example
cagemate pair per genotype (WT-WT, SH-WT, GT-WT).

Movement bouts are contiguous runs in which the animal's speed exceeds
THRESH_PX_PER_FRAME px/frame (converted to px/s at FPS), lasting at least
MIN_BOUT_LEN_FR frames, over the first MINUTES_TO_ANALYZE minutes of the
session. For each bout the script takes the mean relative velocity along
the subject-cagemate axis for both the subject and the cagemate. Two axes
per pair:

  Left: a mirrored histogram of per-bout relative velocity, subject upward
        and cagemate downward, in bins of 50 px/s over [-1000, 1000] px/s,
        expressed as a percentage of that animal's bouts. Negative bins are
        colored as "Away" and positive bins as "Toward". The y limit is set
        per panel to the next multiple of 5 above the tallest bar, with a
        floor of 5.

  Right: a swarm of the same per-bout means for subject and cagemate, with a
        black mean +/- s.d. marker and the bout count under each column.

Sign convention: the upstream computation gives positive values for
increasing inter-animal distance (avoiding) and negative for decreasing
distance (approaching). The plotted quantities are multiplied by -1, so on
the panel positive is approaching ("Toward", right) and negative is avoiding
("Away", left). Only the plotted sign is changed.

The swarm offsets are deterministic (binning plus a sqrt(k / kmax) spread),
with no random component.

The script reads only the subject and cagemate track columns from the .mat
files, which also hold two unused animals.

No statistical test is applied; the panel is descriptive.

Data: tracks_<session>.mat for the sessions named in EXAMPLES
"""

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig3", "fig3_e_velocity_bias")

KP_FRONT = 0         # keypoint index of the front of the head (column pair 0)
KP_HEADCTR = 3       # keypoint index of the head center (column pair 3)

FPS = 60                    # tracking frame rate
MINUTES_TO_ANALYZE = 30     # session duration analyzed
THRESH_PX_PER_FRAME = 5     # speed threshold defining a movement bout
MIN_BOUT_LEN_FR = 6         # minimum bout length, in frames

FIXED_LIM = (-1000, 1000)                    # relative-velocity axis, px/s
FIXED_BINS = np.arange(-1000, 1000 + 1, 50)  # histogram bin edges, px/s

# (panel label, session base name)
EXAMPLES = [
    ("WT-WT", "tahini_pairtrial_dusty_FA_3"),
    ("SH-WT", "onion_pairtrial_taquito_FA_3"),
    ("GT-WT", "lemon_pairtrial_filo_FA_3"),
]


def _keypoints_xy(XY, idx_ctr, idx_front):
    xi = lambda k: 2 * k
    yi = lambda k: 2 * k + 1
    C = np.column_stack([XY[:, xi(idx_ctr)], XY[:, yi(idx_ctr)]])
    F = np.column_stack([XY[:, xi(idx_front)], XY[:, yi(idx_front)]])
    return C, F


def _diff_pad(XY):
    D = np.diff(XY, axis=0)
    if D.shape[0] == 0:
        return np.zeros_like(XY)
    return np.vstack([D[0:1, :], D])


def _unit_rows(X):
    r = np.sqrt((X ** 2).sum(axis=1))
    u = np.zeros_like(X)
    ok = r > 0
    u[ok, :] = X[ok, :] / r[ok, None]
    return u


def _find_bouts(mask, min_len):
    mask = np.asarray(mask, dtype=bool).ravel()
    d = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    bouts = np.column_stack([starts, ends]) if starts.size else np.zeros((0, 2), int)
    if bouts.shape[0] == 0:
        return bouts
    lens = bouts[:, 1] - bouts[:, 0] + 1
    return bouts[lens >= max(1, min_len)]


def _bout_means_pair(x, y, bouts):
    if bouts.shape[0] == 0:
        return np.array([]), np.array([])
    xmean = np.full(bouts.shape[0], np.nan)
    ymean = np.full(bouts.shape[0], np.nan)
    for b, (s, e) in enumerate(bouts):
        if s <= e:
            xmean[b] = np.nanmean(x[s:e + 1])
            ymean[b] = np.nanmean(y[s:e + 1])
    ok = np.isfinite(xmean) & np.isfinite(ymean)
    return xmean[ok], ymean[ok]


def _swarm_jitter(v, halfw):
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    xj = np.zeros(n)
    if n == 0:
        return xj
    nb = max(6, round(2.2 * np.sqrt(n)))
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return xj
    edges = np.linspace(lo, hi + np.finfo(float).eps, nb + 1)
    b = np.clip(np.digitize(v, edges[1:-1], right=False), 0, nb - 1)
    counts = np.bincount(b, minlength=nb)
    kmax = counts.max() if counts.max() > 0 else 1
    for bi in range(nb):
        idx = np.flatnonzero(b == bi)
        k = idx.size
        if k == 0:
            continue
        order = idx[np.argsort(v[idx])]
        spread = halfw * np.sqrt(k / kmax)
        off = np.zeros(k) if k == 1 else np.linspace(-spread, spread, k)
        xj[order] = off
    return xj


def rel_vel(avi_base, name, data_dir, out_dir, STY):
    path = os.path.join(data_dir, f"tracks_{avi_base}.mat")
    S = sio.loadmat(path, variable_names=["Blue", "Red"])
    XYs, XYo = S["Blue"], S["Red"]

    Sctr, _ = _keypoints_xy(XYs, KP_HEADCTR, KP_FRONT)
    Octr, _ = _keypoints_xy(XYo, KP_HEADCTR, KP_FRONT)

    N = min(Sctr.shape[0], Octr.shape[0])
    cap_frames = min(N, round(FPS * 60 * MINUTES_TO_ANALYZE))
    Sctr, Octr = Sctr[:cap_frames], Octr[:cap_frames]

    Vs = _diff_pad(Sctr) * FPS
    Vo = _diff_pad(Octr) * FPS
    Rpos = Sctr - Octr
    u = _unit_rows(Rpos)

    subj_rel_ps = np.nansum(Vs * u, axis=1)
    cmat_rel_ps = np.nansum(Vo * (-u), axis=1)
    speed_subj_ps = np.sqrt((Vs ** 2).sum(axis=1))
    speed_cmat_ps = np.sqrt((Vo ** 2).sum(axis=1))

    vth_ps = THRESH_PX_PER_FRAME * FPS
    bouts_subj = _find_bouts(speed_subj_ps > vth_ps, MIN_BOUT_LEN_FR)
    bouts_cmat = _find_bouts(speed_cmat_ps > vth_ps, MIN_BOUT_LEN_FR)

    print(f"[{name}] {avi_base}: thr={vth_ps:.1f} px/s, subj bouts={bouts_subj.shape[0]}, "
          f"cagemate bouts={bouts_cmat.shape[0]}, min={MIN_BOUT_LEN_FR} fr")

    subj_on_subjB, _ = _bout_means_pair(subj_rel_ps, cmat_rel_ps, bouts_subj)
    _, cmat_on_cmatB = _bout_means_pair(subj_rel_ps, cmat_rel_ps, bouts_cmat)

    # Display sign convention: positive is approaching (see module docstring).
    subj_on_subjB = -subj_on_subjB
    cmat_on_cmatB = -cmat_on_cmatB

    COL_NEG, COL_POS, COL_PT = STY.clr.neg, STY.clr.pos, STY.clr.pt_gray

    edges = FIXED_BINS
    centers = edges[:-1] + np.diff(edges) / 2.0
    top_counts = np.histogram(subj_on_subjB, edges)[0]
    bot_counts = np.histogram(cmat_on_cmatB, edges)[0]
    top_pct = 100 * top_counts / max(top_counts.sum(), 1)
    bot_pct = 100 * bot_counts / max(bot_counts.sum(), 1)
    # Per-panel y limit: the next multiple of 5 above the tallest bar, with a
    # floor of 5 per cent.
    ymax = max(5.0, np.ceil(max(top_pct.max() if top_pct.size else 0,
                                  bot_pct.max() if bot_pct.size else 0, 1e-9) / 5) * 5)

    AX_H, HIST_W, SW_W = 0.95, 0.78, 0.50
    ML, MR, MB, MT, GAP = 0.42, 0.06, 0.44, 0.20, 0.20
    FIG_W = ML + HIST_W + GAP + SW_W + MR
    FIG_H = MB + AX_H + MT

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="w")

    axH = fig.add_axes([ML / FIG_W, MB / FIG_H, HIST_W / FIG_W, AX_H / FIG_H])
    bw = np.diff(edges)
    for k in range(centers.size):
        cc = COL_NEG if centers[k] < 0 else COL_POS
        axH.bar(edges[k], top_pct[k], width=bw[k], align="edge", color=cc, edgecolor="none")
        axH.bar(edges[k], -bot_pct[k], width=bw[k], align="edge", color=cc, edgecolor="none")
    axH.plot([0, 0], [-ymax, ymax], "--", color=(0.2, 0.2, 0.2), lw=STY.lw.hair)
    axH.plot(FIXED_LIM, [0, 0], "-", color="w", lw=STY.lw.hair)
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
    A, B = subj_on_subjB.ravel(), cmat_on_cmatB.ravel()
    axS.plot([0.4, 2.6], [0, 0], "--", color=(0.4, 0.4, 0.4), lw=STY.lw.hair)
    xA = 1 + _swarm_jitter(A, 0.18)
    xB = 2 + _swarm_jitter(B, 0.18)
    axS.scatter(xA, A, s=STY.mk.pt, c=[COL_PT], alpha=0.8, edgecolors="none")
    axS.scatter(xB, B, s=STY.mk.pt, c=[COL_PT], alpha=0.8, edgecolors="none")
    if A.size:
        mA, sA = np.mean(A), np.std(A, ddof=1) if A.size > 1 else 0.0
        axS.plot([1, 1], [mA - sA, mA + sA], "-", color="k", lw=STY.lw.se)
        axS.scatter([1], [mA], s=STY.mk.mean, c="k")
    if B.size:
        mB, sB = np.mean(B), np.std(B, ddof=1) if B.size > 1 else 0.0
        axS.plot([2, 2], [mB - sB, mB + sB], "-", color="k", lw=STY.lw.se)
        axS.scatter([2], [mB], s=STY.mk.mean, c="k")
    axS.set_xlim(0.4, 2.6)
    axS.set_ylim(*FIXED_LIM)
    axS.set_xticks([1, 2])
    axS.set_xticklabels(["Subj.", "Cm."])
    axS.set_yticks([-1000, -500, 0, 500, 1000])
    gt_axes(axS, STY)
    axS.set_ylabel("Relative velocity per bout [px/s]", fontname=STY.font.name, fontsize=STY.font.lab)
    yl = axS.get_ylim()
    ny = yl[0] - 0.20 * (yl[1] - yl[0])
    axS.text(1, ny, f"{A.size}", ha="center", fontname=STY.font.name, fontsize=STY.font.ns,
              color=(0.35, 0.35, 0.35))
    axS.text(2, ny, f"{B.size}", ha="center", fontname=STY.font.name, fontsize=STY.font.ns,
              color=(0.35, 0.35, 0.35))
    axS.text(0.55, ny, "n =", ha="right", fontname=STY.font.name, fontsize=STY.font.ns,
              color=(0.35, 0.35, 0.35))

    gt_figStyle(fig, STY)
    gt_saveFig(fig, f"fig3_e_velocity_bias_{name}", out_dir)
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    figs = []
    for i, (name, avi_base) in enumerate(EXAMPLES):
        figs.append(rel_vel(avi_base, f"p{i+1}", data_dir, out_dir, STY))
    return figs


if __name__ == "__main__":
    main()
