"""
Figure 1, panel b: radar summary of seven endpoints, one plot per group.

Groups: WT, SH (Shank3 mutant + AAV-GFP), GT (Shank3 mutant + AAV-miniSHANK3),
all from the high-dose batch. Each spoke carries one endpoint in its own
native units. This script plots the group mean polygon and a +/- s.e.m. band;
on the SH and GT plots the WT mean polygon is repeated as a dashed reference.

All spokes are in raw units. The rotarod spoke is trial 3. The
electrophysiology spoke is the postsynaptic plateau (n = 13 / 14 / 13). Every
spoke is measured per animal. Western blot endpoints are not included, since
WT is normalized to 1.0 there and carries no spread.

Display conventions that affect reading the panel: each spoke is linearly
rescaled from the P_LO-P_HI percentile range of the pooled values onto
[INNER_HOLE, 1], so some animals fall outside the plotted window and values
beyond it are pinned to the spoke ends. Radial distance is a display
convention, not a magnitude, and the enclosed area depends on spoke ordering
and is not meaningful. Quantitative comparisons are made in the bar panels.

Percentiles follow the `prctile` convention: plotting positions
p_k = 100 * (k - 0.5) / n, linear interpolation between them, and clamping to
the sample minimum and maximum outside that range. `_matlab_prctile`
implements this; numpy's default `np.percentile` uses a different
plotting-position convention and would shift the spoke ranges.

No statistical test is applied; the panel is descriptive.

Data: panel_radar.mat
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig1", "fig1_b_radar")

# Spoke scaling and drawing parameters.
RANGE_MODE = "percentile"   # 'percentile' | 'minmax' | 'zero_to_max' | 'manual'
P_LO = 10                   # lower percentile defining the inner end of a spoke
P_HI = 90                   # upper percentile defining the outer end of a spoke
INNER_HOLE = 0.2            # radius at the inner end of a spoke, in axes units
MANUAL_LO = None            # spoke lower bounds, used only when RANGE_MODE = 'manual'
MANUAL_HI = None            # spoke upper bounds, used only when RANGE_MODE = 'manual'
SE_ALPHA = 0.30             # opacity of the +/- s.e.m. band, pre-blended toward white
SE_WHISKER = False          # draw radial s.e.m. whiskers at each vertex

MEAN_LW = 1.3
VERT_SIZE = 20   # mean-marker area, matching STY.mk.mean
REF_LW = 0.9
WHISK_LW = 1.0
WCAP_LW = 0.8
SHOW_UNITS = False

FIG_WIDTH_IN = 5.70
FIG_HEIGHT_IN = 2.00


def _matlab_prctile(v, p):
    """Percentile using the `prctile` convention: plotting positions
    100 * (k - 0.5) / n, linear interpolation between them, clamped to the
    sample minimum and maximum outside that range."""
    x = np.sort(np.asarray(v, dtype=float))
    n = x.size
    if n == 0:
        return np.nan
    pk = 100 * (np.arange(1, n + 1) - 0.5) / n
    if p <= pk[0]:
        return x[0]
    if p >= pk[-1]:
        return x[-1]
    return np.interp(p, pk, x)


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    S = load_struct(os.path.join(data_dir, "panel_radar.mat"))

    names = list(S.names)
    units = list(S.units)
    nS = int(np.ravel(S.nspokes)[0])
    gnames = ["WT", "SH", "GT"]
    cmean = STY.cmean

    pooled = np.asarray(S.pooled, dtype=float)

    lo = np.full(nS, np.nan)
    hi = np.full(nS, np.nan)
    for k in range(nS):
        v = pooled[k, :]
        v = v[~np.isnan(v)]
        if RANGE_MODE == "percentile":
            lo[k] = _matlab_prctile(v, P_LO)
            hi[k] = _matlab_prctile(v, P_HI)
        elif RANGE_MODE == "minmax":
            lo[k] = v.min()
            hi[k] = v.max()
        elif RANGE_MODE == "zero_to_max":
            lo[k] = 0
            hi[k] = v.max()
        elif RANGE_MODE == "manual":
            lo[k] = MANUAL_LO[k]
            hi[k] = MANUAL_HI[k]
        if hi[k] <= lo[k]:
            hi[k] = lo[k] + 1

    def rad(val, k):
        return INNER_HOLE + (1 - INNER_HOLE) * min(max((val - lo[k]) / (hi[k] - lo[k]), 0), 1)

    # Report any group mean that falls outside its spoke range and is
    # therefore pinned to the spoke end.
    nclamp = 0
    mu_by_g = {}
    se_by_g = {}
    for g, gn in enumerate(gnames):
        mu = np.ravel(np.asarray(S[f"mu_{gn}"], dtype=float))
        se = np.ravel(np.asarray(S[f"se_{gn}"], dtype=float))
        mu_by_g[gn] = mu
        se_by_g[gn] = se
        for k in range(nS):
            if mu[k] < lo[k] or mu[k] > hi[k]:
                nclamp += 1
                warnings.warn(
                    f"{gn} mean on \"{names[k]}\" ({mu[k]:.3g}) is OUTSIDE the "
                    f"spoke range [{lo[k]:.3g} {hi[k]:.3g}] and is pinned to the "
                    f"edge. Widen the range.")
    if nclamp == 0:
        print(f"Radar: no group means clamped (range {RANGE_MODE} {P_LO}-{P_HI}).")
    else:
        print(f"Radar: WARNING - {nclamp} group mean(s) clamped to the spoke edge.")

    theta = -np.pi / 2 + 2 * np.pi * np.arange(nS) / nS
    close_i = list(range(nS)) + [0]

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), facecolor="w")
    axes = [fig.add_subplot(1, 3, g + 1) for g in range(3)]

    for g, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.axis("off")

        tt = np.linspace(0, 2 * np.pi, 240)
        ax.plot(INNER_HOLE * np.cos(tt), INNER_HOLE * np.sin(tt), "-",
                color=(0.78, 0.78, 0.74), linewidth=0.8)
        for f in (0.25, 0.5, 0.75, 1.0):
            rr = INNER_HOLE + (1 - INNER_HOLE) * f
            ax.plot(rr * np.cos(tt), rr * np.sin(tt), "-",
                    color=(0.88, 0.88, 0.86), linewidth=0.6)
        for k in range(nS):
            ax.plot([INNER_HOLE * np.cos(theta[k]), np.cos(theta[k])],
                    [INNER_HOLE * np.sin(theta[k]), np.sin(theta[k])],
                    "-", color=(0.78, 0.78, 0.74), linewidth=0.7)

        mu = mu_by_g[gnames[g]]
        se = se_by_g[gnames[g]]
        rmu = np.array([rad(mu[k], k) for k in range(nS)])
        rlo = np.array([rad(mu[k] - se[k], k) for k in range(nS)])
        rhi = np.array([rad(mu[k] + se[k], k) for k in range(nS)])

        xb = np.concatenate([rhi[close_i] * np.cos(theta[close_i]),
                             (rlo[close_i] * np.cos(theta[close_i]))[::-1]])
        yb = np.concatenate([rhi[close_i] * np.sin(theta[close_i]),
                             (rlo[close_i] * np.sin(theta[close_i]))[::-1]])
        ax.fill(xb, yb, facecolor=gt_ablend(cmean[g], SE_ALPHA),
                edgecolor=gt_ablend(cmean[g], 0.35), linewidth=0.5)

        if g > 0:
            rw = np.array([rad(mu_by_g["WT"][k], k) for k in range(nS)])
            ax.plot(rw[close_i] * np.cos(theta[close_i]),
                    rw[close_i] * np.sin(theta[close_i]),
                    "--", color=cmean[0], linewidth=REF_LW)

        if SE_WHISKER:
            for k in range(nS):
                ax.plot([rlo[k] * np.cos(theta[k]), rhi[k] * np.cos(theta[k])],
                        [rlo[k] * np.sin(theta[k]), rhi[k] * np.sin(theta[k])],
                        "-", color=cmean[g], linewidth=WHISK_LW)
                capw = 0.022
                tx, ty = -np.sin(theta[k]), np.cos(theta[k])
                for rr in (rlo[k], rhi[k]):
                    px, py = rr * np.cos(theta[k]), rr * np.sin(theta[k])
                    ax.plot([px - capw * tx, px + capw * tx],
                            [py - capw * ty, py + capw * ty],
                            "-", color=cmean[g], linewidth=WCAP_LW)

        ax.plot(rmu[close_i] * np.cos(theta[close_i]),
                rmu[close_i] * np.sin(theta[close_i]),
                "-", color=cmean[g], linewidth=MEAN_LW)
        ax.scatter(rmu * np.cos(theta), rmu * np.sin(theta), s=VERT_SIZE,
                  c=[cmean[g]], edgecolors="none", zorder=5)

        for k in range(nS):
            lx, ly = 1.19 * np.cos(theta[k]), 1.19 * np.sin(theta[k])
            if abs(lx) < 0.15:
                ha = "center"
            elif lx > 0:
                ha = "left"
            else:
                ha = "right"
            ax.text(lx, ly, names[k], ha=ha, va="center", fontsize=STY.font.tick,
                    fontweight="bold")
            if SHOW_UNITS:
                ax.text(lx, ly - 0.11, f"{units[k]}: {lo[k]:.3g}-{hi[k]:.3g}",
                        ha=ha, va="center", fontsize=6.5, color=(0.55, 0.55, 0.53))

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(gnames[g], color=tuple(cmean[g]), fontweight="bold",
                    fontsize=STY.font.title)

    print(f"\nSpoke ranges ({RANGE_MODE} {P_LO}-{P_HI}) - put these in the caption:")
    for k in range(nS):
        print(f"  {names[k]:<13} {units[k]:<11} {lo[k]:.3g} to {hi[k]:.3g}")

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig1_b_radar", out_dir)
    return fig


if __name__ == "__main__":
    main()
