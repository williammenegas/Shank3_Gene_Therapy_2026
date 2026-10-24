"""
edfig9_a_control_shwt.py -- Extended Data Fig. 9, panel a.

SH-versus-WT effect size per endpoint in the CONTROL condition: the
control-condition counterpart of the test-condition contrast in Figure 6c.

WHAT IS PLOTTED. A forest plot of Hedges' g with its 95% confidence interval,
one row per endpoint, across the 16 endpoints of the four domains (behavior,
social, cognition, fMRI), with a q-value strip alongside. GT contributes 7 to
8 animals.

DATA. marm_{beh,soc,cog,fmri}_all.mat, the per-domain endpoint files shared
by the Figure 6 panel family. Values are z-scored to the WT distribution
separately within each condition, so WT sits at 0 +/- 1 in both conditions by
construction.

STATISTICS. Two-sided permutation test on the difference of group means,
20000 permutations per endpoint, adjusted by Benjamini-Hochberg across the 16
endpoints of this contrast. The FDR family is this panel's 16 endpoints
alone: it is not extended across the three contrasts of panels a to c, nor
against the test-condition forests. A permutation p-value at this number of
permutations carries a Monte-Carlo standard deviation of about 0.003.

DISPLAY. The confidence intervals are marginal (g +/- 1.96 se) and are not
adjusted for multiplicity, whereas the asterisks in the q strip are
FDR-adjusted; a confidence interval crossing zero beside an asterisk reflects
that difference of convention. Color depth in the q strip encodes -log10(q).

The `seed` argument seeds NumPy's global generator; this panel makes no
unseeded random draws, so it does not affect the output.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403
from _f6common import (  # noqa: E402
    DOMS, DOMLAB, local_build, local_forest, local_sigstrip, local_contrast,
    compute_perm_p, load_permcache, save_permcache,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# The four per-domain endpoint files shared with the Figure 6 panels.
DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")

WHICH = "shwt"
PANEL_NAME = "edfig9_a_control_shwt"


def main(data_dir=DATA, out_dir=HERE, nperm=20000, use_cache=True, seed=1):
    STY = gt_init()
    STY.fs_tick = 7
    STY.fs_ticksm = 6
    STY.fs_lab = 7
    STY.fs_title = 8
    STY.fs_star = 8
    CMEAN = STY.cmean
    CGTWT = STY.clr.gtwt

    if seed is not None:
        np.random.seed(seed)

    S = local_build(data_dir, DOMS, "c")   # control condition

    AX_W, AX_H = 1.30, 2.30
    MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.62, 0.42, 0.42, 0.32
    STRIP_W, STRIP_GAP = 0.16, 0.04
    FIG_W = MARGIN_L + AX_W + STRIP_GAP + STRIP_W + MARGIN_R
    FIG_H = MARGIN_B + AX_H + MARGIN_T

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="w")
    ax = fig.add_axes([MARGIN_L / FIG_W, MARGIN_B / FIG_H, AX_W / FIG_W, AX_H / FIG_H])
    local_forest(ax, S, WHICH, "SH - WT", True, CMEAN, CGTWT, STY)
    gt_panelLetter(ax, "a", STY)

    axs = fig.add_axes([(MARGIN_L + AX_W + STRIP_GAP) / FIG_W, MARGIN_B / FIG_H,
                         STRIP_W / FIG_W, AX_H / FIG_H])

    cache = load_permcache(PANEL_NAME) if use_cache else None
    p = compute_perm_p(S, WHICH, nperm, cache=cache)
    if use_cache:
        save_permcache(PANEL_NAME, cache)
    q = gt_bh(p)
    local_sigstrip(axs, S, WHICH, q, CMEAN, CGTWT, STY)

    print("Extended Data Figure 9a: Control, SH - WT deficit")
    g = np.full(len(S), np.nan)
    lo = g.copy()
    hi = g.copy()
    for i, ep in enumerate(S):
        gi, _, ci = local_contrast(ep, WHICH)
        g[i], lo[i], hi[i] = gi, ci[0], ci[1]
    for i, ep in enumerate(S):
        print(f"  {ep.name:<6} g = {g[i]:+5.2f} [{lo[i]:+5.2f} {hi[i]:+5.2f}]  "
              f"p = {p[i]:<9.3g} q = {q[i]:.3g}")
    nsig = int(np.sum(q < PSTAR_T1))
    print(f"  {nsig}/{len(S)} endpoints with q < {PSTAR_T1:g}")

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig9_a_control_shwt", out_dir)
    return fig


if __name__ == "__main__":
    main()
