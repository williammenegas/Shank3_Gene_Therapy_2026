"""
fig6_e1_forest_shwt.py

Figure 6, panel e, first forest plot: the SH-versus-WT deficit per endpoint,
test condition.

Data. marm_beh_all.mat, marm_soc_all.mat, marm_cog_all.mat and
marm_fmri_all.mat, test-condition fields (<domain>_<group>_t), giving 16
endpoints in four domains. Values are z-scored to WT separately within each
condition.

Display. One row per endpoint, in domain order. The marker is Hedges' g for
the SH-minus-WT contrast and the horizontal line is its 95 per cent
confidence interval, against a vertical reference at zero. The endpoint that
is not rescued by gene therapy is shaded.

Statistics. A two-sided permutation test per endpoint, with `nperm`
permutations (20000 by default), corrected by Benjamini-Hochberg FDR across
the 16 endpoints. The strip at the right of the panel shows q: cell shade is
-log10(q), saturating at q = 1e-4, with the usual star label overlaid.
Permutation p-values are cached on disk, keyed by endpoint, contrast and
permutation count; pass use_cache=False to recompute them.

Effect sizes are Hedges' g, Cohen's d with the small-sample correction that
removes the upward bias of d at these group sizes.
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
DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")

WHICH = "shwt"
PANEL_NAME = "fig6_e1_forest_shwt"


def main(data_dir=DATA, out_dir=HERE, nperm=20000, use_cache=True):
    STY = gt_init()
    STY.fs_tick = 7
    STY.fs_ticksm = 6
    STY.fs_lab = 7
    STY.fs_title = 8
    STY.fs_star = 8
    CMEAN = STY.cmean
    CGTWT = STY.clr.gtwt

    S = local_build(data_dir, DOMS, "t")

    AX_W, AX_H = 1.30, 2.30
    MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.62, 0.42, 0.42, 0.32
    STRIP_W, STRIP_GAP = 0.16, 0.04
    FIG_W = MARGIN_L + AX_W + STRIP_GAP + STRIP_W + MARGIN_R
    FIG_H = MARGIN_B + AX_H + MARGIN_T

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="w")
    ax = fig.add_axes([MARGIN_L / FIG_W, MARGIN_B / FIG_H, AX_W / FIG_W, AX_H / FIG_H])
    local_forest(ax, S, WHICH, "SH - WT", True, CMEAN, CGTWT, STY)
    gt_panelLetter(ax, "e", STY)

    axs = fig.add_axes([(MARGIN_L + AX_W + STRIP_GAP) / FIG_W, MARGIN_B / FIG_H,
                         STRIP_W / FIG_W, AX_H / FIG_H])

    cache = load_permcache(PANEL_NAME) if use_cache else None
    p = compute_perm_p(S, WHICH, nperm, cache=cache)
    if use_cache:
        save_permcache(PANEL_NAME, cache)
    q = gt_bh(p)
    local_sigstrip(axs, S, WHICH, q, CMEAN, CGTWT, STY)

    print("Figure 6e: Test, SH - WT deficit")
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
    gt_saveFig(fig, "fig6_e1_forest_shwt", out_dir)
    return fig


if __name__ == "__main__":
    main()
