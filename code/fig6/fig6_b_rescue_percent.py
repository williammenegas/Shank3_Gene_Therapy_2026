"""
fig6_b_rescue_percent.py

Figure 6, panel b: proportion of each SH deficit closed by gene therapy, per
endpoint, in the test condition.

Data. marm_beh_all.mat, marm_soc_all.mat, marm_cog_all.mat and
marm_fmri_all.mat, test-condition fields (<domain>_<group>_t), giving 16
endpoints in four domains. Values are z-scored to WT separately within each
condition.

Metric. For each endpoint, two Hedges' g effect sizes are computed, SH versus
WT (the deficit) and GT versus WT (the residual), and the bar height is

    closed = 100 * (1 - g_GTWT / g_SHWT)

so that 100 per cent means the GT-versus-WT residual has gone to zero. The
ratio is formed only when |g_SHWT| > 1e-6, since an endpoint with no deficit
has none to close. Endpoints with |g_SHWT| < 0.5 are labeled "weak" on the
figure: the denominator is small enough that the percentage is unstable. Bars
are clipped to the plotted range for display only; the console table prints
the unclipped values.

Display. One bar per endpoint, colored by domain, ordered by domain. The
shaded band marks 80-120 per cent. The endpoint that is not rescued is
outlined and labeled.

No confidence interval is drawn: this is a ratio of two effect sizes and
inherits the instability of its denominator at these group sizes. The
per-endpoint effect sizes and their intervals are in panel e.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403
from _f6common import DOMS, DOMLAB, local_build, local_deficit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")

DOMCOL = np.array([[0.45, 0.55, 0.75], [0.55, 0.70, 0.55],
                    [0.80, 0.60, 0.45], [0.65, 0.55, 0.72]])


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    STY.fs_tick = 7
    STY.fs_ticksm = 6
    STY.fs_lab = 7
    STY.fs_title = 8

    S = local_build(data_dir, DOMS, "t")

    AX_W, AX_H = 2.70, 1.35
    MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.62, 0.14, 0.62, 0.30
    FIG_W = MARGIN_L + AX_W + MARGIN_R
    FIG_H = MARGIN_B + AX_H + MARGIN_T

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="w")
    ax = fig.add_axes([MARGIN_L / FIG_W, MARGIN_B / FIG_H, AX_W / FIG_W, AX_H / FIG_H])

    print("Figure 6b: deficit closed by GT (test condition)")
    closed, disease = local_deficit(ax, S, DOMS, DOMLAB, DOMCOL, STY)
    gt_panelLetter(ax, "b", STY)

    hd = [plt.Rectangle((0, 0), 1, 1, facecolor=DOMCOL[d], edgecolor="none") for d in range(4)]
    fig.legend(hd, DOMLAB, loc="lower center", ncol=4, frameon=False,
               fontsize=STY.fs_tick, bbox_to_anchor=(0.5 + MARGIN_L / (2 * FIG_W), 0.0))

    print("  deficit closed by GT (%), test condition:")
    for i, ep in enumerate(S):
        flag = "   <- weak disease effect, ratio unreliable" if abs(disease[i]) < 0.5 else ""
        cval = closed[i]
        cstr = f"{cval:8.0f}%" if not np.isnan(cval) else "     nan"
        print(f"    {ep.name:<6} {cstr}{flag}")

    nweak = int(np.sum(np.abs(disease) < 0.5))
    inband = int(np.sum((closed >= 80) & (closed <= 120)))
    print(f"  {inband}/{len(S)} endpoints in the 80-120% band")
    print(f'  {nweak} endpoint(s) flagged "weak": |g_SH-WT| < 0.5')

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig6_b_rescue_percent", out_dir)
    return fig


if __name__ == "__main__":
    main()
