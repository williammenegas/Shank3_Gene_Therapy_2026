"""
Figure 1, panel k: postsynaptic component of the field response in acute
slices.

Two axes. Left: input-output curve, population spike amplitude (mV) against
stimulus intensity (mA), mean +/- s.e.m. across animals. Right: plateau
summary, the per-animal mean over stimulus levels 5-11, as a bar plot with a
beeswarm and a dotted line at the WT mean.

Groups: WT, SH (Shank3 mutant + AAV-GFP), GT (Shank3 mutant + AAV-miniSHANK3).
The unit of analysis is the animal; each point is one animal.

Raw recordings are negative deflections; the script multiplies them by -1
for display.

Statistics (plateau summary): Levene's test for equal variance, then one-way
ANOVA with Tukey-Kramer post-hoc, or Welch's ANOVA with Games-Howell where
equal variance is rejected (POST_HOC_CORRECTION = 'levene'), with Cohen's d
for each pair. Significance labels use the house scale: * p < 0.05,
** p < 0.01, *** p < 0.001, the literal '0.05' for p in [0.050, 0.060), and
'n/s' otherwise.

Layout: the script places both axes from absolute margins (0.46 in left,
0.10 in right, 0.34 in bottom, 0.26 in top) and pins the plateau axes to
3.2 x-units at 0.319 in per x-unit, i.e. 1.0208 in wide, matching the other
bar panels.

Data: panel_k_postsynaptic.mat
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig1", "fig1_k_postsynaptic")

FIG_WIDTH_IN = 3.15
FIG_HEIGHT_IN = 2.00
YMAX_EPHYS = 2.5     # y limit of the plateau bar axes, in mV

MARGIN_L_IN = 0.46
MARGIN_R_IN = 0.10
MARGIN_B_IN = 0.34
MARGIN_T_IN = 0.26
GAP_IN = 0.35


# Levene-gated omnibus test and post-hoc; stated at the call site rather than
# left to the gt_compare3 default.
POST_HOC_CORRECTION = "levene"


def main(data_dir=DATA, out_dir=HERE, panel_letter="k",
         mat_name="panel_k_postsynaptic.mat", out_name="fig1_k_postsynaptic"):
    STY = gt_init()
    S = load_struct(os.path.join(data_dir, mat_name))

    # Raw recordings are negative deflections; the sign is flipped for display.
    wt = -1 * np.asarray(S.wt_ephys, dtype=float)
    sh = -1 * np.asarray(S.sh_ephys, dtype=float)
    gt = -1 * np.asarray(S.gt_ephys, dtype=float)
    stim = np.ravel(np.asarray(S.stim, dtype=float))
    mats = [wt, sh, gt]
    gnames = ["WT", "SH", "GT"]

    cmean = STY.cmean
    cscat = STY.cscat
    cbarfill = STY.cbarfill

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), facecolor="w")

    ax2_w_in = STY.geom.xspan * STY.geom.in_per_xunit   # pinned bar-panel width, 1.0208 in
    ax_h_in = FIG_HEIGHT_IN - MARGIN_B_IN - MARGIN_T_IN
    ax1_w_in = FIG_WIDTH_IN - MARGIN_L_IN - GAP_IN - ax2_w_in - MARGIN_R_IN

    ax1 = fig.add_axes([MARGIN_L_IN / FIG_WIDTH_IN, MARGIN_B_IN / FIG_HEIGHT_IN,
                        ax1_w_in / FIG_WIDTH_IN, ax_h_in / FIG_HEIGHT_IN])
    ax2_x0_in = MARGIN_L_IN + ax1_w_in + GAP_IN
    ax2 = fig.add_axes([ax2_x0_in / FIG_WIDTH_IN, MARGIN_B_IN / FIG_HEIGHT_IN,
                        ax2_w_in / FIG_WIDTH_IN, ax_h_in / FIG_HEIGHT_IN])

    # Left axes: input-output curve, mean +/- s.e.m. across animals.
    for g in range(3):
        M = mats[g]
        m = np.nanmean(M, axis=1)
        se = np.nanstd(M, axis=1, ddof=1) / np.sqrt(M.shape[1])
        ax1.fill_between(stim, m - se, m + se, color=gt_ablend(cmean[g], 0.18),
                         edgecolor="none", zorder=1)
        ax1.plot(stim, m, color=cmean[g], linewidth=2, zorder=2, label=gnames[g])

    for y in np.arange(0.5, 2.5 + 1e-9, 0.5):
        ax1.axhline(y, linestyle=":", color=(0.87, 0.87, 0.87), linewidth=0.6,
                    zorder=0)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 2.5)
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_xlabel("Stimulus intensity (mA)")
    ax1.set_ylabel("Population\nspike amplitude (mV)")
    ax1.set_title(str(S.ptitle))
    # No label list here: each mean line already carries label=gnames[g]. Passing
    # gnames would reassign the labels positionally to the first three artists
    # drawn, which are the WT band, the WT line and the SH band.
    ax1.legend(loc="upper left", frameon=False, fontsize=STY.font.leg)

    # Right axes: plateau summary, per-animal mean over stimulus levels 5-11
    # (1-based), i.e. rows 4:11 here.
    plat = [np.nanmean(mats[g][4:11, :], axis=0) for g in range(3)]
    plat = [p[~np.isnan(p)] for p in plat]
    wt_mean = np.nanmean(plat[0])

    ax2.set_xlim(*STY.geom.xlim3)
    href = ax2.axhline(wt_mean, linestyle=":", color="k", linewidth=0.6,
                       zorder=1.5)   # above the bar fills (z=1), below the data (z=2)

    gt_bar3(ax2, plat, STY)

    w, s, t = plat
    # Statistics: variance-gated omnibus test, pairwise contrasts and
    # Cohen's d (gtstyle.stats.gt_compare3).
    R = gt_compare3(plat, POST_HOC_CORRECTION)
    p_anova, Fstat, df1, df2 = R["p_anova"], R["F"], R["df1"], R["df2"]
    p_wtsh, p_shgt, p_wtgt = R["q"]
    d_wtsh, d_shgt, d_wtgt = R["d"]

    print(f"{S.ptitle}: ANOVA F({df1},{df2})={Fstat:.2f} p={p_anova:.2e} | "
          f"WT-SH p={p_wtsh:.3g} d={d_wtsh:.2f} | "
          f"SH-GT p={p_shgt:.3g} d={d_shgt:.2f} | "
          f"WT-GT p={p_wtgt:.3g} d={d_wtgt:.2f}")

    ax2.set_ylim(0, YMAX_EPHYS)
    yl = ax2.get_ylim()
    rng = yl[1] - yl[0]
    h = 0.012 * rng
    base = yl[0] + 0.885 * rng

    gt_drawSig(ax2, 0, 1, base, h, gt_pstar(p_wtsh), STY)
    gt_drawSig(ax2, 1, 2, base, h, gt_pstar(p_shgt), STY)
    gt_drawSig(ax2, 0, 2, base + 0.08 * rng, h, gt_pstar(p_wtgt), STY)

    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(gnames)
    ax2.set_ylabel("Mean spike amplitude")

    # Axis styling. Numeric tick labels are 6 pt on this panel, a local
    # override of the 7 pt house size; tick length and pad follow the house
    # style.
    for ax, has_group_ticks in ((ax1, False), (ax2, True)):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.5)
        ax.tick_params(direction="out", width=STY.lw.axis,
                       length=STY.tick.len, pad=STY.tick.pad, labelsize=6)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname(STY.font.name)
            lbl.set_fontsize(6)
        ax.title.set_fontname(STY.font.name)
        ax.title.set_fontsize(STY.font.title)
        ax.xaxis.label.set_fontname(STY.font.name)
        ax.xaxis.label.set_fontsize(7)
        ax.yaxis.label.set_fontname(STY.font.name)
        ax.yaxis.label.set_fontsize(7)

        # Only the plateau bar carries WT/SH/GT group labels, so only it gets
        # the group palette. The input-output curve also has three x ticks,
        # [0, 0.5, 1], but they are stimulus intensities and stay black, as
        # they are in the published panel.
        xt = ax.get_xticks()
        if has_group_ticks and len(xt) == 3:
            for k, lbl in enumerate(ax.get_xticklabels()):
                lbl.set_color(tuple(cmean[k]))
                lbl.set_fontsize(7)

    # The panel letter sits at the top left of the whole two-axes panel, on
    # the input-output axes rather than the plateau bar axes.
    gt_panelLetter(ax1, "k", STY)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, out_name, out_dir)
    return fig


if __name__ == "__main__":
    main()
