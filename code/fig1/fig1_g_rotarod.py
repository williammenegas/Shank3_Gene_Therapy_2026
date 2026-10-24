"""
Figure 1, panel g: rotarod, latency to fall.

Groups: WT, SH (Shank3 mutant + AAV-GFP), GT (Shank3 mutant + AAV-miniSHANK3),
all from the high-dose batch. n = 26 / 29 / 19. Bars are group means with
s.e.m. whiskers over a beeswarm of individual animals. The script reads the
axis label, title and y limits from the .mat file.

Animal ID encodes batch: 3-digit IDs are the high-dose batch, 4-digit IDs the
low-dose batch. GT animals at this dose exist only in the high-dose batch, so
controls are drawn from the same batch. Batch and dose are therefore the same
variable, and the panel makes no dose comparison.

Statistics: Levene's test for equal variance, then one-way ANOVA with
Tukey-Kramer post-hoc, or Welch's ANOVA with Games-Howell where equal
variance is rejected (POST_HOC_CORRECTION = 'levene'), with Cohen's d for
each pair. The script draws pairwise brackets only when the omnibus test
passes ANOVA_ALPHA; otherwise it annotates the panel 'ANOVA n/s'.

Data: main_d_rotarod.mat
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig1", "fig1_g_rotarod")

POST_HOC_CORRECTION = "levene"   # 'levene' | 'tukey' | 'bh' | 'none' | 'bonferroni' | 'holm'
ANOVA_ALPHA = 0.05               # omnibus test must pass this for brackets to be drawn


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    S = load_struct(os.path.join(data_dir, "main_d_rotarod.mat"))

    # The stored label distinguishes the dose ("GT-HD"); the published panel reads
    # "GT", so the script sets the display name here and leaves the data as is.
    gnames = ["WT", "SH", "GT"]

    # Data, in WT / SH / GT order, with missing values removed.
    G = [np.asarray(v, dtype=float).ravel() for v in (S.wt, S.sh, S.gt)]
    G = [v[~np.isnan(v)] for v in G]

    # Figure geometry is pinned across the three-group bar panels: an axes
    # width of 1.0208 in over xlim [-0.6, 2.6] gives 0.319 in per x-unit,
    # which fixes the physical bar width.
    fig, ax = gt_barpanel_figure(STY)
    gt_bar3(ax, G, STY)

    # Statistics: variance-gated omnibus test, pairwise contrasts and
    # Cohen's d (gtstyle.stats.gt_compare3).
    R = gt_compare3(G, POST_HOC_CORRECTION)
    p_anova, Fstat, df1, df2 = R["p_anova"], R["F"], R["df1"], R["df2"]
    p_adj, d = R["q"], R["d"]

    print(f"{S.ptitle} (batch {int(np.ravel(S.batch_id)[0])})")
    print(f"  ANOVA F({df1},{df2}) = {Fstat:.3f}, p = {p_anova:.4g}")
    print(f"  WT vs SH  p = {p_adj[0]:.4g}  d = {d[0]:+.2f}")
    print(f"  SH vs {str(S.gt_label):<5} p = {p_adj[1]:.4g}  d = {d[1]:+.2f}")
    print(f"  WT vs {str(S.gt_label):<5} p = {p_adj[2]:.4g}  d = {d[2]:+.2f}")

    # Y limits from the .mat file, then significance brackets.
    ylimits = np.ravel(np.asarray(S.ylimits, dtype=float))
    if ylimits.size and not np.all(ylimits == 0):
        ax.set_ylim(ylimits[0], ylimits[1])
    yl = ax.get_ylim()
    rng = yl[1] - yl[0]
    h = 0.012 * rng
    base = yl[0] + 0.860 * rng

    if p_anova < ANOVA_ALPHA:
        gt_drawSig(ax, 0, 1, base, h, gt_pstar(p_adj[0]), STY)
        gt_drawSig(ax, 1, 2, base, h, gt_pstar(p_adj[1]), STY)
        gt_drawSig(ax, 0, 2, base + 0.075 * rng, h, gt_pstar(p_adj[2]), STY)
    else:
        # Pairwise stars under a null omnibus test would be uncontrolled
        # comparisons, so none are drawn.
        ax.text(1, base + 0.05 * rng, "ANOVA n/s", ha="center",
                fontsize=STY.font.ns, color=STY.clr.ns_txt,
                fontname=STY.font.name)

    # Labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(gnames)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    ax.set_ylabel(str(S.ylab))
    ax.set_title(str(S.ptitle), fontname=STY.font.name,
                 fontsize=STY.font.title, pad=STY.geom.title_pad)
    gt_panelLetter(ax, "g", STY)

    ylf = ax.get_ylim()
    for g in range(3):
        # The n= row sits 0.205 of the y range below the axis so that it
        # clears the group tick labels, which sit about 11 pt below it.
        ax.text(g, ylf[0] - 0.205 * (ylf[1] - ylf[0]), f"n={G[g].size}",
                ha="center", fontsize=STY.font.n, color=STY.clr.n_txt,
                fontname=STY.font.name)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig1_g_rotarod", out_dir)
    return fig


if __name__ == "__main__":
    main()
