"""
edfig8_b_cortico_cortical.py -- Extended Data Fig. 8, panel b.

Within-region versus between-region cortico-cortical functional connectivity
in six cortical regions, WT / SH / GT. Companion to panel a, which uses the
same layout and statistics on the cortico-striatal data.

DATA. One file per region, <region>_self_paired_other_data.mat, for
prefrontal, premotor, parietal, temporal, somatosensory and visual cortex.
Each file holds collect_{wt,sh,gt}_r1 and collect_{wt,sh,gt}_r2, where r1 is
within-region ("self") cortico-cortical connectivity and r2 between-region
("other") cortico-cortical connectivity, as per-animal arrays padded with
NaN.

WHAT IS PLOTTED. Two dashboards of six subplots, one column per region in
load order:

  paired comparison  r1 against r2 within each group as a dumbbell plot: one
                     faint line per animal joining that animal's two values,
                     with a heavy line and markers at the group medians.
  box scatter        the within-animal difference r1 - r2 as a box plot with
                     jittered individual points and a mean marker, three
                     boxes (WT, SH, GT) per subplot.

STATISTICS. Within group, a two-sided paired t-test of r1 against r2, one per
group, Holm-Bonferroni corrected down each group's column of six regions and
labeled above each dumbbell. Across groups, the within-animal
difference r1 - r2 is tested for homogeneity of variance with Levene's test,
then compared by one-way ANOVA with Tukey-Kramer when equal variance holds or
by Welch's ANOVA with Games-Howell when it is rejected. Pairwise brackets are
drawn only when the omnibus reaches p < 0.05; otherwise a single 'ANOVA n/s'
note is printed.

DISPLAY. The subplots carry no region titles: regions are distinguished by
their left-to-right order, which is printed to stdout. The paired-comparison
subplots label the three groups A, B and C beneath each dumbbell pair, with
condition ticks 1 and 2; the box-scatter subplots label the groups 1, 2 and
3. In both cases the order is WT, SH, GT. Each drawing routine computes an
auto-scaled y range, positions the in-plot group labels against it, and then
applies a fixed y limit. The paired-comparison limit here runs to 0.8 rather
than the 0.6 used in panel a; the box-scatter limit is 0.6 in both. Figure
sizes match panel a for visual consistency and carry no other meaning. The
box-scatter point jitter is unseeded and cosmetic.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig5", "fig5_d_cortico_boxplots")

# Region load order; sets the left-to-right subplot column order. Same order
# as panel a.
REGIONS = ["prefrontal", "premotor", "parietal", "temporal", "somatosensory", "visual"]
FILE_SUFFIX = "_self_paired_other_data.mat"

# Fixed y limits applied at the end of each drawing routine. The
# paired-comparison upper bound is 0.8 here and 0.6 in panel a.
PAIRED_YLIM_FINAL = (-0.1, 0.8)
BOX_YLIM_FINAL = (-0.1, 0.6)


def _paired_p(y1, y2):
    """Two-sided paired t-test p for one group in one region.

    Drops NaNs pairwise, so a pair counts only if both values are finite,
    and returns NaN where fewer than two complete pairs remain. No label is
    made here: the labels are corrected across the six regions, so they
    cannot be derived from one region alone.
    """
    from scipy.stats import ttest_rel
    y1 = np.asarray(y1, float).ravel()
    y2 = np.asarray(y2, float).ravel()
    ok = ~(np.isnan(y1) | np.isnan(y2))
    a, b = y1[ok], y2[ok]
    if a.size < 2:
        return float("nan")
    try:
        return float(ttest_rel(a, b).pvalue)
    except ValueError:
        return float("nan")

def _plot3_paired_comparison(ax, a1, a2, b1, b2, c1, c2, STY, final_ylim,
                              group_labels=("A", "B", "C"), cond_labels=("1", "2"),
                              sig_labels=("", "", "")):
    """Draw one region's paired-comparison subplot.

    a1/a2, b1/b2, c1/c2 are the r1 and r2 vectors for WT, SH and GT.
    Draws one line per animal with both values finite, a median summary line
    with end markers per group, the group labels, the fixed y limit and a
    paired-t significance label per group.

    Returns (ax, labels), labels being the three strings drawn.
    """
    colors = STY.cmean          # rows are WT, SH, GT, matching a, b, c
    cond_sep, group_gap = 0.55, 1.3
    x_centers = 1 + np.arange(3) * (2 * cond_sep + group_gap)
    x1 = x_centers - cond_sep
    x2 = x_centers + cond_sep

    all_y = []
    pairs = [(a1, a2), (b1, b2), (c1, c2)]
    for g, (y1, y2) in enumerate(pairs):
        y1 = np.asarray(y1, dtype=float).ravel()
        y2 = np.asarray(y2, dtype=float).ravel()
        ok = ~(np.isnan(y1) | np.isnan(y2))
        y1p, y2p = y1[ok], y2[ok]
        if y1p.size:
            all_y.append(y1p)
            all_y.append(y2p)

        col = colors[g]
        # Blended toward white rather than drawn with alpha, so the
        # individual lines survive EPS export.
        light_col = col * 0.18 + np.array([1.0, 1.0, 1.0]) * 0.82

        for i in range(y1p.size):
            ax.plot([x1[g], x2[g]], [y1p[i], y2p[i]], "-", color=light_col,
                    lw=STY.lw.pair, zorder=1)

        s1 = np.median(y1p) if y1p.size else np.nan
        s2 = np.median(y2p) if y2p.size else np.nan

        ax.plot([x1[g], x2[g]], [s1, s2], "-", color=col, lw=2.5, zorder=3)
        ax.plot([x1[g]], [s1], "o", markerfacecolor=col, markeredgecolor=col,
                markersize=10, zorder=4)
        ax.plot([x2[g]], [s2], "o", markerfacecolor=col, markeredgecolor=col,
                markersize=10, zorder=4)

    all_y = np.concatenate(all_y) if all_y else np.array([])

    ax.set_xticks(np.sort(np.concatenate([x1, x2])))
    ax.set_xticklabels(list(cond_labels) * 3, fontsize=STY.font.tick)
    ax.set_xlim(x1.min() - 0.9, x2.max() + 0.9)

    # Auto-scaled y range. It is replaced by the fixed limit below, but the
    # group-label position is computed against it first, so the labels sit a
    # constant fraction below the data rather than below the fixed axis.
    if all_y.size == 0:
        ax.set_ylim(0, 1)
    else:
        yr = [float(all_y.min()), float(all_y.max())]
        if yr[0] == yr[1]:
            yr = [yr[0] - 0.5, yr[1] + 0.5]
        pad = 0.08 * (yr[1] - yr[0])
        ax.set_ylim(yr[0] - pad, yr[1] + pad)

    yl = ax.get_ylim()
    y_text = yl[0] - 0.10 * (yl[1] - yl[0])
    for g in range(3):
        ax.text(x_centers[g], y_text, group_labels[g], ha="center", va="top",
                fontsize=STY.font.tick)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Fixed y limit, applied after the group labels have been placed.
    ax.set_ylim(*final_ylim)

    # Within-group significance of within-region r1 against between-region
    # r2, one label per group. The labels are computed in main() and passed
    # in, because they are Holm-corrected across the six regions and so
    # cannot be derived from one region alone.
    yl = ax.get_ylim(); y_sig = yl[1] - 0.06 * (yl[1] - yl[0])
    for g, lab in enumerate(sig_labels):
        ax.text(x_centers[g], y_sig, lab, ha="center", va="top", fontsize=STY.font.tick)
    return ax, list(sig_labels)


def _plot3_box_scatter(ax, v1, v2, v3, STY, rng, final_ylim,
                        group_labels=("1", "2", "3")):
    """Draw one region's box-scatter subplot of the within-animal difference.

    v1, v2, v3 are the WT, SH and GT difference vectors. Draws an unfilled
    box per group with jittered individual points and a mean marker, applies
    the fixed y limit, and adds the across-group comparison.

    Returns (ax, R), R being the gt_compare3 result dict.
    """
    colors = STY.cmean
    vals = [np.asarray(v, dtype=float).ravel() for v in (v1, v2, v3)]
    vals = [v[~np.isnan(v)] for v in vals]

    # Unfilled box; whiskers extend to 1.5 times the interquartile range,
    # outliers are not drawn separately since every point is in the scatter.
    bp = ax.boxplot(vals, positions=[1, 2, 3], vert=True, widths=0.45,
                     whis=1.5, showfliers=False, patch_artist=False)
    for i in range(3):
        col = colors[i]
        bp["boxes"][i].set_color(col)
        bp["boxes"][i].set_linewidth(1.5)
        bp["medians"][i].set_color(col)
        bp["medians"][i].set_linewidth(1.5)
        for j in (2 * i, 2 * i + 1):
            bp["whiskers"][j].set_color(col)
            bp["whiskers"][j].set_linewidth(1.5)
            bp["caps"][j].set_color(col)
            bp["caps"][j].set_linewidth(1.5)

    for i, y in enumerate(vals):
        if y.size == 0:
            continue
        xj = (i + 1) + (rng.random(y.size) - 0.5) * (2 * 0.22)   # jitter band
        ax.scatter(xj, y, s=STY.mk.pt, c=[colors[i]], edgecolors=colors[i],
                   alpha=0.35, linewidths=0.3, zorder=3)
        m = y.mean()
        ax.plot([i + 1], [m], "o", markerfacecolor=colors[i],
                markeredgecolor=colors[i], markersize=6, zorder=4)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(list(group_labels), fontsize=STY.font.tick)
    ax.set_xlim(0.5, 3.5)

    all_y = np.concatenate(vals) if any(v.size for v in vals) else np.array([])
    if all_y.size == 0:
        ax.set_ylim(0, 1)
    else:
        yr = [float(all_y.min()), float(all_y.max())]
        if yr[0] == yr[1]:
            yr = [yr[0] - 0.5, yr[1] + 0.5]
        pad = 0.05 * (yr[1] - yr[0])
        ax.set_ylim(yr[0] - pad, yr[1] + pad)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Fixed y limit, replacing the auto-scaled range above.
    ax.set_ylim(*final_ylim)

    # Across-group comparison of the within-animal difference (within-region
    # r1 minus between-region r2): Levene's test, then one-way ANOVA with
    # Tukey-Kramer or Welch's ANOVA with Games-Howell. Pairwise brackets are
    # drawn only when the omnibus reaches p < 0.05.
    R = gt_compare3([vals[0], vals[1], vals[2]], "levene")
    yl = ax.get_ylim(); ry = yl[1] - yl[0]
    if R["p_anova"] >= 0.05:
        ax.text(2, yl[1] - 0.04 * ry, "ANOVA n/s", ha="center", va="top",
                fontsize=STY.font.ns, color=STY.clr.ns_txt)
    else:
        base = yl[0] + 0.80 * ry; h = 0.012 * ry
        gt_drawSig(ax, 1, 2, base, h, R["labels"][0], STY)
        gt_drawSig(ax, 2, 3, base, h, R["labels"][1], STY)
        gt_drawSig(ax, 1, 3, base + 0.09 * ry, h, R["labels"][2], STY)
    return ax, R


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    print("edfig8_b_cortico_cortical region order (left->right column):", REGIONS)

    fig3001, axes3001 = plt.subplots(1, 6, figsize=(16.8, 2.6), facecolor="w")
    fig4001, axes4001 = plt.subplots(1, 6, figsize=(12.6, 2.6), facecolor="w")

    # The script loads every region before drawing anything: the
    # within-group paired tests are corrected across the six regions, so
    # all eighteen p-values are needed before one label can be placed.
    loaded = []
    for region in REGIONS:
        S = load_struct(os.path.join(data_dir, f"{region}{FILE_SUFFIX}"))
        loaded.append((region,
                       np.asarray(S.collect_wt_r1, dtype=float).ravel(),
                       np.asarray(S.collect_wt_r2, dtype=float).ravel(),
                       np.asarray(S.collect_sh_r1, dtype=float).ravel(),
                       np.asarray(S.collect_sh_r2, dtype=float).ravel(),
                       np.asarray(S.collect_gt_r1, dtype=float).ravel(),
                       np.asarray(S.collect_gt_r2, dtype=float).ravel()))

    # Holm-Bonferroni down each group's column of six regions, the
    # correction the published labels carry.
    p_raw = np.array([[_paired_p(y1, y2) for y1, y2 in
                       ((w1, w2), (s1, s2), (g1, g2))]
                      for _, w1, w2, s1, s2, g1, g2 in loaded])
    p_adj = np.column_stack([gt_holm(p_raw[:, g]) for g in range(p_raw.shape[1])])
    sig_by_region = [[gt_pstar(v) if np.isfinite(v) else "n/s" for v in row]
                     for row in p_adj]

    for k, (region, wt_r1, wt_r2, sh_r1, sh_r2, gt_r1, gt_r2) in enumerate(loaded):
        r1 = wt_r1 - wt_r2
        r2 = sh_r1 - sh_r2
        r3 = gt_r1 - gt_r2

        ax1 = axes3001[k]
        _, sig = _plot3_paired_comparison(ax1, wt_r1, wt_r2, sh_r1, sh_r2, gt_r1, gt_r2,
                                          STY, PAIRED_YLIM_FINAL,
                                          sig_labels=sig_by_region[k])
        gt_axes(ax1, STY)

        ax2 = axes4001[k]
        _, Rbox = _plot3_box_scatter(ax2, r1, r2, r3, STY, rng, BOX_YLIM_FINAL)
        gt_axes(ax2, STY)

        n_wt = int(np.sum(~(np.isnan(wt_r1) | np.isnan(wt_r2))))
        n_sh = int(np.sum(~(np.isnan(sh_r1) | np.isnan(sh_r2))))
        n_gt = int(np.sum(~(np.isnan(gt_r1) | np.isnan(gt_r2))))
        print(f"  {region:14s} n = {n_wt}/{n_sh}/{n_gt}  |  within-group paired-t (WT/SH/GT) = {sig}"
              f"  |  across-group Delta p={Rbox['p_anova']:.3g} stars = {list(Rbox['labels'])}  |  "
              f"mean(r1-r2) wt/sh/gt = "
              f"{np.nanmean(r1):+.4f}/{np.nanmean(r2):+.4f}/{np.nanmean(r3):+.4f}")

    fig3001.subplots_adjust(left=0.03, right=0.995, top=0.92, bottom=0.22, wspace=0.55)
    fig4001.subplots_adjust(left=0.03, right=0.995, top=0.92, bottom=0.12, wspace=0.5)

    gt_figStyle(fig3001, STY)
    gt_figStyle(fig4001, STY)

    gt_saveFig(fig3001, "edfig8_b_cortico_cortical_pairedComparison", out_dir)
    gt_saveFig(fig4001, "edfig8_b_cortico_cortical_boxScatter", out_dir)

    return [fig3001, fig4001]


if __name__ == "__main__":
    main()
