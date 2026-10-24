"""
_common.py

Drawing and statistics helpers shared by the endpoint panels built from the
new_data sets (EEG, sleep, activity strength, merged endpoints, western blot,
RNA-seq, rotarod and related). This module collects only the primitives used
by more than one of those panels; every panel loads its own data, runs its own
analysis and writes its own figure.

This panel family uses its own WT/SH/GT palette (near-black, muted red, muted
blue), defined in GROUP_COLORS below. It is close to, but not identical with,
the gt_style palette used by the main figure panels.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as _stats

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../python
from gtstyle import gt_style, gt_pstar, gt_compare3, gt_cohenD  # noqa: E402

STY = gt_style()

# Group colors shared by the sleep, activity-strength, merged-endpoint,
# rotarod, western-blot and cross-correlation panels.
GROUP_COLORS = {
    "WT": np.array([0.15, 0.15, 0.15]),
    "SH": np.array([0.75, 0.22, 0.22]),
    "GT": np.array([0.24, 0.35, 0.65]),
    "GT_LD": np.array([0.50, 0.58, 0.80]),
    "GT_HD": np.array([0.24, 0.35, 0.65]),
    "KO": np.array([0.75, 0.22, 0.22]),
}


def sig_stars(p, ns_label=None):
    """Significance label: a thin alias for the house gt_pstar, with tiers at
    0.05, 0.01 and 0.001, '0.05' in the borderline band and 'n/s' otherwise.
    `ns_label` is accepted and ignored, for call-site compatibility.
    """
    return gt_pstar(p)


def house_posthoc(clean):
    """House post-hoc comparison for a set of groups, used by plot_boxes and
    plot_meanerror.

    Three groups: one-way ANOVA with Tukey-Kramer post-hoc and Cohen's d, the
    same procedure as the other three-group endpoint comparisons. Two groups:
    two-sample Student t-test and Cohen's d, with no omnibus test or
    multiplicity correction, neither of which applies to a single comparison.
    Any other group count: pairwise Student t-tests.

    Returns (rows, p_anova), where rows lists (g1, g2, p, d) with 1-based
    group indices for every pair, including non-significant ones. The caller
    draws these as 'n/s' rather than omitting them. p_anova is None when no
    omnibus test applies.
    """
    from itertools import combinations
    usable = [c for c in clean if c.size >= 2]
    p_anova = None
    if len(clean) == 3 and len(usable) == 3:
        R = gt_compare3(clean)
        p_anova = R["p_anova"]
        pairs = [(1, 2), (2, 3), (1, 3)]
        return [(a, b, float(R["q"][i]), float(R["d"][i]))
                for i, (a, b) in enumerate(pairs)], p_anova

    rows = []
    for g1, g2 in combinations(range(len(clean)), 2):
        v1, v2 = clean[g1], clean[g2]
        if v1.size >= 2 and v2.size >= 2:
            _, pv = _stats.ttest_ind(v1, v2, equal_var=True)
            rows.append((g1 + 1, g2 + 1, float(pv), float(gt_cohenD(v1, v2))))
    return rows, p_anova


def holm_correct(p_raw):
    """Holm step-down family-wise correction. Returns adjusted p-values in the
    order in which they were supplied.
    """
    p_raw = np.asarray(p_raw, dtype=float)
    n = p_raw.size
    order = np.argsort(p_raw, kind="stable")
    sorted_p = p_raw[order]
    holm_p = np.ones(n)
    for ri in range(n):
        holm_p[order[ri]] = min(sorted_p[ri] * (n - ri), 1.0)
    # enforce monotonicity
    for ri in range(1, n):
        holm_p[order[ri]] = max(holm_p[order[ri]], holm_p[order[ri - 1]])
    return holm_p


def bh_correct(p_raw):
    """Benjamini-Hochberg step-up false-discovery-rate correction. Returns
    adjusted p-values in the order in which they were supplied.
    """
    p_raw = np.asarray(p_raw, dtype=float)
    m = p_raw.size
    order = np.argsort(p_raw, kind="stable")
    sorted_p = p_raw[order]
    ranks = np.arange(1, m + 1)
    bh = np.minimum(1.0, sorted_p * m / ranks)
    for k in range(m - 2, -1, -1):
        bh[k] = min(bh[k], bh[k + 1])
    out = np.empty(m)
    out[order] = bh
    return out


def welch_t_one_sided(a, b):
    """One-sided Welch t-test of mean(a) > mean(b), with Welch-Satterthwaite
    degrees of freedom.

    Returns (t, df, p).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    ma, mb = a.mean(), b.mean()
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(sa / na + sb / nb)
    t = (ma - mb) / se
    num = (sa / na + sb / nb) ** 2
    den = (sa / na) ** 2 / (na - 1) + (sb / nb) ** 2 / (nb - 1)
    df = num / den
    p = 1.0 - _stats.t.cdf(t, df)
    return t, df, p


def box_stats(v, whisker=1.5):
    """Quartiles and Tukey whiskers. Percentiles use linear interpolation
    between order statistics; whiskers extend to the furthest observation
    within `whisker` times the interquartile range.
    """
    v = np.asarray(v, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    whi = min(v.max(), q3 + whisker * iqr)
    wlo = max(v.min(), q1 - whisker * iqr)
    return q1, med, q3, wlo, whi


def plot_boxes(ax, grp_vals, group_colors, group_labels, rng=None,
               sig_test="ttest", whisker=1.5, hw=0.28, dot_size=STY.mk.pt):
    """Box (median, quartiles, whiskers) with jittered points and a
    significance bracket for every pair, labeled by gt_pstar so that
    non-significant pairs read 'n/s'. Statistics come from house_posthoc.

    sig_test: ignored; retained for call-site compatibility.

    grp_vals: list of 1-D arrays; NaNs are dropped here. Draws in place and
    returns the NaN-dropped groups.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_groups = len(grp_vals)
    clean = []
    for g in range(n_groups):
        gv = np.asarray(grp_vals[g], dtype=float)
        gv = gv[~np.isnan(gv)]
        clean.append(gv)
        if gv.size == 0:
            continue
        col = np.asarray(group_colors[g], dtype=float)
        fill_col = col * 0.25 + 0.75
        q1, med, q3, wlo, whi = box_stats(gv, whisker)
        x = g + 1  # 1-based x position

        ax.fill([x - hw, x - hw, x + hw, x + hw], [q1, q3, q3, q1],
                color=fill_col, edgecolor=col, linewidth=0.8, zorder=2)
        ax.plot([x - hw, x + hw], [med, med], "-", color=col, lw=2, zorder=4)
        ax.plot([x, x], [wlo, q1], "-", color=col, lw=0.8, zorder=3)
        ax.plot([x, x], [q3, whi], "-", color=col, lw=0.8, zorder=3)
        ax.plot([x - hw / 2, x + hw / 2], [wlo, wlo], "-", color=col, lw=0.8, zorder=3)
        ax.plot([x - hw / 2, x + hw / 2], [whi, whi], "-", color=col, lw=0.8, zorder=3)

        xoff = (rng.random(gv.size) - 0.5) * hw * 0.8
        ax.scatter(x + xoff, gv, s=dot_size, c=[col * 0.7 + 0.3],
                   edgecolors="none", zorder=5)

    # Every pair carries a bracket, including non-significant ones, so that a
    # comparison that was made and failed is distinguishable from one that was
    # never made.
    sig = [(g1, g2, pv) for g1, g2, pv, _d in house_posthoc(clean)[0]]

    if sig:
        yl = ax.get_ylim()
        yr = yl[1] - yl[0]
        ns = len(sig)
        ax.set_ylim(yl[0], yl[1] + 0.07 * yr * ns + 0.05 * yr)
        yl2 = ax.get_ylim()
        by = yl2[1] - 0.05 * (yl2[1] - yl2[0])
        bs = 0.06 * (yl2[1] - yl2[0])
        for si, (g1, g2, pv) in enumerate(sig):
            s = sig_stars(pv)
            yb = by - si * bs
            th = 0.006 * (yl2[1] - yl2[0])
            ax.plot([g1, g1, g2, g2], [yb - th, yb, yb, yb - th], "-",
                    color=(0.3, 0.3, 0.3), lw=0.7, zorder=6)
            ax.text((g1 + g2) / 2, yb + 0.003 * (yl2[1] - yl2[0]), s,
                    ha="center", fontsize=8, color=(0.15, 0.15, 0.15))

    ax.set_xticks(np.arange(1, n_groups + 1))
    ax.set_xticklabels(group_labels)
    # Uniform tick and body text size across the panel set.
    ax.tick_params(direction="out", labelsize=STY.font.tick)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(STY.lw.axis)
    ax.set_xlim(0.3, n_groups + 0.7)
    return clean


def plot_meanerror(ax, grp_vals, group_colors, group_labels, rng=None,
                   sig_test="ttest", err="sd", dot_size=STY.mk.pt, zero_line=False):
    """Mean +/- SD dot plot: faint jittered individual points, a large mean
    marker and a vertical error bar (SD by default; pass err='sem' for the
    standard error of the mean).

    This is the mouse-data glyph convention, used where plot_boxes above draws
    median and interquartile boxes for marmoset data. It uses this module's own
    group colors rather than the gt_style palette, and accepts any number of
    groups; the mouse panels call it with three (WT, SH, GT).

    Statistics and bracket placement are the same as in plot_boxes: the house
    post-hoc from house_posthoc, with a bracket for every pair.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_groups = len(grp_vals)
    clean = []
    if zero_line:
        ax.axhline(0.0, ls="--", color=(0.75, 0.75, 0.75), lw=STY.lw.hair, zorder=0)

    for g in range(n_groups):
        gv = np.asarray(grp_vals[g], dtype=float)
        gv = gv[~np.isnan(gv)]
        clean.append(gv)
        if gv.size == 0:
            continue
        col = np.asarray(group_colors[g], dtype=float)
        x = g + 1  # 1-based x position, matching plot_boxes
        m = gv.mean()
        e = gv.std(ddof=1) if err == "sd" else gv.std(ddof=1) / np.sqrt(gv.size)

        # faint jittered points behind, in the same jitter band as plot_boxes
        xoff = (rng.random(gv.size) - 0.5) * 0.28 * 0.8
        ax.scatter(x + xoff, gv, s=dot_size, c=[col * 0.7 + 0.3], edgecolors="none",
                   alpha=0.6, zorder=2)

        # vertical error bar in the group color
        ax.plot([x, x], [m - e, m + e], "-", color=col, lw=1.2, zorder=3)

        # large mean marker on top
        ax.scatter([x], [m], s=STY.mk.mean, c=[col], edgecolors="none", zorder=4)

    # Same house post-hoc and same all-pairs bracket rule as plot_boxes.
    sig = [(g1, g2, pv) for g1, g2, pv, _d in house_posthoc(clean)[0]]

    if sig:
        yl = ax.get_ylim()
        yr = yl[1] - yl[0]
        ns = len(sig)
        ax.set_ylim(yl[0], yl[1] + 0.07 * yr * ns + 0.05 * yr)
        yl2 = ax.get_ylim()
        by = yl2[1] - 0.05 * (yl2[1] - yl2[0])
        bs = 0.06 * (yl2[1] - yl2[0])
        for si, (g1, g2, pv) in enumerate(sig):
            s = sig_stars(pv)
            yb = by - si * bs
            th = 0.006 * (yl2[1] - yl2[0])
            ax.plot([g1, g1, g2, g2], [yb - th, yb, yb, yb - th], "-",
                    color=(0.3, 0.3, 0.3), lw=0.7, zorder=6)
            ax.text((g1 + g2) / 2, yb + 0.003 * (yl2[1] - yl2[0]), s,
                    ha="center", fontsize=8, color=(0.15, 0.15, 0.15))

    ax.set_xticks(np.arange(1, n_groups + 1))
    ax.set_xticklabels(group_labels)
    ax.tick_params(direction="out", labelsize=STY.font.tick)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(STY.lw.axis)
    ax.set_xlim(0.3, n_groups + 0.7)
    return clean
