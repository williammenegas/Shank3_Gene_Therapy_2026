"""
_common.py -- drawing and statistics helpers shared by the Extended Data
endpoint panels (EEG, sleep, muscle strength, bloodwork and related assays).

Contents: significance labeling (sig_stars), the group comparison used by
both distribution panels (house_posthoc), multiplicity corrections
(holm_correct, bh_correct), a one-sided Welch t-test, box statistics
(box_stats), and the two per-group distribution panels themselves
(plot_boxes, plot_meanerror). The statistical routines delegate to gtstyle,
so the labeling thresholds and the three-group comparison match the rest of
the paper.

This panel family uses its own group palette, GROUP_COLORS below, which is
close to but distinct from gt_style's clr.wt, clr.sh and clr.gt.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as _stats

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../python
from gtstyle import gt_style, gt_pstar, gt_compare3, gt_cohenD  # noqa: E402

STY = gt_style()

# Group colors for this panel family: near-black, muted red, muted blue.
GROUP_COLORS = {
    "WT": np.array([0.15, 0.15, 0.15]),
    "SH": np.array([0.75, 0.22, 0.22]),
    "GT": np.array([0.24, 0.35, 0.65]),
    "GT_LD": np.array([0.50, 0.58, 0.80]),
    "GT_HD": np.array([0.24, 0.35, 0.65]),
    "KO": np.array([0.75, 0.22, 0.22]),
}


def sig_stars(p, ns_label=None):
    """Significance label for a p-value. Alias for gt_pstar: '*' below 0.05,
    '**' below 0.01, '***' below 0.001, the literal '0.05' in the borderline
    band [0.050, 0.060), and 'n/s' otherwise.

    ns_label is accepted and ignored; the non-significant label is 'n/s'.
    """
    return gt_pstar(p)


def house_posthoc(clean):
    """Group comparison for plot_boxes and plot_meanerror.

    Three groups are compared with gt_compare3: one-way ANOVA, Tukey-Kramer
    post hoc and Cohen's d, the same procedure as the other three-group
    endpoint comparisons in the paper. Two groups are compared with a
    two-sample Student t-test and Cohen's d, since neither an omnibus test
    nor a multiplicity correction applies to a single comparison. Any other
    number of groups is compared pairwise with two-sample Student t-tests.

    clean: list of 1-D arrays with NaNs already removed.

    Returns (rows, p_anova). rows is a list of (g1, g2, p, d) with 1-based
    group indices covering every pair, including the non-significant ones,
    which the caller draws as 'n/s' rather than omitting. p_anova is None
    where no omnibus test applies.
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
    """Holm step-down familywise correction.

    p_raw: array of raw p-values. Returns the adjusted values in the original
    order, clipped at 1 and made monotone in the sorted order.
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
    """Benjamini-Hochberg step-up false-discovery-rate correction.

    p_raw: array of raw p-values. Returns the adjusted values in the original
    order, clipped at 1 and made monotone by the reverse cumulative minimum.
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
    """One-sided Welch t-test of the alternative mean(a) > mean(b).

    Uses the unequal-variance standard error and Welch-Satterthwaite degrees
    of freedom. Returns (t, df, p).
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
    """Quartiles and Tukey whiskers for one sample.

    Percentiles are computed by linear interpolation. The whiskers extend to
    the most extreme observation within `whisker` times the interquartile
    range of the nearer quartile.

    Returns (q1, median, q3, whisker_low, whisker_high).
    """
    v = np.asarray(v, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    whi = min(v.max(), q3 + whisker * iqr)
    wlo = max(v.min(), q1 - whisker * iqr)
    return q1, med, q3, wlo, whi


def plot_boxes(ax, grp_vals, group_colors, group_labels, rng=None,
               sig_test="ttest", whisker=1.5, hw=0.28, dot_size=STY.mk.pt):
    """Box plot with jittered points and a significance bracket for every
    pair.

    Per group: a filled box from the first to the third quartile with a
    median rule, Tukey whiskers, and the individual observations as jittered
    points. A bracket is drawn for every pair and labeled by sig_stars, so a
    comparison that was made and did not reach significance reads 'n/s'; a
    missing bracket therefore means a comparison was not made. The comparison
    itself is house_posthoc.

    grp_vals: list of 1-D arrays; NaNs are dropped here.
    whisker: multiple of the interquartile range the whiskers extend to.
    hw: box half-width in data units. dot_size: point area in points^2.
    sig_test: accepted and ignored; house_posthoc selects the test.

    Returns the cleaned per-group arrays; drawing is in place.
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
        x = g + 1  # groups are placed at x = 1, 2, 3, ...

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

    # Every pair is drawn, including the non-significant ones, so that a
    # missing bracket means a comparison was not made.
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
    # Uniform 7 pt tick and body text, matching the rest of the figure set.
    ax.tick_params(direction="out", labelsize=STY.font.tick)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(STY.lw.axis)
    ax.set_xlim(0.3, n_groups + 0.7)
    return clean


def plot_meanerror(ax, grp_vals, group_colors, group_labels, rng=None,
                   sig_test="ttest", err="sd", dot_size=STY.mk.pt, zero_line=False):
    """Mean and error-bar dot plot with a significance bracket for every
    pair, following gtstyle.gt_meanerror's visual design.

    Per group: the individual observations as faint jittered points, a
    vertical error bar, and a large marker at the mean. The mouse endpoint
    panels use this display; the marmoset panels use the box-and-whisker
    display of plot_boxes.

    grp_vals: list of 1-D arrays; NaNs are dropped here.
    err: 'sd' for one standard deviation (default) or 'sem' for one standard
    error. zero_line: draw a dashed reference line at y = 0.
    dot_size: point area in points^2.
    sig_test: accepted and ignored; house_posthoc selects the test.

    The group comparison and the bracket layout are those of plot_boxes; only
    the per-group distribution glyph differs.

    Returns the cleaned per-group arrays; drawing is in place.
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
        x = g + 1  # same x positions as plot_boxes
        m = gv.mean()
        e = gv.std(ddof=1) if err == "sd" else gv.std(ddof=1) / np.sqrt(gv.size)

        # faint jittered points behind, in the jitter band plot_boxes uses
        xoff = (rng.random(gv.size) - 0.5) * 0.28 * 0.8
        ax.scatter(x + xoff, gv, s=dot_size, c=[col * 0.7 + 0.3], edgecolors="none",
                   alpha=0.6, zorder=2)

        # vertical error bar in the group color
        ax.plot([x, x], [m - e, m + e], "-", color=col, lw=1.2, zorder=3)

        # large mean marker on top
        ax.scatter([x], [m], s=STY.mk.mean, c=[col], edgecolors="none", zorder=4)

    # Same comparison and same all-pairs rule as plot_boxes.
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
