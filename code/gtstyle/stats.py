"""
gtstyle.stats -- statistical routines shared by the figure panel scripts.

The distribution functions (regularised incomplete beta, Student t CDF, F
CDF) and the group-comparison routines are implemented directly rather than
delegating to scipy; each docstring names the scipy equivalent, and
tests/test_stats.py checks that the two agree.

Dispersion is always the sample statistic (normalized by N-1), so every call
to np.var and np.std below passes ddof=1.
"""

import numpy as np
from math import lgamma, floor, exp, log


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------
def gt_betainc(x, a, b):
    """Regularised incomplete beta function I_x(a, b).

    Evaluated with Lentz's continued fraction (200 iterations, 1e-10
    convergence tolerance) and the standard x > (a+1)/(a+b+2) reflection,
    which keeps the fraction in its fast-converging regime.

    scipy equivalent: scipy.special.betainc(a, b, x)
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    if x > (a + 1) / (a + b + 2):
        return 1 - gt_betainc(1 - x, b, a)

    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * log(x) + b * log(1 - x) - lbeta) / a

    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 201):
        m = floor(i / 2)
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))

        d = 1 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1 / d

        c = 1 + num / c
        if abs(c) < 1e-30:
            c = 1e-30

        f = f * c * d
        if abs(1 - c * d) < 1e-10:
            break

    return front * (f - 1)


def gt_tcdf(x, v):
    """Student t cumulative distribution function on v degrees of freedom.

    scipy equivalent: scipy.stats.t.cdf(x, v)
    """
    return 1 - 0.5 * gt_betainc(v / (v + x ** 2), v / 2, 0.5)


def gt_fcdf(x, d1, d2):
    """F cumulative distribution function on (d1, d2) degrees of freedom.

    scipy equivalent: scipy.stats.f.cdf(x, d1, d2)
    """
    return gt_betainc(d1 * x / (d1 * x + d2), d1 / 2, d2 / 2)


# ---------------------------------------------------------------------------
# Group comparisons
# ---------------------------------------------------------------------------
def gt_anova1(G):
    """One-way analysis of variance across k independent groups.

    G: sequence of 1-D arrays, one per group.
    Returns (p, F, df1, df2).
    """
    G = [np.asarray(g, dtype=float).ravel() for g in G]
    k = len(G)
    n = np.array([g.size for g in G])
    N = n.sum()

    grand = sum(g.sum() for g in G) / N
    ssb = sum(n[i] * (G[i].mean() - grand) ** 2 for i in range(k))
    ssw = sum(((g - g.mean()) ** 2).sum() for g in G)

    df1 = k - 1
    df2 = N - k
    F = (ssb / df1) / (ssw / df2)
    p = 1 - gt_fcdf(F, df1, df2)
    return p, F, df1, df2


def gt_posthocT(G, mode="none"):
    """Post-hoc two-sample Student t tests on the three pairwise contrasts.

    Pair order is fixed at (WT, SH), (SH, GT), (WT, GT). Panel scripts index
    the returned arrays positionally, so the order is load-bearing.

    mode: 'none' | 'bonferroni' | 'holm'
    Returns (p_raw, p_adj), each a length-3 array.
    """
    G = [np.asarray(g, dtype=float).ravel() for g in G]
    pairs = [(0, 1), (1, 2), (0, 2)]

    p_raw = np.zeros(3)
    for i, (ia, ib) in enumerate(pairs):
        a, b = G[ia], G[ib]
        na, nb = a.size, b.size
        # pooled sample variance
        sp2 = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
        t = (a.mean() - b.mean()) / np.sqrt(sp2 * (1 / na + 1 / nb))
        p_raw[i] = 2 * (1 - gt_tcdf(abs(t), na + nb - 2))

    m = mode.lower()
    if m == "none":
        p_adj = p_raw.copy()
    elif m == "bonferroni":
        p_adj = np.minimum(1, p_raw * 3)
    elif m == "holm":
        order = np.argsort(p_raw, kind="stable")
        srt = p_raw[order]
        adj = np.minimum(1, srt * np.array([3, 2, 1]))
        adj = np.maximum.accumulate(adj)      # enforce monotonicity (cummax)
        p_adj = np.zeros(3)
        p_adj[order] = adj
    else:
        raise ValueError("mode must be none | bonferroni | holm")

    return p_raw, p_adj


def gt_cohenD(a, b):
    """Cohen's d for two independent samples, using the pooled standard
    deviation.

    NaNs are not stripped; callers pass pre-cleaned data.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = a.size, b.size
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def gt_hedges(a, b):
    """Hedges' g with the small-sample correction, its standard error and a
    95% confidence interval.

    g = d * J, where d is Cohen's d and J = 1 - 3/(4*(na+nb)-9). NaNs are
    dropped per sample. Degenerate input (fewer than two finite values in
    either sample, or a zero pooled standard deviation) returns NaN.

    Returns (g, se, ci) where ci is a length-2 array.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = a.size, b.size

    nan2 = np.array([np.nan, np.nan])
    if na < 2 or nb < 2:
        return np.nan, np.nan, nan2

    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan, np.nan, nan2

    d = (a.mean() - b.mean()) / sp
    J = 1 - 3 / (4 * (na + nb) - 9)          # small-sample correction
    g = d * J
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
    ci = np.array([g - 1.96 * se, g + 1.96 * se])
    return g, se, ci


def gt_permp(a, b, nperm=10000):
    """Two-sided permutation test on the difference of two group means.

    NaNs are dropped per sample. p is the proportion of permutations whose
    absolute mean difference is at least the observed one, with the usual
    (count + 1) / (nperm + 1) correction.

    nperm: number of label permutations. The Monte-Carlo standard deviation
    on p is approximately 1/sqrt(nperm); the Figure 6 panels use
    nperm = 20000. The MT19937 stream is reseeded to 12345 on every call, so
    repeated calls with the same (a, b, nperm) return the same value.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return np.nan

    rs = np.random.RandomState(12345)         # fixed seed: deterministic p
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na = a.size
    N = pool.size

    cnt = 0
    for _ in range(nperm):
        idx = rs.permutation(N)
        if abs(pool[idx[:na]].mean() - pool[idx[na:]].mean()) >= obs - 1e-12:
            cnt += 1

    return (cnt + 1) / (nperm + 1)


def gt_bh(p):
    """Benjamini-Hochberg false-discovery-rate adjustment.

    Returns q-values in the input order, clipped to [0, 1] and made monotone
    by the reverse cumulative minimum.
    """
    p = np.asarray(p, dtype=float).ravel()
    n = p.size
    order = np.argsort(p, kind="stable")
    ps = p[order]
    qs = ps * n / np.arange(1, n + 1)
    qs = np.minimum.accumulate(qs[::-1])[::-1]   # enforce monotonicity
    q = np.zeros(n)
    q[order] = np.clip(qs, 0, 1)
    return q


def gt_holm(p):
    """Holm-Bonferroni family-wise adjustment.

    Returns adjusted p-values in the input order, clipped to [0, 1] and made
    monotone by the running maximum over the ascending order. This is the
    adjustment a panel uses when it runs the same test once per region and
    the figure corrects across that family of regions. See the Extended Data
    Fig. 8 panels, whose within-group paired tests are corrected down each
    group's column of six cortical regions.
    """
    p = np.asarray(p, dtype=float).ravel()
    n = p.size
    order = np.argsort(p, kind="stable")
    ps = p[order] * (n - np.arange(n))
    ps = np.maximum.accumulate(ps)              # enforce monotonicity
    out = np.zeros(n)
    out[order] = np.clip(ps, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Significance labels
# ---------------------------------------------------------------------------
# Significance thresholds. A single scale is applied to every panel in the
# paper, paired with a family-wise corrected post hoc (Tukey-Kramer; see
# gt_compare3), which carries the multiplicity adjustment.
PSTAR_T1 = 0.05      # *
PSTAR_T2 = 0.01      # **
PSTAR_T3 = 0.001     # ***

# Borderline band: a p-value in [0.050, 0.060) is labeled with the literal
# string '0.05' rather than 'n/s'. Only values above PSTAR_T1 are relabeled;
# a value that earns an asterisk is never downgraded.
PSTAR_BORDER_LO = 0.050
PSTAR_BORDER_HI = 0.060
PSTAR_BORDER_LABEL = "0.05"

NS_LABEL = "n/s"     # comparison performed, not significant
NA_LABEL = "-"       # comparison not performed or undefined


def gt_pstar(p):
    """Significance label for a p-value.

    '***' p<0.001, '**' p<0.01, '*' p<0.05; '0.05' in the borderline band
    [0.050, 0.060); 'n/s' otherwise; '-' if p is None or NaN.

    Every panel routes its labels through this function, including
    FDR-corrected q-values (see gt_stars).
    """
    if p is None:
        return NA_LABEL
    p = float(p)
    if np.isnan(p):
        return NA_LABEL
    if p < PSTAR_T3:
        return "***"
    if p < PSTAR_T2:
        return "**"
    if p < PSTAR_T1:
        return "*"
    if PSTAR_BORDER_LO <= p < PSTAR_BORDER_HI:
        return PSTAR_BORDER_LABEL
    return NS_LABEL


def gt_stars(q):
    """Significance label for an FDR-corrected q-value.

    Alias for gt_pstar, so that a given number of asterisks corresponds to
    the same threshold throughout the paper. The figure caption states
    whether a raw p or a BH q was labeled.
    """
    return gt_pstar(q)


def gt_pairedt(a, b):
    """Two-sided paired Student t test on a - b. Returns (t, p, n).

    The test for a within-subject two-condition contrast, where each subject
    contributes one matched pair (for example the toward-versus-away contrast
    in Figure 3b, measured on the same animal). NaNs are dropped pairwise: a
    pair is retained only if both values are finite. p is computed through
    gt_tcdf and is labeled with gt_pstar like any other p-value.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = d.size
    if n < 2:
        return np.nan, np.nan, n
    sd = d.std(ddof=1)
    if sd == 0:
        return np.nan, np.nan, n
    t = d.mean() / (sd / np.sqrt(n))
    p = 2 * (1 - gt_tcdf(abs(t), n - 1))
    return t, p, n


def gt_tukey(G):
    """Tukey-Kramer pairwise p-values for three groups.

    For each pair the studentized-range statistic
        q = |mean_i - mean_j| / sqrt(MSE/2 * (1/n_i + 1/n_j))
    is referred to the studentized range distribution on (k, dfe), where MSE
    is the pooled within-group mean square and dfe its degrees of freedom.
    scipy.stats.studentized_range supplies the distribution. This is the
    standard Tukey-Kramer procedure for unequal group sizes.

    G: sequence of three 1-D arrays. Returns a length-3 array of p-values in
    pair order (WT,SH), (SH,GT), (WT,GT), matching gt_posthocT.
    """
    from scipy.stats import studentized_range

    G = [np.asarray(g, dtype=float).ravel() for g in G]
    k = len(G)
    n = np.array([g.size for g in G])
    dfe = n.sum() - k
    mse = sum(((g - g.mean()) ** 2).sum() for g in G) / dfe
    m = np.array([g.mean() for g in G])

    out = np.zeros(3)
    for idx, (i, j) in enumerate([(0, 1), (1, 2), (0, 2)]):
        se = np.sqrt(mse / 2 * (1 / n[i] + 1 / n[j]))
        out[idx] = studentized_range.sf(abs(m[i] - m[j]) / se, k, dfe)
    return out


def gt_levene(G):
    """Levene's test for homogeneity of variance, median-centered
    (Brown-Forsythe) variant.

    G: sequence of 1-D arrays, one per group; NaNs are dropped per group.
    Returns the p-value."""
    from scipy.stats import levene
    G = [np.asarray(g, float).ravel() for g in G]
    G = [g[~np.isnan(g)] for g in G]
    return float(levene(*G, center="median")[1])


def gt_welch_anova(G):
    """Welch's ANOVA: the omnibus test for k groups with unequal variances.

    G: sequence of 1-D arrays, one per group; NaNs are dropped per group.
    Returns (p, F, df1, df2), with df2 the Welch-Satterthwaite denominator
    degrees of freedom."""
    G = [np.asarray(g, float).ravel() for g in G]
    G = [g[~np.isnan(g)] for g in G]
    k = len(G)
    n = np.array([g.size for g in G], float)
    m = np.array([g.mean() for g in G])
    v = np.array([g.var(ddof=1) for g in G])
    w = n / v
    W = w.sum()
    xbar = (w * m).sum() / W
    A = (w * (m - xbar) ** 2).sum() / (k - 1)
    tmp = ((1 - w / W) ** 2 / (n - 1)).sum()
    F = (A / (1 + (2 * (k - 2) / (k ** 2 - 1)) * tmp))
    df1 = k - 1
    df2 = 1.0 / ((3.0 / (k ** 2 - 1)) * tmp)
    p = 1 - gt_fcdf(F, df1, df2)
    return p, F, df1, df2


def gt_gameshowell(G):
    """Games-Howell pairwise p-values: the unequal-variance post hoc that
    accompanies Welch's ANOVA.

    Each pair uses a Welch standard error and Welch-Satterthwaite degrees of
    freedom, referred to the studentized range distribution. NaNs are dropped
    per group. Returns a length-3 array in pair order (WT,SH), (SH,GT),
    (WT,GT), matching gt_tukey."""
    from scipy.stats import studentized_range
    G = [np.asarray(g, float).ravel() for g in G]
    G = [g[~np.isnan(g)] for g in G]
    k = len(G)
    out = np.zeros(3)
    for idx, (i, j) in enumerate([(0, 1), (1, 2), (0, 2)]):
        a, b = G[i], G[j]
        na, nb = a.size, b.size
        va, vb = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt((va / na + vb / nb) / 2.0)
        q = abs(a.mean() - b.mean()) / se
        df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        out[idx] = studentized_range.sf(q, k, df)
    return out


def gt_compare3(G, correction="tukey"):
    """Three-group endpoint comparison: omnibus test, pairwise post hoc and
    effect size, in a single call.

    One-way ANOVA across the three groups, then pairwise post-hoc contrasts
    on (WT,SH), (SH,GT), (WT,GT), with Cohen's d for each pair. NaNs are
    dropped per group.

    correction selects the post hoc:
      'tukey' (default)  Tukey-Kramer, controlling the family-wise error
                         rate across the three comparisons.
      'levene'           Test homogeneity of variance with Levene's test,
                         then use one-way ANOVA with Tukey-Kramer when equal
                         variance holds, or Welch's ANOVA with Games-Howell
                         when it is rejected (p < 0.05).
      'bh'               Benjamini-Hochberg, for results combined across
                         endpoints, as in the Figure 6 forest plots.
      'none'             Uncorrected two-sample Student t. Used for
                         diagnostics and for the protein-adjusted western
                         contrasts, which apply Tukey to their own OLS design
                         matrix instead.
      'bonferroni', 'holm'  Passed through to gt_posthocT.

    The 0.05 / 0.01 / 0.001 thresholds in PSTAR_T1..3 are paired with a
    family-wise corrected post hoc, which carries the multiplicity
    adjustment; the two are chosen together.

    Returns a dict with keys: p_anova, F, df1, df2; p (length-3 uncorrected
    Student t, for reference); q (length-3 after `correction`, the values
    that are labeled); d (length-3 Cohen's d); labels (length-3 gt_pstar of
    q); n (length-3 group sizes); pairs (name tuples); correction; and
    levene_p, variance_homogeneous, test_used, which are None unless
    correction selects the Levene-gated route.
    """
    G = [np.asarray(g, dtype=float).ravel() for g in G]
    G = [g[~np.isnan(g)] for g in G]
    if len(G) != 3:
        raise ValueError(f"gt_compare3 expects 3 groups, got {len(G)}")

    p_anova, F, df1, df2 = gt_anova1(G)
    p_raw, _ = gt_posthocT(G, "none")

    c = correction.lower()
    levene_p = None
    variance_homogeneous = None
    test_used = None
    if c in ("levene", "levene-gated"):
        # Test homogeneity of variance with Levene's test (median-centered),
        # then use one-way ANOVA with Tukey-Kramer when equal variance holds,
        # or Welch's ANOVA with Games-Howell when it is rejected (p < 0.05).
        levene_p = gt_levene(G)
        variance_homogeneous = levene_p >= 0.05
        if variance_homogeneous:
            q = gt_tukey(G)
            test_used = "one-way ANOVA + Tukey-Kramer"
        else:
            p_anova, F, df1, df2 = gt_welch_anova(G)
            q = gt_gameshowell(G)
            test_used = "Welch's ANOVA + Games-Howell"
    elif c in ("tukey", "tukey-kramer"):
        q = gt_tukey(G)
    elif c in ("none", ""):
        q = np.asarray(p_raw, dtype=float).copy()
    elif c in ("bh", "fdr"):
        q = gt_bh(p_raw)
    else:
        _, q = gt_posthocT(G, c)

    d = np.array([gt_cohenD(G[0], G[1]),
                  gt_cohenD(G[1], G[2]),
                  gt_cohenD(G[0], G[2])])

    return {
        "p_anova": p_anova, "F": F, "df1": df1, "df2": df2,
        "p": np.asarray(p_raw, dtype=float), "q": q, "d": d,
        "labels": [gt_pstar(v) for v in q],
        "n": np.array([g.size for g in G]),
        "pairs": (("WT", "SH"), ("SH", "GT"), ("WT", "GT")),
        "correction": c,
        "levene_p": levene_p,
        "variance_homogeneous": variance_homogeneous,
        "test_used": test_used,
    }


def gt_compare3_np(G, correction="none"):
    """Nonparametric three-group comparison: the rank-based analogue of
    gt_compare3, for data not assumed to be normally distributed (the fMRI
    connectivity endpoints in Figure 5).

    Omnibus: Kruskal-Wallis H test. Post hoc: two-sided Mann-Whitney U
    (rank-sum) on (WT,SH), (SH,GT), (WT,GT). Effect size is Cohen's d, kept
    for continuity with the parametric panels; the tests themselves are
    rank-based. NaNs are dropped per group.

    correction: 'holm' (Holm step-down across the three pairs), 'none'
    (the Kruskal-Wallis omnibus gates the
    pairwise tests, mirroring how the ANOVA gates the parametric pairwise
    tests) or 'bh' (Benjamini-Hochberg across the three pairs).

    Returns the same dict shape as gt_compare3, so gt_boxsig can consume
    either: p_anova is the Kruskal-Wallis p, F is the H statistic, q is the
    pairwise (corrected) p, labels is gt_pstar(q).
    """
    from scipy import stats as _ss

    G = [np.asarray(g, dtype=float).ravel() for g in G]
    G = [g[~np.isnan(g)] for g in G]
    if len(G) != 3:
        raise ValueError(f"gt_compare3_np expects 3 groups, got {len(G)}")

    H, p_kw = _ss.kruskal(*G)
    pairs = [(0, 1), (1, 2), (0, 2)]
    p_raw = np.array([_ss.mannwhitneyu(G[i], G[j],
                                       alternative="two-sided").pvalue
                      for i, j in pairs])

    c = correction.lower()
    if c == "holm":
        # Holm step-down across the three pairwise contrasts, with the
        # monotonicity constraint applied in ascending order of raw p.
        m = p_raw.size
        order = np.argsort(p_raw)
        q = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * float(p_raw[i]))
            q[i] = min(running, 1.0)
    elif c in ("bh", "fdr"):
        q = gt_bh(p_raw)
    else:
        q = p_raw.copy()

    d = np.array([gt_cohenD(G[0], G[1]),
                  gt_cohenD(G[1], G[2]),
                  gt_cohenD(G[0], G[2])])

    return {
        "p_anova": float(p_kw), "F": float(H), "df1": 2, "df2": None,
        "p": p_raw, "q": q, "d": d,
        "labels": [gt_pstar(v) for v in q],
        "n": np.array([g.size for g in G]),
        "pairs": (("WT", "SH"), ("SH", "GT"), ("WT", "GT")),
        "correction": c, "nonparametric": True,
    }


def gt_pfmt(p):
    """Format a p-value for display.

    Scientific notation below 1e-4, four decimal places above. A p-value that
    underflows to exactly zero in double precision (the t and F tail
    integrals bottom out) is reported as the bound '<1e-16'. None and NaN are
    reported as '-'.
    """
    if p is None:
        return "-"
    p = float(p)
    if np.isnan(p):
        return "-"
    if p <= 0.0:
        return "<1e-16"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"
