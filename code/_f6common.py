"""
_f6common.py -- shared helpers for the Figure 6 panel family and the
Extended Data panels built on the same analysis.

Each panel script loads its own data, computes its own statistics and writes
its own figure. This module holds only the machinery they share: endpoint
assembly, effect-size contrasts, permutation p-values and their on-disk
cache, and the drawing routines for the radar, the forest and its
significance strip, the deficit-closed bars, the ridge-logistic link score
and its S-curve and box plots, the correlation matrices and the paired
control-to-test panels.

Two sources of run-to-run variation:

  1. gt_permp (from gtstyle) and local_pairedperm below reseed a MT19937
     stream to 12345 on every call, so a given (data, nperm) always yields
     the same p-value. Those p-values still carry the Monte-Carlo error
     inherent to a permutation test, about 0.003 at nperm = 20000.

  2. local_kfold, which assigns cross-validation folds for the
     ridge-logistic link score, is unseeded. The ridge weights w and b are
     fit on all training rows and are deterministic; the cross-validated
     AUROC and the out-of-fold GT eta are averaged over five folds and vary
     slightly between runs. The point jitter in local_vbox is likewise
     unseeded, and is purely cosmetic.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtstyle import (  # noqa: E402
    gt_ablend, gt_hedges, gt_permp, gt_bh,
    gt_stars, PSTAR_T1, load_struct,
)

DOMS = ["beh", "soc", "cog", "fmri"]
DOMLAB = ["Behavioral", "Social", "Cognition", "fMRI"]
# Domain names drawn on the radar rim: one label per domain, placed at the
# centroid of its arc, in place of a tag on each of the 16 spokes.
RADAR_DOMLAB = {"beh": "Behavioral", "soc": "Social", "cog": "Cognitive", "fmri": "fMRI"}


# ---------------------------------------------------------------------------
# Endpoint assembly
# ---------------------------------------------------------------------------
class Endpoint:
    """One endpoint: its name (for example 'beh2'), its domain, and the three
    group columns as loaded, still containing NaN for animals not measured on
    that endpoint."""
    __slots__ = ("name", "dom", "wt", "sh", "gt")

    def __init__(self, name, dom, wt, sh, gt):
        self.name = name
        self.dom = dom
        self.wt = wt
        self.sh = sh
        self.gt = gt


def local_build(data_dir, doms, cond):
    """Assemble the endpoint list for one condition.

    data_dir holds one marm_<dom>_all.mat per domain, each with the arrays
    <dom>_{wt,sh,gt}_{c,t}. cond is 'c' (control) or 't' (test).

    Returns a list of Endpoint, one per column of each domain's arrays, named
    <dom><k> with k counting from 1."""
    S = []
    for dom in doms:
        M = load_struct(os.path.join(data_dir, f"marm_{dom}_all.mat"))
        wt = np.atleast_2d(np.asarray(M[f"{dom}_wt_{cond}"], dtype=float))
        sh = np.atleast_2d(np.asarray(M[f"{dom}_sh_{cond}"], dtype=float))
        gt = np.atleast_2d(np.asarray(M[f"{dom}_gt_{cond}"], dtype=float))
        for k in range(wt.shape[1]):
            S.append(Endpoint(f"{dom}{k + 1}", dom, wt[:, k], sh[:, k], gt[:, k]))
    return S


def local_iscog4(name):
    return name == "cog4"


def local_epk(name):
    """Return the trailing digit of an endpoint name as an integer."""
    return int(name[-1])


def _contrast_pair(ep, which):
    if which == "shwt":
        return ep.sh, ep.wt
    elif which == "gtwt":
        return ep.gt, ep.wt
    else:
        return ep.gt, ep.sh


# ---------------------------------------------------------------------------
# Effect-size contrasts
#
# Effect size for these panels is Hedges' g rather than Cohen's d. This
# family pools 16 endpoints across 4 to 8 animals per group; Hedges' g
# multiplies d by J = 1 - 3/(4*(na+nb)-9), removing d's upward bias at small
# sample size, a shrinkage of roughly 5 to 10 per cent at these group sizes.
# ---------------------------------------------------------------------------
def local_contrast(ep, which):
    """Hedges' g, its standard error and 95% confidence interval for one
    endpoint under one contrast.

    which: 'shwt' (SH minus WT), 'gtwt' (GT minus WT) or 'gtsh' (GT minus
    SH). Returns (g, se, ci)."""
    a, b = _contrast_pair(ep, which)
    return gt_hedges(a, b)


# ---------------------------------------------------------------------------
# Permutation p-value cache, held on disk so that a run at nperm = 20000 need
# not be repeated. gt_permp reseeds to 12345 on every call, so a cached value
# is identical to a fresh call at the same nperm rather than an
# approximation of it.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_permcache")


def load_permcache(panel_name):
    fp = os.path.join(_CACHE_DIR, panel_name + ".json")
    if os.path.exists(fp):
        with open(fp) as f:
            raw = json.load(f)
        return {tuple(json.loads(k)): v for k, v in raw.items()}
    return {}


def save_permcache(panel_name, cache):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fp = os.path.join(_CACHE_DIR, panel_name + ".json")
    raw = {json.dumps(list(k)): v for k, v in cache.items()}
    with open(fp, "w") as f:
        json.dump(raw, f)


def compute_perm_p(S, which, nperm, cache=None):
    """Two-sided permutation p-value per endpoint for one contrast.

    S: the endpoint list. which: 'shwt', 'gtwt' or 'gtsh'. nperm: number of
    label permutations. cache: an optional dict from load_permcache, keyed by
    (endpoint name, contrast, nperm).

    A permutation test is used rather than a t-test because normality cannot
    be assessed at 4 to 8 animals per group, and these panels pool 16
    endpoints, so a per-endpoint p-value with an optimistic bias would
    propagate into the BH-FDR step and the count of significant endpoints.
    The permutation test assumes only exchangeability under the null.

    Computed once per panel run; the same array feeds the printed table and
    local_sigstrip. Returns a length-len(S) array."""
    p = np.full(len(S), np.nan)
    for i, ep in enumerate(S):
        a, b = _contrast_pair(ep, which)
        key = (ep.name, which, nperm)
        if cache is not None and key in cache:
            p[i] = cache[key]
            continue
        val = gt_permp(a, b, nperm)
        p[i] = val
        if cache is not None:
            cache[key] = val
    return p


# ---------------------------------------------------------------------------
# Forest plot and significance strip
# ---------------------------------------------------------------------------
def local_forest(ax, S, which, ttl, showy, CMEAN, CGTWT, STY):
    """Draw the forest plot for one contrast: Hedges' g with its 95%
    confidence interval, one row per endpoint, ordered top to bottom as in S.

    which selects the contrast and hence the marker color. showy draws the
    endpoint names on the y axis. The cog4 row is shown on a highlight band."""
    n = len(S)
    y = np.arange(n, 0, -1)
    if which == "shwt":
        col = CMEAN[1]
    elif which == "gtwt":
        col = CGTWT
    else:
        col = CMEAN[2]

    for i in range(n):
        if local_iscog4(S[i].name):
            ax.fill(
                [-3.5, 3.5, 3.5, -3.5],
                [y[i] - 0.5, y[i] - 0.5, y[i] + 0.5, y[i] + 0.5],
                color=gt_ablend(STY.clr.hilite, 0.35), edgecolor="none", zorder=0,
            )
    ax.plot([0, 0], [0.4, n + 0.6], "-", color=STY.clr.gray_hair, lw=STY.lw.hair)
    for i in range(n):
        g, _, ci = local_contrast(S[i], which)
        if np.isnan(g):
            continue
        ax.plot(ci, [y[i], y[i]], "-", color=col, lw=STY.lw.ci)
        ax.plot(g, y[i], "o", color=col, markerfacecolor=col,
                 markersize=STY.mk.pt, markeredgecolor="w", markeredgewidth=0.25)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(0.4, n + 0.6)
    ax.set_yticks(np.arange(1, n + 1))
    ax.set_xticks(np.arange(-3, 4, 1))
    if showy:
        ax.set_yticklabels([S[n - 1 - i].name for i in range(n)])
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Hedges' g (95% CI)", fontsize=STY.fs_lab, fontname=STY.font.name)
    ax.set_title(ttl, fontsize=STY.fs_title, fontname=STY.font.name, fontweight="normal")
    # Both axes carry explicit ticks, so the automatic locator is disabled:
    # applied to the categorical y axis it would discard the
    # one-endpoint-per-row layout.
    from gtstyle import gt_axes
    gt_axes(ax, STY, matlab_ticks=False)


def local_ramp(rgb, m):
    """Sequential colormap of m steps running from white to rgb."""
    x = np.linspace(0, 1, m)
    rgb = np.asarray(rgb, dtype=float)
    return np.column_stack([1 - (1 - rgb[0]) * x, 1 - (1 - rgb[1]) * x, 1 - (1 - rgb[2]) * x])


def local_sigstrip(ax, S, which, q, CMEAN, CGTWT, STY):
    """Draw the q-value strip beside a forest plot, one cell per endpoint.

    q is the BH-FDR-adjusted p-value array, already computed by the caller
    (see compute_perm_p and gt_bh). Cell color depth encodes -log10(q),
    saturating at q = 1e-3, and each cell carries its gt_stars label in
    white or black according to the depth of the fill.

    The strip for the GT-versus-WT contrast uses its own dark gray rather
    than CGTWT."""
    n = len(S)
    y = np.arange(n, 0, -1)
    if which == "shwt":
        rgb = CMEAN[1]
    elif which == "gtwt":
        rgb = np.array([0.20, 0.20, 0.20])
    else:
        rgb = CMEAN[2]

    vmax = 3
    ramp = local_ramp(rgb, 64)
    for i in range(n):
        qi = q[i]
        t = min(max(-np.log10(max(qi, 1e-4)) / vmax, 0), 1)
        idx = int(round(t * (ramp.shape[0] - 1)))
        col = ramp[idx]
        ax.fill([0, 1, 1, 0], [y[i] - 0.5, y[i] - 0.5, y[i] + 0.5, y[i] + 0.5],
                color=col, edgecolor="w", linewidth=0.35)
        s = gt_stars(qi)
        tc = "w" if t > 0.55 else "k"
        ax.text(0.5, y[i], s, ha="center", va="center",
                fontname=STY.font.name, fontsize=STY.fs_star, color=tc)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, n + 0.6)
    ax.set_xticks([0.5])
    ax.set_xticklabels(["q"])
    ax.set_yticks([])
    ax.tick_params(length=0)
    from gtstyle import gt_axes
    gt_axes(ax, STY, matlab_ticks=False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(STY.clr.gray_mid)
    ax.yaxis.set_visible(False)


# ---------------------------------------------------------------------------
# Radar plot
# ---------------------------------------------------------------------------
def local_rof(z, zmin, zmax, r0, rmax):
    """Map an endpoint z-score to a radius on the radar, clamped to the
    annulus between r0 and rmax."""
    t = (z - zmin) / (zmax - zmin)
    t = min(max(t, 0), 1)
    return r0 + (rmax - r0) * t


def local_radar(ax, S, ttl, CMEAN, STY):
    """Draw the radar plot: one spoke per endpoint, one closed line per group
    at the group mean, with a shaded band of one standard error either side.

    Spokes are ordered as in S, so each domain occupies a contiguous quadrant
    of the rim; the rim is drawn as four arcs with a divider at each domain
    boundary and one domain label at each arc centroid.

    Returns the three line handles, for a figure-level legend."""
    n = len(S)
    ang = np.arange(n) * 2 * np.pi / n
    zmin, zmax, R0, RMAX = -2.4, 1.5, 0.15, 0.9
    th = np.linspace(0, 2 * np.pi, 240)
    OUTLINE = (0.15, 0.15, 0.15)

    # concentric grid rings, unlabeled
    for zg in (-1.5, -0.5, 0.5, 1.5):
        rr = local_rof(zg, zmin, zmax, R0, RMAX)
        ax.plot(rr * np.cos(th), rr * np.sin(th), "-",
                color=STY.clr.gray_lite, lw=STY.lw.hair, zorder=0)
    r0ring = local_rof(0, zmin, zmax, R0, RMAX)
    ax.plot(r0ring * np.cos(th), r0ring * np.sin(th), "-",
            color=(0.78, 0.78, 0.78), lw=0.6, zorder=0)
    ax.plot(R0 * np.cos(th), R0 * np.sin(th), "-",
            color=STY.clr.gray_lite, lw=STY.lw.hair, zorder=0)

    # radial spokes, one per endpoint
    for a in ang:
        ax.plot([R0 * np.cos(a), RMAX * np.cos(a)], [R0 * np.sin(a), RMAX * np.sin(a)],
                "-", color=(0.90, 0.90, 0.90), lw=0.4, zorder=0)

    # outer rim as four arcs, one per domain, separated by gaps and dividers
    step = 2 * np.pi / n
    b0 = ang[3] + step / 2                       # first domain boundary
    bnds = b0 + np.arange(4) * (np.pi / 2)       # four quadrant boundaries
    gap = np.radians(5)
    for k in range(4):
        a1 = bnds[k] + gap
        a2 = (bnds[k + 1] if k < 3 else bnds[0] + 2 * np.pi) - gap
        tt = np.linspace(a1, a2, 90)
        ax.plot(RMAX * np.cos(tt), RMAX * np.sin(tt), "-", color=OUTLINE,
                lw=1.1, zorder=1, solid_capstyle="round")
    tk = 0.028
    for a in ang:                                # outward tick per spoke
        ax.plot([RMAX * np.cos(a), (RMAX + tk) * np.cos(a)],
                [RMAX * np.sin(a), (RMAX + tk) * np.sin(a)],
                "-", color=OUTLINE, lw=0.8, zorder=1)
    for bb in bnds:                              # divider at each domain edge
        ax.plot([(RMAX - 0.035) * np.cos(bb), (RMAX + 0.075) * np.cos(bb)],
                [(RMAX - 0.035) * np.sin(bb), (RMAX + 0.075) * np.sin(bb)],
                "-", color=OUTLINE, lw=1.5, zorder=2, solid_capstyle="round")

    # group mean line with a standard-error band; no per-spoke whiskers
    gs = ("wt", "sh", "gt")
    handles = [None, None, None]
    for gi in range(3):
        mu = np.zeros(n)
        se = np.zeros(n)
        for i in range(n):
            v = getattr(S[i], gs[gi])
            v = v[~np.isnan(v)]
            mu[i] = v.mean()
            if v.size > 1:
                se[i] = v.std(ddof=1) / np.sqrt(v.size)
        r = np.array([local_rof(mu[i], zmin, zmax, R0, RMAX) for i in range(n)])
        rlo = np.array([local_rof(mu[i] - se[i], zmin, zmax, R0, RMAX) for i in range(n)])
        rhi = np.array([local_rof(mu[i] + se[i], zmin, zmax, R0, RMAX) for i in range(n)])

        angc = np.append(ang, ang[0])
        rhic = np.append(rhi, rhi[0])
        rloc = np.append(rlo, rlo[0])
        xb = np.concatenate([rhic * np.cos(angc), (rloc * np.cos(angc))[::-1]])
        yb = np.concatenate([rhic * np.sin(angc), (rloc * np.sin(angc))[::-1]])
        ax.fill(xb, yb, color=gt_ablend(CMEAN[gi], 0.15), edgecolor="none", zorder=2)

        xs = np.append(r * np.cos(ang), r[0] * np.cos(ang[0]))
        ys = np.append(r * np.sin(ang), r[0] * np.sin(ang[0]))
        (h,) = ax.plot(xs, ys, "-", color=CMEAN[gi], lw=1.6, zorder=3,
                       solid_joinstyle="round")
        handles[gi] = h

    # one domain label per arc centroid, outside the rim
    for dom in DOMS:
        idx = [i for i in range(n) if S[i].dom == dom]
        ac = float(np.mean(ang[idx]))
        rl = RMAX + 0.15
        ha = "left" if np.cos(ac) > 0.2 else ("right" if np.cos(ac) < -0.2 else "center")
        va = "bottom" if np.sin(ac) > 0.2 else ("top" if np.sin(ac) < -0.2 else "center")
        ax.text(rl * np.cos(ac), rl * np.sin(ac), RADAR_DOMLAB.get(dom, dom),
                fontname=STY.font.name, fontsize=STY.fs_tick + 1,
                ha=ha, va=va, color="k")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(ttl, fontname=STY.font.name, fontsize=STY.fs_title, fontweight="normal")
    return handles


# ---------------------------------------------------------------------------
# Deficit-closed bar panel
# ---------------------------------------------------------------------------
def local_deficit(ax, S, doms, domlab, domcol, STY):
    """Draw the per-endpoint bar of the proportion of the SH deficit closed by
    GT, as 100 * (1 - g_gtwt / g_shwt) per cent, colored by domain.

    Bars are clipped to the axis range for display; endpoints whose SH-versus-
    WT effect is smaller than |g| = 0.5 are marked 'weak', since the ratio is
    unstable when its denominator is near zero. Endpoints with |g_shwt| below
    1e-6 are left blank.

    Returns (closed, disease): the per-endpoint percentage closed and the
    per-endpoint SH-versus-WT Hedges' g."""
    n = len(S)
    closed = np.full(n, np.nan)
    disease = np.full(n, np.nan)
    for i in range(n):
        g_shwt, _, _ = gt_hedges(S[i].sh, S[i].wt)
        g_gtwt, _, _ = gt_hedges(S[i].gt, S[i].wt)
        disease[i] = g_shwt
        if abs(g_shwt) > 1e-6:
            closed[i] = 100 * (1 - g_gtwt / g_shwt)

    YLO, YHI = -60, 150
    ax.fill([0.3, n + 0.7, n + 0.7, 0.3], [80, 80, 120, 120],
            color=gt_ablend([0.2, 0.6, 0.3], 0.08), edgecolor="none")
    ax.plot([0.3, n + 0.7], [100, 100], "-", color=STY.clr.gray_mid, lw=STY.lw.hair)
    ax.plot([0.3, n + 0.7], [0, 0], "-", color=STY.clr.gray_hair, lw=STY.lw.hair)

    for i in range(n):
        di = doms.index(S[i].dom)
        yv = max(min(closed[i], YHI), YLO)
        if local_iscog4(S[i].name):
            ax.bar(i + 1, yv, width=0.72, color=domcol[di], edgecolor=STY.clr.hilite_tx, lw=0.7)
        else:
            ax.bar(i + 1, yv, width=0.72, color=domcol[di], edgecolor="none")
        if abs(disease[i]) < 0.5:
            ax.text(i + 1, 4, "weak", ha="center", va="bottom", rotation=90,
                    fontname=STY.font.name, fontsize=4.2, color=STY.clr.gray_mid)

    icog4 = next((i for i in range(n) if local_iscog4(S[i].name)), None)
    if icog4 is not None:
        ax.text(icog4 + 1, YLO + 8, "not rescued", ha="center", va="bottom",
                fontname=STY.font.name, fontsize=STY.fs_tick, color=STY.clr.hilite_tx)

    ax.set_xticks(np.arange(1, n + 1))
    ax.set_xticklabels([s.name for s in S], rotation=45, ha="right",
                        fontname=STY.font.name, fontsize=STY.fs_ticksm)
    ax.set_ylabel("deficit closed by GT (%)", fontname=STY.font.name, fontsize=STY.fs_lab)
    ax.set_ylim(YLO, YHI + 5)
    ax.set_xlim(0.3, n + 0.7)
    ax.set_title("How much of each SH deficit did GT close?",
                 fontname=STY.font.name, fontsize=STY.fs_title, fontweight="normal")

    from gtstyle import gt_axes
    gt_axes(ax, STY, matlab_ticks=False)
    return closed, disease


# ---------------------------------------------------------------------------
# Ridge-logistic link score
# ---------------------------------------------------------------------------
def local_nan0(X):
    """Replace NaN with zero, for the design matrix passed to the ridge fit.
    Missing endpoints are handled by the availability weighting in
    local_etaavail, not by this substitution."""
    return np.nan_to_num(X, nan=0.0)


def local_zcols(X):
    """Z-score each column of X over all groups pooled, ignoring NaN.

    The standardization uses the population standard deviation (ddof=0). A
    column that is constant or entirely missing is left on a unit scale."""
    Z = X.copy()
    for j in range(X.shape[1]):
        c = X[:, j]
        ok = ~np.isnan(c)
        if not ok.any():
            continue
        m = c[ok].mean()
        s = c[ok].std(ddof=0)
        if s == 0 or np.isnan(s):
            s = 1.0
        Z[:, j] = (c - m) / s
    return Z


def local_cov0(X, mu):
    """Sample covariance of X about the supplied mean mu."""
    Xc = X - mu
    n = X.shape[0]
    if n <= 1:
        return np.zeros((X.shape[1], X.shape[1]))
    return (Xc.T @ Xc) / (n - 1)


def local_ridge(X, y, lam):
    """Fit an L2-penalized logistic regression by iteratively reweighted
    least squares.

    X: design matrix, animals by endpoints. y: binary labels. lam: ridge
    penalty, applied to the endpoint coefficients but not to the intercept.
    Iterates to a coefficient change below 1e-8, capped at 100 iterations.

    Returns (w, b): the endpoint weights and the intercept."""
    n, p = X.shape
    Xa = np.column_stack([np.ones(n), X])
    beta = np.zeros(p + 1)
    R = lam * np.eye(p + 1)
    R[0, 0] = 0
    for _ in range(100):
        eta = Xa @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = np.maximum(mu * (1 - mu), 1e-6)
        z = eta + (y - mu) / W
        WX = Xa * W[:, None]
        bn = np.linalg.solve(Xa.T @ WX + R, Xa.T @ (W * z))
        if np.linalg.norm(bn - beta) < 1e-8:
            beta = bn
            break
        beta = bn
    return beta[1:], beta[0]


def local_etaavail(w, b, X, O):
    """Availability-weighted linear predictor eta, one value per animal.

    w, b: ridge weights and intercept. X: standardized endpoint matrix.
    O: boolean mask of the endpoints observed for each animal.

    An animal's score sums only its observed endpoints, then rescales by the
    ratio of the total absolute weight to the absolute weight actually
    available, so animals measured on different subsets of endpoints are
    placed on a common scale. An animal with no observed endpoint receives
    the intercept."""
    Wtot = np.sum(np.abs(w))
    n = X.shape[0]
    eta = np.zeros(n)
    Xf = np.nan_to_num(X, nan=0.0)
    for i in range(n):
        o = O[i, :]
        den = np.sum(np.abs(w[o]))
        if den <= 0:
            eta[i] = b
        else:
            eta[i] = b + (Wtot / den) * np.sum(w[o] * Xf[i, o])
    return eta


def local_etalim(eg):
    """Return (lo, hi) y limits covering every group's eta with a margin of
    one unit on each side."""
    allv = np.concatenate([eg["wt"], eg["sh"], eg["gt"]])
    allv = allv[~np.isnan(allv)]
    return allv.min() - 1.0, allv.max() + 1.0


def local_kfold(y, K):
    """Assign each observation to one of K cross-validation folds, stratified
    by label so both classes are spread evenly across folds.

    The permutation is unseeded, so fold membership differs between runs
    unless the caller seeds NumPy's global RNG beforehand. Returns a vector
    of 1-based fold indices."""
    y = np.asarray(y)
    idv = np.zeros(y.shape[0], dtype=int)
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        idx = idx[np.random.permutation(idx.size)]
        idv[idx] = np.mod(np.arange(idx.size), K) + 1
    return idv


def local_ldadir(X, y):
    """Regularised linear-discriminant direction separating the two classes.

    The pooled within-class covariance is shrunk toward a scaled identity by
    a factor lam = p / (p + n1 + n0), clamped to [0.05, 0.9], so the solve
    stays conditioned when there are more endpoints than animals.

    Returns (w, c0): the unit-norm discriminant direction and the projection
    of the class midpoint onto it, which serves as the threshold."""
    X1 = X[y == 1]
    X0 = X[y == 0]
    mu1 = X1.mean(axis=0)
    mu0 = X0.mean(axis=0)
    n1, n0 = X1.shape[0], X0.shape[0]
    p = X.shape[1]
    Sp = ((n1 - 1) * local_cov0(X1, mu1) + (n0 - 1) * local_cov0(X0, mu0)) / max(n1 + n0 - 2, 1)
    mv = np.mean(np.diag(Sp))
    if mv <= 0 or np.isnan(mv):
        mv = 1.0
    lam = min(max(p / (p + n1 + n0), 0.05), 0.9)
    Sig = (1 - lam) * Sp + lam * mv * np.eye(p)
    wl = np.linalg.solve(Sig, mu1 - mu0)
    nrm = np.linalg.norm(wl)
    if nrm > 0:
        wl = wl / nrm
    c0 = wl @ ((mu1 + mu0) / 2)
    return wl, c0


def local_auroc(y, s):
    """Area under the receiver-operating-characteristic curve for scores s
    against binary labels y, by trapezoidal integration of the empirical
    curve. NaN scores are dropped. Returns NaN if either class is empty."""
    ok = ~np.isnan(s)
    y = y[ok]
    s = s[ok]
    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        return np.nan
    order = np.argsort(-s, kind="stable")
    y2 = y[order]
    tp = np.cumsum(y2 == 1)
    fp = np.cumsum(y2 == 0)
    tpr = np.concatenate([[0], tp / P])
    fpr = np.concatenate([[0], fp / N])
    return float(np.trapz(tpr, fpr))


def local_domain_eta(data_dir, dom, cond):
    """Fit the link score within a single domain and score every animal.

    The ridge-logistic model is trained on WT against SH and the GT animals
    are projected onto the resulting axis. Point estimates come from a fit on
    all training rows; the reported AUROC and the GT eta are averaged over
    five cross-validation folds, so GT is never scored by a model fit to it.

    Returns (eg, au, aul): the per-group eta dict, the out-of-fold AUROC of
    the ridge score, and the out-of-fold AUROC of the linear-discriminant
    score computed on the same folds."""
    M = load_struct(os.path.join(data_dir, f"marm_{dom}_all.mat"))
    wt = np.atleast_2d(np.asarray(M[f"{dom}_wt_{cond}"], dtype=float))
    sh = np.atleast_2d(np.asarray(M[f"{dom}_sh_{cond}"], dtype=float))
    gt = np.atleast_2d(np.asarray(M[f"{dom}_gt_{cond}"], dtype=float))
    wt = wt[~np.all(np.isnan(wt), axis=1)]
    sh = sh[~np.all(np.isnan(sh), axis=1)]
    gt = gt[~np.all(np.isnan(gt), axis=1)]
    X = np.vstack([wt, sh, gt])
    grp = np.concatenate([np.full(wt.shape[0], 1), np.full(sh.shape[0], 2), np.full(gt.shape[0], 3)])
    X = local_zcols(X)
    O = ~np.isnan(X)
    isWT, isSH, isGT = grp == 1, grp == 2, grp == 3
    tr = (isWT | isSH) & O.any(axis=1)
    ytr = isWT[tr].astype(float)
    Xtr, Otr = X[tr], O[tr]
    w, b = local_ridge(local_nan0(Xtr), ytr, 5.0)
    fold = local_kfold(ytr, 5)
    oof = np.full(ytr.shape, np.nan)
    oofl = np.full(ytr.shape, np.nan)
    gtcv = np.zeros(int(isGT.sum()))
    nf = 0
    for ff in range(1, 6):
        trm, te = fold != ff, fold == ff
        wf, bf = local_ridge(local_nan0(Xtr[trm]), ytr[trm], 5.0)
        oof[te] = local_etaavail(wf, bf, Xtr[te], Otr[te])
        gtcv = gtcv + local_etaavail(wf, bf, X[isGT], O[isGT])
        wl, c0 = local_ldadir(local_nan0(Xtr[trm]), ytr[trm])
        oofl[te] = local_nan0(Xtr[te]) @ wl - c0
        nf += 1
    gtcv = gtcv / nf
    eta = np.full(grp.shape, np.nan)
    eta[tr] = local_etaavail(w, b, Xtr, Otr)
    eta[isGT] = gtcv
    eg = {"wt": eta[isWT], "sh": eta[isSH], "gt": eta[isGT]}
    au = local_auroc(ytr, oof)
    aul = local_auroc(ytr, oofl)
    return eg, au, aul


def local_combined_eta(data_dir, doms, cond, min_domains=1):
    """Combined multi-domain link score: one ridge-logistic model over all
    endpoints of all domains, one eta per animal.

    The model is trained on WT against SH and the GT animals are projected
    onto the resulting axis, with the GT eta averaged over five
    cross-validation folds.

    min_domains: animals observed in fewer than this many of the four domains
    are excluded before the model is fit or scored. At min_domains = 2 every
    animal's score integrates at least two domains; a one-domain animal would
    otherwise contribute a single domain rescaled to the full endpoint set by
    local_etaavail. min_domains = 1, the default, keeps every animal.

    Endpoint coverage remains associated with group, so the combined score
    compares groups whose scores rest on different numbers of observed
    endpoints; the per-group coverage summary returned in `cov` records this.

    Returns (eg, au, cov): the per-group eta dict, the out-of-fold AUROC, and
    a per-group coverage dict giving n, the median number of observed
    endpoints, the count of animals with complete coverage, the number
    excluded, and min_domains.
    """
    gs = ("wt", "sh", "gt")
    Xg = {}
    dom_w = []
    for g in gs:
        cols = []
        widths = []
        for dom in doms:
            M = load_struct(os.path.join(data_dir, f"marm_{dom}_all.mat"))
            block = np.atleast_2d(np.asarray(M[f"{dom}_{g}_{cond}"], dtype=float))
            cols.append(block)
            widths.append(block.shape[1])
        Xg[g] = np.hstack(cols)
        if not dom_w:
            dom_w = widths
    for g in gs:
        Xg[g] = Xg[g][~np.all(np.isnan(Xg[g]), axis=1)]
    X = np.vstack([Xg["wt"], Xg["sh"], Xg["gt"]])
    grp = np.concatenate([np.full(Xg["wt"].shape[0], 1),
                           np.full(Xg["sh"].shape[0], 2),
                           np.full(Xg["gt"].shape[0], 3)])

    # Domain index of each column of the stacked matrix.
    domcol = np.concatenate([np.full(w, di) for di, w in enumerate(dom_w)])

    # Exclude animals observed in fewer than min_domains domains.
    Oraw = ~np.isnan(X)
    ndom = np.array([np.unique(domcol[Oraw[i]]).size for i in range(X.shape[0])])
    excluded = {g: int(np.sum((grp == gi) & (ndom < min_domains)))
                for gi, g in zip((1, 2, 3), gs)}
    keep = ndom >= min_domains
    X = X[keep]
    grp = grp[keep]

    Oraw = ~np.isnan(X)
    cov = {}
    for gi, g in zip((1, 2, 3), gs):
        k = Oraw[grp == gi].sum(axis=1)
        cov[g] = dict(n=int(k.size),
                      median_obs=float(np.median(k)) if k.size else float("nan"),
                      complete=int(np.sum(k == X.shape[1])),
                      excluded=excluded[g], min_domains=min_domains)

    X = local_zcols(X)
    O = ~np.isnan(X)
    isWT, isSH, isGT = grp == 1, grp == 2, grp == 3
    tr = (isWT | isSH) & O.any(axis=1)
    ytr = isWT[tr].astype(float)
    Xtr, Otr = X[tr], O[tr]
    w, b = local_ridge(local_nan0(Xtr), ytr, 5.0)
    fold = local_kfold(ytr, 5)
    oof = np.full(ytr.shape, np.nan)
    gtcv = np.zeros(int(isGT.sum()))
    nf = 0
    for ff in range(1, 6):
        trm, te = fold != ff, fold == ff
        wf, bf = local_ridge(local_nan0(Xtr[trm]), ytr[trm], 5.0)
        oof[te] = local_etaavail(wf, bf, Xtr[te], Otr[te])
        gtcv = gtcv + local_etaavail(wf, bf, X[isGT], O[isGT])
        nf += 1
    gtcv = gtcv / nf
    eta = np.full(grp.shape, np.nan)
    eta[tr] = local_etaavail(w, b, Xtr, Otr)
    eta[isGT] = gtcv
    eg = {"wt": eta[isWT], "sh": eta[isSH], "gt": eta[isGT]}
    au = local_auroc(ytr, oof)
    return eg, au, cov


def local_scurve(ax, eg, ttl, showy, CMEAN, STY):
    """Draw the link-score S-curve: the logistic function of eta with each
    animal placed on it at its own eta, colored by group.

    The y axis is the modeled probability of being wild-type-like. Reference
    lines mark eta = 0 and P = 0.5, where the two classes are equally
    likely."""
    lo, hi = local_etalim(eg)
    xs = np.linspace(lo, hi, 400)
    ax.plot([0, 0], [-0.04, 1.06], "-", color=STY.clr.gray_lite, lw=STY.lw.hair)
    ax.plot([lo, hi], [0.5, 0.5], "-", color=STY.clr.gray_lite, lw=STY.lw.hair)
    ax.plot(xs, 1 / (1 + np.exp(-xs)), "-", color=STY.clr.gray_mid, lw=1.0)
    for gi, g in enumerate(("wt", "sh", "gt")):
        v = eg[g]
        v = v[~np.isnan(v)]
        ax.scatter(v, 1 / (1 + np.exp(-v)), s=STY.mk.pt, c=[CMEAN[gi]],
                   edgecolors="w", linewidths=0.2, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"link score $\eta$", fontname=STY.font.name, fontsize=STY.fs_lab)
    if showy:
        ax.set_ylabel("P(WT-like)", fontname=STY.font.name, fontsize=STY.fs_lab)
    else:
        ax.set_yticklabels([])
    ax.set_title(ttl, fontname=STY.font.name, fontsize=STY.fs_title - 1, fontweight="normal")
    from gtstyle.plotting import gt_ticks_matlab
    from gtstyle import gt_axes
    gt_ticks_matlab(ax, axis="x")          # x ticks chosen automatically
    gt_axes(ax, STY, matlab_ticks=False)   # y ticks are explicit: 0, 0.5, 1


# ---------------------------------------------------------------------------
# Shared-axis tick chooser
# ---------------------------------------------------------------------------
def local_niceTicks(lo, hi):
    """Choose at most six ticks spanning [lo, hi] on a round step drawn from
    1, 2, 2.5, 5 or 10 times a power of ten.

    Used where several panels share one y range and must therefore carry the
    same ticks. Returns an empty array if no two ticks fit."""
    MAXT = 6
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.array([])
    raw = span / MAXT
    mag = 10 ** np.floor(np.log10(raw))
    cand = np.array([1, 2, 2.5, 5, 10]) * mag
    ge = cand[cand >= raw]
    step = ge[0] if ge.size else 10 * mag
    first = np.ceil(lo / step) * step
    last = np.floor(hi / step) * step
    if last < first:
        return np.array([])
    t = np.arange(first, last + step / 2, step)
    t = np.round(t / step) * step
    if t.size < 2:
        return np.array([])
    return t


def local_prctile(x, p):
    """Percentile p of x by linear interpolation, with NaNs dropped. Returns
    NaN for an empty sample."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, p))


def local_vbox(ax, eg, lo, hi, showy, CMEAN, CSCAT, CBARF, STY, yticks=None, rng=None):
    """Draw the WT / SH / GT box plots of the link score eta.

    Per group: a pale filled box from the first to the third quartile, the
    individual animals as jittered points, black whiskers to the extremes,
    and a black median rule. lo, hi set the y limits and yticks the tick
    positions, so several panels can share one axis.

    The point jitter is cosmetic and unseeded unless an rng is supplied."""
    if rng is None:
        rng = np.random.default_rng()
    w = 0.55
    gs = ("wt", "sh", "gt")
    ax.plot([-0.6, 2.6], [0, 0], "-", color=STY.clr.gray_lite, lw=STY.lw.hair)
    for gi, g in enumerate(gs):
        v = eg[g]
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        x = gi
        q1 = local_prctile(v, 25)
        med = local_prctile(v, 50)
        q3 = local_prctile(v, 75)
        vlo, vhi = v.min(), v.max()
        # box fill behind the points; black edges, whiskers and median over
        ax.fill([x - w / 2, x - w / 2, x + w / 2, x + w / 2], [q1, q3, q3, q1],
                color=CBARF[gi], edgecolor="none", zorder=1)
        jx = x + (rng.random(v.size) - 0.5) * w * 0.55
        ax.scatter(jx, v, s=STY.mk.pt, c=[CSCAT[gi]], edgecolors="none", zorder=2)
        ax.plot([x, x], [vlo, q1], "-", color="k", lw=0.5, zorder=3)
        ax.plot([x, x], [q3, vhi], "-", color="k", lw=0.5, zorder=3)
        ax.plot([x - w / 2, x - w / 2, x + w / 2, x + w / 2, x - w / 2],
                [q1, q3, q3, q1, q1], "-", color="k", lw=0.7, zorder=3)
        ax.plot([x - w / 2, x + w / 2], [med, med], "-", color="k", lw=STY.lw.axis, zorder=4)

    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["WT", "SH", "GT"])
    if yticks is not None and len(yticks):
        ax.set_yticks(yticks)
    if showy:
        ax.set_ylabel(r"link score $\eta$", fontname=STY.font.name, fontsize=STY.fs_lab)
    else:
        ax.set_ylabel(r"$\eta$", fontname=STY.font.name, fontsize=STY.fs_lab)
    from gtstyle import gt_axes, gt_color_xticklabels
    gt_axes(ax, STY, matlab_ticks=False)
    gt_color_xticklabels(ax, STY)


# ---------------------------------------------------------------------------
# Correlation matrices
# ---------------------------------------------------------------------------
def local_corr(a, b):
    """Pearson correlation of two equal-length vectors. Returns NaN if either
    has zero variance."""
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    da = np.sqrt(np.sum(a ** 2))
    db = np.sqrt(np.sum(b ** 2))
    if da == 0 or db == 0:
        return np.nan
    return float(np.sum(a * b) / (da * db))


def local_divmap(m):
    """Diverging colormap of m steps running from amber through white to
    dark gray.

    Positive correlations read as gray and negative correlations as amber, so
    the sign of a coefficient is legible without reading the printed value.
    A matrix whose coefficients are all positive therefore renders as a plain
    white-to-gray ramp."""
    anchors = np.array([[0.85, 0.55, 0.15], [1, 1, 1], [0.15, 0.15, 0.15]])
    x = np.linspace(0, 1, anchors.shape[0])
    xi = np.linspace(0, 1, m)
    return np.column_stack([np.interp(xi, x, anchors[:, c]) for c in range(3)])


def local_corrmat(data_dir, dom, cond):
    """Pearson correlation matrix among the four endpoints of one domain,
    over all three groups pooled. A cell with three or fewer animals having
    both endpoints observed is left NaN. Returns a 4x4 array."""
    M = load_struct(os.path.join(data_dir, f"marm_{dom}_all.mat"))
    Xs = []
    for g in ("wt", "sh", "gt"):
        v = np.atleast_2d(np.asarray(M[f"{dom}_{g}_{cond}"], dtype=float))
        Xs.append(v[~np.all(np.isnan(v), axis=1)])
    X = np.vstack(Xs)
    C = np.full((4, 4), np.nan)
    for i in range(4):
        for j in range(4):
            a, b = X[:, i], X[:, j]
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() > 3:
                C[i, j] = local_corr(a[ok], b[ok])
    return C


def local_corrpanel(ax, C, dom, domlab, STY):
    """Draw one domain's correlation matrix as a lower-triangular heat map
    with the coefficient printed in each cell. The title carries the largest
    absolute off-diagonal coefficient."""
    cmap = local_divmap(64)
    for i in range(4):
        for j in range(4):
            if j > i:
                continue
            v = C[i, j]
            if np.isnan(v):
                col = (1, 1, 1)
            else:
                t = min(max((v + 1) / 2, 0), 1)
                col = cmap[int(round(t * (cmap.shape[0] - 1)))]
            ax.fill([j + 0.5, j + 1.5, j + 1.5, j + 0.5],
                    [i + 0.5, i + 0.5, i + 1.5, i + 1.5], color=col,
                    edgecolor="w", linewidth=0.5)
            if not np.isnan(v):
                if i == j:
                    ax.text(j + 1, i + 1, "1", ha="center", va="center",
                            fontname=STY.font.name, fontsize=STY.fs_cell, color="w")
                else:
                    tc = "w" if abs(v) > 0.6 else "k"
                    ax.text(j + 1, i + 1, f"{v:.2f}", ha="center", va="center",
                            fontname=STY.font.name, fontsize=STY.fs_cell, color=tc)
    ax.invert_yaxis()
    labs = [f"{dom}{k}" for k in range(1, 5)]
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_yticks(range(1, 5))
    ax.set_yticklabels(labs)
    ax.tick_params(length=0)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(4.5, 0.5)
    ax.set_aspect("equal")
    off = [C[i, j] for i in range(1, 4) for j in range(i)]
    off = [v for v in off if not np.isnan(v)]
    mx = max(abs(v) for v in off) if off else np.nan
    ax.set_title(f"{domlab}\nmax |r| = {mx:.2f}", fontname=STY.font.name,
                 fontsize=STY.fs_title, fontweight="normal")
    from gtstyle import gt_axes
    gt_axes(ax, STY, matlab_ticks=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def local_corrbar(ax, STY):
    """Draw the color scale for the correlation matrices, spanning r = -1
    to r = +1."""
    cmap = local_divmap(64)
    n = cmap.shape[0]
    xs = np.linspace(-1, 1, n + 1)
    for k in range(n):
        ax.fill([xs[k], xs[k + 1], xs[k + 1], xs[k]], [0, 0, 1, 1], color=cmap[k], edgecolor="none")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(np.arange(-1, 1.5, 0.5))
    ax.set_xlabel("Pearson r", fontname=STY.font.name, fontsize=STY.fs_lab)
    ax.tick_params(direction="out", length=1.2 * 6)
    from gtstyle import gt_axes
    gt_axes(ax, STY, matlab_ticks=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(STY.lw.axis)
        ax.spines[side].set_color(STY.clr.axis)


# ---------------------------------------------------------------------------
# Paired control-to-test change
# ---------------------------------------------------------------------------
def local_pairedct(D, dom, k, g):
    """Return the (control, test) value pairs for endpoint k of domain dom in
    group g, keeping only animals present in both conditions with both values
    observed."""
    c = np.atleast_2d(np.asarray(D[f"{dom}_{g}_c"], dtype=float))
    t = np.atleast_2d(np.asarray(D[f"{dom}_{g}_t"], dtype=float))
    shd = (~np.all(np.isnan(c), axis=1)) & (~np.all(np.isnan(t), axis=1))
    cv = c[shd, k - 1]
    tv = t[shd, k - 1]
    ok = ~np.isnan(cv) & ~np.isnan(tv)
    return cv[ok], tv[ok]


def local_pairedperm(cv, tv, nperm):
    """Two-sided sign-flip permutation test on the within-animal change.

    The statistic is the absolute mean of the paired differences
    test - control; the null distribution is generated by independently
    flipping the sign of each animal's difference. Fewer than three usable
    pairs returns NaN. The MT19937 stream is reseeded to 12345 on every call,
    so the p-value is deterministic for a given (data, nperm)."""
    d = np.asarray(tv, dtype=float) - np.asarray(cv, dtype=float)
    d = d[~np.isnan(d)]
    n = d.size
    if n < 3:
        return np.nan
    obs = abs(d.mean())
    rs = np.random.RandomState(12345)
    cnt = 0
    for _ in range(nperm):
        sgn = (rs.random_sample(n) > 0.5).astype(float) * 2 - 1
        if abs(np.mean(d * sgn)) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (nperm + 1)


def local_pairedq(data_dir, doms, nperm, cache=None):
    """Paired sign-flip p-values for every endpoint in the SH and GT groups,
    adjusted by Benjamini-Hochberg across all of them as a single family.

    Returns a dict keyed '<dom><k>_<group>' holding the q-value."""
    gs = ("sh", "gt")
    keys = []
    ps = []
    for dom in doms:
        D = load_struct(os.path.join(data_dir, f"marm_{dom}_all.mat"))
        for k in range(1, 5):
            for g in gs:
                cv, tv = local_pairedct(D, dom, k, g)
                key = f"{dom}{k}_{g}"
                keys.append(key)
                cache_key = (key, nperm)
                if cache is not None and cache_key in cache:
                    ps.append(cache[cache_key])
                else:
                    p = local_pairedperm(cv, tv, nperm)
                    ps.append(p)
                    if cache is not None:
                        cache[cache_key] = p
    q = gt_bh(np.array(ps))
    return dict(zip(keys, q))


def local_paired(ax, data_dir, dom, k, showy, PQ, CMEAN, CSCAT, STY):
    """Draw one endpoint's control-to-test change: one faint line per animal
    joining its two conditions, plus a heavy group mean line per group.

    The SH and GT significance labels are read from PQ, the BH-adjusted
    q-values from local_pairedq. The cog4 panel is drawn on a highlight band
    with a colored bold title."""
    D = load_struct(os.path.join(data_dir, f"marm_{dom}_all.mat"))
    is_c4 = local_iscog4(f"{dom}{k}")
    if is_c4:
        ax.fill([-0.35, 1.35, 1.35, -0.35], [-3.5, -3.5, 3.5, 3.5],
                color=gt_ablend(STY.clr.hilite, 0.30), edgecolor="none")
    ax.plot([-0.35, 1.35], [0, 0], "-", color=STY.clr.gray_lite, lw=STY.lw.hair)

    for gi, g in enumerate(("sh", "gt")):
        key = f"{dom}{k}_{g}"
        s = gt_stars(PQ[key]) if key in PQ else "-"
        ax.text(0.30 + 0.40 * gi, 0.985, s, transform=ax.transAxes,
                ha="center", va="top", fontname=STY.font.name,
                fontsize=STY.fs_star, fontweight="bold", color=CMEAN[gi + 1])

    for gi, g in enumerate(("wt", "sh", "gt")):
        cv, tv = local_pairedct(D, dom, k, g)
        for j in range(cv.size):
            ax.plot([0, 1], [cv[j], tv[j]], "-", color=CSCAT[gi], lw=STY.lw.pair)
        if cv.size:
            ax.plot([0, 1], [cv.mean(), tv.mean()], "-", color=CMEAN[gi], lw=STY.lw.mean)
            ax.plot([0, 1], [cv.mean(), tv.mean()], "o", color=CMEAN[gi],
                    markerfacecolor=CMEAN[gi], markersize=2.6,
                    markeredgecolor="w", markeredgewidth=0.25)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["C", "T"])
    ax.set_yticks(np.arange(-3, 3.1, 1.5))
    tcol, tw = "k", "normal"
    if is_c4:
        tcol, tw = STY.clr.hilite_tx, "bold"
    ax.set_title(f"{dom}{k}", fontname=STY.font.name, fontsize=STY.fs_title - 1,
                 color=tcol, fontweight=tw)
    if not showy:
        ax.set_yticklabels([])
    from gtstyle import gt_axes
    gt_axes(ax, STY, matlab_ticks=False)


# ---------------------------------------------------------------------------
# Endpoint-pair scatter
# ---------------------------------------------------------------------------
def local_scatterpair(ax, data_dir, dom, k1, k2, cond, CMEAN, STY):
    """Scatter endpoint k1 against endpoint k2 within one domain, colored by
    group, with a least-squares line through all groups pooled.

    The Pearson correlation and the number of animals contributing to it are
    printed in the corner. Returns (r, n)."""
    M = load_struct(os.path.join(data_dir, f"marm_{dom}_all.mat"))
    A, B = [], []
    for gi, g in enumerate(("wt", "sh", "gt")):
        v = np.atleast_2d(np.asarray(M[f"{dom}_{g}_{cond}"], dtype=float))
        v = v[~np.all(np.isnan(v), axis=1)]
        a, b = v[:, k1 - 1], v[:, k2 - 1]
        ok = ~np.isnan(a) & ~np.isnan(b)
        ax.scatter(a[ok], b[ok], s=STY.mk.pt, c=[gt_ablend(CMEAN[gi], 0.75)], edgecolors="none")
        A.append(a[ok])
        B.append(b[ok])
    A = np.concatenate(A)
    B = np.concatenate(B)
    r = local_corr(A, B)
    if A.size > 2:
        pf = np.polyfit(A, B, 1)
        xs = np.array([A.min(), A.max()])
        ax.plot(xs, pf[0] * xs + pf[1], "-", color=STY.clr.gray_mid, lw=STY.lw_fit)
    ax.set_xlabel(f"{dom}{k1}", fontname=STY.font.name, fontsize=STY.fs_lab)
    ax.set_ylabel(f"{dom}{k2}", fontname=STY.font.name, fontsize=STY.fs_lab)
    ax.text(0.04, 0.97, f"r = {r:.2f}\nn = {A.size}", transform=ax.transAxes,
            ha="left", va="top", fontname=STY.font.name, fontsize=STY.fs_tick,
            color=STY.clr.axis)

    xt = ax.get_xticks()
    if xt.size > 4:
        ax.set_xticks(xt[::2])
    yt = ax.get_yticks()
    if yt.size > 4:
        ax.set_yticks(yt[::2])
    from gtstyle import gt_axes
    gt_axes(ax, STY, matlab_ticks=False)
    return r, A.size
