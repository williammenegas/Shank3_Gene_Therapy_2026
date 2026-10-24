"""
Figure 1, panel c: multivariate classifier separating WT from SH, and the
position of GT on that classifier axis. Four sub-panels.

Groups: WT, SH (Shank3 mutant + AAV-GFP), GT (Shank3 mutant + AAV-miniSHANK3),
all from the high-dose batch. n = 26 / 29 / 19 (74 animals).

Model. Five behavioral constructs (locomotion, rearing, zero maze,
fractional social state, rotarod) are z-scored, grouped by single-linkage
clustering on |r| at THRESH, averaged within cluster, z-scored again, and fed
to an L2-penalized logistic regression (penalty LAMBDA, intercept
unpenalized) trained to discriminate WT from SH. Out-of-fold predictions come
from NFOLD-fold stratified cross-validation. Animals missing a construct are
scored by the available-case linear predictor, which rescales the observed
weights to the total weight, so no animal is dropped. Within this batch only
the fractional social state has missing values (11 / 26 WT, 10 / 29 SH,
7 / 19 GT).

  a  Group separation on the link scale, bar plus beeswarm. eta = 0 is the
     decision boundary P = 0.5, not a zero quantity, so the bar encodes the
     mean as an edge. Group differences are two-sided Mann-Whitney U tests
     on the three pairs.

  b  Pairwise correlation between constructs, lower triangle, on complete
     pairs. Color ramps white to black over r = 0 to 0.80, the clustering
     threshold, so the panel shows directly that no pair merges. All
     correlations are positive, hence a sequential rather than a diverging
     ramp. The diagonal is exactly 1 and clamps to black.

  c  ROC for WT versus SH, from the cross-validated out-of-fold predictions.

  d  Discriminant projection. The x axis is eta, the projection onto the
     fitted weight vector, so the decision boundary is the line x = 0; the
     y axis is the leading direction orthogonal to that weight vector (first
     right singular vector of the residual), i.e. the variation the model
     discards. eta here is the cross-validated value, as in panel c.

GT animals appear in sub-panel a only; sub-panels c and d characterise the
WT versus SH classifier the model was trained on. Batch is confounded with
dose in this experiment, so no dose comparison is made.

Implementation. Rank-sum, tied ranks, correlation and percentiles are
computed locally rather than through a toolbox; `scipy.stats.rankdata`
supplies average ranks for ties. Fill colors are pre-blended toward white
with gt_ablend because EPS carries no transparency and a true alpha would
force the panel to rasterize.

Cross-validation fold assignment is drawn from NumPy's Mersenne Twister
(RNG_SEED). The original implementation's `rng`/`randperm` consume their
stream differently, so fold membership, and hence the out-of-fold AUROC, the
ROC curve in c and the cross-validated eta in d, are close to but not
bit-identical with it. NPERM is retained as a named constant for parity with
that implementation; the permutation-based construct importance and the
univariate per-assay fits it controls are not read by any sub-panel and are
not computed here.

Data: panel_modelA_fourpanel.mat
"""

import os
import sys
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig1", "fig1_c_classifier")

THRESH = 0.80        # |r| above which two constructs join the same cluster
LAMBDA = 5.0         # L2 penalty on the logistic weights (intercept unpenalized)
NFOLD = 5            # stratified cross-validation folds
NPERM = 200          # permutation count constant, retained for parity; unused here
RNG_SEED = 0         # seeds the fold assignment

USE_A = ["Locomotion", "Rearing", "Zero maze", "FracSocial", "Rotarod"]


# Numeric helpers
def _pstar(p):
    """Significance label on the house scale (0.05 / 0.01 / 0.001), with a
    NaN guard.

    A model fit can fail to yield a p-value; gt_pstar raises on NaN, so this
    wrapper returns 'n/a' in that case. The thresholds are the house ones.
    """
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return gt_pstar(p)


def _zscore(X):
    m = np.nanmean(X, axis=0)
    s = np.nanstd(X, axis=0, ddof=1)
    s = np.where(s < 1e-9, 1.0, s)
    return (X - m) / s


def _pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if d < 1e-12:
        return 0.0
    return float(np.sum(x * y) / d)


def _nancorr(Z):
    p = Z.shape[1]
    C = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            m = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
            r = _pearson(Z[m, i], Z[m, j]) if m.sum() > 3 else 0.0
            C[i, j] = C[j, i] = r
    return C


def _cluster(A, t):
    """Single-linkage clustering on |correlation|: a construct joins a
    cluster if its correlation with any member exceeds t."""
    p = A.shape[0]
    un = list(range(p))
    cl = []
    while un:
        cur = [un[0]]
        changed = True
        while changed:
            changed = False
            remaining = [j for j in un if j not in cur]
            for j in remaining:
                if any(A[j, c] > t for c in cur):
                    cur.append(j)
                    changed = True
        cl.append(sorted(cur))
        un = [u for u in un if u not in cur]
    return cl


def _ridge_logit(X, y, lam):
    """L2-penalized logistic regression by iteratively reweighted least
    squares. The intercept is not penalized."""
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0
    for _ in range(200):
        eta = X @ w + b
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(mu * (1 - mu), 1e-6)
        z = eta + (y - mu) / W
        Xa = np.column_stack([np.ones(n), X])
        Rmat = np.diag([0.0] + [lam] * p)
        A = Xa.T @ (W[:, None] * Xa) + Rmat
        rhs = Xa.T @ (W * z)
        bnew = np.linalg.solve(A, rhs)
        cur = np.concatenate([[b], w])
        if np.max(np.abs(bnew - cur)) < 1e-9:
            b, w = bnew[0], bnew[1:]
            break
        b, w = bnew[0], bnew[1:]
    return w, b


def _eta_avail(w, b, X, O):
    """Available-case linear predictor: the observed weights are rescaled to
    the total absolute weight, so an animal missing a construct is still
    scored rather than dropped. O flags which constructs are observed."""
    Wtot = np.sum(np.abs(w))
    n = X.shape[0]
    eta = np.zeros(n)
    for i in range(n):
        o = O[i, :]
        if not o.any():
            eta[i] = b
            continue
        Wobs = np.sum(np.abs(w[o]))
        eta[i] = b + (Wtot / max(Wobs, 1e-9)) * np.sum(w[o] * X[i, o])
    return eta


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _stratified_folds(y, k, rng):
    """Class-stratified cross-validation fold ids (0-based)."""
    f = np.zeros(y.shape, dtype=int)
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        idx = idx[rng.permutation(idx.size)]
        f[idx] = np.arange(idx.size) % k
    return f


def _auroc(y, s):
    rk = rankdata(s, method="average")
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((np.sum(rk[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def _roc(y, s):
    idx = np.argsort(-s, kind="stable")
    ys = y[idx]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    tpr = np.concatenate([[0], tp / max(np.sum(y == 1), 1)])
    fpr = np.concatenate([[0], fp / max(np.sum(y == 0), 1)])
    return fpr, tpr


def _ranksum(a, b):
    """Two-sided Mann-Whitney U test, normal approximation with tie and
    continuity corrections. Implemented here rather than via
    scipy.stats.mannwhitneyu, whose default tie and continuity handling
    differs."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n1, n2 = a.size, b.size
    if n1 < 1 or n2 < 1:
        return np.nan
    x = np.concatenate([a, b])
    rk = rankdata(x, method="average")
    R1 = np.sum(rk[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    _, counts = np.unique(x, return_counts=True)
    N = n1 + n2
    tie = np.sum(counts ** 3 - counts)
    sd = math.sqrt(n1 * n2 / 12 * ((N + 1) - tie / (N * (N - 1)))) if N > 1 else 0.0
    if sd < 1e-12:
        return 1.0
    z = (abs(U1 - mu) - 0.5) / sd
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return float(min(max(p, 0.0), 1.0))


def _ranksum_safe(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return np.nan
    return _ranksum(a, b)


def _prctile_local(v, p):
    """Percentile by linear interpolation of the sorted sample at plotting
    positions (k - 0.5) / n, extrapolating linearly beyond the end points."""
    x = np.sort(np.asarray(v, dtype=float))
    n = x.size
    pos = (np.arange(n) + 0.5) / n
    p = np.atleast_1d(np.asarray(p, dtype=float)) / 100.0
    out = np.empty_like(p)
    for i, pp in enumerate(p):
        if n == 1:
            out[i] = x[0]
        elif pp <= pos[0]:
            slope = (x[1] - x[0]) / (pos[1] - pos[0])
            out[i] = x[0] + slope * (pp - pos[0])
        elif pp >= pos[-1]:
            slope = (x[-1] - x[-2]) / (pos[-1] - pos[-2])
            out[i] = x[-1] + slope * (pp - pos[-1])
        else:
            out[i] = np.interp(pp, pos, x)
    return out if out.size > 1 else out[0]


def _kde(v, xs):
    """Gaussian kernel density estimate with Silverman's bandwidth."""
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    if n < 2:
        return np.zeros_like(xs)
    s = v.std(ddof=1)
    iq = np.diff(_prctile_local(v, [25, 75]))[0]
    a = min(s, iq / 1.349)
    if a <= 0:
        a = s
    h = 0.9 * a * n ** (-1 / 5)
    if h <= 0:
        h = 1.0
    d = np.zeros_like(xs)
    for vi in v:
        d = d + np.exp(-0.5 * ((xs - vi) / h) ** 2)
    return d / (n * h * math.sqrt(2 * math.pi))


def _fit_ridge(X, assays, level, use_assays, drop_level, thresh, lam, nfold, rng):
    """Fit the classifier for one construct set: z-score, cluster, average
    within cluster, z-score again, then cross-validated and full-sample
    L2-penalized logistic fits, plus the scores for the held-out group."""
    keep = [assays.index(a) for a in use_assays]
    Xs = X[:, keep]
    anms = [assays[k] for k in keep]

    Zs = _zscore(Xs)
    Cmat = _nancorr(Zs)
    clust = _cluster(np.abs(Cmat), thresh)
    nC = len(clust)

    IDX = np.full((Zs.shape[0], nC), np.nan)
    with warnings.catch_warnings():
        # An animal missing every construct in a cluster gives an all-NaN
        # row, which correctly yields NaN for that cluster; the resulting
        # "Mean of empty slice" RuntimeWarning is expected and suppressed.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for c, members in enumerate(clust):
            IDX[:, c] = np.nanmean(Zs[:, members], axis=1)
    IDX = _zscore(IDX)
    obsIDX = ~np.isnan(IDX)

    haveAny = obsIDX.any(axis=1)
    if drop_level > 0:
        usable = haveAny & (level != drop_level)
    else:
        usable = haveAny

    istr = usable & ((level == 1) | (level == 2))
    Xtr = IDX[istr, :]
    Otr = obsIDX[istr, :]
    ytr = (level[istr] == 1).astype(float)
    Xfit = Xtr.copy()
    Xfit[np.isnan(Xfit)] = 0

    folds = _stratified_folds(ytr, nfold, rng)
    oof = np.full(ytr.shape, np.nan)
    etaOof = np.full(ytr.shape, np.nan)
    Wfold = []
    for f in range(nfold):
        te = folds == f
        tr = ~te
        w, b = _ridge_logit(Xfit[tr, :], ytr[tr], lam)
        etaOof[te] = _eta_avail(w, b, Xtr[te, :], Otr[te, :])
        oof[te] = _sigmoid(etaOof[te])
        Wfold.append((w, b))
    auroc = _auroc(ytr, oof)

    wFull, bFull = _ridge_logit(Xfit, ytr, lam)
    eta_tr = _eta_avail(wFull, bFull, Xtr, Otr)

    R = dict(anms=anms, nAssay=len(anms), nC=nC, Zs=Zs, IDX=IDX, obsIDX=obsIDX,
             usable=usable, istr=istr, Xtr=Xtr, Otr=Otr, ytr=ytr,
             oof=oof, auroc=auroc, nTrain=ytr.size, etaOof=etaOof, Wfold=Wfold,
             wFull=wFull, bFull=bFull, eta_tr=eta_tr,
             pWT=oof[ytr == 1], pSH=oof[ytr == 0])

    R["pGL"] = np.array([]); R["eGL"] = np.array([]); R["mGL"] = np.array([])
    R["pGH"] = np.array([]); R["eGH"] = np.array([]); R["mGH"] = np.array([])
    for g in (4,):   # GT is the only non-training level present in this batch
        m = usable & (level == g)
        if not m.any():
            continue
        Xg = IDX[m, :]
        Og = obsIDX[m, :]
        pg = np.zeros(int(m.sum()))
        for f in range(nfold):
            w, b = Wfold[f]
            pg = pg + _sigmoid(_eta_avail(w, b, Xg, Og))
        pg = pg / nfold
        eg = _eta_avail(wFull, bFull, Xg, Og)
        mg = ~Og.all(axis=1)
        if g == 3:
            R["pGL"], R["eGL"], R["mGL"] = pg, eg, mg
        else:
            R["pGH"], R["eGH"], R["mGH"] = pg, eg, mg

    R["p_gh_sh"] = _ranksum_safe(R["pGH"], R["pSH"])
    R["p_gh_wt"] = _ranksum_safe(R["pGH"], R["pWT"])
    R["p_gl_sh"] = _ranksum_safe(R["pGL"], R["pSH"])
    R["p_gl_wt"] = _ranksum_safe(R["pGL"], R["pWT"])
    return R


# Sub-panel drawing
def _eta_group_plot(ax, R, cmean, cscat, bar_w, swarm_w, pt_size, bar_lw, se_lw, STY):
    names = ["WT", "SH", "GT-high"]
    vals = [R["eta_tr"][R["ytr"] == 1], R["eta_tr"][R["ytr"] == 0], np.ravel(R["eGH"])]

    topAll, botAll = -np.inf, np.inf
    means, ses = [], []
    for k in (1, 2, 3):
        v = vals[k - 1]
        v = v[~np.isnan(v)]
        m = v.mean()
        se = v.std(ddof=1) / np.sqrt(v.size)
        means.append(m)
        ses.append(se)
        x0, x1 = k - bar_w / 2, k + bar_w / 2

        ax.fill([x0, x0, x1, x1], [0, m, m, 0], color=gt_ablend(cmean[k - 1], 0.08),
                edgecolor="none", zorder=1)

        xs = gt_beeswarm(v, k, swarm_w)
        xs = np.clip(xs, x0 + 0.02, x1 - 0.02)
        ax.scatter(xs, v, s=pt_size, c=[cscat[k - 1]], edgecolors="none", zorder=2)

        ax.plot([x0, x0, x1, x1], [0, m, m, 0], "-", color=cmean[k - 1],
                linewidth=bar_lw, zorder=3)
        ax.plot([k, k], [m - se, m + se], "-", color="k", linewidth=se_lw, zorder=4)
        ax.plot([k - 0.09, k + 0.09], [m - se, m - se], "-", color="k",
                linewidth=se_lw, zorder=4)
        ax.plot([k - 0.09, k + 0.09], [m + se, m + se], "-", color="k",
                linewidth=se_lw, zorder=4)

        topAll = max(topAll, v.max())
        botAll = min(botAll, v.min())

    ax.plot([0.4, 3.6], [0, 0], "-", color=(0.35, 0.35, 0.35), linewidth=0.7, zorder=0)

    p12 = _ranksum_safe(vals[0], vals[1])
    p23 = _ranksum_safe(vals[1], vals[2])
    p13 = _ranksum_safe(vals[0], vals[2])

    rng_ = topAll - botAll
    step = 0.09 * rng_
    h = 0.05 * rng_
    y1 = topAll + step
    gt_drawSig(ax, 1, 2, y1, h, _pstar(p12), STY)
    gt_drawSig(ax, 2, 3, y1, h, _pstar(p23), STY)
    gt_drawSig(ax, 1, 3, y1 + step * 1.6, h, _pstar(p13), STY)

    ax.set_xlim(0.4, 3.6)
    ax.set_ylim(botAll - 0.06 * (topAll - botAll), y1 + step * 3.2)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(names)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    gt_color_xticklabels(ax, STY, labels=("WT", "SH", "GT-high"))
    return p12, p23, p13


def _corr_tri(ax, Zs, anms):
    CMAX = 0.80
    C = _nancorr(Zs)
    n = C.shape[0]

    for i in range(n):
        for j in range(i + 1):
            x, y = j, n - 1 - i
            r = C[i, j]
            t = min(max(abs(r) / CMAX, 0), 1)
            fc = (1 - t, 1 - t, 1 - t)
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fc,
                                   edgecolor=(0.75, 0.75, 0.75), linewidth=0.5))
            lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
            tc = (1, 1, 1) if lum < 0.55 else (0.15, 0.15, 0.15)
            fw = "bold" if abs(r) > 0.5 else "normal"
            s = "1" if i == j else f"{r:.2f}".replace("0.", ".")
            ax.text(x + 0.5, y + 0.5, s, ha="center", va="center", fontsize=7,
                    color=tc, fontweight=fw)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(anms, rotation=40, ha="right", fontsize=6)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(list(reversed(anms)), fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cmap = plt.get_cmap("gray_r")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=CMAX))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, location="right")
    cb.set_label("r  (all positive)", fontsize=6)
    cb.set_ticks([0, 0.2, 0.4, 0.6, CMAX])
    cb.ax.tick_params(labelsize=6)


def _disc_proj(ax, R, cmean, STY):
    USE_OOF = True
    if USE_OOF and np.any(~np.isnan(R["etaOof"])):
        eta = R["etaOof"].ravel()
    else:
        eta = R["eta_tr"].ravel()

    w = R["wFull"].ravel()
    Wn = w / np.linalg.norm(w)

    Xf = R["Xtr"].copy()
    Xf[np.isnan(Xf)] = 0
    Rr = Xf - np.outer(Xf @ Wn, Wn)
    Rc = Rr - Rr.mean(axis=0)
    _, Sv, Vt = np.linalg.svd(Rc, full_matrices=False)
    v = Vt[0, :]
    yproj = Xf @ v
    varexp = Sv[0] ** 2 / np.sum(Sv ** 2)

    xl = (eta.min() - 0.8, eta.max() + 0.8)
    yl = (yproj.min() - 2.2, yproj.max() + 0.6)
    ax.fill([xl[0], 0, 0, xl[0]], [yl[0], yl[0], yl[1], yl[1]],
            color=gt_ablend(cmean[1], 0.05), edgecolor="none", zorder=0)
    ax.fill([0, xl[1], xl[1], 0], [yl[0], yl[0], yl[1], yl[1]],
            color=gt_ablend(cmean[0], 0.05), edgecolor="none", zorder=0)

    ybot = yl[0] + 0.15
    xs = np.linspace(xl[0], xl[1], 300)
    for g in (1, 0):
        sel = R["ytr"] == g
        d = _kde(eta[sel], xs)
        d = d / d.max() * 1.3
        c = cmean[2 - g - 1]   # g == 1 is WT (row 0), g == 0 is SH (row 1)
        ax.fill(np.concatenate([xs, xs[::-1]]),
                np.concatenate([ybot + d, np.full_like(xs, ybot)]),
                color=gt_ablend(c, 0.22), edgecolor="none", zorder=1)
        ax.plot(xs, ybot + d, "-", color=c, linewidth=0.9, zorder=2)

    ax.plot([0, 0], list(yl), "-", color="k", linewidth=1.3, zorder=3)

    sel = R["ytr"] == 1
    ax.scatter(eta[sel], yproj[sel], s=STY.mk.pt, facecolors="none",
              edgecolors=[cmean[0]], linewidths=0.7, zorder=4)
    sel = R["ytr"] == 0
    ax.scatter(eta[sel], yproj[sel], s=STY.mk.pt, facecolors="none",
              edgecolors=[cmean[1]], linewidths=0.7, zorder=4)

    acc = np.mean((eta > 0) == (R["ytr"].ravel() == 1))
    au = _auroc(R["ytr"], _sigmoid(eta))
    ax.text(xl[0] + 0.15, yl[1] - 0.15,
            f"{eta.size} animals | {acc * 100:.0f}% correct | AUROC {au:.3f}",
            fontsize=STY.font.n, color=(0.25, 0.25, 0.25), va="top")

    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xlabel(r"$\eta$  =  projection onto w")
    ax.set_ylabel(f"orthogonal direction ({varexp * 100:.0f}% resid. var.)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    return au


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    S = load_struct(os.path.join(data_dir, "panel_modelA_fourpanel.mat"))

    X = np.asarray(S.X, dtype=float)
    assays = list(S.assays)
    level = np.ravel(np.asarray(S.level, dtype=float))

    rng = np.random.RandomState(RNG_SEED)   # seeds cross-validation fold assignment
    A = _fit_ridge(X, assays, level, USE_A, 0, THRESH, LAMBDA, NFOLD, rng)

    cmean = np.vstack([STY.cmean, [0.102, 0.647, 0.804]])
    cscat = np.vstack([STY.cscat, [0.612, 0.847, 0.914]])

    BAR_W, SWARM_W, PT_SIZE = STY.geom.bar_w, STY.geom.swarm_w, STY.mk.swarm
    BAR_LW, SE_LW = STY.lw.bar, STY.lw.se

    fig = plt.figure(figsize=(7.2, 6.4), facecolor="w")
    gt_figStyle(fig, STY)

    # a: group separation on the link scale
    ax = fig.add_subplot(2, 2, 1)
    p12, p23, p13 = _eta_group_plot(ax, A, cmean, cscat, BAR_W, SWARM_W, PT_SIZE,
                                    BAR_LW, SE_LW, STY)
    ax.set_ylabel("Classifier score")
    ax.set_title("Group separation on the link scale",
                 fontname=STY.font.name, fontsize=STY.font.title)
    gt_panelLetter(ax, "c", STY)

    # b: construct correlation, lower triangle
    ax = fig.add_subplot(2, 2, 2)
    _corr_tri(ax, A["Zs"], A["anms"])
    ax.set_title("Assay correlation (pairwise complete)",
                 fontname=STY.font.name, fontsize=STY.font.title)

    # c: cross-validated ROC, WT versus SH
    ax = fig.add_subplot(2, 2, 3)
    ax.plot([0, 1], [0, 1], ":", color=(0.6, 0.6, 0.6), linewidth=0.7)
    fa, ta = _roc(A["ytr"], A["oof"])
    ax.fill(np.concatenate([fa, [1, 0]]), np.concatenate([ta, [0, 0]]),
            color=gt_ablend(np.array([0, 0, 0]), 0.07), edgecolor="none")
    ax.plot(fa, ta, "-", color="k", linewidth=1.4)
    jj = int(np.argmax(ta - fa))
    ax.plot(fa[jj], ta[jj], "o", markersize=4, markerfacecolor=(0.15, 0.15, 0.15),
            markeredgecolor="none")
    ax.text(fa[jj] + 0.05, ta[jj] - 0.10, f"sens {ta[jj]:.2f}\nspec {1 - fa[jj]:.2f}",
            fontsize=STY.font.n, color=(0.25, 0.25, 0.25))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"WT vs SH, cross-validated (AUROC {A['auroc']:.3f}, n={A['nTrain']})",
                 fontname=STY.font.name, fontsize=STY.font.title)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # d: discriminant projection
    ax = fig.add_subplot(2, 2, 4)
    _disc_proj(ax, A, cmean, STY)
    ax.set_title("Discriminant projection",
                 fontname=STY.font.name, fontsize=STY.font.title)

    print("panel_modelA_fourpanel:")
    print(f"  n = {A['nTrain']} (WT {int(np.sum(A['ytr'] == 1))} / "
          f"SH {int(np.sum(A['ytr'] == 0))}), cross-validated AUROC = {A['auroc']:.3f}")
    print(f"  eta:  WT-SH p={_pstar(p12)} ({p12:.3g})  "
          f"SH-GTHD p={_pstar(p23)} ({p23:.3g})  WT-GTHD p={_pstar(p13)} ({p13:.3g})")
    print(f"  posterior p(WT): GT-HD vs SH p={A['p_gh_sh']:.3g}  "
          f"GT-HD vs WT p={A['p_gh_wt']:.3g}")
    print("  NOTE: cross-validation folds use a NumPy RNG stream, not MATLAB's, "
          "so exact fold membership (and hence AUROC/eta_oof) will not be "
          "bit-identical to the MATLAB run -- see module docstring.")

    gt_saveFig(fig, "fig1_c_classifier", out_dir)
    return fig


if __name__ == "__main__":
    main()
