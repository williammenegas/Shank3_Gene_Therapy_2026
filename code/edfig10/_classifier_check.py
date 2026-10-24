"""
_classifier_check.py -- console companion to edfig10_d_wt_vs_gt_control.py.
Writes no figure; it is not itself a panel.

Reports the cross-validated separability of the treated group from wild type
across the 16 endpoints of the four domains, using the Figure 6 link-score
pipeline with the training contrast changed from WT against SH to WT against
GT. Everything else is unchanged: the same marm_{beh,soc,cog,fmri}_all.mat
inputs, pooled column z-scoring, L2-penalized logistic regression with
penalty 5, availability-weighted eta, five-fold cross-validation and the
filter requiring at least two observed domains per animal.

The script reports five quantities, each as a median with an interquartile
range over repeats, since the fold assignment is unseeded upstream:

  1. WT against SH, all animals: the contrast used for the Figure 6 link
     score.
  2. WT against GT.
  3. WT against SH with SH downsampled to the size of the treated group: a
     power reference, giving the value the analysis returns for a difference
     of known size at that sample size.
  4. WT against GT with the labels permuted: the chance distribution for this
     design, and an empirical p-value for row 2 against it.
  5. WT split at random against itself: a second chance distribution, built
     from data containing no group difference.

The script takes the condition from the first command line argument, 't' for
test (the default) or 'c' for control.

Run:  python _classifier_check.py [t|c]
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from _f6common import (load_struct, local_zcols, local_nan0, local_ridge,
                       local_kfold, local_etaavail, local_auroc)

DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")
DOMS = ("beh", "soc", "cog", "fmri")
COND = "t"          # 't' = test, 'c' = control
CONDLAB = {"t": "test", "c": "control"}
LAM = 5.0           # ridge penalty
NFOLD = 5
NREP = 200          # cross-validation repeats; fold assignment is unseeded
NPERM = 2000        # label permutations for the null
MIN_DOMAINS = 2     # minimum observed domains per animal


# ---------------------------------------------------------------------------
def load_matrix(cond=COND, min_domains=MIN_DOMAINS):
    """Return X, the animals-by-endpoints matrix with NaN for unobserved
    endpoints, and grp, a group vector coding WT, SH and GT as 1, 2 and 3.

    Assembly matches local_combined_eta, including the removal of all-NaN
    rows and the min_domains coverage filter, but stops before the model is
    fit so that the caller can choose the training contrast.
    """
    gs = ("wt", "sh", "gt")
    Xg, dom_w = {}, []
    for g in gs:
        cols, widths = [], []
        for dom in DOMS:
            M = load_struct(os.path.join(DATA, f"marm_{dom}_all.mat"))
            block = np.atleast_2d(np.asarray(M[f"{dom}_{g}_{cond}"], dtype=float))
            cols.append(block)
            widths.append(block.shape[1])
        Xg[g] = np.hstack(cols)
        if not dom_w:
            dom_w = widths
    for g in gs:
        Xg[g] = Xg[g][~np.all(np.isnan(Xg[g]), axis=1)]

    X = np.vstack([Xg[g] for g in gs])
    grp = np.concatenate([np.full(Xg[g].shape[0], i + 1) for i, g in enumerate(gs)])
    domcol = np.concatenate([np.full(w, di) for di, w in enumerate(dom_w)])

    O = ~np.isnan(X)
    ndom = np.array([np.unique(domcol[O[i]]).size for i in range(X.shape[0])])
    keep = ndom >= min_domains
    return X[keep], grp[keep]


def cv_auroc(X, y, nfold=NFOLD, lam=LAM):
    """Cross-validated AUROC for one ridge-logistic fit.

    y: binary labels, 1 for the positive class. Every animal is scored by a
    model fit without its fold. Returns NaN if any fold leaves the training
    set with a single class or empty."""
    O = ~np.isnan(X)
    Z = local_zcols(X)
    fold = local_kfold(y, nfold)
    oof = np.full(y.shape, np.nan)
    for ff in range(1, nfold + 1):
        trm, te = fold != ff, fold == ff
        if len(np.unique(y[trm])) < 2 or not te.any():
            return np.nan
        w, b = local_ridge(local_nan0(Z[trm]), y[trm], lam)
        oof[te] = local_etaavail(w, b, Z[te], O[te])
    ok = ~np.isnan(oof)
    return local_auroc(y[ok], oof[ok])


def repeat_auroc(X, y, nrep=NREP, seed0=0):
    """Cross-validated AUROC over nrep repeats, each with its own fold
    assignment seeded from seed0. Returns the array of finite values."""
    vals = []
    for r in range(nrep):
        np.random.seed(seed0 + r)
        a = cv_auroc(X, y)
        if not np.isnan(a):
            vals.append(a)
    v = np.asarray(vals)
    return v


def summarize(name, v, extra=""):
    """Print one row of the summary table: median AUROC with its
    interquartile range."""
    if v.size == 0:
        print(f"  {name:<44} n/a")
        return
    print(f"  {name:<44} AUROC {np.median(v):.3f}   "
          f"[IQR {np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f}]{extra}")


# ---------------------------------------------------------------------------
def main(cond=None):
    cond = cond or (sys.argv[1][:1].lower()
                    if len(sys.argv) > 1 else COND)
    X, grp = load_matrix(cond=cond)
    iWT, iSH, iGT = grp == 1, grp == 2, grp == 3
    nWT, nSH, nGT = int(iWT.sum()), int(iSH.sum()), int(iGT.sum())
    print("=" * 74)
    print(f"WT-vs-GT separability check   ({CONDLAB[cond]} condition, "
          ">=2 domains, lambda=5)")
    print("=" * 74)
    print(f"  n: WT {nWT}   SH {nSH}   GT {nGT}    endpoints: {X.shape[1]}")
    print(f"  {NREP} CV repeats per row; unseeded folds, so median [IQR] is shown.\n")

    # 1. the contrast used for the Figure 6 link score
    m = iWT | iSH
    v_shwt = repeat_auroc(X[m], iWT[m].astype(float))
    summarize("1. WT vs SH  (Figure 6 link-score contrast)", v_shwt)

    # 2. WT against GT
    m = iWT | iGT
    Xg, yg = X[m], iWT[m].astype(float)
    v_gtwt = repeat_auroc(Xg, yg)
    summarize("2. WT vs GT", v_gtwt)

    # 3. power reference: SH downsampled to the size of the treated group
    idxWT, idxSH = np.flatnonzero(iWT), np.flatnonzero(iSH)
    vals = []
    for r in range(NREP):
        rng = np.random.RandomState(1000 + r)
        sub = rng.choice(idxSH, size=nGT, replace=False)
        sel = np.concatenate([idxWT, sub])
        y = np.concatenate([np.ones(idxWT.size), np.zeros(nGT)])
        np.random.seed(1000 + r)
        a = cv_auroc(X[sel], y)
        if not np.isnan(a):
            vals.append(a)
    v_down = np.asarray(vals)
    summarize(f"3. WT vs SH, SH downsampled to n={nGT}", v_down)

    # 4. chance distribution: WT against GT with the labels permuted
    null = []
    for r in range(NPERM):
        rng = np.random.RandomState(50000 + r)
        yp = yg[rng.permutation(yg.size)]
        np.random.seed(50000 + r)
        a = cv_auroc(Xg, yp)
        if not np.isnan(a):
            null.append(a)
    null = np.asarray(null)
    obs = np.median(v_gtwt)
    p = (np.sum(null >= obs) + 1) / (null.size + 1)
    summarize("4. WT vs GT, labels permuted (null)", null,
              extra=f"   95th pct {np.percentile(null, 95):.3f}")
    print(f"  {'   -> empirical p for row 2 vs this null':<44} p = {p:.4f}")

    # 5. second chance distribution: WT split at random against itself
    vals = []
    for r in range(NREP):
        rng = np.random.RandomState(9000 + r)
        perm = rng.permutation(idxWT)
        half = idxWT.size // 2
        sel = np.concatenate([perm[:half], perm[half:2 * half]])
        y = np.concatenate([np.ones(half), np.zeros(half)])
        np.random.seed(9000 + r)
        a = cv_auroc(X[sel], y)
        if not np.isnan(a):
            vals.append(a)
    summarize("5. WT vs WT, random split (floor)", np.asarray(vals))

    # summary comparison of rows 2, 3 and 4
    print("\n" + "-" * 74)
    floor = np.percentile(null, 95)
    print(f"  WT vs GT median AUROC ......... {np.median(v_gtwt):.3f}")
    print(f"  null 95th percentile .......... {floor:.3f}")
    print(f"  WT vs SH at the same n ........ {np.median(v_down):.3f}")
    if np.median(v_gtwt) <= floor and np.median(v_down) > floor:
        print("\n  => GT is NOT separable from WT, and the sample size does not")
        print("     explain it: a real difference (SH) IS detected at n = "
              f"{nGT}.")
    elif np.median(v_gtwt) > floor:
        print("\n  => GT is separable from WT above chance: the median AUROC")
        print("     exceeds the 95th percentile of the permutation null.")
    else:
        print("\n  => Underpowered: even the true SH difference is not detected")
        print(f"     at n = {nGT}.")
    print("-" * 74)
    print("\nNOTE: a null result here is still not a formal equivalence test.")
    print("It bounds how large a WT-GT difference could be and remain undetected;")
    print("it does not establish equivalence at any pre-specified margin.")


if __name__ == "__main__":
    main()
