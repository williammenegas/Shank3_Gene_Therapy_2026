"""
edfig10_d_wt_vs_gt_control.py -- Extended Data Fig. 10, panel d.

Separability of the treated group from wild type across the 16 endpoints of
the four domains (behavior, social, cognition, fMRI). Companion to
_classifier_check.py in this directory, which reports the same quantities to
the console over the same number of cross-validation repeats.

WHAT IS PLOTTED. Four sub-panels:
  a  radar of the group means for WT, SH and GT, one spoke per endpoint, with
     a band of one standard error either side.
  b  forest of Hedges' g for GT minus WT, one row per endpoint, with a
     q-value strip alongside.
  c  S-curve of a link score retrained on WT against GT rather than on WT
     against SH, showing each animal at its cross-validated eta.
  d  cross-validated AUROC by contrast: the median over cross-validation
     repeats with an interquartile whisker, over a shuffled-label null band.
     Three contrasts are shown: WT against SH using all SH animals, WT
     against SH with SH downsampled to the size of the treated group, and WT
     against GT. The downsampled contrast is a power reference, showing what
     the analysis returns for a difference of known size at the treated
     group's sample size.

UNIT OF ANALYSIS. Sub-panels a to c plot one point per animal. Sub-panel d
plots one point per contrast, because an AUROC is a property of the cohort
rather than of an animal.

DATA. marm_{beh,soc,cog,fmri}_all.mat, the per-domain endpoint files shared
by the Figure 6 panel family. The script takes the condition from the first
command line argument, 't' for test (the default) or 'c' for control; the
control output filenames carry a _control suffix.

STATISTICS. The link score is the linear predictor of an L2-penalized
logistic regression (penalty 5) on pooled column z-scores, with
availability weighting for animals not measured on every endpoint and the
Figure 6 filter requiring at least two observed domains. Only the training
contrast differs between sub-panels. Effect sizes in sub-panel b are Hedges'
g with a marginal 95% confidence interval; their p-values are two-sided
permutation tests over 20000 permutations, adjusted by Benjamini-Hochberg
across the 16 endpoints. Each AUROC is summarized over 200 cross-validation
repeats, and each null band over 2000 label permutations.

OUTPUT. The script prints the AUROC medians, interquartile ranges and null
bounds, and writes them to edfig10_d_wt_vs_gt_control_stats.csv.

Run:  python edfig10_d_wt_vs_gt_control.py [t|c]
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from gtstyle import *                                                # noqa
import _f6common as F                                                # noqa
from _f6common import (DOMS, local_build, local_radar, local_forest,  # noqa
                       local_sigstrip, compute_perm_p, local_scurve,
                       local_zcols, local_nan0, local_ridge, local_kfold,
                       local_etaavail, local_auroc, local_etalim)
from _classifier_check import load_matrix, cv_auroc            # noqa

DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")
OUT = central_output_dir(HERE)
NPERM = 20000
COND = (sys.argv[1][:1].lower() if len(sys.argv) > 1 else "t")
CONDLAB = {"t": "test", "c": "control"}[COND]
SUF = "" if COND == "t" else "_control"
NREP = 200
NPERM_AU = 2000

STY = gt_init()
STY.fs_tick = 7; STY.fs_title = 8; STY.fs_lab = 7
STY.fs_star = 8; STY.fs_ticksm = 6
CMEAN = STY.cmean
CGTWT = STY.clr.gtwt


# ---------------------------------------------------------------------------
def eta_wt_vs_gt(X, grp, seed=0):
    """Refit the combined link score with GT rather than SH as the training
    contrast, and return the out-of-fold eta per animal.

    Every animal is scored by a model fit without it: an in-sample eta from a
    model fit on the same animals is separated by construction. The folds are
    seeded here so that the drawn panel is reproducible; the AUROC in its
    title is the median over repeats computed for sub-panel d, not this
    single draw.

    Returns ({'wt': ..., 'sh': empty, 'gt': ...}, auroc) for this draw.
    """
    iWT, iGT = grp == 1, grp == 3
    Z = local_zcols(X)
    O = ~np.isnan(X)
    tr = (iWT | iGT) & O.any(axis=1)
    ytr = iWT[tr].astype(float)
    Ztr, Otr = Z[tr], O[tr]

    np.random.seed(seed)
    fold = local_kfold(ytr, 5)
    oof = np.full(ytr.shape, np.nan)
    for ff in range(1, 6):
        trm, te = fold != ff, fold == ff
        wf, bf = local_ridge(local_nan0(Ztr[trm]), ytr[trm], 5.0)
        oof[te] = local_etaavail(wf, bf, Ztr[te], Otr[te])

    eta = np.full(grp.shape, np.nan)
    eta[tr] = oof
    return ({"wt": eta[iWT], "sh": np.array([]), "gt": eta[iGT]},
            local_auroc(ytr, oof))


def auroc_repeats(X, grp):
    """Compute the AUROC distributions summarized in sub-panel d.

    Returns a dict of arrays: 'shwt' and 'gtwt' over NREP cross-validation
    repeats; 'shwt_down' over NREP repeats, each drawing a fresh SH subsample
    of the size of the treated group; and one shuffled-label null per
    contrast ('null_shwt', 'null_shwt_down', 'null_gtwt') over NPERM_AU label
    permutations. 'null' is the WT-versus-GT null, of which 'null_gtwt' is an
    alias."""
    iWT, iSH, iGT = grp == 1, grp == 2, grp == 3
    idxWT, idxSH = np.flatnonzero(iWT), np.flatnonzero(iSH)
    nGT = int(iGT.sum())
    out = {}

    m = iWT | iSH
    out["shwt"] = np.array([_rep(X[m], iWT[m].astype(float), s)
                            for s in range(NREP)])
    m = iWT | iGT
    Xg, yg = X[m], iWT[m].astype(float)
    out["gtwt"] = np.array([_rep(Xg, yg, s) for s in range(NREP)])

    vals = []
    for r in range(NREP):
        rng = np.random.RandomState(1000 + r)
        sel = np.concatenate([idxWT, rng.choice(idxSH, nGT, replace=False)])
        y = np.concatenate([np.ones(idxWT.size), np.zeros(nGT)])
        vals.append(_rep(X[sel], y, 1000 + r))
    out["shwt_down"] = np.array(vals)

    vals = []
    for r in range(NPERM_AU):
        rng = np.random.RandomState(50000 + r)
        vals.append(_rep(Xg, yg[rng.permutation(yg.size)], 50000 + r))
    out["null"] = np.array(vals)

    # One null per contrast. The width of a permutation null depends on the
    # group sizes, so the WT-versus-GT null is not a valid reference for the
    # WT-versus-SH contrasts, which have different designs. Each observed
    # point is therefore read against a null built from its own design.
    m = iWT | iSH
    Xs, ys = X[m], iWT[m].astype(float)
    out["null_shwt"] = np.array(
        [_rep(Xs, ys[np.random.RandomState(60000 + r).permutation(ys.size)],
              60000 + r) for r in range(NPERM_AU)])
    vals = []
    for r in range(NPERM_AU):
        rng = np.random.RandomState(64000 + r)
        sel = np.concatenate([idxWT, rng.choice(idxSH, nGT, replace=False)])
        y = np.concatenate([np.ones(idxWT.size), np.zeros(nGT)])
        vals.append(_rep(X[sel], y[rng.permutation(y.size)], 64000 + r))
    out["null_shwt_down"] = np.array(vals)
    out["null_gtwt"] = out["null"]
    return out


def _rep(X, y, seed):
    """One cross-validated AUROC at a given seed for the fold assignment."""
    np.random.seed(seed)
    return cv_auroc(X, y)


def panel_auroc(ax, A, nGT, STY):
    """Draw sub-panel d: AUROC by contrast, as a median with an
    interquartile whisker over a per-contrast shuffled-label null band.

    Each contrast is shown as a single summary rather than a scatter of
    repeats, because an AUROC is a property of the cohort and not of an
    animal. The spread across repeats is also not the same quantity in every
    column: for the observed contrasts it comes from the fold assignment
    alone, whereas for the shuffled null it comes from the fold assignment
    and the label permutation together. The null is therefore drawn as a
    reference interval, the 5th to 95th percentile of its distribution.

    The whiskers are the interquartile range over cross-validation repeats,
    that is, the stability of the estimate under refolding. They are not
    confidence intervals on the AUROC.
    """
    keys = ["shwt", "shwt_down", "gtwt"]
    labs = ["WT vs SH\n(all)", f"WT vs SH\n(n={nGT})", "WT vs GT"]
    cols = [CMEAN[1], CMEAN[1], CMEAN[2]]

    # one null band per contrast; see auroc_repeats
    for i, k in enumerate(keys):
        nl = A[f"null_{k}"]
        nl = nl[~np.isnan(nl)]
        lo, hi = np.percentile(nl, [5, 95])
        ax.add_patch(plt.Rectangle((i + 0.62, lo), 0.76, hi - lo,
                                    color=gt_ablend(STY.clr.gray_mid, 0.16),
                                    zorder=0, lw=0))
    ax.axhline(0.5, color=STY.clr.gray_mid, lw=STY.lw.hair,
               zorder=1, ls=(0, (3, 2)))

    for i, k in enumerate(keys):
        v = A[k][~np.isnan(A[k])]
        med = float(np.median(v))
        q1, q3 = np.percentile(v, [25, 75])
        ax.plot([i + 1, i + 1], [q1, q3], "-", color=cols[i],
                lw=1.2, zorder=3, solid_capstyle="round")
        ax.plot(i + 1, med, "o", color=cols[i], ms=5.2,
                markeredgecolor="w", markeredgewidth=0.5, zorder=4)
        dy = -0.085 if med < 0.6 else 0.05      # keep the label clear of the band
        ax.text(i + 1, med + dy, f"{med:.2f}", ha="center",
                fontname=STY.font.name, fontsize=STY.fs_ticksm,
                color=cols[i], zorder=5)

    ax.set_xlim(0.45, len(keys) + 0.55)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels(labs, fontname=STY.font.name, fontsize=STY.fs_ticksm)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("cross-validated AUROC", fontname=STY.font.name,
                  fontsize=STY.fs_lab)
    ax.set_title(f"By contrast ({CONDLAB})", fontname=STY.font.name,
                 fontsize=STY.fs_title, fontweight="normal")
    ax.text(0.52, 1.005, "gray: shuffled-label null (5-95%), per contrast",
            ha="left", va="bottom", fontname=STY.font.name,
            fontsize=STY.fs_ticksm - 0.5, color=STY.clr.gray_mid)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
def main():
    S_t = local_build(DATA, DOMS, COND)
    X, grp = load_matrix(cond=COND)
    nGT = int(np.sum(grp == 3))
    eg_gt, au_gt = eta_wt_vs_gt(X, grp)
    A = auroc_repeats(X, grp)

    fig = plt.figure(figsize=(13.0, 3.6), facecolor="w")

    # a: radar of the group means
    axA = fig.add_axes([0.035, 0.13, 0.20, 0.74])
    h = local_radar(axA, S_t, f"All assays ({CONDLAB})", CMEAN, STY)
    gt_panelLetter(axA, "d", STY)
    fig.legend(h, ["WT", "SH", "GT"], loc="lower center",
               bbox_to_anchor=(0.135, -0.01), frameon=False, fontsize=7,
               ncol=3, handlelength=1.4)

    # b: GT minus WT forest with its q strip
    axB = fig.add_axes([0.295, 0.20, 0.135, 0.66])
    local_forest(axB, S_t, "gtwt", "GT - WT", True, CMEAN, CGTWT, STY)
    axS = fig.add_axes([0.434, 0.20, 0.016, 0.66])
    p = compute_perm_p(S_t, "gtwt", NPERM)
    q = gt_bh(p)
    local_sigstrip(axS, S_t, "gtwt", q, CMEAN, CGTWT, STY)

    # c: S-curve of the WT-versus-GT-trained link score, at out-of-fold eta
    au_med = np.median(A["gtwt"][~np.isnan(A["gtwt"])])
    axC = fig.add_axes([0.545, 0.20, 0.165, 0.66])
    local_scurve(axC, eg_gt,
                 f"Retrained WT vs GT (AUROC={au_med:.2f})", True, CMEAN, STY)

    # d: AUROC by contrast
    axD = fig.add_axes([0.795, 0.20, 0.175, 0.66])
    panel_auroc(axD, A, nGT, STY)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig10_d_wt_vs_gt_control" + SUF, OUT)

    # console summary and CSV
    print("\n=== WT vs GT separability ===")
    print(f"  n: WT {int(np.sum(grp==1))}  SH {int(np.sum(grp==2))}  GT {nGT}")
    rows = []
    for k, lab in [("shwt", "WT vs SH (all)"),
                   ("shwt_down", f"WT vs SH (SH n={nGT})"),
                   ("gtwt", "WT vs GT"),
                   ("null", "WT vs GT, shuffled")]:
        v = A[k][~np.isnan(A[k])]
        print(f"  {lab:<26} AUROC {np.median(v):.3f} "
              f"[{np.percentile(v,25):.3f}-{np.percentile(v,75):.3f}]")
        nk = {"WT vs SH (all)": "null_shwt",
              f"WT vs SH (SH n={nGT})": "null_shwt_down",
              "WT vs GT": "null_gtwt"}.get(lab)
        if nk:
            nn = A[nk][~np.isnan(A[nk])]
            n5, n95 = np.percentile(nn, [5, 95])
            print(f"      null 5-95% for this contrast: {n5:.3f}-{n95:.3f}"
                  f"   {'ABOVE null' if np.median(v) > n95 else 'within null'}")
        else:
            n5 = n95 = float("nan")
        rows.append([lab, round(float(np.median(v)), 4),
                     round(float(np.percentile(v, 25)), 4),
                     round(float(np.percentile(v, 75)), 4), int(v.size),
                     round(float(n5), 4), round(float(n95), 4)])
    obs = np.median(A["gtwt"][~np.isnan(A["gtwt"])])
    nl = A["null"][~np.isnan(A["null"])]
    pv = (np.sum(nl >= obs) + 1) / (nl.size + 1)
    print(f"  empirical p (WT vs GT vs shuffled null) = {pv:.4f}")
    print(f"  single-fit retrained AUROC shown in panel c = {au_gt:.3f}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "edfig10_d_wt_vs_gt_control_stats" + SUF + ".csv"),
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["contrast", "AUROC_median", "IQR_lo", "IQR_hi",
                    "n_repeats", "null_p5", "null_p95"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["empirical_p_WTvsGT_vs_null", round(float(pv), 5)])
    print("saved", OUT)


if __name__ == "__main__":
    main()
