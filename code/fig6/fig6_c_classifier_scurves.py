"""
fig6_c_classifier_scurves.py

Figure 6, panel c: probability of being WT-like as a function of the link
score, per domain and combined, in the test condition.

Data. marm_beh_all.mat, marm_soc_all.mat, marm_cog_all.mat and
marm_fmri_all.mat, test-condition fields (<domain>_<group>_t).

Method. Within each domain the four endpoints are pooled z-scored across all
three groups, which preserves group separation, and a ridge-penalized
logistic regression is fitted to the WT-versus-SH contrast (lambda = 5.0,
fitted by IRLS, intercept unpenalized). There is no separate step to collapse
redundant endpoints: the ridge penalty makes correlated endpoints share
weight instead of each contributing in full. The endpoint correlation
structure itself is shown in panel f.

The link score eta is the linear predictor of that model,
eta = b + sum_j w_j * z_ij, where w_j are the fitted ridge coefficients and
z_ij is animal i's pooled z-score on endpoint j. Animals with missing
endpoints are rescaled by the total absolute weight of the endpoints they do
have.

GT animals are not used to fit the model. They are scored on the held-out
WT-versus-SH axis by averaging the predictions of the five cross-validation
fold models. The AUROC in each title is the out-of-fold AUROC of the
WT-versus-SH classifier under the same 5-fold split.

Display. The curve is the logistic map eta -> P(WT-like) = 1/(1 + exp(-eta)),
with eta = 0 (P = 0.5) the decision boundary. Each point is one animal and
lies on the curve by construction, so the curve acts as a ruler and the
information in the panel is the horizontal position of each group. The eta
scale is per domain, because the ridge weights are fitted separately in each
domain, and the x limits therefore differ between columns; the y axis is a
probability, shared and labeled once, on the left.

The rightmost column is a single model fitted on all 16 endpoints at once, so
that each animal has one cross-domain link score. Animals are matched across
the four files by row index. Animals observed in only one domain are excluded
(min_domains=2). Endpoint coverage remains associated with group, GT animals
being more completely measured than WT and SH, so the combined column
compares partly extrapolated WT and SH against more fully measured GT; the
per-domain columns are not affected. The script prints coverage counts to
the console.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403
from _f6common import (  # noqa: E402
    DOMS, DOMLAB, local_domain_eta, local_combined_eta, local_scurve,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")


def main(data_dir=DATA, out_dir=HERE, seed=30):
    STY = gt_init()
    STY.fs_tick = 7
    STY.fs_lab = 7
    STY.fs_title = 8

    # The 5-fold cross-validation split is seeded so that the reported AUROC
    # values and the combined link scores are reproducible and agree with the
    # link-score panels. Pass seed=None for an unseeded split.
    if seed is not None:
        np.random.seed(seed)

    CMEAN = STY.cmean

    NCOL = len(DOMS) + 1
    PAN, PAN_GAP, COMB_GAP = 1.10, 0.34, 0.24
    AX_H = PAN
    MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.60, 0.12, 0.62, 0.42
    FIG_W = MARGIN_L + NCOL * PAN + (NCOL - 1) * PAN_GAP + COMB_GAP + MARGIN_R
    FIG_H = MARGIN_B + AX_H + MARGIN_T

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="w")

    print("Figure 6c: link-score S curves (test condition)")
    au = np.full(NCOL, np.nan)
    for d, dom in enumerate(DOMS):
        eg, au[d], _ = local_domain_eta(data_dir, dom, "t")
        x = MARGIN_L + d * (PAN + PAN_GAP)
        ax = fig.add_axes([x / FIG_W, MARGIN_B / FIG_H, PAN / FIG_W, PAN / FIG_H])
        local_scurve(ax, eg, f"Test {DOMLAB[d]}\nAUROC {au[d]:.2f}", d == 0, CMEAN, STY)
        if d == 0:
            gt_panelLetter(ax, "c", STY)
        print(f"  {DOMLAB[d]:<10} AUROC {au[d]:.2f}   n = {eg['wt'].size}/{eg['sh'].size}/{eg['gt'].size}")

    egC, auC, covC = local_combined_eta(data_dir, DOMS, "t", min_domains=2)
    au[NCOL - 1] = auC
    xC = MARGIN_L + (NCOL - 1) * (PAN + PAN_GAP) + COMB_GAP
    axC = fig.add_axes([xC / FIG_W, MARGIN_B / FIG_H, PAN / FIG_W, PAN / FIG_H])
    local_scurve(axC, egC, f"Test COMBINED\nAUROC {auC:.2f}", False, CMEAN, STY)
    print(f"  {'COMBINED':<10} AUROC {auC:.2f}   n = {egC['wt'].size}/{egC['sh'].size}/{egC['gt'].size}"
          f"   (>=2 domains; excluded 1-domain: "
          f"WT {covC['wt']['excluded']}, SH {covC['sh']['excluded']}, GT {covC['gt']['excluded']})")
    print("  COMBINED coverage (endpoints observed per animal, of 16):")
    for g in ("wt", "sh", "gt"):
        c = covC[g]
        print(f"    {g.upper()} n={c['n']} median={c['median_obs']:g} all-16={c['complete']}")
    print("    NOTE: animals covered in only one domain are excluded (>=2-domain")
    print("    filter). GT is still better covered than WT/SH, so the combined")
    print("    column compares partly-extrapolated WT/SH against measured GT.")

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig6_c_classifier_scurves", out_dir)
    return fig


if __name__ == "__main__":
    main()
