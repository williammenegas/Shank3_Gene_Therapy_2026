"""
fig6_df_linkscores.py

Figure 6, panel d and the upper part of panel f: distributions of the link
score by group, per domain and combined, in the test condition.

Data. marm_beh_all.mat, marm_soc_all.mat, marm_cog_all.mat and
marm_fmri_all.mat, test-condition fields (<domain>_<group>_t).

Link score. Eta is the linear predictor of the ridge-penalized logistic
regression fitted on the WT-versus-SH contrast within each domain (lambda =
5.0, IRLS, intercept unpenalized): eta = b + sum_j w_j * z_ij, where w_j are
the fitted ridge coefficients and z_ij is animal i's pooled z-score on
endpoint j. The ridge penalty makes correlated endpoints share weight rather
than each contributing in full. GT animals are not used to fit the model;
they are scored on the held-out axis by averaging the five cross-validation
fold models. Panel c shows the same models as S curves.

Display. One box per group: median and quartiles, with whiskers running to
the data extremes rather than to 1.5*IQR, since at these group sizes a Tukey
fence would mark ordinary points as outliers. Every animal is drawn as a
jittered point.

The eta scale is per domain, because the ridge weights are fitted separately
in each domain. Behavior and Social are drawn on one shared axis, whose
limits are the union of their own ranges so that nothing is clipped;
Cognition and fMRI keep their own axes. A shared axis is not a shared scale:
eta remains a different quantity in each domain, and equal heights across
domains are not equal biological effects. The script raises an error if any
animal would fall outside its axis.

The rightmost column is a single model fitted on all 16 endpoints, giving one
cross-domain link score per animal, with animals matched across the four
files by row index and animals observed in only one domain excluded
(min_domains=2). It keeps its own axis. Endpoint coverage remains associated
with group, GT animals being more completely measured than WT and SH, so the
combined column compares partly extrapolated WT and SH against more fully
measured GT. Coverage counts are printed to the console.

Point jitter is unseeded, so horizontal offsets of individual animals differ
slightly between runs.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403
from _f6common import (  # noqa: E402
    DOMS, DOMLAB, local_domain_eta, local_combined_eta, local_etalim,
    local_vbox, local_niceTicks,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")

AXGROUPS = [["beh", "soc"], ["cog", "fmri"]]


def main(data_dir=DATA, out_dir=HERE, seed=30):
    STY = gt_init()
    STY.fs_tick = 7
    STY.fs_lab = 7
    FS_TITLE = 8

    # The 5-fold cross-validation split is seeded with the same value as the
    # S-curve panel, so the link scores drawn here match those in panel c.
    # Pass seed=None for an unseeded split.
    if seed is not None:
        np.random.seed(seed)

    CMEAN, CSCAT, CBARF = STY.cmean, STY.cscat, STY.cbarfill

    NCOL = len(DOMS) + 1
    PAN, PAN_GAP, COMB_GAP = 1.10, 0.42, 0.24
    AX_H = PAN
    MARGIN_L, MARGIN_R, MARGIN_B, MARGIN_T = 0.60, 0.10, 0.48, 0.42
    FIG_W = MARGIN_L + NCOL * PAN + (NCOL - 1) * PAN_GAP + COMB_GAP + MARGIN_R
    FIG_H = MARGIN_B + AX_H + MARGIN_T

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="w")

    print("Figure 6d: link scores per domain (test condition)")
    EG = []
    LO = np.full(len(DOMS), np.nan)
    HI = np.full(len(DOMS), np.nan)
    for d, dom in enumerate(DOMS):
        eg, _, _ = local_domain_eta(data_dir, dom, "t")
        EG.append(eg)
        LO[d], HI[d] = local_etalim(eg)

    YTICKS = [None] * len(DOMS)
    for grp in AXGROUPS:
        ix = [DOMS.index(m) for m in grp if m in DOMS]
        if len(ix) < 2:
            continue
        glo, ghi = LO[ix].min(), HI[ix].max()
        own_lo, own_hi = LO[ix].copy(), HI[ix].copy()
        LO[ix] = glo
        HI[ix] = ghi
        tk = local_niceTicks(glo, ghi)
        for m in ix:
            YTICKS[m] = tk
        print(f"  shared axis [{' + '.join(DOMLAB[i] for i in ix)}]: "
              f"ylim [{glo:.2f} {ghi:.2f}], ticks {list(np.round(tk, 3))}")
        for j, m in enumerate(ix):
            print(f"    {DOMLAB[m]:<10} own range would have been [{own_lo[j]:.2f} {own_hi[j]:.2f}]")

    for d in range(len(DOMS)):
        v = np.concatenate([EG[d]["wt"], EG[d]["sh"], EG[d]["gt"]])
        v = v[~np.isnan(v)]
        if v.min() < LO[d] or v.max() > HI[d]:
            raise RuntimeError(
                f"fig6h:clipped {DOMLAB[d]} eta [{v.min():.2f} {v.max():.2f}] falls outside "
                f"its axis [{LO[d]:.2f} {HI[d]:.2f}]. Data would be clipped off the plot.")

    mw = np.full(len(DOMS), np.nan)
    ms = mw.copy()
    mg = mw.copy()
    for d, dom in enumerate(DOMS):
        eg, lo, hi = EG[d], LO[d], HI[d]
        x = MARGIN_L + d * (PAN + PAN_GAP)
        ax = fig.add_axes([x / FIG_W, MARGIN_B / FIG_H, PAN / FIG_W, PAN / FIG_H])
        local_vbox(ax, eg, lo, hi, d == 0, CMEAN, CSCAT, CBARF, STY, YTICKS[d])
        ax.set_title(DOMLAB[d], fontname="Arial", fontsize=FS_TITLE, fontweight="normal")
        if d == 0:
            gt_panelLetter(ax, "d", STY)
        mw[d], ms[d], mg[d] = np.median(eg["wt"]), np.median(eg["sh"]), np.median(eg["gt"])
        print(f"  {DOMLAB[d]:<10} median eta  WT {mw[d]:+5.2f}  SH {ms[d]:+5.2f}  GT {mg[d]:+5.2f}")

    egC, _, covC = local_combined_eta(data_dir, DOMS, "t", min_domains=2)
    loC, hiC = local_etalim(egC)
    xC = MARGIN_L + (NCOL - 1) * (PAN + PAN_GAP) + COMB_GAP
    axC = fig.add_axes([xC / FIG_W, MARGIN_B / FIG_H, PAN / FIG_W, PAN / FIG_H])
    local_vbox(axC, egC, loC, hiC, False, CMEAN, CSCAT, CBARF, STY, None)
    axC.set_title("COMBINED", fontname="Arial", fontsize=FS_TITLE, fontweight="normal")
    mwC, msC, mgC = np.median(egC["wt"]), np.median(egC["sh"]), np.median(egC["gt"])
    print(f"  {'COMBINED':<10} median eta  WT {mwC:+5.2f}  SH {msC:+5.2f}  GT {mgC:+5.2f}")
    print("  COMBINED coverage (endpoints observed per animal, of 16):")
    for g in ("wt", "sh", "gt"):
        c = covC[g]
        print(f"    {g.upper()} n={c['n']} median={c['median_obs']:g} all-16={c['complete']}")
    print("    NOTE: animals covered in only one domain are excluded (>=2-domain")
    print("    filter). GT is still better covered than WT/SH, so the combined")
    print("    column compares partly-extrapolated WT/SH against measured GT.")

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig6_df_linkscores", out_dir)
    return fig


if __name__ == "__main__":
    main()
