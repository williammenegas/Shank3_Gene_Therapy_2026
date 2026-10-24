"""
edfig10_b_male_only_groups.py -- Extended Data Fig. 10, panel b.

The Figure 6 analysis restricted to male animals: WT males, SH males and GT
males across the 16 endpoints of the four domains (behavior, social,
cognition, fMRI), in the test condition.

WHAT IS PLOTTED. Six sub-panels:
  a      radar of the group means, one spoke per endpoint, with a band of one
         standard error either side.
  b,c,d  forests of Hedges' g for SH minus WT, GT minus WT and GT minus SH,
         one row per endpoint, each with a q-value strip alongside.
  e      S-curve of the ridge-logistic link score, titled with the
         cross-validated AUROC for WT males against SH males.
  f      box plots of the link score eta by group.

DATA. data_male/marm_{beh,soc,cog,fmri}_all.mat, written by
_build_subsets.py from the shared per-domain endpoint files; run it first if
the directory is absent. This script does that automatically.

STATISTICS. Two-sided permutation test on the difference of group means,
20000 permutations per endpoint, adjusted by Benjamini-Hochberg across the 16
endpoints of each contrast separately. Effect size is Hedges' g with a
marginal 95% confidence interval. The link score is fit on WT against SH with
GT projected onto the resulting axis; its AUROC is cross-validated over five
folds. The fold assignment is seeded (CV_SEED), so the reported value is
reproducible. Animals observed in fewer than two of the four domains are
excluded (min_domains=2), matching the combined classifier in Figure 6.

OUTPUT. The per-endpoint sample sizes, effect sizes and q values for all
three contrasts are printed and written to
edfig10_b_male_only_groups_stats.csv.
"""

# The male/female and male-only endpoint subsets are written by the builder
# script in this directory; run it so its output is present.
def _run_prereq(sub):
    # Only build when the subset is absent. The .mat files are tracked, so
    # rebuilding on every run would leave the repository dirty after a render.
    import subprocess, sys, os
    d = os.path.dirname(os.path.abspath(__file__))
    need = [f"marm_{dom}_all.mat" for dom in ("beh", "soc", "cog", "fmri")]
    if all(os.path.exists(os.path.join(d, sub, f)) for f in need):
        return
    subprocess.run([sys.executable, "_build_subsets.py"], cwd=d, check=False,
                   capture_output=True)
_run_prereq("data_male")

import os, sys, csv, numpy as np, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import *                                    # noqa
import _f6common as F                                    # noqa
from _f6common import (DOMS, local_build, local_radar, local_forest, local_sigstrip,    # noqa
                       local_contrast, compute_perm_p, local_combined_eta, local_scurve, local_vbox)

DATA = os.path.join(HERE, "data_male")
OUT = central_output_dir(HERE)
NPERM = 20000
CV_SEED = 30          # 5-fold cross-validation split; fixes the reported AUROC
np.random.seed(CV_SEED)
STY = gt_init(); STY.fs_tick=7; STY.fs_title=8; STY.fs_lab=7; STY.fs_star=8; STY.fs_ticksm=6
CMEAN = STY.cmean; CGTWT = STY.clr.gtwt          # WT / SH / GT group colors

S_t = local_build(DATA, DOMS, "t")
fig = plt.figure(figsize=(13.0, 4.6), facecolor="w")

axA = fig.add_axes([0.035, 0.10, 0.225, 0.82])
h = local_radar(axA, S_t, "Male only: test endpoints", CMEAN, STY)
gt_panelLetter(axA, "b", STY)
fig.legend(h, ["WT male","SH male","GT male"], loc="lower center",
           bbox_to_anchor=(0.145,0.0), frameon=False, fontsize=7, ncol=3, handlelength=1.4)

forests = [("shwt","SH - WT","b"), ("gtwt","GT - WT","c"), ("gtsh","GT - SH","d")]
x0 = 0.30; fw = 0.088; strip = 0.014; gap = 0.052
allrows = {}
for k,(which,ttl,letter) in enumerate(forests):
    xf = x0 + k*(fw+strip+gap)
    axf = fig.add_axes([xf, 0.14, fw, 0.74])
    local_forest(axf, S_t, which, ttl, k==0, CMEAN, CGTWT, STY)
    axs = fig.add_axes([xf+fw+0.004, 0.14, strip, 0.74])
    p = compute_perm_p(S_t, which, NPERM); q = gt_bh(p)
    local_sigstrip(axs, S_t, which, q, CMEAN, CGTWT, STY)
    allrows[which] = (p, q)

eg, au, cov = local_combined_eta(DATA, DOMS, "t", min_domains=2)
lo, hi = F.local_etalim(eg)
axE = fig.add_axes([0.755, 0.16, 0.15, 0.72]); local_scurve(axE, eg, f"Link score (AUROC={au:.2f})", True, CMEAN, STY)
axF = fig.add_axes([0.935, 0.16, 0.06, 0.72])
local_vbox(axF, eg, lo, hi, True, CMEAN, [gt_ablend(CMEAN[i],0.75) for i in range(3)],
           [gt_ablend(CMEAN[i],0.25) for i in range(3)], STY)
gt_figStyle(fig, STY); gt_saveFig(fig, "edfig10_b_male_only_groups", OUT)

print("\n=== Males only: per-endpoint Hedges' g, perm p, BH q ===")
hdr=["endpoint","nWT","nSH","nGT"]+[f"{c}_{s}" for c in ["shwt","gtwt","gtsh"] for s in ("g","p","q")]
rows=[]
for i,ep in enumerate(S_t):
    nWT=np.sum(~np.isnan(ep.wt)); nSH=np.sum(~np.isnan(ep.sh)); nGT=np.sum(~np.isnan(ep.gt))
    row=[ep.name,int(nWT),int(nSH),int(nGT)]; line=f"  {ep.name:<6} n={nWT}/{nSH}/{nGT} "
    for which in ["shwt","gtwt","gtsh"]:
        g,_,ci=local_contrast(ep,which); p,q=allrows[which]
        row+=[round(g,3),round(float(p[i]),4),round(float(q[i]),4)]
        line+=f" | {which}: g={g:+.2f} q={q[i]:.3g}"
    print(line); rows.append(row)
print(f"\n  combined ridge AUROC (WT-male vs SH-male) = {au:.3f}   [GT-male projected]")
print(f"  coverage: WT n={cov['wt']['n']}, SH n={cov['sh']['n']}, GT n={cov['gt']['n']}")
with open(os.path.join(OUT,"edfig10_b_male_only_groups_stats.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(hdr); w.writerows(rows)
    w.writerow([]); w.writerow(["ridge_AUROC_WTmale_vs_SHmale",round(au,4)])
print("saved", OUT)
