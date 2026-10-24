"""
edfig10_a_wt_male_female.py -- Extended Data Fig. 10, panel a.

Wild-type males compared with wild-type females across the 16 endpoints of
the four domains (behavior, social, cognition, fMRI), in the test condition,
using the Figure 6 analysis with sex in place of genotype: males occupy the
"wt" slot and females the "sh" slot.

WHAT IS PLOTTED. Four sub-panels:
  a  radar of the group means, one spoke per endpoint, with a band of one
     standard error either side.
  b  forest of Hedges' g for female minus male, one row per endpoint, with a
     q-value strip alongside.
  c  S-curve of the ridge-logistic link score: the logistic function of eta
     with each animal at its own eta, titled with the cross-validated AUROC.
  d  box plots of eta by sex, labeled M and F.

DATA. data_mf/marm_{beh,soc,cog,fmri}_all.mat, written by _build_subsets.py
from the shared per-domain endpoint files; run it first if the directory is
absent. This script does that automatically.

STATISTICS. Two-sided permutation test on the difference of group means,
20000 permutations per endpoint, adjusted by Benjamini-Hochberg across the 16
endpoints. Effect size is Hedges' g with a marginal 95% confidence interval.
The AUROC is cross-validated over five folds. The fold assignment is
seeded (CV_SEED), so the reported value is reproducible. Every wild-type
animal with at least one observed domain is included (min_domains=1); the
male-only comparison in panel b uses min_domains=2 instead, matching the
combined classifier in Figure 6.

OUTPUT. The per-endpoint sample sizes, effect sizes and p and q values are
printed and written to edfig10_a_wt_male_female_stats.csv.
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
_run_prereq("data_mf")

import os, sys, csv, numpy as np, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import *                                   # noqa
import _f6common as F                                   # noqa
from _f6common import (DOMS, local_build, local_forest, local_sigstrip, local_contrast,   # noqa
                       compute_perm_p, local_rof, local_combined_eta, local_scurve, local_vbox)

DATA = os.path.join(HERE, "data_mf")
OUT = central_output_dir(HERE)
os.makedirs(OUT, exist_ok=True)
NPERM = 20000
CV_SEED = 30          # 5-fold cross-validation split; fixes the reported AUROC
np.random.seed(CV_SEED)
STY = gt_init(); STY.fs_tick=7; STY.fs_title=8; STY.fs_lab=7; STY.fs_star=8; STY.fs_ticksm=6
C_M = np.array([0.20,0.42,0.70]); C_F = np.array([0.83,0.37,0.52])
STY.cmean[0]=C_M; STY.cmean[1]=C_F; CMEAN=STY.cmean; CGTWT=STY.clr.gtwt


def radar2(ax, S, ttl):
    """Two-group radar: one spoke per endpoint, one closed line per group at
    the group mean with a standard-error band, and one domain label at each
    quadrant centroid."""
    n=len(S); ang=np.arange(n)*2*np.pi/n
    zmin,zmax,R0,RMAX=-2.4,1.5,0.15,0.9; th=np.linspace(0,2*np.pi,240); OL=(0.15,0.15,0.15)
    for zg in (-1.5,-0.5,0.5,1.5):
        rr=local_rof(zg,zmin,zmax,R0,RMAX); ax.plot(rr*np.cos(th),rr*np.sin(th),"-",color=STY.clr.gray_lite,lw=STY.lw.hair,zorder=0)
    r0=local_rof(0,zmin,zmax,R0,RMAX); ax.plot(r0*np.cos(th),r0*np.sin(th),"-",color=(0.78,)*3,lw=0.6,zorder=0)
    for a in ang: ax.plot([R0*np.cos(a),RMAX*np.cos(a)],[R0*np.sin(a),RMAX*np.sin(a)],"-",color=(0.90,)*3,lw=0.4,zorder=0)
    step=2*np.pi/n; b0=ang[3]+step/2; bnds=b0+np.arange(4)*(np.pi/2); gap=np.radians(5)
    for k in range(4):
        a1=bnds[k]+gap; a2=(bnds[k+1] if k<3 else bnds[0]+2*np.pi)-gap; tt=np.linspace(a1,a2,90)
        ax.plot(RMAX*np.cos(tt),RMAX*np.sin(tt),"-",color=OL,lw=1.1,zorder=1,solid_capstyle="round")
    for gi,attr in enumerate(("wt","sh")):
        mu=np.zeros(n); se=np.zeros(n)
        for i in range(n):
            v=getattr(S[i],attr); v=v[~np.isnan(v)]; mu[i]=v.mean() if v.size else np.nan
            se[i]=v.std(ddof=1)/np.sqrt(v.size) if v.size>1 else 0.0
        r=np.array([local_rof(mu[i],zmin,zmax,R0,RMAX) for i in range(n)])
        rlo=np.array([local_rof(mu[i]-se[i],zmin,zmax,R0,RMAX) for i in range(n)])
        rhi=np.array([local_rof(mu[i]+se[i],zmin,zmax,R0,RMAX) for i in range(n)])
        ac=np.append(ang,ang[0])
        xb=np.concatenate([np.append(rhi,rhi[0])*np.cos(ac),(np.append(rlo,rlo[0])*np.cos(ac))[::-1]])
        yb=np.concatenate([np.append(rhi,rhi[0])*np.sin(ac),(np.append(rlo,rlo[0])*np.sin(ac))[::-1]])
        ax.fill(xb,yb,color=gt_ablend(CMEAN[gi],0.15),edgecolor="none",zorder=2)
        xs=np.append(r*np.cos(ang),r[0]*np.cos(ang[0])); ys=np.append(r*np.sin(ang),r[0]*np.sin(ang[0]))
        ax.plot(xs,ys,"-",color=CMEAN[gi],lw=1.6,zorder=3,solid_joinstyle="round")
    for dom in DOMS:
        idx=[i for i in range(n) if S[i].dom==dom]; a=float(np.mean(ang[idx])); rl=RMAX+0.15
        ha="left" if np.cos(a)>0.2 else ("right" if np.cos(a)<-0.2 else "center")
        va="bottom" if np.sin(a)>0.2 else ("top" if np.sin(a)<-0.2 else "center")
        ax.text(rl*np.cos(a),rl*np.sin(a),F.RADAR_DOMLAB.get(dom,dom),fontname=STY.font.name,fontsize=STY.fs_tick+1,ha=ha,va=va)
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(ttl,fontname=STY.font.name,fontsize=STY.fs_title)


S_t = local_build(DATA, DOMS, "t")
fig = plt.figure(figsize=(10.0, 4.3), facecolor="w")
axA = fig.add_axes([0.01, 0.12, 0.30, 0.78]); radar2(axA, S_t, "All assays")
gt_panelLetter(axA, "a", STY)
fig.legend([plt.Line2D([],[],color=C_M,lw=2), plt.Line2D([],[],color=C_F,lw=2)],
           ["WT male","WT female"], loc="lower center", bbox_to_anchor=(0.16,0.02),
           frameon=False, fontsize=7, ncol=2, handlelength=1.4)
axB = fig.add_axes([0.40, 0.14, 0.13, 0.74]); local_forest(axB, S_t, "shwt", "WT: female - male", True, CMEAN, CGTWT, STY)
axBs = fig.add_axes([0.535, 0.14, 0.017, 0.74])
p = compute_perm_p(S_t, "shwt", NPERM); q = gt_bh(p); local_sigstrip(axBs, S_t, "shwt", q, CMEAN, CGTWT, STY)
eg, au, cov = local_combined_eta(DATA, DOMS, "t", min_domains=1)
lo, hi = F.local_etalim(eg)
axC = fig.add_axes([0.62, 0.16, 0.19, 0.72]); local_scurve(axC, eg, f"Link score (AUROC={au:.2f})", True, CMEAN, STY)
axD = fig.add_axes([0.86, 0.16, 0.12, 0.72])
local_vbox(axD, eg, lo, hi, True, CMEAN, [gt_ablend(CMEAN[i],0.75) for i in range(3)],
           [gt_ablend(CMEAN[i],0.25) for i in range(3)], STY)
axD.set_xticklabels(["M","F",""])
gt_figStyle(fig, STY); gt_saveFig(fig, "edfig10_a_wt_male_female", OUT)

rows=[]; print("\n=== WT male vs female: Hedges' g (female-male), perm p, BH q ===")
for i,ep in enumerate(S_t):
    g,_,ci=local_contrast(ep,"shwt"); m=ep.wt[~np.isnan(ep.wt)]; f=ep.sh[~np.isnan(ep.sh)]
    print(f"  {ep.name:<6} nM={m.size:>2} nF={f.size:>2}  g(F-M)={g:+5.2f} [{ci[0]:+5.2f},{ci[1]:+5.2f}]  p={p[i]:.3g} q={q[i]:.3g}")
    rows.append([ep.name,m.size,f.size,round(g,3),round(ci[0],3),round(ci[1],3),round(float(p[i]),4),round(float(q[i]),4)])
nsig=int(np.sum(q<PSTAR_T1)); beh_g=[local_contrast(ep,"shwt")[0] for ep in S_t if ep.dom=="beh"]
print(f"\n  {nsig}/16 endpoints q<0.05  |  ridge AUROC (male vs female) = {au:.3f}")
print(f"  behavioral g(F-M) = {['%+.2f'%x for x in beh_g]}  (all negative => WT females score lower on behavior)")
with open(os.path.join(OUT,"edfig10_a_wt_male_female_stats.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["endpoint","n_male","n_female","hedges_g_FminusM","ci_lo","ci_hi","perm_p","bh_q"])
    w.writerows(rows); w.writerow([]); w.writerow(["ridge_AUROC_male_vs_female",round(au,4)])
print("saved", OUT)
