"""
edfig10_c_founder_lines.py -- Extended Data Fig. 10, panel c.

Wild-type-versus-mutant ridge link score by domain, with mutant and treated
animals colored by founder line: Founder-K (43 bp deletion, Thr1305fs) drawn
as open markers and Founder-S (single-G insertion, Arg1320fs) as filled
markers. Founder line is known for every mutant and treated animal, and the
script prints the composition of each group by founder line to stdout.

WHAT IS PLOTTED. Five box panels of the per-animal link score eta: one for
each of the four domains (behavior, social, cognition, fMRI) and one for the
combined score over all endpoints. Each panel shows WT, SH and GT as a box
from the first to the third quartile with a median rule, whiskers to the
extremes, and the individual animals as jittered points. Points are open with
a colored edge for Founder-K animals and filled for Founder-S animals; WT
animals carry no founder line and are drawn filled.

The link score is the linear predictor of a ridge-logistic model fit to
separate WT from SH, with the GT animals projected onto the resulting axis.
The per-domain models are fit on that domain's endpoints alone; the combined
model is fit on all endpoints together. Behavior and social share one y
range and cognition and fMRI share another, so panels on a shared range carry
identical ticks.

DATA. marm_{beh,soc,cog,fmri}_all.mat, the per-domain endpoint files shared
by the Figure 6 panel family, in the test condition. Founder line is joined
from animal_lines.csv, which gives the founder line of each animal that
carries the mutant allele against its unique identifier, already resolved
from the founder worksheets; the identifier indexes the rows of the
endpoint arrays.

STATISTICS. None. This panel is a descriptive display of the distribution of
the link score with founder line marked.
"""

import os, sys, csv, numpy as np, scipy.io as sio, matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import *  # noqa
import _f6common as F
from _f6common import DOMS, Endpoint, local_forest, local_contrast, compute_perm_p, local_rof, RADAR_DOMLAB

SRC = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")
OWN = os.path.join(HERE, "..", "..", "data", "edfig10")
OUT = central_output_dir(HERE)
NPERM=20000

# Founder line per animal, keyed by unique identifier, which is the row index
# (plus one) into the endpoint arrays. animal_lines.csv carries the line
# already resolved, so no colony identifier is needed here.
uid2line={}
with open(os.path.join(OWN,"animal_lines.csv"),newline="",encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        uid2line[int(row["Unique_ID"])]=row["Founder_line"] or None

def group_mat(g):
    """All endpoints of one group as a single animals-by-endpoints matrix,
    with the row indices of the animals actually present."""
    cols=[np.asarray(sio.loadmat(os.path.join(SRC,f"marm_{dom}_all.mat"))[f"{dom}_{g}_t"],float) for dom in DOMS]
    X=np.hstack(cols); pres=np.where(~np.all(np.isnan(X),axis=1))[0]
    return X, pres

WT,WTp=group_mat("wt"); SH,SHp=group_mat("sh"); GT,GTp=group_mat("gt")
sh_line=np.array([uid2line.get(i+1) for i in SHp]); gt_line=np.array([uid2line.get(i+1) for i in GTp])
print("SH:",{l:int((sh_line==l).sum()) for l in ["Kelle","Spock",None]})
print("GT:",{l:int((gt_line==l).sum()) for l in ["Kelle","Spock",None]})

STY=gt_init(); STY.fs_tick=7; STY.fs_title=8; STY.fs_lab=7; STY.fs_star=7.5
C_K=np.array([0.85,0.45,0.10]); C_S=np.array([0.25,0.50,0.72])   # founder-line hues
CMEAN=STY.cmean  # group colors: WT dark gray, SH red, GT blue

# Endpoint list holding the WT reference in the wt slot and the two founder
# lines of the mutant group in the sh and gt slots.
S=[]
kmask=SHp[sh_line=="Kelle"]; smask=SHp[sh_line=="Spock"]
for dom in DOMS:
    Msh=np.asarray(sio.loadmat(os.path.join(SRC,f"marm_{dom}_all.mat"))[f"{dom}_sh_t"],float)
    Mwt=np.asarray(sio.loadmat(os.path.join(SRC,f"marm_{dom}_all.mat"))[f"{dom}_wt_t"],float)
    wtp=np.where(~np.all(np.isnan(Mwt),axis=1))[0]
    for k in range(4):
        # wt slot: WT reference; sh slot: Founder-S mutants; gt slot:
        # Founder-K mutants.
        S.append(Endpoint(f"{dom}{k+1}",dom, Mwt[wtp,k], Msh[smask,k], Msh[kmask,k]))

def radar2(ax,S,ttl):
    """Two-group radar: one spoke per endpoint, one closed line per group at
    the group mean with a standard-error band, and one domain label at each
    quadrant centroid."""
    n=len(S); ang=np.arange(n)*2*np.pi/n; zmin,zmax,R0,RMAX=-2.4,1.5,0.15,0.9
    th=np.linspace(0,2*np.pi,240); OL=(0.15,0.15,0.15)
    for zg in (-1.5,-0.5,0.5,1.5):
        rr=local_rof(zg,zmin,zmax,R0,RMAX); ax.plot(rr*np.cos(th),rr*np.sin(th),"-",color=STY.clr.gray_lite,lw=STY.lw.hair,zorder=0)
    r0=local_rof(0,zmin,zmax,R0,RMAX); ax.plot(r0*np.cos(th),r0*np.sin(th),"-",color=(0.78,)*3,lw=0.6,zorder=0)
    for a in ang: ax.plot([R0*np.cos(a),RMAX*np.cos(a)],[R0*np.sin(a),RMAX*np.sin(a)],"-",color=(0.9,)*3,lw=0.4,zorder=0)
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
        ax.plot(xs,ys,"-",color=CMEAN[gi],lw=1.6,zorder=3)
    for dom in DOMS:
        idx=[i for i in range(n) if S[i].dom==dom]; a=float(np.mean(ang[idx])); rl=RMAX+0.15
        ha="left" if np.cos(a)>0.2 else ("right" if np.cos(a)<-0.2 else "center")
        va="bottom" if np.sin(a)>0.2 else ("top" if np.sin(a)<-0.2 else "center")
        ax.text(rl*np.cos(a),rl*np.sin(a),RADAR_DOMLAB.get(dom,dom),fontname=STY.font.name,fontsize=STY.fs_tick,ha=ha,va=va)
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(ttl,fontname=STY.font.name,fontsize=STY.fs_title)

# Combined link score: ridge-logistic fit of WT (label 1) against SH (label
# 0) over all endpoints, with GT projected onto the resulting axis. eta is
# the per-animal linear predictor.
Xall=np.vstack([WT[WTp],SH[SHp],GT[GTp]])
mu=np.nanmean(Xall,0); sd=np.nanstd(Xall,0); sd=np.where(~np.isfinite(sd)|(sd==0),1.0,sd)
z=lambda X:(X-mu)/sd
np.random.seed(0)
w,b=F.local_ridge(F.local_nan0(np.vstack([z(WT[WTp]),z(SH[SHp])])),
                  np.concatenate([np.ones(len(WTp)),np.zeros(len(SHp))]),5.0)
eta=lambda X:F.local_etaavail(w,b,z(X),~np.isnan(z(X)))
eta_wt,eta_sh,eta_gt=eta(WT[WTp]),eta(SH[SHp]),eta(GT[GTp])

# Per-domain link score, one value per animal, with the founder line of each
# mutant and treated animal.
def present(A):
    """Row indices of the animals with at least one observed endpoint."""
    return np.where(~np.all(np.isnan(A),axis=1))[0]
def domain_eta(dom):
    """Fit the link score on one domain's endpoints alone and score every
    animal. Returns the WT, SH and GT eta vectors followed by the founder
    line of each SH and each GT animal."""
    M=sio.loadmat(os.path.join(SRC,f"marm_{dom}_all.mat"))
    wt=np.asarray(M[f"{dom}_wt_t"],float); sh=np.asarray(M[f"{dom}_sh_t"],float); gt=np.asarray(M[f"{dom}_gt_t"],float)
    wp,sp,gp=present(wt),present(sh),present(gt)
    Xw,Xs,Xg=wt[wp],sh[sp],gt[gp]
    mu=np.nanmean(np.vstack([Xw,Xs,Xg]),0); sd=np.nanstd(np.vstack([Xw,Xs,Xg]),0)
    sd=np.where(~np.isfinite(sd)|(sd==0),1.0,sd); zz=lambda X:(X-mu)/sd
    np.random.seed(0)
    w_,b_=F.local_ridge(F.local_nan0(np.vstack([zz(Xw),zz(Xs)])),
                        np.concatenate([np.ones(len(wp)),np.zeros(len(sp))]),5.0)
    et=lambda X:F.local_etaavail(w_,b_,zz(X),~np.isnan(zz(X)))
    return et(Xw),et(Xs),et(Xg),np.array([uid2line.get(i+1) for i in sp]),np.array([uid2line.get(i+1) for i in gp])

DOMTITLE={"beh":"Behavioral","soc":"Social","cog":"Cognition","fmri":"fMRI"}
DOMD=[(DOMTITLE[d],)+domain_eta(d) for d in DOMS]
COMB=("COMBINED",eta_wt,eta_sh,eta_gt,sh_line,gt_line)

from gtstyle import gt_color_xticklabels
from _f6common import local_etalim, local_niceTicks
CMEAN,CSCAT,CBARF=STY.cmean,STY.cscat,STY.cbarfill
STY.fs_tick=7; STY.fs_lab=7; FS_TITLE=8

def vbox_founder(ax,ew,es,eg,sl,gl,lo,hi,showy,yticks):
    """Draw one panel of WT / SH / GT link-score box plots.

    ew, es, eg are the three eta vectors and sl, gl the founder line of each
    SH and each GT animal. Founder-K animals are drawn as open markers with a
    colored edge, Founder-S animals as filled markers; WT animals, which
    carry no founder line, are drawn filled.

    lo, hi set the y limits and yticks the tick positions, so panels sharing
    a range carry identical ticks. showy selects the long y-axis label."""
    w=0.55
    ax.plot([-0.6,2.6],[0,0],"-",color=STY.clr.gray_lite,lw=STY.lw.hair)
    for gi,(v,ln) in enumerate([(ew,None),(es,sl),(eg,gl)]):
        v=np.asarray(v,float); ok=~np.isnan(v)
        lnn=(ln[ok] if ln is not None else None); v=v[ok]
        if v.size==0: continue
        x=gi; q1,med,q3=np.percentile(v,[25,50,75]); vlo,vhi=v.min(),v.max()
        ax.fill([x-w/2,x-w/2,x+w/2,x+w/2],[q1,q3,q3,q1],color=CBARF[gi],edgecolor="none",zorder=1)
        rng=np.random.default_rng(gi); jx=x+(rng.random(v.size)-0.5)*w*0.55
        if lnn is None:
            ax.scatter(jx,v,s=STY.mk.pt,c=[CSCAT[gi]],edgecolors="none",zorder=2)
        else:
            kk=lnn=="Kelle"; ss=~kk
            ax.scatter(jx[ss],v[ss],s=STY.mk.pt,c=[CSCAT[gi]],edgecolors="none",zorder=2)
            ax.scatter(jx[kk],v[kk],s=STY.mk.pt,facecolor="white",edgecolors=CMEAN[gi],linewidths=0.9,zorder=3)
        ax.plot([x,x],[vlo,q1],"-",color="k",lw=0.5,zorder=3)
        ax.plot([x,x],[q3,vhi],"-",color="k",lw=0.5,zorder=3)
        ax.plot([x-w/2,x-w/2,x+w/2,x+w/2,x-w/2],[q1,q3,q3,q1,q1],"-",color="k",lw=0.7,zorder=3)
        ax.plot([x-w/2,x+w/2],[med,med],"-",color="k",lw=STY.lw.axis,zorder=4)
    ax.set_xlim(-0.6,2.6); ax.set_ylim(lo,hi); ax.set_xticks([0,1,2]); ax.set_xticklabels(["WT","SH","GT"])
    if yticks is not None and len(yticks): ax.set_yticks(yticks)
    ax.set_ylabel("link score $\\eta$" if showy else "$\\eta$",fontname=STY.font.name,fontsize=STY.fs_lab)
    gt_axes(ax,STY,matlab_ticks=False); gt_color_xticklabels(ax,STY)

# Shared y ranges: behavior and social share one range, cognition and fMRI
# another, so panels on a common scale carry identical ticks.
LO=np.array([local_etalim({"wt":d[1],"sh":d[2],"gt":d[3]})[0] for d in DOMD])
HI=np.array([local_etalim({"wt":d[1],"sh":d[2],"gt":d[3]})[1] for d in DOMD])
YT=[None]*4
for grp in [["beh","soc"],["cog","fmri"]]:
    ix=[DOMS.index(m) for m in grp]
    glo,ghi=LO[ix].min(),HI[ix].max(); LO[ix]=glo; HI[ix]=ghi
    tk=local_niceTicks(glo,ghi)
    for m in ix: YT[m]=tk

PAN,PAN_GAP,COMB_GAP=1.10,0.42,0.24
ML,MR,MB,MT=0.60,0.10,0.48,0.52
NCOL=5
FIG_W=ML+NCOL*PAN+(NCOL-1)*PAN_GAP+COMB_GAP+MR; FIG_H=MB+PAN+MT
fig=plt.figure(figsize=(FIG_W,FIG_H),facecolor="w")
for d,(title,ew,es,eg,sl,gl) in enumerate(DOMD):
    x=ML+d*(PAN+PAN_GAP)
    ax=fig.add_axes([x/FIG_W,MB/FIG_H,PAN/FIG_W,PAN/FIG_H])
    vbox_founder(ax,ew,es,eg,sl,gl,LO[d],HI[d],d==0,YT[d])
    ax.set_title(title,fontname=STY.font.name,fontsize=FS_TITLE,fontweight="normal")
    if d==0: gt_panelLetter(ax,"c",STY)
# combined-endpoint panel
tC,ewC,esC,egC,slC,glC=COMB
loC,hiC=local_etalim({"wt":ewC,"sh":esC,"gt":egC})
xC=ML+4*(PAN+PAN_GAP)+COMB_GAP
axC=fig.add_axes([xC/FIG_W,MB/FIG_H,PAN/FIG_W,PAN/FIG_H])
vbox_founder(axC,ewC,esC,egC,slC,glC,loC,hiC,False,None)
axC.set_title(tC,fontname=STY.font.name,fontsize=FS_TITLE,fontweight="normal")
fig.suptitle("WT-vs-SH link score by domain and overall - Founder-K line (open) among Founder-S line (filled)",
             fontsize=8.5,fontname=STY.font.name,y=0.995)
fig.legend([plt.Line2D([],[],marker='o',color='w',markerfacecolor='white',markeredgecolor='k',markersize=6,markeredgewidth=0.9),
            plt.Line2D([],[],marker='o',color='w',markerfacecolor=(0.4,)*3,markersize=6)],
           ["Founder-K (43 bp del)","Founder-S (G ins)"],loc="lower center",bbox_to_anchor=(0.5,-0.01),
           ncol=2,frameon=False,fontsize=6.5,columnspacing=1.8)
gt_figStyle(fig,STY)
gt_saveFig(fig,"edfig10_c_founder_lines",OUT)

