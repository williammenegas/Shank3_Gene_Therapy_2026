"""
Figure 3, panels f and l, with the dyad as the unit of analysis.

Two box-and-scatter plots, groups WT / SH / GT, in the style of the other
Figure 3 box panels:

  f  Inter-animal similarity (%), 100 times 1 minus the velocity-similarity
     measure. y limits [40, 100], as in the published panel and in the
     matching Extended Data Fig. 6f.

  l  Call probability (%). y limits [30, 80].

Both cagemates of a WT dyad carry the same value, so the script collapses WT
values to one per dyad with np.unique; an SH or GT dyad contributes its
mutant animal once. Each plot shows the box (median, quartiles and whiskers)
with a jittered scatter over it.

Statistics: Levene's test for equal variance, then one-way ANOVA with
Tukey-Kramer post-hoc, or Welch's ANOVA with Games-Howell where equal
variance is rejected (gt_boxsig). Pairwise brackets are drawn only when the
omnibus test passes; the call is made after the y limits are set.

Data: vel_quantified_data.mat, voc_data.mat

Outputs: fig3_f_velocity_bias, fig3_l_call_probability
"""

import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa
HERE=os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig3", "fig3_fl_dyad_endpoints")
OUT = central_output_dir(HERE)

def nn(x): x=np.asarray(x,float).ravel(); return x[~np.isnan(x)]

def panel(name, matfile, keys, transform, ylabel, ylim, title):
    STY=gt_init(); rng=np.random.default_rng(1)
    S=load_struct(os.path.join(DATA,matfile))
    wt=transform(nn(getattr(S,keys[0]))); sh=transform(nn(getattr(S,keys[1]))); gt=transform(nn(getattr(S,keys[2])))
    wt=np.unique(wt)                       # one value per WT dyad, not per animal
    fig,ax=gt_boxscatter_figure(STY)
    gt_colorboxplot(ax,[wt,sh,gt],STY,rng=rng)
    ax.set_xticks([1,2,3]); ax.set_xticklabels(["WT","SH","GT"]); ax.set_xlim(0.5,3.5); ax.set_ylim(*ylim)
    gt_boxsig(ax,STY)
    ax.set_ylabel(ylabel,fontname=STY.font.name,fontsize=STY.font.lab)
    gt_axes(ax,STY); gt_color_xticklabels(ax,STY); gt_figStyle(fig,STY)
    gt_saveFig(fig,name,OUT)
    print(f"{name}: n={wt.size}/{sh.size}/{gt.size} means={wt.mean():.3f}/{sh.mean():.3f}/{gt.mean():.3f}")

panel("fig3_f_velocity_bias","vel_quantified_data.mat",("wt_f1_mat","sh_f1_mat","gt_f1_mat"),
      lambda x:100.0*(1-x),"Inter-animal similarity (%)",(40,100),"Fig. 3f (dyad-level)")
panel("fig3_l_call_probability","voc_data.mat",("voc_wt","voc_sh","voc_gt"),
      lambda x:x,"Call probability (%)",(30,80),"Fig. 3l (dyad-level)")
print("saved to",OUT)
