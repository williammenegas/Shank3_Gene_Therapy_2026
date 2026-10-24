"""
Figure 1, panel n: striatum, synaptic protein levels by western blot.

Seven synaptic proteins on the x axis. For each protein, the script plots SH
and GT relative expression, normalized to WT = 1.0 (dotted reference line).
Light gray lines connect the SH and GT lanes collected on the same gel set.
Bold markers are the mean with s.e.m. whiskers.

Statistics: every gel set carries one WT, one SH and one GT lane per protein,
so averaging across the seven proteins within a gel set gives one value per
group per gel set. The gel set is the experimental unit (2 mice pooled per
preparation): n = 6 sets for striatum. Those gel-set means are tested with a
one-way repeated-measures ANOVA across the three groups, followed by paired
t-tests on the same gel-set means with Holm adjustment across the three
pairs. Significance labels use the house scale: * p < 0.05, ** p < 0.01,
*** p < 0.001, the literal '0.05' for p in [0.050, 0.060), and 'n/s'
otherwise.

The data file is named for an earlier panel lettering: cortex data are in
panel_n_cortex_western.mat and striatum data in panel_m_striatum_western.mat,
while the published panels are m = cortex and n = striatum. The panel letter
does not follow from the file name.

Data: panel_m_striatum_western.mat
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import studentized_range

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig1", "fig1_n_striatum_western")

FIG_WIDTH_IN = 3.39
FIG_HEIGHT_IN = 2.00
MARGIN_L_IN = 0.46
MARGIN_R_IN = 0.10
MARGIN_B_IN = 0.34
MARGIN_T_IN = 0.26
PT_SIZE_W = 6      # marker area for one gel set, matching STY.mk.pt
DX = 0.19          # horizontal offset of the SH and GT columns from the protein tick



def _gelset_rm_anova(value, group, setnum, gnames=("WT", "SH", "GT")):
    """Gel-set repeated-measures ANOVA, the test annotated on the panel and
    reported in Supplementary Table 1.

    Every gel set carries one WT, one SH and one GT lane per protein, so
    averaging across the seven proteins within a gel set gives one value per
    group per gel set. The gel set is the experimental unit (2 mice pooled
    per preparation): n = 4 sets for cortex, 6 for striatum. That matrix is
    tested with a one-way repeated-measures ANOVA across the three groups,
    then paired t-tests on the same gel-set means with Holm adjustment across
    the three pairs.
    """
    import numpy as np
    from scipy import stats as _ss

    value = np.asarray(value, float).ravel()
    group = np.asarray(group).ravel()
    sets = sorted(set(np.asarray(setnum, float).ravel().tolist()))
    sn = np.asarray(setnum, float).ravel()

    Y = np.full((len(sets), 3), np.nan)
    for i, s in enumerate(sets):
        for j, g in enumerate(gnames):
            m = (sn == s) & (group == g)
            if m.any():
                Y[i, j] = np.nanmean(value[m])
    Y = Y[~np.isnan(Y).any(axis=1)]
    n, k = Y.shape

    gm = Y.mean()
    ss_c = n * ((Y.mean(0) - gm) ** 2).sum()
    ss_s = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_e = ((Y - gm) ** 2).sum() - ss_c - ss_s
    df_c, df_e = k - 1, (k - 1) * (n - 1)
    F = float((ss_c / df_c) / (ss_e / df_e))
    p_omni = float(_ss.f.sf(F, df_c, df_e))

    idx = [(0, 1), (1, 2), (0, 2)]
    t = np.empty(3); p = np.empty(3)
    for m, (a, b) in enumerate(idx):
        t[m], p[m] = _ss.ttest_rel(Y[:, a], Y[:, b])

    order = np.argsort(p)
    q = np.empty(3); run = 0.0
    for r, m in enumerate(order):
        run = max(run, (3 - r) * float(p[m]))
        q[m] = min(1.0, run)

    return dict(Y=Y, n_sets=n, F=F, df1=df_c, df2=df_e, p_omnibus=p_omni,
                labels=[("WT", "SH"), ("SH", "GT"), ("WT", "GT")],
                t=t, p=p, q=q)


def main(data_dir=DATA, out_dir=HERE, panel_letter="n",
         mat_name="panel_m_striatum_western.mat", out_name = "fig1_n_striatum_western"):
    STY = gt_init()
    S = load_struct(os.path.join(data_dir, mat_name))

    protein = np.asarray(S.protein)
    group = np.asarray(S.group)
    setnum = np.ravel(np.asarray(S.setnum, dtype=float))
    value = np.ravel(np.asarray(S.value, dtype=float))
    prots = list(S.protein_order)
    nP = len(prots)

    c_sh = STY.clr.sh
    c_gt = STY.clr.gt
    cscat_sh = STY.cscat[1]
    cscat_gt = STY.cscat[2]

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), facecolor="w")
    ax_w_in = FIG_WIDTH_IN - MARGIN_L_IN - MARGIN_R_IN
    ax_h_in = FIG_HEIGHT_IN - MARGIN_B_IN - MARGIN_T_IN
    ax = fig.add_axes([MARGIN_L_IN / FIG_WIDTH_IN, MARGIN_B_IN / FIG_HEIGHT_IN,
                       ax_w_in / FIG_WIDTH_IN, ax_h_in / FIG_HEIGHT_IN])

    ax.set_xlim(-0.6, nP - 0.4)
    ax.axhline(1.0, linestyle=":", color="k", linewidth=0.6, zorder=0)  # WT reference

    for pi, pn in enumerate(prots):
        is_p = protein == pn
        sh_v = value[is_p & (group == "SH")]
        sh_s = setnum[is_p & (group == "SH")]
        gt_v = value[is_p & (group == "GT")]
        gt_s = setnum[is_p & (group == "GT")]

        common = np.intersect1d(sh_s, gt_s)
        nj = common.size
        for kk, cs in enumerate(common):
            jx = ((kk - (nj + 1) / 2) / max(nj, 1)) * DX * 0.5
            xs = pi - DX + jx
            xg = pi + DX + jx
            ys = sh_v[np.flatnonzero(sh_s == cs)[0]]
            yg = gt_v[np.flatnonzero(gt_s == cs)[0]]
            ax.plot([xs, xg], [ys, yg], "-", color=(0.81, 0.81, 0.81),
                    linewidth=0.6, zorder=1)
            ax.scatter(xs, ys, s=PT_SIZE_W, c=[cscat_sh], edgecolors="none", zorder=2)
            ax.scatter(xg, yg, s=PT_SIZE_W, c=[cscat_gt], edgecolors="none", zorder=2)

        ms, ses = sh_v.mean(), sh_v.std(ddof=1) / np.sqrt(sh_v.size)
        mg, seg = gt_v.mean(), gt_v.std(ddof=1) / np.sqrt(gt_v.size)
        ax.errorbar(pi - DX, ms, yerr=ses, color="k", elinewidth=1.3,
                    capsize=0, zorder=3)
        ax.errorbar(pi + DX, mg, yerr=seg, color="k", elinewidth=1.3,
                    capsize=0, zorder=3)
        ax.scatter(pi - DX, ms, s=STY.mk.mean, c=[c_sh], edgecolors="none", zorder=4)
        ax.scatter(pi + DX, mg, s=STY.mk.mean, c=[c_gt], edgecolors="none", zorder=4)

    ax.set_xticks(range(nP))
    ax.set_xticklabels(prots, rotation=30, ha="right")
    ax.set_ylim(0.4, 1.5)
    ax.set_ylabel("Relative abundance")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
    ax.tick_params(direction="out", width=STY.lw.axis,
                       length=STY.tick.len, pad=STY.tick.pad, labelsize=6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(STY.font.name)
        lbl.set_fontsize(6)
    ax.xaxis.label.set_fontname(STY.font.name)
    ax.xaxis.label.set_fontsize(7)
    ax.yaxis.label.set_fontname(STY.font.name)
    ax.yaxis.label.set_fontsize(7)

    gt_panelLetter(ax, panel_letter, STY)

    # Statistics: gel-set repeated-measures ANOVA, as annotated on the panel.
    R = _gelset_rm_anova(value, group, setnum)
    p_omni = R["p_omnibus"]
    t_shgt, p_shgt, q_shgt = R["t"][1], R["p"][1], R["q"][1]

    pstr = (f"F({R['df1']},{R['df2']}) = {R['F']:.1f}, "
            + ("p<0.0001" if p_omni < 1e-4 else f"p = {p_omni:.2g}"))
    titstr = f"{S.region} - synaptic protein level"
    ax.set_title(titstr, fontweight="normal", fontsize=STY.font.title)

    print(f"{S.region}: gel-set RM-ANOVA  {pstr}   (n = {R['n_sets']} gel sets, "
          f"2 mice pooled per prep)")
    for (a, b), tt, pr, qq in zip(R["labels"], R["t"], R["p"], R["q"]):
        print(f"  {a} vs {b}: paired t({R['n_sets']-1}) = {tt:+.2f}  "
              f"p = {gt_pfmt(pr)}  (Holm-adjusted p = {gt_pfmt(qq)})")

    gt_figStyle(fig, STY)
    gt_saveFig(fig, out_name, out_dir)
    return fig


if __name__ == "__main__":
    main()
