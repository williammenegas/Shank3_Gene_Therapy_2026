"""Extended Data Figure 2, panels a-c: low-dose gene-therapy endpoints.

This script draws three endpoints, one panel each: locomotion (panel a,
distance traveled), rearing (panel b, rearing time) and social state
(panel c, fraction of the session spent in the social state). Groups are WT,
SH and low-dose GT.

Data: HD_LD_Merged_Data.xlsx, first sheet, header on the second row. The
script reads only the columns these three endpoints require. Each animal
carries a Group (WT+GFP, Mut+GFP, Mut+MiniShank3) and a Dose (High, Low);
low dose is 4e13 vg/kg and high dose is 2e14 vg/kg.

Group assignment: WT and SH pool animals from both dose batches; GT is the
low-dose animals only, and high-dose GT animals are not plotted.

Batch normalization: locomotion and rearing are expressed as a percentage of
the wild-type mean within the animal's own dose batch, so 100% is that batch's
own wild-type average and both batches share one axis. The social-state
fraction is bounded and is plotted raw.

Display: each panel's y axis is capped (F1_YMAX). Bars, error bars and all
statistics are computed from the full data; off-scale points are named beneath
the axis.

Statistics: one-way ANOVA across the three groups with Tukey-Kramer post-hoc
contrasts. Pairwise brackets are drawn only when the ANOVA is significant;
otherwise the panel is annotated 'ANOVA n/s'.
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import (gt_init, gt_style, gt_barpanel_figure, gt_bar3,  # noqa: E402
                     gt_compare3, gt_drawSig, gt_pstar, gt_axes,
                     gt_color_xticklabels, gt_offscale_note, gt_saveFig,
                     gt_figStyle)
sys.path.insert(0, os.path.dirname(HERE))          # parent directory
from _common import house_posthoc, sig_stars  # noqa: E402
from scipy import stats as sstats  # noqa: E402

# Display ceiling per endpoint. The axis is capped so bars and error bars stay
# legible; bars, SEM and every statistic are computed from the full data, and
# off-scale points are named under the axis.
F1_YMAX = {"Locomotion": 200.0, "Rearing": 200.0, "Social State": 1.0}
F1_YLO = {}                                         # all default to 0
F1_REF = {"Locomotion": 100.0, "Rearing": 100.0}    # dashed guide line


def _panel_f1(G, gnames, title, ylab, out_dir, fname):
    """Draw and save one three-group bar panel.

    Bar geometry and body come from gt_barpanel_figure/gt_bar3; pairwise
    Tukey-Kramer brackets are drawn only when the omnibus ANOVA is significant;
    tick labels are group-colored and per-group n is printed beneath the axis.
    Saves <fname>.pdf and <fname>.eps in out_dir.
    """
    STY = gt_style()
    ymax = F1_YMAX.get(title, float(np.nanmax(np.concatenate(G))) * 1.1)
    ylo = F1_YLO.get(title, 0.0)
    ref = F1_REF.get(title, None)

    fig, ax = gt_barpanel_figure(STY)
    if ref is not None:
        ax.axhline(ref, ls="--", color=(0.75, 0.75, 0.75), lw=STY.lw.hair, zorder=0)
    n_off = gt_bar3(ax, G, STY, ymax=ymax)

    R = gt_compare3(G, "tukey")
    p_anova, p_adj = R["p_anova"], R["q"]

    ax.set_ylim(ylo, ymax)
    yl = ax.get_ylim(); rng = yl[1] - yl[0]; h = 0.012 * rng
    vis_max = max([v[v <= ymax].max() for v in G if (v <= ymax).any()] or [ylo])
    base = min(max(yl[0] + 0.860 * rng, vis_max + 0.035 * rng), yl[1] - 0.055 * rng)
    if p_anova < 0.05:
        gt_drawSig(ax, 0, 1, base, h, gt_pstar(p_adj[0]), STY)
        gt_drawSig(ax, 1, 2, base, h, gt_pstar(p_adj[1]), STY)
        gt_drawSig(ax, 0, 2, base + 0.075 * rng, h, gt_pstar(p_adj[2]), STY)
    else:
        ax.text(1, base + 0.05 * rng, "ANOVA n/s", ha="center",
                fontsize=STY.font.ns, color=STY.clr.ns_txt, fontname=STY.font.name)

    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(gnames)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY, labels=tuple(gnames))
    ax.get_xticklabels()[2].set_color(STY.cmean[2])     # GT-low label in the GT color
    ax.set_ylabel(ylab)
    ax.set_title(title, fontname=STY.font.name, fontsize=STY.font.title,
                 pad=STY.geom.title_pad)
    ylf = ax.get_ylim()
    for g in range(3):
        ax.text(g, ylf[0] - 0.205 * (ylf[1] - ylf[0]), f"n={G[g].size}",
                ha="center", fontsize=STY.font.n, color=STY.clr.n_txt,
                fontname=STY.font.name)
    gt_offscale_note(ax, n_off, gnames, ymax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, fname, out_dir)
    plt.close(fig)

STY = gt_style()

DATA = os.path.join(HERE, "..", "..", "data", "edfig2")

# Three display groups. WT and SH pool both dose batches, each normalized
# within its own batch for locomotion and rearing; GT is the low-dose group
# only, and the high-dose GT animals are excluded.
GROUP_LABELS = ["WT", "SH", "GT-low"]
GROUP_COLORS = [
    np.array([0.15, 0.15, 0.15]),
    np.array([0.75, 0.22, 0.22]),
    np.array([0.50, 0.58, 0.80]),
]

# Spreadsheet column, panel title, raw y label, raw y limits, batch-normalize.
ENDPOINTS = [
    ("Distance\nTraveled (m)", "Locomotion", "Distance (m)", (0, 15000), True),
    ("Rearing\nTime (s)", "Rearing", "Time (s)", (0, 400), True),
    ("Frac Social\nState", "Social State", "Fraction of session", (0, 1.0), False),
]


def _assign_display_group(group_raw, dose_raw):
    is_low = dose_raw.str.lower() == "low"
    is_high = dose_raw.str.lower() == "high"
    g = np.zeros(len(group_raw), dtype=int)
    g[group_raw.str.contains("WT", na=False)] = 1
    g[group_raw.str.contains("Mut", na=False) & group_raw.str.contains("GFP", na=False) &
      ~group_raw.str.contains("Mini", na=False)] = 2
    g[group_raw.str.contains("MiniShank", na=False) & is_low] = 3
    g[group_raw.str.contains("MiniShank", na=False) & is_high] = 4
    return g, is_low.to_numpy(), is_high.to_numpy()


def _batch_norm(raw_vals, is_low, is_high, display_group, mode):
    if mode == "none":
        return raw_vals.copy()
    plot_vals = np.full(raw_vals.shape, np.nan)
    for bm in (is_low, is_high):
        if bm.sum() == 0:          # batch not represented in this selection
            continue
        if mode == "pct_wt":
            wt_in_batch = bm & (display_group == 1)
            mu_wt = np.nanmean(raw_vals[wt_in_batch])
            if abs(mu_wt) < np.finfo(float).eps:
                mu_wt = 1.0
            plot_vals[bm] = raw_vals[bm] / mu_wt * 100
        elif mode == "zscore":
            bd = raw_vals[bm]
            mu_b, sd_b = np.nanmean(bd), np.nanstd(bd, ddof=1)
            if sd_b < np.finfo(float).eps:
                sd_b = 1.0
            plot_vals[bm] = (raw_vals[bm] - mu_b) / sd_b
        elif mode == "pct":
            bd = raw_vals[bm]
            mu_b = np.nanmean(bd)
            if abs(mu_b) < np.finfo(float).eps:
                mu_b = 1.0
            plot_vals[bm] = raw_vals[bm] / mu_b * 100
    return plot_vals


def _plot_endpoint(ax, group_data, ttl, ylab, ylims, norm_mode, rng):
    x_positions = [1, 2, 3]
    hw = 0.30
    n_groups = 3

    if norm_mode == "zscore":
        ax.axhline(0, ls="--", color=(0.75, 0.75, 0.75), lw=STY.lw.hair)
    elif norm_mode in ("pct", "pct_wt"):
        ax.axhline(100, ls="--", color=(0.75, 0.75, 0.75), lw=STY.lw.hair)
    else:
        wt = group_data[0]
        if wt.size:
            ax.plot([0.3, 5.0], [wt.mean(), wt.mean()], "--", color=(0.75, 0.75, 0.75), lw=STY.lw.hair)

    for gi in range(n_groups):
        gv = group_data[gi]
        if gv.size == 0:
            continue
        xc = x_positions[gi]
        m, se = gv.mean(), gv.std(ddof=1) / np.sqrt(gv.size)
        col = GROUP_COLORS[gi]
        ax.bar(xc, m, width=2 * hw, color=col * 0.25 + 0.75, edgecolor=col, lw=0.5, zorder=2)
        xoff = (rng.random(gv.size) - 0.5) * hw
        ax.scatter(xc + xoff, gv, s=STY.mk.pt, c=[col * 0.7 + 0.3], edgecolors="none", zorder=3)
        ax.plot([xc, xc], [m - se, m + se], "-", color=col, lw=1.2, zorder=4)
        ax.plot([xc - 0.06, xc + 0.06], [m + se, m + se], "-", color=col, lw=1.2, zorder=4)
        ax.plot([xc - 0.06, xc + 0.06], [m - se, m - se], "-", color=col, lw=1.2, zorder=4)

    # Group comparison (see _common.house_posthoc); every pair is bracketed,
    # including non-significant ones.
    rows, _p_anova = house_posthoc([group_data[g] for g in range(n_groups)])
    sig = [(g1 - 1, g2 - 1, p) for g1, g2, p, _d in rows]
    if sig:
        yl = ax.get_ylim(); yr = yl[1] - yl[0]; ns = len(sig)
        ax.set_ylim(yl[0], yl[1] + 0.06 * yr * ns + 0.05 * yr)
        yl2 = ax.get_ylim()
        by = yl2[1] - 0.05 * (yl2[1] - yl2[0])
        bs = 0.06 * (yl2[1] - yl2[0])
        for si, (g1, g2, p) in enumerate(sig):
            s = sig_stars(p)
            x1, x2 = x_positions[g1], x_positions[g2]
            yb = by - si * bs
            th = 0.006 * (yl2[1] - yl2[0])
            ax.plot([x1, x1, x2, x2], [yb - th, yb, yb, yb - th], "-", color=(0.3, 0.3, 0.3), lw=0.7)
            ax.text((x1 + x2) / 2, yb + 0.003 * (yl2[1] - yl2[0]), s, ha="center", fontsize=8,
                    color=(0.15, 0.15, 0.15))

    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.set_xlim(0.4, 3.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(GROUP_LABELS)
    ax.set_ylabel(ylab, fontsize=STY.font.lab)
    ax.set_title(ttl, fontsize=STY.font.title, fontweight="bold")
    ax.tick_params(direction="out", labelsize=STY.font.tick)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(STY.lw.axis)

    yl_f = ax.get_ylim()
    for gi in range(n_groups):
        n = group_data[gi].size
        if n > 0:
            ax.text(x_positions[gi], yl_f[0] - 0.10 * (yl_f[1] - yl_f[0]), f"n={n}",
                    ha="center", fontsize=7, color=(0.6, 0.6, 0.6), clip_on=False)
    ax.tick_params(axis="x", pad=12)


def main(data_dir=DATA, out_dir=HERE, rng_seed=7, show=False):
    gt_init()
    rng = np.random.default_rng(rng_seed)

    excel_path = os.path.join(data_dir, "HD_LD_Merged_Data.xlsx")
    T = pd.read_excel(excel_path, sheet_name=0, header=1)
    print(f"  {len(T)} animals loaded (HD+LD merged)")

    # WT and SH pool both batches, normalized within batch; GT is low-dose only.
    group_raw = T.iloc[:, 1].astype(str)
    dose_raw = T.iloc[:, 2].astype(str)
    display_group, is_low, is_high = _assign_display_group(group_raw, dose_raw)
    for g in range(1, 4):   # WT, SH, GT-low; group 4 (GT high dose) is not plotted
        print(f"  {GROUP_LABELS[g-1]}: n={np.sum(display_group==g)}")

    os.makedirs(out_dir, exist_ok=True)

    # Panel letter per endpoint, in the order they appear in ENDPOINTS.
    panel_letters = ["a", "b", "c"]

    for (col_name, ttl, ylab, ylims, do_norm), letter in zip(ENDPOINTS, panel_letters):
        raw_vals = T[col_name].to_numpy(dtype=float)
        mode = "pct_wt" if do_norm else "none"
        plot_vals = _batch_norm(raw_vals, is_low, is_high, display_group, mode)
        ylab_eff = "% of WT mean" if do_norm else ylab
        ylims_eff = None if do_norm else ylims

        group_data = [plot_vals[display_group == g] for g in (1, 2, 3)]
        group_data = [v[~np.isnan(v)] for v in group_data]
        if all(v.size == 0 for v in group_data):
            continue

        out_stem = f"edfig2_{letter}_" + ttl.replace(" ", "_").lower()
        _panel_f1(group_data, GROUP_LABELS, ttl, ylab_eff, out_dir, out_stem)
        print(f"  saved {out_stem}.pdf  (F1 style)")

    print("\nDone.")


if __name__ == "__main__":
    main()
