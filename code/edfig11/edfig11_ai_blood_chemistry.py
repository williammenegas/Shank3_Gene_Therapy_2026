"""
edfig11_ai_blood_chemistry.py -- Extended Data Fig. 11, panels a to i.

Serum chemistry markers of hepatic, muscle and renal toxicity in treated and
control animals.

ANALYTES, one panel each:
  liver   AST, ALT, GGT, alkaline phosphatase, total bilirubin, albumin
  muscle  creatine kinase
  kidney  urea nitrogen, creatinine
Units are given on each y-axis label.

DATA. bloodwork_data.csv, one row per blood draw with the animal name, its
group (Control or Treatment) and its age in days. The script averages draws
within animal within each one-year age bin for the time course, and within
animal across all draws for the summary.

WHAT IS PLOTTED. Two figures of nine panels each:
  edfig11_ai_blood_chemistry_timecourse
                         the time course over five one-year age bins from 0-1
                         to 4-5 years: a faint trace per animal, the group
                         median as a line, and the 25th to 75th percentile as
                         a shaded band, for each of Control and Treatment.
  edfig11_ai_blood_chemistry_summary
                         the per-animal average over all draws, as a box plot
                         with jittered points, Control against Treatment.

This script draws marmoset reference ranges as gray dashed lines in every
panel, and sets the y limits so that the reference range occupies the middle
third of the axis, placing every panel on the same relative span. The ranges
are the literature consensus values of Bhatt et al. 2022 (J. Am. Assoc. Lab.
Anim. Sci.), Wegman et al. 2024 (Animals) and Schultz-Darken et al. 2016.

STATISTICS. The summary figure applies a two-sided Mann-Whitney U test of
Treatment against Control on the per-animal means, computed only where both
groups have at least three animals, followed by Benjamini-Hochberg FDR
correction across the analytes. Significance markers use the adjusted value.
The time-course figure is descriptive; no test is applied to it.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches
from scipy import stats as sstats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import gt_init, gt_style  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))          # new_data_panels/
from gtstyle import central_output_dir  # noqa: E402
from _common import sig_stars  # noqa: E402

STY = gt_style()

DATA = os.path.join(HERE, "..", "..", "data", "edfig11")

AAV_ANALYTES = ["AST_SGOT", "ALT_SGPT", "GGT", "ALK_Phosphatase", "TOTAL_BILIRUBIN",
                "ALBUMIN", "CK", "BUN", "CREATININE"]
AAV_LABELS = {
    "AST_SGOT": "AST [U/L]", "ALT_SGPT": "ALT [U/L]", "GGT": "GGT [U/L]",
    "ALK_Phosphatase": "ALP [U/L]", "TOTAL_BILIRUBIN": "Total Bilirubin [mg/dL]",
    "ALBUMIN": "Albumin [g/dL]", "CK": "CK [U/L]", "BUN": "BUN [mg/dL]",
    "CREATININE": "Creatinine [mg/dL]",
}
AAV_CATEGORY = {
    "AST_SGOT": "Liver", "ALT_SGPT": "Liver", "GGT": "Liver", "ALK_Phosphatase": "Liver",
    "TOTAL_BILIRUBIN": "Liver", "ALBUMIN": "Liver", "CK": "Muscle", "BUN": "Kidney",
    "CREATININE": "Kidney",
}
REF_RANGES = {
    "AST_SGOT": (50, 350), "ALT_SGPT": (0, 40), "GGT": (0, 15),
    "ALK_Phosphatase": (40, 400), "TOTAL_BILIRUBIN": (0, 0.5), "ALBUMIN": (3.0, 5.5),
    "CK": (100, 5000), "BUN": (10, 40), "CREATININE": (0.1, 0.6),
}

# Panel colors: Control salmon and Treatment lavender, with light tinted box
# fills, near-black box edges and medians, color-matched semi-transparent
# scatter, and gray dashed reference lines.
C_CTRL = np.array([0.840, 0.450, 0.490])       # salmon (Control) point/accent
C_TREAT = np.array([0.520, 0.500, 0.780])      # lavender (Treatment) point/accent
C_CTRL_L = np.array([0.905, 0.745, 0.755])     # salmon box fill
C_TREAT_L = np.array([0.825, 0.810, 0.915])    # lavender box fill
C_EDGE = np.array([0.10, 0.10, 0.13])          # near-black box edge / median
C_REF = (0.67, 0.67, 0.68)                     # gray dashed reference line

BIN_EDGES = [0, 365, 730, 1095, 1461, 1826]
BIN_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-5"]


def _box(ax, v, x0, hw, point_color, fill):
    """Draw one box at x0: a tinted fill from the first to the third
    quartile, a near-black edge and median rule, and thin whiskers extending
    to 1.5 times the interquartile range. Samples of fewer than two values
    are skipped."""
    if v.size < 2:
        return
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    whi = min(v.max(), q3 + 1.5 * iqr)
    wlo = max(v.min(), q1 - 1.5 * iqr)
    ax.add_patch(matplotlib.patches.Rectangle(
        (x0 - hw, q1), 2 * hw, max(q3 - q1, 1e-3), facecolor=fill, alpha=0.9,
        edgecolor=C_EDGE, lw=0.9, zorder=2))
    ax.plot([x0 - hw, x0 + hw], [q2, q2], "-", color=C_EDGE, lw=1.4, zorder=3)
    ax.plot([x0, x0], [wlo, q1], "-", color=C_EDGE, lw=0.8, zorder=2)
    ax.plot([x0, x0], [q3, whi], "-", color=C_EDGE, lw=0.8, zorder=2)
    ax.plot([x0 - hw / 2, x0 + hw / 2], [wlo, wlo], "-", color=C_EDGE, lw=0.8, zorder=2)
    ax.plot([x0 - hw / 2, x0 + hw / 2], [whi, whi], "-", color=C_EDGE, lw=0.8, zorder=2)


def main(data_dir=DATA, out_dir=central_output_dir(HERE), rng_seed=7, show=False):
    gt_init()
    rng = np.random.default_rng(rng_seed)

    T = pd.read_csv(os.path.join(data_dir, "bloodwork_data.csv"))
    T["Unique_ID"] = T["Unique_ID"].astype(str)
    T["TimeBin"] = pd.cut(T["Age_days"], bins=BIN_EDGES, labels=False, right=True) + 1

    T_binned = T.groupby(["Unique_ID", "Group", "TimeBin"], as_index=False)[AAV_ANALYTES].mean()
    print(f"Time-binned: {len(T_binned)} rows (was {len(T)}) -- each row = "
          f"one animal in one year bin")

    T_animal = T.groupby(["Unique_ID", "Group"], as_index=False)[AAV_ANALYTES].mean()
    print(f"Per-animal:   {len(T_animal)} animals (Treatment="
          f"{(T_animal['Group']=='Treatment').sum()}, Control="
          f"{(T_animal['Group']=='Control').sum()})")

    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: time course by age bin. Faint per-animal traces, the group
    # median as a line, and a shaded 25th-to-75th percentile band.
    fig1, axes1 = plt.subplots(3, 3, figsize=(14, 11))
    centers = np.arange(1, len(BIN_LABELS) + 1) - 0.5     # bin midpoints, years
    # Data column values stay "Control"/"Treatment"; the paper labels them SH and GT.
    GRP_STYLE = [("SH", "Control", C_CTRL), ("GT", "Treatment", C_TREAT)]

    for i, col in enumerate(AAV_ANALYTES):
        ax = axes1.flat[i]
        for lab, grp, c in GRP_STYLE:
            sub = T_binned[T_binned["Group"] == grp]
            # faint per-animal trajectories
            for _, g in sub.groupby("Unique_ID"):
                g = g.sort_values("TimeBin")
                x = g["TimeBin"].to_numpy(float) - 0.5
                y = g[col].to_numpy(float)
                m = ~np.isnan(y)
                if m.sum() >= 1:
                    ax.plot(x[m], y[m], "-", color=(*c, 0.22), lw=STY.lw.trace, zorder=2)
            # group median with a 25th-to-75th percentile band, per bin
            xc, med, q1, q3 = [], [], [], []
            for b in range(1, len(BIN_LABELS) + 1):
                v = sub.loc[sub["TimeBin"] == b, col].dropna().to_numpy(float)
                if v.size >= 1:
                    a, b25, b75 = np.percentile(v, [50, 25, 75])
                    xc.append(b - 0.5); med.append(a); q1.append(b25); q3.append(b75)
            xc, med, q1, q3 = map(np.array, (xc, med, q1, q3))
            if xc.size:
                ax.fill_between(xc, q1, q3, color=c, alpha=0.18,
                                edgecolor="none", zorder=3)
                ax.plot(xc, med, "-", color=c, lw=2.0, marker="o", ms=3.2,
                        label=lab, zorder=4)

        rr = REF_RANGES[col]
        ax.axhline(rr[0], ls="--", color=C_REF, lw=STY.lw.hair, zorder=1)
        ax.axhline(rr[1], ls="--", color=C_REF, lw=STY.lw.hair, zorder=1)
        rr_span = rr[1] - rr[0]
        ax.set_ylim(rr[0] - rr_span, rr[1] + rr_span)
        ax.set_xticks(centers)
        ax.set_xticklabels(BIN_LABELS, fontsize=STY.font.tick)
        ax.set_xlim(0.3, len(BIN_LABELS) + 0.2)
        ax.set_xlabel("Age (years)", fontsize=STY.font.lab)
        ax.set_ylabel(AAV_LABELS[col], fontsize=STY.font.lab)
        ax.set_title(f"{col.replace('_',' ')}  ({AAV_CATEGORY[col]})", fontsize=STY.font.title,
                    fontweight="bold")
        ax.tick_params(direction="out", labelsize=STY.font.tick)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes1.flat[0].legend(frameon=False, fontsize=STY.font.tick, loc="upper right")
    # The published figure carries no sheet title; its legend explains what the
    # lines, bands and dashed rules mean instead.
    fig1.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = os.path.join(out_dir, "edfig11_ai_blood_chemistry_timecourse.png")
    fig1.savefig(p1, dpi=200, facecolor="w")
    fig1.savefig(p1.replace(".png", ".pdf"), facecolor="w")   # vector PDF
    print(f"\nSaved: {p1}")

    # Figure 2: per-animal summary over all draws, Control against Treatment.
    fig2, axes2 = plt.subplots(3, 3, figsize=(14, 11))
    idx_treat_a = T_animal["Group"] == "Treatment"
    idx_ctrl_a = T_animal["Group"] == "Control"

    print("\n--- Per-animal summary (each animal averaged across all draws) ---")
    print(f"{'Analyte':<20s}  {'Category':<8s}  {'T mean':>8s}  {'T SD':>8s}  "
          f"{'C mean':>8s}  {'C SD':>8s}  {'p (raw)':>10s}  {'q (BH)':>10s}  {'Sig':>5s}")
    print("-" * 97)

    # Two-sided Mann-Whitney U on per-animal means, then Benjamini-Hochberg
    # FDR across the analytes. Significance markers use the adjusted value.
    p_raw = np.full(len(AAV_ANALYTES), np.nan)
    for i, col in enumerate(AAV_ANALYTES):
        a = T_animal.loc[idx_treat_a, col].dropna().to_numpy(float)
        b = T_animal.loc[idx_ctrl_a, col].dropna().to_numpy(float)
        if a.size >= 3 and b.size >= 3:
            p_raw[i] = sstats.mannwhitneyu(a, b, alternative="two-sided").pvalue

    q_adj = np.full(len(AAV_ANALYTES), np.nan)
    ok = ~np.isnan(p_raw)
    if ok.any():
        pv = p_raw[ok]
        m = pv.size
        order = np.argsort(pv)
        ranked = pv[order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        qv = np.empty(m)
        qv[order] = np.minimum(ranked, 1.0)
        q_adj[ok] = qv

    for i, col in enumerate(AAV_ANALYTES):
        ax = axes2.flat[i]
        v_t = T_animal.loc[idx_treat_a, col].dropna().to_numpy(float)
        v_c = T_animal.loc[idx_ctrl_a, col].dropna().to_numpy(float)

        _box(ax, v_c, 1, 0.25, C_CTRL, C_CTRL_L)
        _box(ax, v_t, 2, 0.25, C_TREAT, C_TREAT_L)
        ax.scatter(1 + 0.15 * (rng.random(v_c.size) - 0.5), v_c, s=STY.mk.pt, c=[C_CTRL],
                  alpha=0.8, edgecolors="none", zorder=4)
        ax.scatter(2 + 0.15 * (rng.random(v_t.size) - 0.5), v_t, s=STY.mk.pt, c=[C_TREAT],
                  alpha=0.8, edgecolors="none", zorder=4)

        p, q = p_raw[i], q_adj[i]
        sig = sig_stars(q)
        if not np.isnan(q):
            ax.text(1.5, 0.94, sig if sig else "n/s", ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=STY.font.tick,
                    color=(0.35, 0.35, 0.35))
        print(f"{col.replace('_',' '):<20s}  {AAV_CATEGORY[col]:<8s}  {v_t.mean():8.1f}  "
              f"{v_t.std(ddof=1):8.1f}  {v_c.mean():8.1f}  {v_c.std(ddof=1):8.1f}  "
              f"{p:10.4f}  {q:10.4f}  {sig:>5s}")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["SH", "GT"], fontsize=STY.font.tick)
        ax.set_xlim(0.4, 2.6)
        ax.set_ylabel(AAV_LABELS[col], fontsize=STY.font.lab)
        ax.set_title(f"{col.replace('_',' ')}  ({AAV_CATEGORY[col]})", fontsize=STY.font.title,
                    fontweight="bold")
        rr = REF_RANGES[col]
        ax.axhline(rr[0], ls="--", color=C_REF, lw=STY.lw.hair, zorder=1)
        ax.axhline(rr[1], ls="--", color=C_REF, lw=STY.lw.hair, zorder=1)
        rr_span = rr[1] - rr[0]
        ax.set_ylim(rr[0] - rr_span, rr[1] + rr_span)
        ax.tick_params(direction="out", labelsize=STY.font.tick)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    p2 = os.path.join(out_dir, "edfig11_ai_blood_chemistry_summary.png")
    fig2.savefig(p2, dpi=200, facecolor="w")
    fig2.savefig(p2.replace(".png", ".pdf"), facecolor="w")   # vector PDF
    print(f"\nSaved: {p2}")

    if not show:
        plt.close(fig1)
        plt.close(fig2)
    return fig1, fig2


if __name__ == "__main__":
    main()
