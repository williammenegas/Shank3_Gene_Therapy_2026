"""
edfig12_bcd_swd_rates.py -- Extended Data Fig. 12, panels b, c and d.

EEG spike-wave-discharge burden across six genotype and treatment groups.

GROUPS:
    WT_GFP           wild type with AAV-GFP
    Shank3_HOM_GFP   Shank3 homozygous null with AAV-GFP
    Shank3_HOM_GT    Shank3 homozygous null with AAV-miniSHANK3
    SCN2A            SCN2A model, positive seizure control
    Shank3_HET_GFP   Shank3 heterozygous with AAV-GFP
    Shank3_HET_GT    Shank3 heterozygous with AAV-miniSHANK3

DATA. EEG_data.csv, one column per group, holding the P7 spike-wave-discharge
rate of each animal, as events per hour.

WHAT IS PLOTTED. Five figures:
  edfig12_bcd_swd_rates_all_groups
                        all six groups, group mean with a standard-error
                        whisker and the individual animals as a beeswarm,
                        homozygous and heterozygous blocks separated.
  edfig12_bcd_swd_rates_no_SCN2A
                        the same excluding the positive control, on a y range
                        that resolves the remaining groups.
  edfig12_bcd_swd_rates_homozygous
                        wild type, Shank3-null with GFP and Shank3-null with
                        gene therapy, in the pinned three-group bar geometry.
  edfig12_bcd_swd_rates_heterozygous
                        the heterozygous counterpart, on the same y range.
  edfig12_bcd_swd_rates_assay_validation
                        wild type against the SCN2A positive control, on the
                        same y range.
  edfig12_bcd_swd_rates_example_swds
                        three representative discharge traces from the
                        positive control, read from
                        Seizure_Representative_SWDs.xlsx.
The three-group panels and the validation panel share one y maximum so their
bar heights are directly comparable.

STATISTICS. Discharge counts are not assumed normally distributed, so the
tests are rank-based throughout.
  Omnibus: Kruskal-Wallis across all six groups, and again excluding the
  positive control.
  Pairwise: seven planned contrasts, not all pairs, compared by two-sided
  Mann-Whitney U with Holm step-down correction across the seven tests. The
  correction is applied because the seven contrasts are reported as one
  table.
  Effect size: rank-biserial correlation, r = 1 - 2U / (n1 n2).
  The three-group panels apply a Kruskal-Wallis-gated Mann-Whitney
  comparison, drawing pairwise brackets only when the omnibus reaches
  p < 0.05; the validation panel applies a single Mann-Whitney U.

DISPLAY. This panel family uses its own six-color palette rather than the
three-group WT/SH/GT palette, so the script draws the multi-group figures
with local helpers. Point jitter is the deterministic center-out beeswarm
from gtstyle; it encodes local density only, and no plotted value or
statistic depends on it.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sstats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # .../python
sys.path.insert(0, os.path.dirname(HERE))                            # .../new_data_panels
from gtstyle import (gt_init, gt_beeswarm, gt_style, gt_saveFig, central_output_dir,  # noqa: E402
                     gt_barpanel_figure, gt_bar3, gt_compare3_np, gt_drawSig,
                     gt_pstar, gt_axes, gt_color_xticklabels, gt_figStyle)
from _common import holm_correct, sig_stars  # noqa: E402


def _eeg_panel_f1(G, title, out_dir, fname, ymax=2.5, ylab="Events per hour"):
    """Draw one genotype block in the pinned three-group bar geometry.

    G: [WT, SH, GT] arrays of the spike rate. ymax: display ceiling and y
    maximum. The comparison is rank-based: a Kruskal-Wallis omnibus gates
    pairwise Mann-Whitney brackets, and a 'K-W n/s' note is printed in their
    place when the omnibus does not reach p < 0.05."""
    STY = gt_style()
    fig, ax = gt_barpanel_figure(STY)
    n_off = gt_bar3(ax, G, STY, ymax=ymax)

    R = gt_compare3_np(G, "none")   # Kruskal-Wallis omnibus, Mann-Whitney pairwise
    p_kw, p_adj = R["p_anova"], R["q"]

    ax.set_ylim(0.0, ymax)
    yl = ax.get_ylim(); rng = yl[1] - yl[0]; h = 0.012 * rng
    vis_max = max([v[v <= ymax].max() for v in G if (v <= ymax).any()] or [0.0])
    base = min(max(yl[0] + 0.860 * rng, vis_max + 0.035 * rng), yl[1] - 0.055 * rng)
    if p_kw < 0.05:
        gt_drawSig(ax, 0, 1, base, h, gt_pstar(p_adj[0]), STY)
        gt_drawSig(ax, 1, 2, base, h, gt_pstar(p_adj[1]), STY)
        gt_drawSig(ax, 0, 2, base + 0.075 * rng, h, gt_pstar(p_adj[2]), STY)
    else:
        ax.text(1, base + 0.05 * rng, "K-W n/s", ha="center",
                fontsize=STY.font.ns, color=STY.clr.ns_txt, fontname=STY.font.name)

    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["WT", "SH", "GT"])
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontname=STY.font.name, fontsize=STY.font.title,
                 pad=STY.geom.title_pad)
    ylf = ax.get_ylim()
    for g in range(3):
        ax.text(g, ylf[0] - 0.205 * (ylf[1] - ylf[0]), f"n={G[g].size}",
                ha="center", fontsize=STY.font.n, color=STY.clr.n_txt,
                fontname=STY.font.name)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, fname, out_dir)
    plt.close(fig)

STY = gt_style()

DATA = os.path.join(HERE, "..", "..", "data", "edfig12", "edfig12_bcd_swd_rates")

GROUP_KEYS = ["WT_GFP", "Shank3_HOM_GFP", "Shank3_HOM_GT", "SCN2A",
              "Shank3_HET_GFP", "Shank3_HET_GT"]
GROUP_LABELS = ["WT+GFP", "Shank3$^{-/-}$+GFP", "Shank3$^{-/-}$+GT",
                "SCN2A", "Shank3$^{+/-}$+GFP", "Shank3$^{+/-}$+GT"]

COLORS = np.array([
    [0.00, 0.00, 0.00],
    [0.80, 0.00, 0.00],
    [0.20, 0.20, 0.80],
    [0.60, 0.40, 0.00],
    [0.80, 0.40, 0.40],
    [0.40, 0.40, 0.80],
])
DOT_COLORS = np.array([
    [0.72, 0.72, 0.72],
    [1.00, 0.65, 0.65],
    [0.68, 0.68, 0.92],
    [0.85, 0.75, 0.50],
    [0.90, 0.75, 0.75],
    [0.75, 0.75, 0.90],
])

# The seven planned pairwise comparisons, as index pairs into GROUP_KEYS.
COMPARISONS = [
    (0, 1, "WT vs Shank3-/- GFP     (disease effect, hom)"),
    (1, 2, "Shank3-/- GFP vs GT     (gene therapy, hom)"),
    (0, 2, "WT vs Shank3-/- GT      (WT vs treated, hom)"),
    (0, 4, "WT vs Shank3+/- GFP     (disease effect, het)"),
    (4, 5, "Shank3+/- GFP vs GT     (gene therapy, het)"),
    (0, 5, "WT vs Shank3+/- GT      (WT vs treated, het)"),
    (0, 3, "WT vs SCN2A             (positive control)"),
]


def _plot_groups(ax, data, x_positions, colors, dot_colors, labels, ttl, ylab, ylims):
    """Draw the multi-group figure: per group, the individual animals as a
    beeswarm, a horizontal rule at the mean, and a standard-error whisker
    with caps. Sample sizes are printed beneath each group."""
    bar_hw, cap_hw = 0.35, 0.18
    for gi, vals in enumerate(data):
        if vals.size == 0:
            continue
        xc = x_positions[gi]
        m = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(vals.size)

        offs = gt_beeswarm(vals, center=0.0, halfw=0.28)
        ax.scatter(xc + offs, vals, s=STY.mk.pt, c=[dot_colors[gi]], edgecolors="none", zorder=2)
        ax.plot([xc - bar_hw, xc + bar_hw], [m, m], "-", color=colors[gi], lw=2.5, zorder=3)
        ax.plot([xc, xc], [m - sem, m + sem], "-", color=colors[gi], lw=1, zorder=3)
        ax.plot([xc - cap_hw, xc + cap_hw], [m + sem, m + sem], "-", color=colors[gi], lw=1, zorder=3)
        ax.plot([xc - cap_hw, xc + cap_hw], [m - sem, m - sem], "-", color=colors[gi], lw=1, zorder=3)

    ax.set_xlim(min(x_positions) - 0.7, max(x_positions) + 0.7)
    ax.set_ylim(*ylims)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=STY.font.tick)
    # Uniform 7 pt tick and body text, matching the rest of the figure set.
    ax.tick_params(direction="out", labelsize=STY.font.tick)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(STY.lw.axis)
    ax.set_ylabel(ylab, fontsize=STY.font.lab)
    ax.set_title(ttl, fontsize=STY.font.title, fontweight="bold")

    for gi, vals in enumerate(data):
        ax.text(x_positions[gi], ylims[0] - 0.04 * (ylims[1] - ylims[0]),
                f"n={vals.size}", ha="center", fontsize=7, color=(0.6, 0.6, 0.6))


def _eeg_validation_panel(wt, scn2a, out_dir, fname, ymax=45.0):
    """Draw the assay-validation panel: wild type against the SCN2A positive
    control, as two centered bars in the pinned three-group geometry, on the
    same 0 to ymax scale as the genotype panels so the positive control's
    magnitude is legible against them. The bracket is a two-sided
    Mann-Whitney U."""
    STY = gt_style()
    fig, ax = gt_barpanel_figure(STY)
    pos = [0.5, 1.5]
    fills = [STY.cbarfill[0], np.array([0.86, 0.79, 0.56])]   # gray, olive
    scats = [STY.cscat[0], np.array([0.72, 0.60, 0.30])]
    means = [STY.cmean[0], np.array([0.55, 0.40, 0.00])]
    G = [np.asarray(wt, float).ravel(), np.asarray(scn2a, float).ravel()]
    bar_w, cap_hw = STY.geom.bar_w, STY.geom.cap_hw
    for i, v in enumerate(G):
        m = v.mean(); se = v.std(ddof=1) / np.sqrt(v.size); x = pos[i]
        xs = gt_beeswarm(v, x, STY.geom.swarm_w)
        ax.bar(x, m, width=bar_w, color=fills[i], edgecolor="none", zorder=1)
        ax.scatter(xs, v, s=STY.mk.swarm, c=[scats[i]], edgecolors="none", zorder=2)
        ax.plot([x, x], [m - se, m + se], "-", color=means[i], lw=STY.lw.se, zorder=3)
        ax.plot([x - cap_hw, x + cap_hw], [m + se, m + se], "-", color=means[i], lw=STY.lw.se, zorder=3)
        ax.plot([x - cap_hw, x + cap_hw], [m - se, m - se], "-", color=means[i], lw=STY.lw.se, zorder=3)
        ax.plot([x - bar_w / 2, x - bar_w / 2, x + bar_w / 2, x + bar_w / 2],
                [0, m, m, 0], "-", color=means[i], lw=STY.lw.bar, zorder=3)

    _, p = sstats.mannwhitneyu(G[0], G[1], alternative="two-sided")
    ax.set_ylim(0.0, ymax)
    yl = ax.get_ylim(); rng = yl[1] - yl[0]
    base = max(G[1].max(), G[0].max()) + 0.06 * rng
    gt_drawSig(ax, pos[0], pos[1], base, 0.012 * rng, gt_pstar(p), STY)

    ax.set_xticks(pos); ax.set_xticklabels(["WT", "SCN2A"])
    gt_axes(ax, STY)
    ax.get_xticklabels()[1].set_color((0.55, 0.40, 0.00))
    ax.set_ylabel("Events per hour")
    ax.set_title("Assay validation", fontname=STY.font.name,
                 fontsize=STY.font.title, pad=STY.geom.title_pad)
    ylf = ax.get_ylim()
    for i in range(2):
        ax.text(pos[i], ylf[0] - 0.205 * (ylf[1] - ylf[0]), f"n={G[i].size}",
                ha="center", fontsize=STY.font.n, color=STY.clr.n_txt,
                fontname=STY.font.name)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, fname, out_dir)
    plt.close(fig)


def plot_example_swds(data_dir=DATA, out_dir=HERE, channel="EEG EEG1A-B"):
    """Draw three representative spike-wave discharges from the positive
    control, showing what the assay counts.

    Reads sheets SWD_01 to SWD_03 of Seizure_Representative_SWDs.xlsx, each
    a 2 s trace at 1 kHz on the named channel, and stacks them on one axis
    with 0.5 s and 200 uV scale bars. Fully vector output."""
    STY = gt_style()
    xl = pd.ExcelFile(os.path.join(data_dir, "Seizure_Representative_SWDs.xlsx"))
    sheets = [s for s in ("SWD_01", "SWD_02", "SWD_03") if s in xl.sheet_names]
    traces = []
    for s in sheets:
        df = xl.parse(s)
        t = df["Time from Start"].to_numpy(float)
        t = t - t.min()
        traces.append((t, df[channel].to_numpy(float)))

    off = 950.0                                   # trace spacing, uV
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for i, (t, y) in enumerate(traces):
        base = (len(traces) - 1 - i) * off
        ax.plot(t, y + base, "-", color=(0.12, 0.12, 0.12), lw=0.35, zorder=2)
        ax.text(-0.02, base, f"SWD {i+1}", ha="right", va="center",
                fontsize=STY.font.tick, color=(0.4, 0.4, 0.4))

    # scale bars: 0.5 s horizontal, 200 uV vertical
    x0, y0 = 1.55, -off * 0.55
    ax.plot([x0, x0 + 0.5], [y0, y0], "-", color="k", lw=1.2)
    ax.plot([x0, x0], [y0, y0 + 200], "-", color="k", lw=1.2)
    ax.text(x0 + 0.25, y0 - off * 0.10, "0.5 s", ha="center", va="top", fontsize=6)
    ax.text(x0 - 0.03, y0 + 100, "200 uV", ha="right", va="center", fontsize=6)

    ax.set_xlim(-0.35, 2.05)
    ax.set_ylim(-off * 0.9, (len(traces) - 1) * off + off * 0.7)
    ax.axis("off")
    ax.set_title("SCN2A - representative SWDs", fontname=STY.font.name,
                 fontsize=STY.font.title, fontweight="bold", color=(0.4, 0.28, 0.0))
    fig.tight_layout()
    gt_saveFig(fig, "edfig12_bcd_swd_rates_example_swds", out_dir)
    plt.close(fig)
    print("  saved edfig12_bcd_swd_rates_example_swds.pdf  (representative spike traces)")


def main(data_dir=DATA, out_dir=HERE, show=False):
    gt_init()
    csv_path = os.path.join(data_dir, "EEG_data.csv")
    T = pd.read_csv(csv_path)

    data = [T[k].dropna().to_numpy(dtype=float) for k in GROUP_KEYS]

    print("EEG Spike Data Summary")
    print("=" * 62)
    for k, lbl, d in zip(GROUP_KEYS, GROUP_LABELS, data):
        print(f"  {lbl:<22s}  n={d.size}  mean={d.mean():.3f}  "
              f"med={np.median(d):.3f}  SD={d.std(ddof=1):.3f}")
    print()

    # Figure: all six groups.
    fig1, ax1 = plt.subplots(figsize=(700 / 100, 420 / 100))
    x_pos = [1, 2, 3, 4.5, 5.5, 6.5]
    _plot_groups(ax1, data, x_pos, COLORS, DOT_COLORS, GROUP_LABELS,
                 "EEG Spike Rate", "Events per hour", (0, 45))
    ax1.axvline(3.75, color=(0.85, 0.85, 0.85), lw=STY.lw.hair, zorder=0)
    ax1.text(2, 43, "Homozygous (-/-)", ha="center", fontsize=9, color=(0.5, 0.5, 0.5))
    ax1.text(5.5, 43, "Heterozygous (+/-)", ha="center", fontsize=9, color=(0.5, 0.5, 0.5))
    fig1.tight_layout()
    gt_saveFig(fig1, "edfig12_bcd_swd_rates_all_groups", out_dir, also_eps=True, also_pdf=True)
    fig1.savefig(os.path.join(central_output_dir(out_dir), "edfig12_bcd_swd_rates_all_groups.png"), dpi=300)

    # Figure: the same groups without the positive control, on a y range that
    # resolves the remainder.
    idx_no_scn2a = [0, 1, 2, 4, 5]
    data_zoom = [data[i] for i in idx_no_scn2a]
    labels_zoom = [GROUP_LABELS[i] for i in idx_no_scn2a]
    colors_zoom = COLORS[idx_no_scn2a]
    dots_zoom = DOT_COLORS[idx_no_scn2a]
    x_pos_zoom = [1, 2, 3, 4.5, 5.5]

    fig2, ax2 = plt.subplots(figsize=(620 / 100, 420 / 100))
    _plot_groups(ax2, data_zoom, x_pos_zoom, colors_zoom, dots_zoom, labels_zoom,
                 "EEG Spike Rate (excl. SCN2A)", "Events per hour", (0, 2.5))
    ax2.axvline(3.75, color=(0.85, 0.85, 0.85), lw=STY.lw.hair, zorder=0)
    ax2.text(2, 2.35, "Homozygous (-/-)", ha="center", fontsize=9, color=(0.5, 0.5, 0.5))
    ax2.text(5, 2.35, "Heterozygous (+/-)", ha="center", fontsize=9, color=(0.5, 0.5, 0.5))
    fig2.tight_layout()
    gt_saveFig(fig2, "edfig12_bcd_swd_rates_no_SCN2A", out_dir, also_eps=True, also_pdf=True)
    fig2.savefig(os.path.join(central_output_dir(out_dir), "edfig12_bcd_swd_rates_no_SCN2A.png"), dpi=300)

    # Figures: the homozygous and heterozygous blocks in the pinned
    # three-group geometry, plus the assay-validation panel. The positive
    # control is not one of the three groups in either genotype panel; it
    # appears in the validation panel and in the all-groups panel above.
    G_hom = [data[0], data[1], data[2]]          # WT, Shank3-/- GFP, Shank3-/- GT
    G_het = [data[0], data[4], data[5]]          # WT, Shank3+/- GFP, Shank3+/- GT
    ymax = 40.0                                  # y maximum shared by all three panels
    _eeg_panel_f1(G_hom, "Homozygous (-/-)", out_dir,
                  "edfig12_bcd_swd_rates_homozygous", ymax=ymax)
    _eeg_panel_f1(G_het, "Heterozygous (+/-)", out_dir,
                  "edfig12_bcd_swd_rates_heterozygous", ymax=ymax)
    _eeg_validation_panel(data[0], data[3], out_dir,
                          "edfig12_bcd_swd_rates_assay_validation", ymax=ymax)
    print("  saved edfig12_bcd_swd_rates_homozygous.pdf / "
          "edfig12_bcd_swd_rates_heterozygous.pdf / "
          "edfig12_bcd_swd_rates_assay_validation.pdf  (F1 style)")
    plot_example_swds(data_dir, out_dir)

    # Statistics
    print("\n" + "=" * 62)
    print("  STATISTICAL ANALYSIS")
    print("=" * 62)

    all_vals = np.concatenate(data)
    all_grp = np.concatenate([np.full(d.size, gi) for gi, d in enumerate(data)])
    H, p_kw = sstats.kruskal(*data)
    print(f"\n  Kruskal-Wallis (all 6 groups): chi2={H:.2f}, p={p_kw:.6f}")

    idx_no = all_grp != 3  # the positive control is group index 3
    groups_no = [all_vals[(all_grp == g) & idx_no] for g in range(6) if g != 3]
    H2, p_kw2 = sstats.kruskal(*groups_no)
    print(f"  Kruskal-Wallis (excl. SCN2A):  chi2={H2:.2f}, p={p_kw2:.6f}")

    print("\n  Planned Pairwise Comparisons (Mann-Whitney U, Holm-corrected)")
    print("  " + "-" * 63)

    raw_p = np.zeros(len(COMPARISONS))
    Us = np.zeros(len(COMPARISONS))
    for ci, (g1, g2, label) in enumerate(COMPARISONS):
        x, y = data[g1], data[g2]
        U, p = sstats.mannwhitneyu(x, y, alternative="two-sided")
        Us[ci] = U
        raw_p[ci] = p

    holm_p = holm_correct(raw_p)

    print(f"  {'Comparison':<40s}  {'U-stat':>8s}  {'p(raw)':>8s}  {'p(Holm)':>8s}  {'Sig':>4s}")
    for ci, (g1, g2, label) in enumerate(COMPARISONS):
        # Labels use the Holm-corrected value: the seven planned comparisons
        # are reported as one table, so a multiplicity correction applies.
        sig = sig_stars(holm_p[ci])
        print(f"  {label:<40s}  {Us[ci]:8.1f}  {raw_p[ci]:8.4f}  {holm_p[ci]:8.4f}  {sig:>4s}")

    print("\n  Effect Sizes (rank-biserial correlation)")
    print("  " + "-" * 43)
    for ci, (g1, g2, label) in enumerate(COMPARISONS):
        n1, n2 = data[g1].size, data[g2].size
        r = 1 - 2 * Us[ci] / (n1 * n2)
        mag = "large" if abs(r) > 0.5 else ("medium" if abs(r) > 0.3 else "small")
        print(f"  {label:<40s}  r={r:.3f}  ({mag})")

    print("\n" + "=" * 62)
    print(f"  Done. Figures saved to {out_dir}/")
    print("=" * 62)

    if not show:
        plt.close(fig1)
        plt.close(fig2)
    return fig1, fig2


if __name__ == "__main__":
    main()
