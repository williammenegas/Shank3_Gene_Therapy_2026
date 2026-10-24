"""
fig6_h_shank3_rna.py

Figure 6, panel h: total SHANK3 expression in AAV-positive versus
AAV-negative cells.

Data. total_shank3_per_cell.csv, one row per filtered single cell or nucleus.
total_shank3_umis is the sum of endogenous and miniSHANK3 UMI counts, and
aav_status is AAV+ or AAV-, defined by a WPRE3_hGH UMI count above zero.
Cells are analyzed per animal (Ravioli, Ezra) and pooled across animals
("All").

Statistics. A one-sided Welch t-test per group of mean(AAV+) > mean(AAV-),
with Welch-Satterthwaite degrees of freedom. Benjamini-Hochberg FDR is
applied across the two animal-level tests; the pooled column carries its
uncorrected p-value. Labels use the house three-tier significance scale.

This script writes three figures:
  1. Grouped bars, AAV- left and AAV+ right within each group, with standard
     error bars, jittered individual cells (AAV- subsampled to 300 points per
     group for legibility) and significance brackets.
  2. Kernel-density panels, one per group, with density on the x axis and UMI
     count on the y axis, a Gaussian kernel of bandwidth 1.5, and group means
     marked by dashed lines.
  3. Combined violin panel: per-animal AAV- and AAV+ kernel densities with a
     point swarm (subsampled to 400 points) and a mean with standard error.

UMI counts are clipped at 10 for the density estimates and the y axes; the
statistics use the unclipped values. Point subsampling and jitter use a
seeded generator (rng_seed), so the drawn points are reproducible.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import central_output_dir  # noqa: E402
from gtstyle import gt_init, gt_style  # noqa: E402
from _common import welch_t_one_sided, bh_correct, sig_stars  # noqa: E402
from gtstyle import gt_pstar  # noqa: E402

STY = gt_style()

DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_h_shank3_rna")

AAV_NEG_COL = np.array([0.75, 0.75, 0.75])
AAV_POS_COL = np.array([0.15, 0.65, 0.15])


def _sig_label_4(p):
    """House significance label: three tiers (0.05, 0.01, 0.001), a '0.05'
    label in the borderline band and 'n/s' otherwise.
    """
    return gt_pstar(p)


def _kde_smooth(vals, x_eval, bw):
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    d = np.zeros(x_eval.size)
    for j, xe in enumerate(x_eval):
        d[j] = np.sum(np.exp(-0.5 * ((vals - xe) / bw) ** 2)) / (n * bw * np.sqrt(2 * np.pi))
    return d


def _violin_combined(disp_names, raw_neg, raw_pos, mean_neg, mean_pos,
                     sem_neg, sem_pos, out_dir, rng, STY):
    """Combined panel: per-animal AAV- and AAV+ kernel-density violins with a
    point swarm and a mean with standard error, one group per column."""
    nG = len(disp_names)
    y_eval = np.linspace(0, 10, 300)
    bw = 1.5
    HALF, OFF = 0.30, 0.24          # violin max half-width, AAV-/+ offset
    centers = np.arange(1, nG + 1)

    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    xticks, xticklabels = [], []
    for i in range(nG):
        for sign, vals, m, se, col in (
                (-1, raw_neg[i], mean_neg[i], sem_neg[i], AAV_NEG_COL),
                (+1, raw_pos[i], mean_pos[i], sem_pos[i], AAV_POS_COL)):
            c = centers[i] + sign * OFF
            v = np.minimum(vals, 10.0)
            d = _kde_smooth(v, y_eval, bw)
            w = HALF * d / d.max() if d.max() > 0 else d * 0.0
            # violin (mirrored KDE)
            ax.fill(np.concatenate([c + w, (c - w)[::-1]]),
                    np.concatenate([y_eval, y_eval[::-1]]),
                    facecolor=tuple(col), alpha=0.5, edgecolor=tuple(col * 0.6),
                    lw=0.8, zorder=1)
            # points, jittered within the violin width
            vv = vals.copy()
            if vv.size > 400:
                vv = rng.choice(vv, 400, replace=False)
            wv = np.interp(np.minimum(vv, 10.0), y_eval, w)
            xj = c + (rng.random(vv.size) - 0.5) * 2 * wv * 0.8
            ax.scatter(xj, vv, s=2.5, c=[col * 0.75], alpha=0.30,
                       edgecolors="none", zorder=2)
            # mean +/- SEM
            ax.plot([c, c], [m - se, m + se], color="k", lw=1.0, zorder=3)
            ax.plot(c, m, "o", ms=5.5, mfc="k", mec="w", mew=0.6, zorder=4)
            xticks.append(c)
            xticklabels.append("-" if sign < 0 else "+")
        # group label + underline above the group (below the title)
        ax.text(centers[i], 10.35, disp_names[i], ha="center", va="bottom",
                fontsize=STY.font.title, clip_on=False)
        ax.plot([centers[i] - 0.42, centers[i] + 0.42], [10.2, 10.2], "k-",
                lw=0.8, clip_on=False)

    ax.set_ylim(0, 10)
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlim(0.45, nG + 0.55)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=STY.font.tick)
    ax.set_ylabel("SHANK3 UMIs per filtered cell", fontsize=STY.font.lab)
    ax.text((nG + 1) / 2.0, 11.35, "Total SHANK3 expression", ha="center",
            va="bottom", fontsize=STY.font.title, clip_on=False)
    ax.tick_params(direction="out", labelsize=STY.font.tick, length=2.7, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(STY.lw.axis)
    ax.text(-0.01, -0.055, "AAV:", transform=ax.transAxes, ha="right", va="top",
            fontsize=STY.font.tick)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig6_h_shank3_rna_violin_by_animal.png")
    fig.savefig(p, dpi=300, facecolor="w", bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), facecolor="w", bbox_inches="tight")
    print(f"saved {p}")
    return fig


def main(data_dir=DATA, out_dir=central_output_dir(HERE), rng_seed=7, show=False):
    gt_init()
    rng = np.random.default_rng(rng_seed)

    T = pd.read_csv(os.path.join(data_dir, "total_shank3_per_cell.csv"))
    import re
    T["animal"] = T["sample"].apply(lambda s: re.match(r"^\d+_([A-Za-z]+)_", s).group(1))

    group_names = ["Ravioli", "Ezra", "All"]
    nG = len(group_names)
    raw_pos, raw_neg = [], []
    for g in group_names:
        mask = np.ones(len(T), dtype=bool) if g == "All" else (T["animal"] == g).to_numpy()
        raw_pos.append(T.loc[mask & (T["aav_status"] == "AAV+"), "total_shank3_umis"].to_numpy(float))
        raw_neg.append(T.loc[mask & (T["aav_status"] == "AAV-"), "total_shank3_umis"].to_numpy(float))

    p_raw = np.zeros(nG)
    for i in range(nG):
        _, _, p_raw[i] = welch_t_one_sided(raw_pos[i], raw_neg[i])

    p_animal_bh = bh_correct(p_raw[:2])
    p_adj = np.concatenate([p_animal_bh, [p_raw[2]]])
    sig_lab = [_sig_label_4(p) for p in p_adj]

    mean_pos = np.array([v.mean() for v in raw_pos])
    sem_pos = np.array([v.std(ddof=1) / np.sqrt(v.size) for v in raw_pos])
    mean_neg = np.array([v.mean() for v in raw_neg])
    sem_neg = np.array([v.std(ddof=1) / np.sqrt(v.size) for v in raw_neg])

    xtl = [f"{g}\nn={raw_pos[i].size + raw_neg[i].size} (AAV+ n={raw_pos[i].size})"
           for i, g in enumerate(group_names)]

    print(f"\n{'Group':<12s}  {'n+':>6s}  {'n-':>6s}  {'mean+':>8s}  {'mean-':>8s}  "
          f"{'p_adj':>10s}  {'p_raw':>10s}  {'sig':>6s}")
    for i, g in enumerate(group_names):
        print(f"{g:<12s}  {raw_pos[i].size:6d}  {raw_neg[i].size:6d}  "
              f"{mean_pos[i]:8.3f}  {mean_neg[i]:8.3f}  {p_adj[i]:10.2e}  "
              f"{p_raw[i]:10.2e}  {sig_lab[i]:>6s}")

    # Figure 1: grouped bars with standard error, points and brackets.
    fig, ax = plt.subplots(figsize=(7.5, 6))
    barW = 0.30
    xpos = np.arange(1, nG + 1)

    for i in range(nG):
        ax.add_patch(matplotlib.patches.Rectangle(
            (xpos[i] - barW / 2 - barW / 2, 0), barW, mean_neg[i],
            facecolor=AAV_NEG_COL, alpha=0.65, edgecolor="k", lw=0.8))
        ax.add_patch(matplotlib.patches.Rectangle(
            (xpos[i] + barW / 2 - barW / 2, 0), barW, mean_pos[i],
            facecolor=AAV_POS_COL, alpha=0.65, edgecolor="k", lw=0.8))
        ax.errorbar(xpos[i] - barW / 2, mean_neg[i], yerr=sem_neg[i], color="k",
                    lw=1.2, capsize=5, ls="none")
        ax.errorbar(xpos[i] + barW / 2, mean_pos[i], yerr=sem_pos[i], color="k",
                    lw=1.2, capsize=5, ls="none")

    maxPts = 300
    for i in range(nG):
        vn = raw_neg[i]
        if vn.size > maxPts:
            vn = rng.choice(vn, maxPts, replace=False)
        xj = xpos[i] - barW / 2 + (rng.random(vn.size) - 0.5) * barW * 0.8
        ax.scatter(xj, vn, s=STY.mk.pt, c=[AAV_NEG_COL * 0.8], alpha=0.20, edgecolors="none")

        vp = raw_pos[i]
        xj = xpos[i] + barW / 2 + (rng.random(vp.size) - 0.5) * barW * 0.8
        ax.scatter(xj, vp, s=STY.mk.pt, c=[AAV_POS_COL * 0.85], alpha=0.35, edgecolors="none")

    for i in range(nG):
        y_top = max(mean_neg[i] + sem_neg[i], mean_pos[i] + sem_pos[i])
        y_raw_max = max(raw_pos[i].max(), raw_neg[i].max())
        y_bracket = max(y_top, y_raw_max) + 1.5
        xL, xR = xpos[i] - barW / 2, xpos[i] + barW / 2
        ax.plot([xL, xL, xR, xR], [y_bracket - 0.4, y_bracket, y_bracket, y_bracket - 0.4],
                "k-", lw=1.0)
        ax.text(xpos[i], y_bracket + 0.6, sig_lab[i], ha="center", fontsize=11)

    ax.set_ylim(0, 10)
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlim(0.4, nG + 0.6)
    ax.set_xticks(xpos)
    # Uniform tick and body text size across the panel set.
    ax.set_xticklabels(xtl, fontsize=STY.font.tick)
    ax.tick_params(direction="out", labelsize=STY.font.tick)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(STY.lw.axis)
    ax.set_ylabel("Mean total SHANK3 UMIs per filtered cell", fontsize=STY.font.lab)
    ax.set_title("Total SHANK3 expression in WPRE3_hGH-defined AAV+ vs AAV- cells", fontsize=STY.font.title)

    h_neg = matplotlib.patches.Patch(facecolor=AAV_NEG_COL, alpha=0.65, edgecolor="k",
                                     label="AAV- (WPRE3_hGH = 0)")
    h_pos = matplotlib.patches.Patch(facecolor=AAV_POS_COL, alpha=0.65, edgecolor="k",
                                     label="AAV+ (WPRE3_hGH > 0)")
    ax.legend(handles=[h_neg, h_pos], loc="upper left", fontsize=9, frameon=False)

    fig.text(0.5, 0.01,
             "One-sided Welch t-test; stars show BH-adjusted p-values for animal-level "
             "tests and raw p-value for pooled column (* <0.05, ** <0.01, *** <0.001, "
             "**** <0.0001)", fontsize=8, ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, "fig6_h_shank3_rna_by_animal.png")
    fig.savefig(p1, dpi=200, facecolor="w")
    fig.savefig(p1.replace(".png", ".pdf"), facecolor="w")   # vector PDF for figure assembly
    print(f"\nsaved {p1}")

    # Figure 2: kernel-density panels, one per group.
    all_vals = np.concatenate(raw_pos + raw_neg)
    fig2, axes2 = plt.subplots(1, nG, figsize=(12, 5))
    y_eval = np.linspace(0, 10, 300)
    kde_bw = 1.5

    for i in range(nG):
        ax2 = axes2[i]
        vn = np.minimum(raw_neg[i], 10)
        vp = np.minimum(raw_pos[i], 10)
        d_neg = _kde_smooth(vn, y_eval, kde_bw)
        d_pos = _kde_smooth(vp, y_eval, kde_bw)

        ax2.fill(np.concatenate([d_neg, np.zeros_like(y_eval)]),
                 np.concatenate([y_eval, y_eval[::-1]]), color=AAV_NEG_COL, alpha=0.45,
                 edgecolor=AAV_NEG_COL * 0.6, lw=1.4)
        ax2.fill(np.concatenate([d_pos, np.zeros_like(y_eval)]),
                 np.concatenate([y_eval, y_eval[::-1]]), color=AAV_POS_COL, alpha=0.45,
                 edgecolor=AAV_POS_COL * 0.6, lw=1.4)
        ax2.axhline(raw_neg[i].mean(), ls="--", color=AAV_NEG_COL * 0.6, lw=1.3)
        ax2.axhline(raw_pos[i].mean(), ls="--", color=AAV_POS_COL * 0.6, lw=1.3)

        n_total = raw_pos[i].size + raw_neg[i].size
        ax2.set_title(f"{group_names[i]}  (n={n_total}, AAV+ n={raw_pos[i].size})", fontsize=STY.font.title)
        ax2.set_xlabel("Density", fontsize=STY.font.lab)
        if i == 0:
            ax2.set_ylabel("Total SHANK3 UMIs per cell", fontsize=STY.font.lab)
        ax2.set_ylim(0, 10)
        # Uniform tick and body text size across the panel set.
        ax2.tick_params(direction="out", labelsize=STY.font.tick)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax2.spines[s].set_linewidth(STY.lw.axis)
        xl = ax2.get_xlim()
        ax2.text(xl[1] * 0.7, 9.2, sig_lab[i], fontsize=13, fontweight="bold")

    axes2[-1].legend(["AAV-", "AAV+"], loc="upper right", fontsize=9, frameon=False)
    fig2.suptitle("Total SHANK3 UMI distribution: AAV+ vs AAV-", fontsize=STY.font.title)
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    p2 = os.path.join(out_dir, "fig6_h_shank3_rna_kde_by_animal.png")
    fig2.savefig(p2, dpi=200, facecolor="w")
    fig2.savefig(p2.replace(".png", ".pdf"), facecolor="w")   # vector PDF for figure assembly
    print(f"saved {p2}")

    # Figure 3: combined violin, point swarm and mean panel.
    disp_names = {"Ravioli": "Animal R", "Ezra": "Animal E", "All": "Pooled"}
    disp = [disp_names.get(g, g) for g in group_names]
    fig3 = _violin_combined(disp, raw_neg, raw_pos, mean_neg, mean_pos,
                            sem_neg, sem_pos, out_dir, rng, STY)

    if not show:
        plt.close(fig)
        plt.close(fig2)
        plt.close(fig3)
    return fig, fig2, fig3


if __name__ == "__main__":
    main()
