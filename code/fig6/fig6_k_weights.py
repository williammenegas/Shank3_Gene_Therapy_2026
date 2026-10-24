"""
fig6_k_weights.py

Figure 6, panel k: body weight against age in SH and GT marmosets, as a
growth and general-health readout.

Data. marmoset_weights.mat, holding the struct array `animals` (19 animals,
each with uid, group in {SH, GT}, age_months and weight_g). The plotted
window is 12 to 48 months, and the plotted points are the raw per-visit
age and weight pairs.

This script writes two figures:
  1. fig6_k_weights_SH_vs_GT: every animal's trajectory drawn faintly, with the
     group mean and a band of +/- one standard deviation computed in 3-month
     bins; a bin needs at least two weighings to contribute a standard
     deviation, and single-weighing bins are drawn with zero width. SH and GT
     are overlaid on one axis.
  2. fig6_k_weights_boxplots_12_48mo: SH versus GT boxes (median, quartiles,
     whiskers at 1.5*IQR) of each animal's weighing nearest to 12 months and
     nearest to 48 months, in two panels that share y limits, with every
     animal drawn as a jittered point.

No statistical test is applied; the panel is a descriptive growth comparison.
Point jitter uses a seeded generator (rng_seed), so the drawn offsets are
reproducible.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import central_output_dir  # noqa: E402
from gtstyle import gt_init, loadmat, gt_style  # noqa: E402

STY = gt_style()

DATA = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_k_weights")

GROUPS = ["SH", "GT"]
GCOLORS = {"SH": np.array([0.839, 0.153, 0.157]), "GT": np.array([0.122, 0.467, 0.706])}


def main(data_dir=DATA, out_dir=central_output_dir(HERE), rng_seed=7, show=False):
    gt_init()
    rng = np.random.default_rng(rng_seed)

    S = loadmat(os.path.join(data_dir, "marmoset_weights.mat"))
    animals = S["animals"]
    if not isinstance(animals, list):
        animals = [animals]

    groups_of = np.array([str(a["group"]) for a in animals])

    # Figure 1: SH and GT trajectories with the group mean +/- SD band.
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for g in GROUPS:
        c = GCOLORS[g]
        idx = np.flatnonzero(groups_of == g)
        all_age, all_w = [], []
        first = True
        for k in idx:
            a = animals[k]
            age = np.ravel(np.asarray(a["age_months"], dtype=float))
            w = np.ravel(np.asarray(a["weight_g"], dtype=float))
            order = np.argsort(age)
            age, w = age[order], w[order]
            all_age.append(age); all_w.append(w)
            ax.plot(age, w, "-", color=(*c, 0.40), lw=STY.lw.trace,
                    label=(g if first else None))
            first = False
        all_age = np.concatenate(all_age)
        all_w = np.concatenate(all_w)

        bin_w = 3
        edges = np.arange(12, 48 + 1, bin_w)
        ctrs = edges[:-1] + bin_w / 2
        mu = np.full(ctrs.size, np.nan)
        sd = np.full(ctrs.size, np.nan)
        for b in range(ctrs.size):
            inb = (all_age >= edges[b]) & (all_age < edges[b + 1])
            if inb.sum() >= 2:
                mu[b] = all_w[inb].mean()
                sd[b] = all_w[inb].std(ddof=1)
            elif inb.sum() == 1:
                mu[b] = all_w[inb].mean()
                sd[b] = 0
        v = ~np.isnan(mu)
        cc, mm, ss = ctrs[v].copy(), mu[v], sd[v]
        cc[0] = all_age.min()
        cc[-1] = all_age.max()

        ax.fill(np.concatenate([cc, cc[::-1]]),
                np.concatenate([mm + ss, (mm - ss)[::-1]]), color=c, alpha=0.15,
                edgecolor="none")
        ax.plot(cc, mm, "-", color=c, lw=3, label=f"{g} mean $\\pm$ SD")

    ax.set_xlim(12, 48)
    ax.set_xlabel("Age (months)", fontsize=STY.font.lab)
    ax.set_ylabel("Weight (g)", fontsize=STY.font.lab)
    ax.set_title("Marmoset Weight vs Age -- SH vs GT", fontsize=STY.font.title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, "fig6_k_weights_SH_vs_GT.png")
    fig.savefig(p1, dpi=200, facecolor="w")
    fig.savefig(p1.replace(".png", ".pdf"), facecolor="w")   # vector PDF for figure assembly
    print(f"saved {p1}")

    # Figure 2: boxes at 12 and 48 months, nearest weighing per animal.
    targets = [12, 48]
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 6))
    for ti, t in enumerate(targets):
        ax2 = axes2[ti]
        vals_by_g = {g: [] for g in GROUPS}
        for g in GROUPS:
            idx = np.flatnonzero(groups_of == g)
            for k in idx:
                a = animals[k]
                age = np.ravel(np.asarray(a["age_months"], dtype=float))
                w = np.ravel(np.asarray(a["weight_g"], dtype=float))
                j = np.argmin(np.abs(age - t))
                vals_by_g[g].append(w[j])

        for gi, g in enumerate(GROUPS):
            x = gi + 1
            v = np.array(vals_by_g[g])
            c = GCOLORS[g]
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            iqr = q3 - q1
            whi = min(v.max(), q3 + 1.5 * iqr)
            wlo = max(v.min(), q1 - 1.5 * iqr)
            hw = 0.3
            ax2.add_patch(matplotlib.patches.Rectangle(
                (x - hw, q1), 2 * hw, q3 - q1, facecolor=c, alpha=0.45,
                edgecolor="none"))
            ax2.plot([x - hw, x + hw], [med, med], "k-", lw=1.5)
            ax2.plot([x, x], [wlo, q1], "k-", lw=1)
            ax2.plot([x, x], [q3, whi], "k-", lw=1)
            xj = x + (rng.random(v.size) - 0.5) * 0.12
            ax2.scatter(xj, v, s=STY.mk.pt, facecolors=[c], edgecolors="k", zorder=5)

        ax2.set_title(f"{t // 12} year", fontweight="bold")
        if ti == 0:
            ax2.set_ylabel("Weight (g)")
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(GROUPS)
        ax2.grid(True, alpha=0.3)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)

    ylims = [ax.get_ylim() for ax in axes2]
    ylo = min(y[0] for y in ylims)
    yhi = max(y[1] for y in ylims)
    for ax2 in axes2:
        ax2.set_ylim(ylo, yhi)

    fig2.suptitle("SH vs GT -- nearest weight at 12 and 48 months", fontsize=STY.font.title,
                 fontweight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    p2 = os.path.join(out_dir, "fig6_k_weights_boxplots_12_48mo.png")
    fig2.savefig(p2, dpi=200, facecolor="w")
    fig2.savefig(p2.replace(".png", ".pdf"), facecolor="w")   # vector PDF for figure assembly
    print(f"saved {p2}")

    if not show:
        plt.close(fig)
        plt.close(fig2)
    return fig, fig2


if __name__ == "__main__":
    main()
