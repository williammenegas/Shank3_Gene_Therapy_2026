"""
edfig13_e_strength.py -- Extended Data Fig. 13, panel e.

Muscle strength and body weight in WT, SH and GT animals.

DATA. weight_data.mat (body weight) and strength_data.mat (pull strength),
each holding one NaN-padded vector per group; the script strips the padding
per group on load.

WHAT IS PLOTTED. Three box + scatter panels, WT against SH against GT:
    edfig13_e_strength_body_weight       body weight, g
    edfig13_e_strength_pull              pull strength, g
    edfig13_e_strength_ratio             strength-to-weight ratio, x10^3
Each shows a box from the first to the third quartile with a median rule,
whiskers, and the individual animals as jittered points.

The strength-to-weight ratio pairs pull strength with body weight by position
within the padded vectors, since the source files carry no animal identifier.
Index i in one is the same animal as index i in the other. A weight of zero is
treated as missing. The ratio is scaled by 1000 for readability.

STATISTICS. Each panel applies the three-group comparison with brackets from
gt_boxsig: Levene's test for homogeneity of variance, then one-way ANOVA with
Tukey-Kramer when equal variance holds or Welch's ANOVA with Games-Howell
when it is rejected, with pairwise brackets drawn only when the omnibus
reaches p < 0.05.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import *  # noqa: E402,F403  (box + scatter panel helpers)
from gtstyle import gt_init, loadmat, gt_style  # noqa: E402
from _common import GROUP_COLORS  # noqa: E402

STY = gt_style()

DATA = os.path.join(HERE, "..", "..", "data", "edfig13", "edfig13_e_strength")
GROUP_LABELS = ["WT", "SH", "GT"]
GROUP_COLORS_ARR = np.vstack([GROUP_COLORS["WT"], GROUP_COLORS["SH"], GROUP_COLORS["GT"]])


def _clean(v):
    """Flatten a padded group vector and drop its NaN padding."""
    v = np.ravel(np.asarray(v, dtype=float))
    return v[~np.isnan(v)]


def _compute_ratio(pull_vec, weight_vec):
    """Strength-to-weight ratio, scaled by 1000.

    Pull strength and body weight are paired by position within the padded
    vectors, since the source files carry no animal identifier. Entries
    missing from either vector are dropped, and a weight of zero is treated
    as missing."""
    pull_vec = np.ravel(np.asarray(pull_vec, dtype=float))
    weight_vec = np.ravel(np.asarray(weight_vec, dtype=float))
    both_valid = ~np.isnan(pull_vec) & ~np.isnan(weight_vec)
    p = pull_vec[both_valid]
    w = weight_vec[both_valid].copy()
    w[w == 0] = np.nan
    ratio = (p / w) * 1000.0
    return ratio[~np.isnan(ratio)]


def main(data_dir=DATA, out_dir=HERE, rng_seed=7, show=False):
    gt_init()
    rng = np.random.default_rng(rng_seed)

    W = loadmat(os.path.join(data_dir, "weight_data.mat"))
    P = loadmat(os.path.join(data_dir, "strength_data.mat"))

    weight = [_clean(W["weight_mat_wt"]), _clean(W["weight_mat_sh"]), _clean(W["weight_mat_gt"])]
    pull = [_clean(P["pull_norm_wt"]), _clean(P["pull_norm_sh"]), _clean(P["pull_norm_gt"])]
    ratio = [
        _compute_ratio(P["pull_norm_wt"], W["weight_mat_wt"]),
        _compute_ratio(P["pull_norm_sh"], W["weight_mat_sh"]),
        _compute_ratio(P["pull_norm_gt"], W["weight_mat_gt"]),
    ]

    print("Group sizes:")
    print(f"  Weight:   WT n={weight[0].size}, SH n={weight[1].size}, GT n={weight[2].size}")
    print(f"  Strength: WT n={pull[0].size}, SH n={pull[1].size}, GT n={pull[2].size}")
    print(f"  Ratio:    WT n={ratio[0].size}, SH n={ratio[1].size}, GT n={ratio[2].size}")

    # One box + scatter panel per measure.
    endpoints = [
        (weight, "Body weight (g)", "edfig13_e_strength_body_weight"),
        (pull, "Pull strength (g)", "edfig13_e_strength_pull"),
        (ratio, "Strength : weight (ratio)", "edfig13_e_strength_ratio"),
    ]
    os.makedirs(out_dir, exist_ok=True)
    for grp_vals, ylab, name in endpoints:
        allv = np.concatenate([g for g in grp_vals if g.size])
        lo, hi = float(allv.min()), float(allv.max())
        span = (hi - lo) or 1.0
        fig, ax = gt_boxscatter_figure(STY)
        gt_colorboxplot(ax, list(grp_vals), STY, rng=rng)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(GROUP_LABELS)
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(lo - 0.10 * span, hi + 0.30 * span)
        gt_boxsig(ax, STY)          # three-group comparison and brackets
        ax.set_ylabel(ylab, fontname=STY.font.name, fontsize=STY.font.lab)
        gt_axes(ax, STY)
        gt_color_xticklabels(ax, STY)
        gt_figStyle(fig, STY)
        gt_saveFig(fig, name, out_dir)
        print(f"  saved {name}.pdf")
        if not show:
            plt.close(fig)
    print("\nDone.")
    return None


if __name__ == "__main__":
    main()
