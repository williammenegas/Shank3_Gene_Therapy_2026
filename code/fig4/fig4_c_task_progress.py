"""
fig4_c_task_progress.py

Figure 4, panel c: per-task reward rate for the five touchscreen task types
and for the average of tasks 4 and 5. This script produces one figure per
group (WT, SH, GT).

Data. t_efficiency_type1_data.mat through t_efficiency_type5_data.mat, each
holding eff_wt / eff_sh / eff_gt as 1000-slot NaN-padded per-animal vectors.
Every task is centered by subtracting that task's own WT mean, so the plotted
quantity is reward rate relative to WT on the same task.

Display. The script draws tasks 1-5 as jittered individual animals, a Q1-Q3
error bar in the group color, and a mean dot. A sixth column, "Avg (4+5)",
shows the per-animal mean of the centered tasks 4 and 5 as a box (Q1, median,
Q3, whiskers to the furthest point within 1.5*IQR) with the same jittered
points and mean dot.

Statistics. Per task, a one-sided one-sample t-test of mean < 0, uncorrected
across tasks, marked with a single '#' when p < HASH_ALPHA (0.02). The glyph
is ungraded and uses '#' rather than '*' to distinguish this test against
zero from the pairwise tests used in other panels. WT is centered on its own
mean and is therefore the reference.

Point jitter is unseeded, so the horizontal offsets of individual animals
differ slightly between runs; the plotted values and statistics do not.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_c_task_progress")

TYPE_FILES = [f"t_efficiency_type{t}_data.mat" for t in range(1, 6)]
YLIM = (-0.4, 0.4)
# Marker sizes follow the shared house style: STY.mk.pt for individual
# animals, STY.mk.mean for the group mean dot.


HASH_ALPHA = 0.02   # single-tier significance cutoff for the per-task test


def _hash_label(p):
    """Significance glyph for the per-task one-sided test: a single '#' when
    p < HASH_ALPHA, an empty string otherwise. The glyph is ungraded."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "#" if p < HASH_ALPHA else ""


def _ttest1_below_zero(y):
    """One-sided, uncorrected one-sample t-test that mean(y) < 0.

    p = P(T_{n-1} <= t_obs) with t_obs = mean / (sd / sqrt(n)), evaluated
    with gt_tcdf. Returns NaN for n < 2; for sd == 0 the result degenerates
    to 0 or 1 according to the sign of the mean."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = y.size
    if n < 2:
        return np.nan
    sd = y.std(ddof=1)
    if sd == 0:
        return 0.0 if y.mean() < 0 else 1.0
    t = y.mean() / (sd / np.sqrt(n))
    # gt_tcdf returns the CDF at |x| (a value in [0.5, 1]), so the lower-tail
    # one-sided p = P(T_{n-1} <= t) is assembled as:
    #   t <  0 (mean below 0): p = 1 - CDF(|t|)
    #   t >= 0               : p = CDF(t)
    cdf_abs = gt_tcdf(abs(t), n - 1)
    return (1.0 - cdf_abs) if t < 0 else cdf_abs


def _draw_single_box(ax, y, x_center, box_color, STY):
    """Box summary for one column: Q1-Q3 patch, median line, and whiskers to
    the furthest observation within 1.5*IQR of the box."""
    y = y[~np.isnan(y)]
    if y.size == 0:
        return
    q1, med, q3 = np.percentile(y, [25, 50, 75])
    iqr = q3 - q1
    wlen = 1.5 * iqr
    y_min = max(y.min(), q1 - wlen)
    y_max = min(y.max(), q3 + wlen)

    box_hw = 0.22
    cap_hw = 0.10

    ax.add_patch(plt.Rectangle((x_center - box_hw, q1), 2 * box_hw, q3 - q1,
                                facecolor=box_color, edgecolor="none", alpha=0.45, zorder=2))
    ax.plot([x_center - box_hw, x_center + box_hw], [med, med], "k-", lw=1.5, zorder=3)
    ax.plot([x_center - box_hw, x_center + box_hw], [q1, q1], "k-", lw=0.6, zorder=3)
    ax.plot([x_center - box_hw, x_center + box_hw], [q3, q3], "k-", lw=0.6, zorder=3)
    ax.plot([x_center, x_center], [y_min, q1], "k-", lw=0.6, zorder=3)
    ax.plot([x_center, x_center], [q3, y_max], "k-", lw=0.6, zorder=3)
    ax.plot([x_center - cap_hw, x_center + cap_hw], [y_min, y_min], "k-", lw=0.6, zorder=3)
    ax.plot([x_center - cap_hw, x_center + cap_hw], [y_max, y_max], "k-", lw=0.6, zorder=3)


def _plot_group_tasks(M, group_label, pt_color, box_color, ylim, STY, rng):
    """Draw one group's panel. M is (nAnimals, 5); columns are tasks 1-5."""
    with np.errstate(invalid="ignore"):
        avg_vals = np.nanmean(M[:, 3:5], axis=1)   # tasks 4 and 5 (0-based cols 3,4)

    fig, ax = plt.subplots(figsize=(1.990, 1.590), facecolor="w")

    QCAP_HW = 0.10          # quartile-whisker end-cap half-width
    for k in range(5):
        y = M[:, k]
        valid = ~np.isnan(y)
        x_j = (k + 1) + gt_jitter_binned(y[valid], num_bins=20, min_offset=0.08, max_offset=0.22, rng=rng)
        ax.scatter(x_j, y[valid], s=STY.mk.pt, c=[pt_color], edgecolors="none", alpha=0.25, zorder=2)

        # Q1-Q3 error bar with short end caps, in the group color, drawn over
        # the jittered points and under the mean dot. The marker is the mean
        # and the bar is the interquartile range, so on a skewed task the mean
        # need not lie at the center of the bar. The "Avg (4+5)" column shows
        # its quartiles as a full box instead.
        yv = y[valid]
        if yv.size:
            q25, q75 = np.percentile(yv, [25, 75])
            xk = k + 1
            ax.plot([xk, xk], [q25, q75], "-", color=pt_color, lw=STY.lw.se, zorder=2.5)
            ax.plot([xk - QCAP_HW, xk + QCAP_HW], [q25, q25], "-", color=pt_color,
                    lw=STY.lw.se, zorder=2.5)
            ax.plot([xk - QCAP_HW, xk + QCAP_HW], [q75, q75], "-", color=pt_color,
                    lw=STY.lw.se, zorder=2.5)

        mu = np.nanmean(y)
        ax.scatter([k + 1], [mu], s=STY.mk.mean, c=[pt_color], edgecolors="none", alpha=1.0, zorder=3)

        # Per-task one-sided test (mean < 0), uncorrected, marked with '#'.
        p_below = _ttest1_below_zero(yv)
        lbl = _hash_label(p_below)
        if lbl:
            y_hash = ylim[1] - 0.06 * (ylim[1] - ylim[0])
            ax.text(k + 1, y_hash, lbl, ha="center", va="top",
                    fontname=STY.font.name, fontsize=STY.font.star, color="k")
        print(f"    {group_label} task {k + 1}: one-sided p(mean<0) = "
              f"{p_below:.4g} -> '{lbl or 'n/s'}'  (n={yv.size})")

    x_avg = 6
    _draw_single_box(ax, avg_vals, x_avg, box_color, STY)
    valid_avg = ~np.isnan(avg_vals)
    xj_avg = x_avg + gt_jitter_binned(avg_vals[valid_avg], num_bins=20, min_offset=0.08, max_offset=0.22, rng=rng)
    ax.scatter(xj_avg, avg_vals[valid_avg], s=STY.mk.pt, c=[pt_color], edgecolors="none", alpha=0.25, zorder=4)
    mu_avg = np.nanmean(avg_vals)
    ax.scatter([x_avg], [mu_avg], s=STY.mk.mean, c=[pt_color], edgecolors="none", alpha=1.0, zorder=5)

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["1", "2", "3", "4", "5", "Avg (4+5)"])
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Reward rate (normalized)", fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_title(group_label, fontname=STY.font.name, fontsize=STY.font.title)
    gt_axes(ax, STY)
    gt_figStyle(fig, STY)
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    wt_task = np.full((1000, 5), np.nan)
    sh_task = np.full((1000, 5), np.nan)
    gt_task = np.full((1000, 5), np.nan)

    for t, fn in enumerate(TYPE_FILES):
        S = load_struct(os.path.join(data_dir, fn))
        v_wt = np.asarray(S.eff_wt, dtype=float).ravel()
        v_sh = np.asarray(S.eff_sh, dtype=float).ravel()
        v_gt = np.asarray(S.eff_gt, dtype=float).ravel()
        if v_wt.size != 1000 or v_sh.size != 1000 or v_gt.size != 1000:
            raise ValueError(f"{fn}: eff_* must each have 1000 elements")

        wt_mean = np.nanmean(v_wt)
        wt_task[:, t] = v_wt - wt_mean
        sh_task[:, t] = v_sh - wt_mean
        gt_task[:, t] = v_gt - wt_mean

    groups = [
        ("WT", wt_task, STY.cmean[0], STY.cbarfill[0]),
        ("SH", sh_task, STY.cmean[1], STY.cbarfill[1]),
        ("GT", gt_task, STY.cmean[2], STY.cbarfill[2]),
    ]

    figs = []
    for label, M, pt_color, box_color in groups:
        fig = _plot_group_tasks(M, label, pt_color, box_color, YLIM, STY, rng)
        figs.append(fig)
        n_per_task = [int(np.sum(~np.isnan(M[:, k]))) for k in range(5)]
        with np.errstate(invalid="ignore"):
            avg_vals = np.nanmean(M[:, 3:5], axis=1)
        print(f"{label}: n per task = {n_per_task}, n avg(4+5) = {int(np.sum(~np.isnan(avg_vals)))}, "
              f"mean avg = {np.nanmean(avg_vals):.4f}")

    for i, fig in enumerate(figs, start=1):
        gt_saveFig(fig, f"fig4_c_task_progress_{groups[i-1][0]}", out_dir)

    return figs


if __name__ == "__main__":
    main()
