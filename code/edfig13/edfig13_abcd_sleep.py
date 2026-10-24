"""
edfig13_abcd_sleep.py -- Extended Data Fig. 13, panels a to d.

Circadian activity and the fraction of time active during the sleep and wake
phases, in WT, SH and GT animals.

DATA. sleep_measurement.mat, holding the raw running-wheel counts (y_running),
the phase-aligned 24 h profile (z_running, 17281 bins of 5 s), its
per-group normalized forms and the animal identifiers.

GROUP ASSIGNMENT.
    WT: animal identifier above 200
    SH: identifier below 200 with the gene-therapy flag clear
    GT: identifier below 200 with the gene-therapy flag set

PHASE WINDOWS, as bin ranges of the phase-centered 24 h profile:
    late wake    bins 1 to 4300        early sleep  bins 4300 to 7200
    late sleep   bins 7200 to 13000    early wake   bins 13000 to the end
The sleep measure sums the early and late sleep windows and the wake measure
sums the early and late wake windows. Fraction active is the proportion of
bins in a window exceeding 5 counts.

WHAT IS PLOTTED. Five figures:
  edfig13_abcd_sleep_raw_example
                     one animal per group, the full recording, phase-aligned
                     by the same circadian shift applied to the 24 h profile
                     and smoothed with a 60-bin moving mean.
  edfig13_abcd_sleep_timecourse
                     the 24 h activity profile: a faint trace per WT animal,
                     the group mean as a line and a band of one standard
                     error, with the early and late sleep windows shaded.
  edfig13_abcd_sleep_frac_wake
                     the wake-phase fraction active, as a box plot with
                     jittered points, WT against SH against GT.
  edfig13_abcd_sleep_frac_sleep
                     the same for the sleep phase.
  edfig13_abcd_sleep_heatmap
                     normalized activity by animal, one row per animal,
                     grouped into blocks on a white-to-group-color scale.

STATISTICS. The two fraction-active panels apply the three-group comparison
with brackets from gt_boxsig: Levene's test for homogeneity of variance, then
one-way ANOVA with Tukey-Kramer when equal variance holds or Welch's ANOVA
with Games-Howell when it is rejected, with pairwise brackets drawn only when
the omnibus reaches p < 0.05. The example, time-course and heatmap figures
are descriptive.

DISPLAY. Dense traces are rasterized and everything else stays vector, so the
PDF remains small and editable. The heatmap row blocks are floored at a
minimum height so the smaller groups' rotated labels stay legible.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from scipy import stats as sstats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import central_output_dir  # noqa: E402
from gtstyle import (gt_init, loadmat, gt_style, gt_boxscatter_figure,  # noqa: E402
                     gt_colorboxplot, gt_boxsig, gt_axes, gt_color_xticklabels,
                     gt_figStyle, gt_saveFig)
from _common import plot_meanerror, GROUP_COLORS  # noqa: E402

STY = gt_style()

DATA = os.path.join(HERE, "..", "..", "data", "edfig13", "edfig13_abcd_sleep")

GROUP_LABELS = ["WT", "SH", "GT"]
GROUP_COLORS_ARR = np.vstack([GROUP_COLORS["WT"], GROUP_COLORS["SH"], GROUP_COLORS["GT"]])


def _movmean(x, win):
    """Centered moving average of a 1-D array over a window of `win` samples.

    NaNs are excluded from each local mean, and the window is truncated
    rather than padded at the edges. Implemented with cumulative sums, since
    it is applied to every animal's full-length recording.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    half_lo = win // 2
    half_hi = win - half_lo - 1
    idx = np.arange(n)
    lo = np.clip(idx - half_lo, 0, n)
    hi = np.clip(idx + half_hi + 1, 0, n)

    x0 = np.where(np.isnan(x), 0.0, x)
    cums = np.concatenate([[0.0], np.cumsum(x0)])
    valid = (~np.isnan(x)).astype(float)
    cumc = np.concatenate([[0.0], np.cumsum(valid)])

    counts = cumc[hi] - cumc[lo]
    sums = cums[hi] - cums[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, sums / counts, np.nan)
    return out


def main(data_dir=DATA, out_dir=central_output_dir(HERE), rng_seed=7, show=False):
    gt_init()
    rng = np.random.default_rng(rng_seed)

    D = loadmat(os.path.join(data_dir, "sleep_measurement.mat"))

    ids = np.ravel(D["animal_ID_running"]).astype(float)
    gt_flag = np.ravel(D["gene_therapy_running"]).astype(float)
    n_animals = ids.size

    grp = np.zeros(n_animals, dtype=int)
    grp[ids > 200] = 1
    grp[(ids < 200) & (gt_flag == 0)] = 2
    grp[(ids < 200) & (gt_flag == 1)] = 3
    print(f"Groups: WT n={np.sum(grp==1)}, SH n={np.sum(grp==2)}, GT n={np.sum(grp==3)}")

    z_norm_wt = np.asarray(D["z_running_norm_wt"], dtype=float)
    z_norm_sh = np.asarray(D["z_running_norm_sh"], dtype=float)
    z_norm_gt = np.asarray(D["z_running_norm_gt"], dtype=float)
    ts_data = [z_norm_wt, z_norm_sh, z_norm_gt]
    n_pts = z_norm_wt.shape[1]

    z = np.asarray(D["z_running"], dtype=float)   # animals x time bins

    def _win_mean(a, i0, i1):
        """Mean activity per animal over bins i0 to i1 inclusive, 1-based."""
        seg = a[:, i0 - 1:i1]
        return np.nanmean(seg, axis=1)

    def _win_frac(a, i0, i1, thresh=5):
        """Fraction of bins i0 to i1 inclusive, 1-based, exceeding thresh
        counts, per animal."""
        seg = a[:, i0 - 1:i1]
        with np.errstate(invalid="ignore"):
            return np.nanmean(seg > thresh, axis=1)

    early_sleep = _win_mean(z, 4300, 7200)
    late_sleep = _win_mean(z, 7200, 13000)
    early_wake = _win_mean(z, 13000, n_pts)
    late_wake = _win_mean(z, 1, 4300)

    early_sleep_1 = _win_frac(z, 4300, 7200)
    late_sleep_1 = _win_frac(z, 7200, 13000)
    early_wake_1 = _win_frac(z, 13000, n_pts)
    late_wake_1 = _win_frac(z, 1, 4300)

    sleep_v = early_sleep + late_sleep
    wake_v = early_wake + late_wake
    sleep_1 = early_sleep_1 + late_sleep_1
    wake_1 = early_wake_1 + late_wake_1

    os.makedirs(out_dir, exist_ok=True)
    saved = []

    def _save(fig, name):
        """Write the figure as PNG and PDF. The PDF dpi applies only to the
        rasterized dense traces; axes, text and thin lines stay vector."""
        p1 = os.path.join(out_dir, name + ".png")
        fig.savefig(p1, dpi=200, facecolor="w")
        p2 = os.path.join(out_dir, name + ".pdf")
        fig.savefig(p2, facecolor="w", dpi=300)
        print(f"  saved {p1}")
        saved.append(p1)

    # Raw example: one animal per group, phase-aligned. The example is the
    # animal of that group with the most valid samples.
    y_raw = np.asarray(D["y_running"], dtype=float)
    bins_per_day = 17280
    shift = 2880

    example_idx = np.zeros(3, dtype=int)
    for g in range(1, 4):
        gm = np.flatnonzero(grp == g)
        n_valid = np.sum(~np.isnan(y_raw[gm, :]), axis=1)
        example_idx[g - 1] = gm[np.argmax(n_valid)]

    n_days_each = np.zeros(3, dtype=int)
    for g in range(3):
        trace = y_raw[example_idx[g], :]
        valid_idx = np.flatnonzero(~np.isnan(trace))
        last_valid = valid_idx[-1] + 1 if valid_idx.size else 0   # count of bins
        n_days_each[g] = last_valid // bins_per_day
    n_days = int(n_days_each.min())
    n_bins_aligned = n_days * bins_per_day

    # Three short stacked rows, at the same width as the time-course panel so
    # the two sit together at final size.
    fig0, axes0 = plt.subplots(3, 1, figsize=(4.9, 2.6), sharex=False)
    if n_days >= 1:
        for g in range(3):
            ax = axes0[g]
            trace = y_raw[example_idx[g], :n_bins_aligned]
            trace_days = trace.reshape(n_days, bins_per_day)
            trace_days = np.concatenate(
                [trace_days[:, -shift:], trace_days[:, :-shift]], axis=1)
            trace_aligned = trace_days.reshape(-1)
            trace_smooth = _movmean(trace_aligned, 60)
            t_days_plot = (np.arange(1, trace_smooth.size + 1)) / bins_per_day

            col = GROUP_COLORS_ARR[g]
            ax.fill_between(t_days_plot, 0, trace_smooth, color=col * 0.3 + 0.7,
                            edgecolor=col, linewidth=STY.lw.trace, alpha=0.5,
                            rasterized=True)   # dense full-recording trace
            ax.set_ylabel("Counts", fontsize=STY.font.lab)
            ax.set_title(f"{GROUP_LABELS[g]} - Animal {int(ids[example_idx[g]])}",
                        fontsize=STY.font.title, color=col)
            ax.set_xlim(0, n_days)
            ax.set_ylim(0, 200)
            ax.set_xticks(range(n_days + 1))
            # Uniform 7 pt tick and body text, as elsewhere in the set.
            ax.tick_params(direction="out", labelsize=STY.font.tick)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_linewidth(STY.lw.axis)
            if g < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (days)", fontsize=STY.font.lab)
    fig0.suptitle("Raw Activity - Full Recording (phase-aligned)", fontweight="bold", fontsize=STY.font.title)
    fig0.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig0, "edfig13_abcd_sleep_raw_example")

    # 24 h activity time course. Sized so its text and line weights match the
    # box panels when the two are placed together at final size.
    fig1, ax1 = plt.subplots(figsize=(4.9, 1.9))
    t_hrs = np.linspace(0, 24, n_pts)
    smooth_win = 60

    yl_tmp = (0, 4)
    t_es = np.array([4300, 7200]) / n_pts * 24
    t_ls = np.array([7200, 13000]) / n_pts * 24
    ax1.fill([t_es[0], t_es[0], t_es[1], t_es[1]], [0, yl_tmp[1], yl_tmp[1], 0],
             color=(0.92, 0.92, 0.92), edgecolor="none", zorder=0)
    ax1.fill([t_ls[0], t_ls[0], t_ls[1], t_ls[1]], [0, yl_tmp[1], yl_tmp[1], 0],
             color=(0.85, 0.85, 0.85), edgecolor="none", zorder=0)

    for a in range(z_norm_wt.shape[0]):
        trace = _movmean(z_norm_wt[a, :], smooth_win)
        ax1.plot(t_hrs, trace, "-", color=(*GROUP_COLORS_ARR[0], 0.12), lw=STY.lw.trace,
                 zorder=1, rasterized=True)   # individual WT traces, rasterized

    for g in range(3):
        data_g = ts_data[g]
        n_g = data_g.shape[0]
        data_smooth = np.vstack([_movmean(data_g[a, :], smooth_win) for a in range(n_g)])
        m = np.nanmean(data_smooth, axis=0)
        se = np.nanstd(data_smooth, axis=0, ddof=1) / np.sqrt(n_g)
        col = GROUP_COLORS_ARR[g]
        ax1.fill(np.concatenate([t_hrs, t_hrs[::-1]]),
                 np.concatenate([m + se, (m - se)[::-1]]), color=col, alpha=0.2,
                 edgecolor="none", zorder=2)
        ax1.plot(t_hrs, m, "-", color=col, lw=1.1, zorder=3,
                 label=f"{GROUP_LABELS[g]} (n={n_g})")

    ax1.text(t_es.mean(), yl_tmp[1] * 0.97, "Early sleep", ha="center",
             fontsize=STY.font.tick, color=(0.5, 0.5, 0.5))
    ax1.text(t_ls.mean(), yl_tmp[1] * 0.97, "Late sleep", ha="center",
             fontsize=STY.font.tick, color=(0.5, 0.5, 0.5))
    ax1.legend(loc="upper right", fontsize=STY.font.tick, frameon=False)
    ax1.set_xlabel("Time (hours)", fontsize=STY.font.lab)
    ax1.set_ylabel("Normalized activity", fontsize=STY.font.lab)
    ax1.set_title("24-Hour Activity Profile", fontsize=STY.font.title, fontweight="bold")
    ax1.set_xlim(0, 24)
    ax1.set_ylim(*yl_tmp)
    ax1.set_xticks(range(0, 25, 4))
    # Uniform 7 pt tick and body text, as elsewhere in the set.
    ax1.tick_params(direction="out", labelsize=STY.font.tick)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax1.spines[s].set_linewidth(STY.lw.axis)
    fig1.tight_layout()
    _save(fig1, "edfig13_abcd_sleep_timecourse")

    # Fraction of time active, one box + scatter panel per phase.
    for vals, ttl, name in [(wake_1, "Wake", "edfig13_abcd_sleep_frac_wake"),
                            (sleep_1, "Sleep", "edfig13_abcd_sleep_frac_sleep")]:
        grp_vals = [vals[grp == g] for g in (1, 2, 3)]
        allv = np.concatenate([g for g in grp_vals if g.size])
        lo, hi = float(allv.min()), float(allv.max())
        span = (hi - lo) or 1.0
        fig2, ax2 = gt_boxscatter_figure(STY)
        gt_colorboxplot(ax2, grp_vals, STY, rng=rng)
        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels(GROUP_LABELS)
        ax2.set_xlim(0.5, 3.5)
        ax2.set_ylim(lo - 0.10 * span, hi + 0.30 * span)
        gt_boxsig(ax2, STY)   # three-group comparison and brackets
        ax2.set_ylabel("Average activity", fontname=STY.font.name, fontsize=STY.font.lab)
        gt_axes(ax2, STY)
        gt_color_xticklabels(ax2, STY)
        gt_figStyle(fig2, STY)
        gt_saveFig(fig2, name, out_dir)
        print(f"  saved {name}.pdf")
        if not show:
            plt.close(fig2)

    # Activity heatmap, one row per animal, grouped into blocks.
    z_norm_all = np.asarray(D["z_running_norm"], dtype=float)
    grp_data_heat = [(z_norm_all[grp == 1, :], "WT"),
                     (z_norm_all[grp == 2, :], "SH"),
                     (z_norm_all[grp == 3, :], "GT")]
    n_per_group = np.array([d.shape[0] for d, _ in grp_data_heat])
    total_rows = n_per_group.sum()
    margin_top, margin_bot, gap = 0.08, 0.10, 0.03
    usable_h = 1 - margin_top - margin_bot - gap * 2
    # Block heights track the number of animals, since one row is one animal,
    # but are floored so the smaller groups' labels have room.
    raw_h = (n_per_group / total_rows) * usable_h
    panel_h = np.maximum(raw_h, 0.16 * usable_h)
    panel_h = panel_h / panel_h.sum() * usable_h

    # Authored at its final portrait size with 7 pt text, so it needs no
    # rescaling on placement.
    fig4 = plt.figure(figsize=(2.25, 4.25), facecolor="w")
    y_top = 1 - margin_top
    for g in range(3):
        pos = [0.10, y_top - panel_h[g], 0.78, panel_h[g]]
        ax4 = fig4.add_axes(pos)
        y_top -= panel_h[g] + gap

        data_g, label = grp_data_heat[g]
        n_g = data_g.shape[0]
        target = GROUP_COLORS_ARR[g]
        cmap_g = np.stack([np.linspace(1, target[c], 256) for c in range(3)], axis=1)
        from matplotlib.colors import ListedColormap
        im = ax4.imshow(data_g, aspect="auto", cmap=ListedColormap(cmap_g),
                        vmin=0, vmax=5, extent=[t_hrs[0], t_hrs[-1], n_g, 0])
        ax4.set_ylabel(f"{label} (n={n_g})", fontsize=STY.font.lab, fontweight="bold", color=target)
        ax4.set_yticks([])
        # Uniform 7 pt tick and body text, as elsewhere in the set.
        ax4.tick_params(direction="out", labelsize=STY.font.tick)
        for s in ("top", "right"):
            ax4.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax4.spines[s].set_linewidth(STY.lw.axis)
        if g < 2:
            ax4.set_xticklabels([])
        else:
            ax4.set_xlabel("Time (hours)", fontsize=STY.font.lab)
            ax4.set_xticks(range(0, 25, 6))     # every 6 h, to fit the panel width
    fig4.suptitle("24-Hour Activity - By Group", fontweight="bold", fontsize=STY.font.title)
    _save(fig4, "edfig13_abcd_sleep_heatmap")

    print("\nDone.")

    if not show:
        for f in (fig0, fig1, fig2, fig4):
            plt.close(f)
    return fig0, fig1, fig2, fig4


if __name__ == "__main__":
    main()
