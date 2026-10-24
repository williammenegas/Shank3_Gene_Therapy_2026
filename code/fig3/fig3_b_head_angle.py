"""
Figure 3, panel b: head angle relative to the social window, toward versus
away.

  1. Three polar histograms, one example session per genotype (WT, SH, GT).
     For every frame of the session the script bins the absolute head angle
     relative to the social window (radians, 0 to pi) into 18 bins of 10
     degrees and draws each bin as a colored polar wedge, with a fixed
     per-bin palette read from the .mat file. Theta = 0, at the top of the
     plot, is the head pointing directly at the window ("toward");
     theta = pi is directly away.

  2. A paired toward/away swarm, one tile per genotype. The script plots
     the per-session mean measure in the toward half and the away half of
     the angle distribution: a light line joins each animal's two values, a
     jittered scatter shows up to PLOT_N_MAX subsampled pairs, and black
     markers give the median with the interquartile range of the full,
     unsubsampled distributions. y limits are fixed at FIXED_YLIM for all
     three tiles.

Statistics: the polar histograms are descriptive. The paired swarm carries a
paired Student t-test per genotype on the log ratio log(toward / away), since
toward versus away is a within-animal contrast, drawn as a bracket over the
two columns and labeled on the house scale (0.05 / 0.01 / 0.001). Medians
and quartiles are display marks.

Which pairs appear in the subsampled scatter is drawn from an unseeded
random number generator and varies between runs; the medians and quartiles
are computed on the full data and are reproducible.

Data: hist_example_wt.mat, hist_example_sh.mat, hist_example_gt.mat,
theta.mat
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig3", "fig3_b_head_angle")

PLOT_N_MAX = 250        # maximum toward/away pairs drawn in the scatter
JITTER_AMOUNT = 0.05    # half-width of the horizontal jitter, in x units
LINE_ALPHA = 0.15
POINT_ALPHA = 0.65
POINT_SIZE = 18
FIXED_YLIM = (0.0, 0.66)   # shared y limits across the three genotype tiles


def _polar_hist_panel(fname, title, STY, outname, out_dir):
    S = load_struct(fname)
    cell = S.all_theta_abs_cell
    if isinstance(cell, list):
        parts = [np.asarray(c, dtype=float).ravel() for c in cell]
    else:
        parts = [np.asarray(cell, dtype=float).ravel()]
    theta = np.concatenate(parts)

    edges = np.asarray(S.edges, dtype=float).ravel()
    bin_colors = np.asarray(S.bin_colors, dtype=float)
    n_bins = edges.size - 1

    counts = np.histogram(theta, bins=edges)[0]
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)

    fig = plt.figure(figsize=(1.5, 1.5), facecolor="w")
    ax = fig.add_axes([0.12, 0.12, 0.76, 0.76], projection="polar")
    ax.set_theta_zero_location("N")
    # Clockwise from North, so the 0-180 sweep opens to the right, as the
    # published panel does. Counterclockwise would be equally consistent
    # but would mirror the semicircle onto the left.
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    for b in range(n_bins):
        ax.bar(centers[b], counts[b], width=widths[b], bottom=0,
               color=bin_colors[b], edgecolor="none")

    ax.set_rlim(0, 6e4)
    # The published polar panels carry no title.
    ax.set_title("",
                  fontname=STY.font.name, fontsize=STY.font.title)
    ax.tick_params(labelsize=STY.font.tick)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, outname, out_dir)
    return fig, counts


def _med_quarts(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    return np.median(x), np.percentile(x, 25), np.percentile(x, 75)


def _toward_away_panel(theta_path, STY, out_dir, rng):
    S = load_struct(theta_path)
    conds = [
        ("WT full", "toward_wt_full", "away_wt_full"),
        ("SH full", "toward_sh_full", "away_sh_full"),
        ("GT full", "toward_gt_full", "away_gt_full"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(1.85, 1.35), facecolor="w")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.78, wspace=0.55)

    for i, (name, tow_name, aw_name) in enumerate(conds):
        ax = axes[i]
        tow = np.asarray(S[tow_name], dtype=float).ravel()
        aw = np.asarray(S[aw_name], dtype=float).ravel()
        ok = np.isfinite(tow) & np.isfinite(aw)
        tow, aw = tow[ok], aw[ok]
        N = tow.size

        if N > PLOT_N_MAX:
            idx = rng.choice(N, size=PLOT_N_MAX, replace=False)
            towP, awP = tow[idx], aw[idx]
        else:
            towP, awP = tow, aw

        for k in range(towP.size):
            ax.plot([1, 2], [towP[k], awP[k]], "-", lw=STY.lw.pair,
                    color=(*STY.clr.pairline, LINE_ALPHA))

        x1 = 1 + (rng.random(towP.size) - 0.5) * 2 * JITTER_AMOUNT
        x2 = 2 + (rng.random(awP.size) - 0.5) * 2 * JITTER_AMOUNT
        ax.scatter(x1, towP, s=STY.mk.pt, c=[STY.clr.toward], alpha=POINT_ALPHA,
                   edgecolors="none")
        ax.scatter(x2, awP, s=STY.mk.pt, c=[STY.clr.away], alpha=POINT_ALPHA,
                   edgecolors="none")

        med_t, q25_t, q75_t = _med_quarts(tow)
        med_a, q25_a, q75_a = _med_quarts(aw)
        ax.plot(1, med_t, "ko", mfc="k", ms=STY.mk.mean ** 0.5)
        ax.plot([1, 1], [q25_t, q75_t], "k-", lw=STY.lw.se)
        ax.plot(2, med_a, "ko", mfc="k", ms=STY.mk.mean ** 0.5)
        ax.plot([2, 2], [q25_a, q75_a], "k-", lw=STY.lw.se)

        ax.set_ylim(*FIXED_YLIM)
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Tow.", "Aw."])

        # Toward versus away is a within-animal contrast, so the test is a
        # paired Student t-test on the log ratio log(toward / away), the
        # measures being compositional. Drawn after set_ylim so the bracket
        # is placed relative to the final y range, and labeled on the house
        # 0.05 / 0.01 / 0.001 scale.
        t_stat, p_paired, n_pair = gt_pairedt(np.log(tow), np.log(aw))
        yl = ax.get_ylim()
        h = 0.02 * (yl[1] - yl[0])
        base = yl[0] + 0.88 * (yl[1] - yl[0])
        gt_drawSig(ax, 1, 2, base, h, gt_pstar(p_paired), STY)

        gt_axes(ax, STY)
        # The published panel titles each group with its name alone; the
        # condition and n stay in the console summary below.
        ax.set_title(name.split()[0], fontname=STY.font.name,
                     fontsize=STY.font.title)

        print(f"  {name}: n={N}, median toward={med_t:.4f}, median away={med_a:.4f}, "
              f"log-ratio paired t={t_stat:+.3f} p={p_paired:.4g} -> {gt_pstar(p_paired)}")

    # The published figure carries no overall title.
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "fig3_b_head_angle_p4", out_dir)
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    figs = []
    for i, (fname, title) in enumerate([
        ("hist_example_wt.mat", "WT"),
        ("hist_example_sh.mat", "SH"),
        ("hist_example_gt.mat", "GT"),
    ]):
        fig, counts = _polar_hist_panel(os.path.join(data_dir, fname), title,
                                         STY, f"fig3_b_head_angle_p{i+1}", out_dir)
        print(f"{title}: total frames={counts.sum()}, bins={counts}")
        figs.append(fig)

    print("Toward/Away paired panel:")
    figs.append(_toward_away_panel(os.path.join(data_dir, "theta.mat"), STY, out_dir, rng))

    return figs


if __name__ == "__main__":
    main()
