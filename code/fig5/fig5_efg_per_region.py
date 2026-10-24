"""
fig5_efg_per_region.py

Figure 5, panels e, f and g: cortico-striatal minus midbrain-striatal
connectivity by cortical region. This script produces one figure per group,
giving the three panels: e = WT, f = SH, g = GT.

Outputs: fig5_e_per_region_WT, fig5_f_per_region_SH, fig5_g_per_region_GT

Data. Six region files, prefrontal_str_paired_midbrain_str_data.mat and its
premotor, parietal, temporal, somatosensory and visual counterparts. The
plotted quantity is the per-animal difference collect_*_r1 - collect_*_r2,
that is cortico-striatal minus midbrain-striatal connectivity, for each of
the six regions.

Display. Regions run along the x axis with labels rotated 35 degrees;
normalized correlation is on the y axis, with a dashed reference line at
zero. Each region carries an unfilled box in the group color (median,
quartiles, whiskers at 1.5*IQR, outliers not drawn separately), jittered
individual animals spread uniformly over +/- JITTER (0.22), and a filled mean
marker. The y limits are fixed at [-0.1, 0.6] in all three panels so that the
groups can be compared directly.

No statistical test is drawn in these panels; the corresponding group
comparisons are in panels b and d.

Point jitter is unseeded, so horizontal offsets of individual animals differ
slightly between runs.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig5", "fig5_d_cortico_boxplots")

REGION_FILES = [
    "prefrontal_str_paired_midbrain_str_data.mat",
    "premotor_str_paired_midbrain_str_data.mat",
    "parietal_str_paired_midbrain_str_data.mat",
    "temporal_str_paired_midbrain_str_data.mat",
    "somatosensory_str_paired_midbrain_str_data.mat",
    "visual_str_paired_midbrain_str_data.mat",
]
REGION_LABELS = ["Frontal", "Motor", "Parietal", "Temporal", "Somatosensory", "Visual"]

JITTER = 0.22
# Marker sizes follow the shared house style: STY.mk.pt for individual
# animals and STY.mk.mean for the mean marker. The house convention is a
# marker AREA, while matplotlib's plot() takes a diameter, so the mean marker
# below passes STY.mk.mean ** 0.5.
BOX_WIDTH = 0.45
LINE_WIDTH = 0.6


def _plot6_boxscatter(vals, plot_color, region_labels, STY, rng, title=None):
    """One group, six regions: vertical box, jittered points and mean marker.

    Regions run along the x axis, normalized correlation on the y axis, with a
    dashed reference line at zero, unfilled colored boxes with 1.5*IQR
    whiskers, uniform +/- JITTER point spread and a filled mean marker."""
    vals = [v[~np.isnan(v)] for v in vals]

    fig, ax = plt.subplots(figsize=(2.15, 1.90), facecolor="w")
    # bottom margin holds the rotated region labels ("Somatosensory")
    fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.30)

    ax.axhline(0.0, ls="--", color="k", lw=STY.lw.hair, zorder=0)

    # Vertical unfilled boxes with whiskers at 1.5*IQR.
    bp = ax.boxplot(
        vals, positions=np.arange(1, 7), vert=True, widths=BOX_WIDTH,
        whis=1.5, showfliers=False, patch_artist=False,
    )
    for part in ("boxes", "whiskers", "caps", "medians"):
        for artist in bp[part]:
            artist.set_color(plot_color)
            artist.set_linewidth(LINE_WIDTH)

    for i, y in enumerate(vals, start=1):
        if y.size == 0:
            continue
        xj = i + (rng.random(y.size) - 0.5) * (2 * JITTER)
        ax.scatter(xj, y, s=STY.mk.pt, c=[plot_color], edgecolors=plot_color,
                   alpha=0.35, linewidths=0.3, zorder=3)
        m = y.mean()
        ax.plot([i], [m], "o", markerfacecolor=plot_color, markeredgecolor=plot_color,
                markersize=STY.mk.mean ** 0.5, zorder=4)

    ax.set_xticks(np.arange(1, 7))
    ax.set_xticklabels(region_labels, rotation=35, ha="right")
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(-0.1, 0.6)             # fixed limits, shared by all three groups
    ax.set_yticks([0.0, 0.3, 0.6])
    ax.set_ylabel("Correlation (normalized)", fontname=STY.font.name,
                  fontsize=STY.font.lab)
    if title:
        ax.set_title(title, fontname=STY.font.name, fontsize=STY.font.title)
    gt_axes(ax, STY, matlab_ticks=False)
    gt_figStyle(fig, STY)
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    collect_wt = np.full((1000, 6), np.nan)
    collect_sh = np.full((1000, 6), np.nan)
    collect_gt = np.full((1000, 6), np.nan)

    for j, fn in enumerate(REGION_FILES):
        S = load_struct(os.path.join(data_dir, fn))
        collect_wt[:, j] = (np.asarray(S.collect_wt_r1, dtype=float).ravel()
                             - np.asarray(S.collect_wt_r2, dtype=float).ravel())
        collect_sh[:, j] = (np.asarray(S.collect_sh_r1, dtype=float).ravel()
                             - np.asarray(S.collect_sh_r2, dtype=float).ravel())
        collect_gt[:, j] = (np.asarray(S.collect_gt_r1, dtype=float).ravel()
                             - np.asarray(S.collect_gt_r2, dtype=float).ravel())

    # Panel letter per group: e = WT, f = SH, g = GT.
    groups = [
        ("WT", "e", collect_wt, STY.cmean[0]),
        ("SH", "f", collect_sh, STY.cmean[1]),
        ("GT", "g", collect_gt, STY.cmean[2]),
    ]

    figs = []
    for label, letter, M, color in groups:
        vals = [M[:, j] for j in range(6)]
        fig = _plot6_boxscatter(vals, color, REGION_LABELS, STY, rng, title=label)
        figs.append(fig)
        ns = [int(np.sum(~np.isnan(M[:, j]))) for j in range(6)]
        means = [float(np.nanmean(M[:, j])) for j in range(6)]
        print(f"{label}: n per region = {ns}")
        print(f"{label}: mean per region = {['%.4f' % m for m in means]}")

    for (label, letter, _M, _c), fig in zip(groups, figs):
        gt_saveFig(fig, f"fig5_{letter}_per_region_{label}", out_dir)

    return figs


if __name__ == "__main__":
    main()
