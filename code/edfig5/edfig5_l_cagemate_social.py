"""Extended Data Figure 5, panel l: change in time near the stimulus,
face minus box, WT/SH/GT.

Data: `time_near_box_data.mat` and `time_near_face_data.mat`, each holding
`wt_2_mean`, `sh_2_mean` and `gt_2_mean`, per-animal mean time-near-stimulus
vectors padded to 1000 slots with NaN. The plotted quantity is the per-slot
difference face minus box, computed elementwise over the aligned arrays; NaN
propagates through the subtraction. The script produces one figure.

Plot: box and density-jittered scatter, one group per x position (WT, SH, GT),
figure size 1.821 x 2.000 in, using the same `_enhanced_boxplot` style as
panel g: solid whisker stems from the box edge to q1 - 1.5 x IQR and
q3 + 1.5 x IQR, clipped to the observed data range, with narrow end caps 0.1
data units wide. The y axis is fixed to [-0.6, 0.3]; panel g uses [-0.3, 0.1]
for its object-minus-treat contrast, so the two ranges are not interchangeable.

Statistics: the three-group comparison drawn by gt_boxsig, i.e. a one-way
ANOVA (Welch's ANOVA when Levene's test rejects equal variance) with
Tukey-Kramer (or Games-Howell) post-hoc contrasts. Pairwise brackets appear
only when the omnibus test is significant; otherwise the panel is annotated
'ANOVA n/s'. The script prints counts and group medians at run time.

The scatter jitter is unseeded, so the horizontal offsets of individual points
differ between runs.
"""

import os
import sys
import numpy as np
import matplotlib.patches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_c_novelty_boxplots")

FIG_W, FIG_H = 1.821, 2.000   # figure size in inches


def _enhanced_boxplot(ax, G, STY, positions=(1, 2, 3), box_width=0.6075,
                       whisker=1.5, cap_width=0.1, rng=None):
    """Box plot with solid whisker stems and narrow caps, plus a jittered
    scatter overlay.

    positions: x position per group. box_width: box width in x units.
    whisker: IQR multiplier for the whisker length, clipped to the data range.
    cap_width: half-width of the whisker end caps in x units.
    """
    if rng is None:
        rng = np.random.default_rng()
    hw = box_width / 2.0
    stats = []
    for g in range(3):
        v = np.asarray(G[g], dtype=float).ravel()
        v = v[~np.isnan(v)]
        x = positions[g]

        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        wlen = whisker * iqr
        y_min = max(v.min(), q1 - wlen)
        y_max = min(v.max(), q3 + wlen)
        stats.append((q1, med, q3, y_min, y_max))

        boxcol = STY.cbarfill[g]
        ptcol = STY.cscat[g]

        ax.plot([x, x], [q3, y_max], "-", color="k", lw=0.6, zorder=2)
        ax.plot([x, x], [q1, y_min], "-", color="k", lw=0.6, zorder=2)
        ax.plot([x - cap_width, x + cap_width], [y_max, y_max], "-", color="k",
                lw=0.6, zorder=2)
        ax.plot([x - cap_width, x + cap_width], [y_min, y_min], "-", color="k",
                lw=0.6, zorder=2)

        ax.add_patch(matplotlib.patches.Rectangle(
            (x - hw, q1), box_width, q3 - q1,
            facecolor=boxcol, edgecolor="none", alpha=0.5, zorder=3))

        ax.plot([x - hw, x + hw], [med, med], "-", color="k", lw=STY.lw.axis, zorder=4)
        ax.plot([x - hw, x + hw], [q1, q1], "-", color="k", lw=0.6, zorder=4)
        ax.plot([x - hw, x + hw], [q3, q3], "-", color="k", lw=0.6, zorder=4)

        xj = x + gt_jitter_binned(v, rng=rng)
        ax.scatter(xj, v, s=STY.mk.pt, c=[ptcol], edgecolors="none",
                   alpha=0.66, zorder=5)

    ax._gt_box_groups = [np.asarray(g, float).ravel()
                         [~np.isnan(np.asarray(g, float).ravel())] for g in G]
    ax._gt_box_pos = list(positions)
    return stats


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    rng = np.random.default_rng()

    Sb = load_struct(os.path.join(data_dir, "time_near_box.mat"))
    Sf = load_struct(os.path.join(data_dir, "time_near_face.mat"))

    a1 = np.asarray(Sb.wt_2_mean, dtype=float).ravel()
    b1 = np.asarray(Sb.sh_2_mean, dtype=float).ravel()
    c1 = np.asarray(Sb.gt_2_mean, dtype=float).ravel()
    a2 = np.asarray(Sf.wt_2_mean, dtype=float).ravel()
    b2 = np.asarray(Sf.sh_2_mean, dtype=float).ravel()
    c2 = np.asarray(Sf.gt_2_mean, dtype=float).ravel()

    r1 = 100.0 * (a2 - a1)
    r2 = 100.0 * (b2 - b1)
    r3 = 100.0 * (c2 - c1)

    fig, ax = gt_boxscatter_figure(STY, fig_w=FIG_W, fig_h=FIG_H)
    _enhanced_boxplot(ax, [r1, r2, r3], STY, rng=rng)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT", "SH", "GT"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-60, 30)
    gt_boxsig(ax, STY)
    ax.set_ylabel(r"$\Delta$ Time near stimulus (%)", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_axes(ax, STY)
    gt_color_xticklabels(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig5_l_cagemate_social", out_dir)

    n = [np.sum(~np.isnan(v)) for v in (r1, r2, r3)]
    print("edfig5_l_cagemate_social -- interaction-time shift (face - box), WT/SH/GT")
    print(f"  n = {n[0]}/{n[1]}/{n[2]}, "
          f"medians = {np.nanmedian(r1):.4f}/{np.nanmedian(r2):.4f}/{np.nanmedian(r3):.4f}")

    return fig


if __name__ == "__main__":
    main()
