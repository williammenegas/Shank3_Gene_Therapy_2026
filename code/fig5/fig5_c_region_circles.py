"""
fig5_c_region_circles.py

Figure 5, panel c: circular connectome plots of pairwise ROI-to-ROI
functional-connectivity differences (SH vs WT, GT vs WT, SH vs GT).

Data. fmri_g1.mat, fmri_g2.mat and fmri_g3.mat, each holding corr_mat1 (WT,
n = 13), corr_mat2 (SH, n = 6) and corr_mat3 (GT, n = 7) as 74 x 74
ROI-to-ROI correlation matrices. The three files hold three repeated
within-animal measurements. The script averages them per animal and ROI pair
before any group comparison.

Statistics. Per ROI pair, a two-sample equal-variance (Student) t-test,
two-sided. By default an edge is drawn where its own uncorrected p is below
P_THRESH; with --sig fdr, Benjamini-Hochberg FDR at level q is applied
instead across the 2701 unique edges of each contrast. No correction is
applied across the three contrasts.

Display. Each drawn edge is colored and weighted by the difference in group
means for that same contrast, scaled by MD_SCALE, on a diverging
purple-gray-green colormap built by piecewise-linear interpolation over
control points at 0, 0.1, 0.45, 0.55, 0.9 and 1. Significance mask and edge
weight are always taken from the contrast being drawn.

Nodes sit on a unit circle, one per ROI, starting at (0, -1) and running
clockwise; each ROI pair is drawn once, from the lower triangle. Each edge
follows the curve (1 - t)^2 * P_start + t^2 * P_end over 20 values of t in
(0, 1). This bows the line toward the center of the circle. Edge line width
is 0.9 * max(0, |value| * 10 - 3)^1.25, floored at 0.01 and scaled by
LW_SCALE for the printed figure size. Node marker area is proportional to the
mean absolute weight of the edges touching that node, scaled by NODE_SCALE.
Edges are added in ascending order of absolute weight, so stronger edges are
drawn on top where curves cross.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import t as student_t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig5", "fig5_c_region_circles")

N_ROI = 74
P_THRESH = 0.02   # default uncorrected per-edge p cutoff
N_BUCKET = 50   # half the color-bucket count; 2*N_BUCKET buckets in total
# Edge widths and node areas are set in points and are scaled here to the
# printed panel size.
LW_SCALE = 0.18                         # edge line-width multiplier
NODE_SCALE = 80                         # node marker area multiplier
NODE_COLOR = (151 / 255, 47 / 255, 218 / 255)   # node fill color
MD_SCALE = 5.0                          # mean-difference multiplier; lifts the
                                        # smaller, low-variance edges above the
                                        # line-width floor without saturating
                                        # the strongest edges


def _load_avg_corr(data_dir):
    mats = {}
    for suffix in ("corr_mat1", "corr_mat2", "corr_mat3"):
        stack = []
        for gi in (1, 2, 3):
            S = load_struct(os.path.join(data_dir, f"fmri_g{gi}.mat"))
            stack.append(np.asarray(S[suffix], dtype=float))
        with np.errstate(invalid="ignore"):
            mats[suffix] = np.nanmean(np.stack(stack, axis=-1), axis=-1)
    return mats["corr_mat1"], mats["corr_mat2"], mats["corr_mat3"]   # WT, SH, GT


def _ttest2_grid(gA, gB):
    """Two-sample equal-variance t-test, vectorized over the (74,74) ROI grid.
    gA, gB: (nAnimals, 74, 74)."""
    nA = np.sum(~np.isnan(gA), axis=0).astype(float)
    nB = np.sum(~np.isnan(gB), axis=0).astype(float)
    mA = np.nanmean(gA, axis=0)
    mB = np.nanmean(gB, axis=0)
    with np.errstate(invalid="ignore"):
        vA = np.nanvar(gA, axis=0, ddof=1)
        vB = np.nanvar(gB, axis=0, ddof=1)
    df = nA + nB - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        sp2 = ((nA - 1) * vA + (nB - 1) * vB) / df
        se = np.sqrt(sp2 * (1.0 / nA + 1.0 / nB))
        tstat = (mA - mB) / se
        p = 2.0 * (1.0 - student_t.cdf(np.abs(tstat), df))
    return np.where((nA < 2) | (nB < 2), np.nan, p)


def _cohend_grid(gA, gB):
    """Cohen's d with the (n-1)-weighted pooled standard deviation, vectorized
    over the (74, 74) ROI grid."""
    nA = gA.shape[0]
    nB = gB.shape[0]
    mA = np.nanmean(gA, axis=0)
    mB = np.nanmean(gB, axis=0)
    with np.errstate(invalid="ignore"):
        vA = np.nanvar(gA, axis=0, ddof=1)
        vB = np.nanvar(gB, axis=0, ddof=1)
    sp = np.sqrt(((nA - 1) * vA + (nB - 1) * vB) / (nA + nB - 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (mA - mB) / sp


def _custom_cmap(n=100):
    xcolors = np.array([0.0, 0.1, 0.45, 0.55, 0.9, 1.0])
    purple = np.array([151, 47, 218]) / 255.0
    p_mid = np.array([161, 77, 200]) / 255.0
    gray = np.array([190, 190, 190]) / 255.0
    g_mid = np.array([34, 158, 51]) / 255.0
    green = np.array([31, 158, 51]) / 255.0
    colors = np.vstack([purple, p_mid, gray, gray, g_mid, green])

    xq = np.linspace(0, 1, n)
    cmap = np.column_stack([np.interp(xq, xcolors, colors[:, ch]) for ch in range(3)])
    return cmap


def _plot_circle_graph(ax, r, labels, STY, cmap):
    """Draw one contrast as a circular connectome. r is the (74, 74) matrix of
    masked edge weights.

    The diagonal is a self-comparison: every animal's self-correlation is
    exactly 1, both groups have zero variance and the edge weight is
    undefined. It is zeroed before use so that it contributes to neither the
    edge list nor the node-strength sum.
    """
    sz = r.shape
    r = np.clip(r, -1, 1)
    np.fill_diagonal(r, 0.0)

    tau = 2 * np.pi
    step = tau / sz[0]
    theta = np.arange(-0.25 * tau, 0.75 * tau - step / 2, step)[: sz[0]]
    x = np.cos(theta)
    y = np.sin(theta)

    tf_row, tf_col = np.tril_indices(sz[0], k=-1)
    rvals = r[tf_row, tf_col]

    n_bucket2 = 2 * N_BUCKET
    edges = np.linspace(-1, 1 + np.finfo(float).eps * 100, n_bucket2 + 1)
    bin_idx = np.clip(np.digitize(rvals, edges) - 1, 0, n_bucket2 - 1)

    rvals1 = np.abs(rvals) * 10 - 3
    rvals1[rvals1 < 0] = 0
    # Line width 0.9 * max(0, |value| * 10 - 3)^1.25, rescaled to the printed
    # panel size by LW_SCALE.
    linewidths = LW_SCALE * 0.9 * (rvals1 ** 1.25)
    linewidths[linewidths < 0.01] = 0.01

    t = np.arange(0.025, 1.0, 0.05)
    w_start = (1 - t) ** 2
    w_end = t ** 2

    order = np.argsort(np.abs(rvals))   # weakest first, strongest drawn last (on top)

    segments = []
    seg_colors = []
    seg_widths = []
    for k in order:
        if rvals[k] == 0:               # non-significant edges are masked to 0 and not drawn
            continue
        i, j = tf_row[k], tf_col[k]
        bx = w_start * x[j] + w_end * x[i]
        by = w_start * y[j] + w_end * y[i]
        segments.append(np.column_stack([bx, by]))
        seg_colors.append(cmap[bin_idx[k]])
        seg_widths.append(linewidths[k])

    lc = LineCollection(segments, colors=seg_colors, linewidths=seg_widths)
    ax.add_collection(lc)

    # Node strength is the mean absolute weight of the edges touching that
    # node, taken over both rows and columns and left unnormalized.
    full = np.abs(r) + np.abs(r).T
    node_strength = full.sum(axis=1) / (2 * max(sz[0] - 1, 1))
    # Marker area in points^2, rescaled to the printed panel size by NODE_SCALE.
    ax.scatter(x, y, s=np.clip(1 + NODE_SCALE * node_strength, 0.5, None),
               c=[NODE_COLOR], edgecolors="k", linewidths=0.3, zorder=5)

    for i in range(sz[0]):
        ha = "left" if x[i] > 0 else "right"
        ax.text(x[i] * 1.12, y[i] * 1.12, labels[i], color=STY.clr.pt_gray,
                fontsize=4, ha=ha, va="center",
                rotation=theta[i] / tau * 360 + (0 if x[i] > 0 else 180))

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")


def _bh_fdr_edges(pmat, q):
    """Benjamini-Hochberg FDR over the unique ROI-to-ROI edges (lower triangle)
    of a symmetric p-value matrix. Returns a symmetric boolean mask of the
    edges that pass at level q; NaN edges never pass."""
    iu = np.tril_indices(pmat.shape[0], -1)
    p = pmat[iu]
    ok = np.isfinite(p)
    passed = np.zeros(p.shape, bool)
    pv = p[ok]
    m = pv.size
    if m:
        order = np.argsort(pv)
        ranked = pv[order]
        crit = q * (np.arange(1, m + 1) / m)
        below = ranked <= crit
        if below.any():
            thr = ranked[np.max(np.flatnonzero(below))]   # largest rank that passes
            passed[ok] = pv <= thr
    h = np.zeros(pmat.shape, bool)
    h[iu] = passed
    return h | h.T                                          # symmetrize


def main(data_dir=DATA, out_dir=HERE, sig="uncorrected", q=0.05, p_thresh=P_THRESH):
    STY = gt_init()

    corr_wt, corr_sh, corr_gt = _load_avg_corr(data_dir)
    all_names = [str(i) for i in range(1, N_ROI + 1)]
    cmap = _custom_cmap()

    contrasts = [
        ("sh_vs_wt", corr_sh, corr_wt),
        ("gt_vs_wt", corr_gt, corr_wt),
        ("sh_vs_gt", corr_sh, corr_gt),
    ]
    lvl = f"BH-FDR q<{q}" if sig == "fdr" else f"uncorrected p<{p_thresh}"
    print(f"schema balls: edge significance = {lvl}")

    figs = []
    for name, gA, gB in contrasts:
        p = _ttest2_grid(gA, gB)
        # Edge color and width scale with the difference in group means,
        # rescaled by MD_SCALE for the drawn line-width range.
        md = np.nanmean(gA, 0) - np.nanmean(gB, 0)   # first group minus second
        if sig == "fdr":                       # BH-FDR per contrast over the 2701 edges
            h = _bh_fdr_edges(p, q)
        else:                                  # uncorrected per-edge threshold
            h = np.nan_to_num(p, nan=1.0) < p_thresh
        graph_to_plot = h * MD_SCALE * md      # masked, scaled mean difference

        fig, ax = plt.subplots(figsize=(1.85, 2.05), facecolor="w")
        _plot_circle_graph(ax, graph_to_plot, all_names, STY, cmap)
        gt_figStyle(fig, STY)
        figs.append(fig)
        n_sig = int(np.sum(np.triu(h, 1)))
        print(f"{name}: n significant edges (upper triangle) = {n_sig} / {N_ROI*(N_ROI-1)//2}")

    for i, fig in enumerate(figs, start=1):
        gt_saveFig(fig, f"fig5_c_region_circles_p{i}", out_dir)

    return figs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Figure 5c schema balls")
    ap.add_argument("--sig", choices=["fdr", "uncorrected"], default="uncorrected",
                    help="edge significance: raw per-edge p threshold (default) or BH-FDR "
                         "(note: FDR over 2701 edges leaves 0 significant on this data)")
    ap.add_argument("--q", type=float, default=0.05, help="FDR level (default 0.05)")
    ap.add_argument("--p-thresh", type=float, default=P_THRESH,
                    help="raw p cutoff for --sig uncorrected (default 0.05)")
    a = ap.parse_args()
    main(sig=a.sig, q=a.q, p_thresh=a.p_thresh)
