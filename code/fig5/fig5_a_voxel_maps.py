"""
fig5_a_voxel_maps.py

Figure 5, panel a: horizontal-slice maps of group differences in
resting-state global functional connectivity (WT vs SH, WT vs GT, SH vs GT).

Data. One whole-brain mask dump per animal, global_fconn.<ID>.maskdump.txt,
with five columns: an unused identifier, a grid x index, a grid y index, a
horizontal slice index and the connectivity value. 26 animals, assigned to
groups by animal ID: WT n = 13, SH n = 6, GT n = 7.

Processing. For each of the 60 horizontal slices and each animal, the script
bins the (x, y) grid into an 80 x 60 matrix by averaging the values in each
integer grid cell; bins are matched to coordinate values 1..80 and 1..60.
Animals whose connectivity values extend below zero are shifted upward by the
magnitude of their own minimum, undoing the upstream normalization. Each
slice matrix is then averaged with its own left-right mirror image (columns
1-29 flipped against columns 29-57), giving an 80 x 29 midline-symmetric map.

Statistics. Per pixel and per slice: Cohen's d between the two groups, with
the pooled standard deviation taken as the root mean square of the two group
standard deviations; and a two-sample equal-variance (Student) t-test,
two-sided, with NaNs excluded pixel by pixel. The t distribution CDF is
evaluated with scipy.stats.t.cdf, vectorized over all pixels and slices. No
multiple-comparison correction is applied across pixels.

Display. This script renders twelve slices (indices 22, 24, ..., 44), one
figure per slice, each holding three pointy-topped hexagon grids for the
three contrasts. Hexagon fill is Cohen's d divided by 1.25, clipped to
[-1, 1], on the fixed diverging colormap in fmri_colormap_gp.mat; hexagons
where the uncorrected p-value is below P_THRESH (0.05) carry a black outline.
A black line traces the convex hull of the non-zero hexagon centers as a
brain outline and encodes no statistical information.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from scipy.stats import t as student_t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig5", "fig5_a_voxel_maps")

GROUP1 = ["cas01", "che01", "duc01", "fla01", "hor01", "lyc01", "par01",
          "rex01", "riv01", "sag01", "spa01", "yet01", "yuz01"]           # WT, n=13
GROUP2 = ["dex01", "kyl01", "mel01", "nat01", "poe01", "pos01"]           # SH, n=6
GROUP3 = ["chu01", "ezr03", "fil01", "hel01", "lar02", "rav03", "tor01"]  # GT, n=7

N_ROW, N_COL_FULL, N_COL_HALF = 80, 60, 29
N_SLICES = 60
PLOT_SLICES = list(range(22, 45, 2))   # 1-based slice indices, 12 slices
P_THRESH = 0.05


def _file_id(fname):
    base = os.path.basename(fname)
    parts = base.split(".")
    return parts[1]   # global_fconn.<ID>.maskdump.txt


def _load_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.maskdump.txt")))
    data = {}
    for fn in files:
        a = np.loadtxt(fn)
        a[a == 0] = np.nan
        data[fn] = a
    return files, data


def _slice_grid(a, slice_loop):
    """Bin one animal's values for one slice.

    a: (N, 5) array of (unused id, grid x, grid y, slice, value) with zeros
    already replaced by NaN. Returns an (80, 60) matrix holding the mean value
    per (row = grid y, column = grid x) cell, with bins matched to coordinate
    values 1..80 and 1..60.
    """
    mask = a[:, 3] == slice_loop
    if not np.any(mask):
        return np.full((N_ROW, N_COL_FULL), np.nan)
    x1 = a[mask, 1]   # grid x
    x2 = a[mask, 2]   # grid y
    x3 = a[mask, 4]   # connectivity value

    valid = (x1 >= 1) & (x1 <= N_COL_FULL) & (x2 >= 1) & (x2 <= N_ROW)
    if not np.any(valid):
        return np.full((N_ROW, N_COL_FULL), np.nan)

    row = x2[valid].astype(int) - 1   # grid y 1..80 -> 0-based row
    col = x1[valid].astype(int) - 1   # grid x 1..60 -> 0-based column
    val = x3[valid]

    df = pd.DataFrame({"row": row, "col": col, "val": val})
    means = df.groupby(["row", "col"])["val"].mean()

    grid = np.full((N_ROW, N_COL_FULL), np.nan)
    idx = means.index.to_frame(index=False)
    grid[idx["row"].to_numpy(), idx["col"].to_numpy()] = means.to_numpy()
    return grid


def _lr_average(x_score):
    """Average each slice map with its own left-right mirror image: columns
    1-29 flipped, averaged against columns 29-57 (1-based)."""
    a1 = np.fliplr(x_score[:, 0:29])       # columns 1-29, mirrored
    a2 = x_score[:, 28:57]                 # columns 29-57
    return (a1 + a2) / 2.0


def _pooled_cohend(m1, s1, m2, s2):
    """Cohen's d with the pooled standard deviation taken as the root mean
    square of the two group standard deviations."""
    pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (m1 - m2) / pooled
    return d


def _ttest2_pvalue(g1, g2):
    """Two-sample, equal-variance (pooled) Student t-test, two-sided, with
    NaNs excluded column by column. Vectorized over the last axis."""
    n1 = np.sum(~np.isnan(g1), axis=0).astype(float)
    n2 = np.sum(~np.isnan(g2), axis=0).astype(float)
    m1 = np.nanmean(g1, axis=0)
    m2 = np.nanmean(g2, axis=0)
    with np.errstate(invalid="ignore"):
        v1 = np.nanvar(g1, axis=0, ddof=1)
        v2 = np.nanvar(g2, axis=0, ddof=1)
    df = n1 + n2 - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        tstat = (m1 - m2) / se
        p = 2.0 * (1.0 - student_t.cdf(np.abs(tstat), df))
    bad = (n1 < 2) | (n2 < 2)
    p = np.where(bad, np.nan, p)
    return p


def _compute_group_maps(data_dir):
    files, data = _load_files(data_dir)
    is_wt = np.array([_file_id(f) in GROUP1 for f in files])
    is_sh = np.array([_file_id(f) in GROUP2 for f in files])
    is_gt = np.array([_file_id(f) in GROUP3 for f in files])
    print(f"  files: {len(files)}  WT={is_wt.sum()} SH={is_sh.sum()} GT={is_gt.sum()}")

    n_cols = N_ROW * N_COL_HALF
    x_collect1 = np.full((N_SLICES, n_cols), np.nan)
    x_collect2 = np.full((N_SLICES, n_cols), np.nan)
    x_collect3 = np.full((N_SLICES, n_cols), np.nan)
    Pdiff12 = np.full((N_SLICES, n_cols), np.nan)
    Pdiff13 = np.full((N_SLICES, n_cols), np.nan)
    Pdiff23 = np.full((N_SLICES, n_cols), np.nan)
    diff12 = np.full((N_SLICES, n_cols), np.nan)
    diff13 = np.full((N_SLICES, n_cols), np.nan)
    diff23 = np.full((N_SLICES, n_cols), np.nan)

    for si in range(N_SLICES):
        slice_loop = si + 1   # slice indices are 1-based in the mask dumps
        x_new = np.full((len(files), n_cols), np.nan)
        for fi, fn in enumerate(files):
            a = data[fn]
            x_mean_col = a[:, 4]   # all of this animal's connectivity values
            grid = _slice_grid(a, slice_loop)
            x_score = _lr_average(grid)
            if np.nanmin(x_mean_col) < 0:
                x_score = x_score + abs(np.nanmin(x_mean_col))
            # Column-major flattening; the plotting loop below reshapes with
            # the same order, which is all the pixel indexing requires.
            x_new[fi, :] = x_score.flatten(order="F")

        g1, g2, g3 = x_new[is_wt], x_new[is_sh], x_new[is_gt]
        with np.errstate(invalid="ignore"):
            x_collect1[si] = np.nanmean(g1, axis=0)
            x_collect2[si] = np.nanmean(g2, axis=0)
            x_collect3[si] = np.nanmean(g3, axis=0)

            m1, m2, m3 = x_collect1[si], x_collect2[si], x_collect3[si]
            s1 = np.nanstd(g1, axis=0, ddof=1)
            s2 = np.nanstd(g2, axis=0, ddof=1)
            s3 = np.nanstd(g3, axis=0, ddof=1)

        diff12[si] = _pooled_cohend(m1, s1, m2, s2)
        diff13[si] = _pooled_cohend(m1, s1, m3, s3)
        diff23[si] = _pooled_cohend(m2, s2, m3, s3)

        Pdiff12[si] = _ttest2_pvalue(g1, g2)
        Pdiff13[si] = _ttest2_pvalue(g1, g3)
        Pdiff23[si] = _ttest2_pvalue(g2, g3)

        print(f"  slice {slice_loop}/{N_SLICES} done", end="\r")
    print()

    return dict(diff12=diff12, diff13=diff13, diff23=diff23,
                Pdiff12=Pdiff12, Pdiff13=Pdiff13, Pdiff23=Pdiff23,
                x_collect1=x_collect1, x_collect2=x_collect2, x_collect3=x_collect3)


def _hexagon_grid_plot(ax, data_matrix, binary_matrix, cmap):
    """Render one slice contrast as a pointy-topped hexagon grid. Fill color
    comes from `data_matrix` via `cmap`; hexagons flagged in `binary_matrix`
    are outlined in black. A convex hull of the non-zero hexagon centers is
    drawn as a brain outline."""
    M, N = data_matrix.shape
    h_size = 1.0

    x_idx, y_idx = np.meshgrid(np.arange(N), np.arange(M))
    x_pos = h_size * np.sqrt(3) * (x_idx + 0.5 * (y_idx % 2))
    y_pos = h_size * 1.5 * y_idx
    x_pos = x_pos.ravel()
    y_pos = y_pos.ravel()
    values = data_matrix.ravel()
    binary_values = binary_matrix.ravel().astype(bool)

    angles = np.arange(6) * np.pi / 3 + np.pi / 6
    x_unit = h_size * np.cos(angles)
    y_unit = h_size * np.sin(angles)

    n_cmap = cmap.shape[0]
    values_clipped = np.clip(values, -1, 1)
    values_norm = (values_clipped + 1) / 2
    idx = np.clip(np.round(values_norm * (n_cmap - 1)).astype(int), 0, n_cmap - 1)
    colors = cmap[idx]

    zero_or_nan = np.isnan(values) | (values == 0)
    colors = colors.copy()
    colors[zero_or_nan] = [1, 1, 1]

    patches = []
    edgecolors = []
    for i in range(x_pos.size):
        verts = np.column_stack([x_pos[i] + x_unit, y_pos[i] + y_unit])
        patches.append(mpatches.Polygon(verts, closed=True))
        edgecolors.append((0, 0, 0, 1) if binary_values[i] else (0, 0, 0, 0))

    pc = PatchCollection(patches, facecolors=colors, edgecolors=edgecolors, linewidths=0.4)
    ax.add_collection(pc)

    non_zero = values != 0
    if np.sum(non_zero) >= 3:
        pts = np.column_stack([x_pos[non_zero], y_pos[non_zero]])
        try:
            hull = ConvexHull(pts)
            hull_pts = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
            ax.plot(hull_pts[:, 0], hull_pts[:, 1], "k-", lw=1.0)
        except Exception:
            pass

    ax.set_xlim(x_pos.min() - 1.5, x_pos.max() + 1.5)
    ax.set_ylim(y_pos.min() - 1.5, y_pos.max() + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    print("fig5_a_voxel_maps: computing pixel-wise group maps (60 slices x 26 animals)...")
    R = _compute_group_maps(data_dir)

    cm = load_struct(os.path.join(data_dir, "fmri_colormap_gp.mat"))
    cm_fmri = np.asarray(cm.cm_fmri, dtype=float)

    figs = []
    for pi, f in enumerate(PLOT_SLICES, start=1):
        row = f - 1
        s12 = R["diff12"][row].reshape(N_ROW, N_COL_HALF, order="F")
        s13 = R["diff13"][row].reshape(N_ROW, N_COL_HALF, order="F")
        s23 = R["diff23"][row].reshape(N_ROW, N_COL_HALF, order="F")
        p12 = R["Pdiff12"][row].reshape(N_ROW, N_COL_HALF, order="F") < P_THRESH
        p13 = R["Pdiff13"][row].reshape(N_ROW, N_COL_HALF, order="F") < P_THRESH
        p23 = R["Pdiff23"][row].reshape(N_ROW, N_COL_HALF, order="F") < P_THRESH

        s12 = np.nan_to_num(s12, nan=0.0)
        s13 = np.nan_to_num(s13, nan=0.0)
        s23 = np.nan_to_num(s23, nan=0.0)
        p12 = np.nan_to_num(p12, nan=0.0).astype(bool)
        p13 = np.nan_to_num(p13, nan=0.0).astype(bool)
        p23 = np.nan_to_num(p23, nan=0.0).astype(bool)

        fig, axes = plt.subplots(1, 3, figsize=(2.45, 1.15), facecolor="w")
        _hexagon_grid_plot(axes[0], s12 / 1.25, p12, cm_fmri)
        _hexagon_grid_plot(axes[1], s13 / 1.25, p13, cm_fmri)
        _hexagon_grid_plot(axes[2], s23 / 1.25, p23, cm_fmri)
        gt_figStyle(fig, STY)
        figs.append(fig)
        gt_saveFig(fig, f"fig5_a_voxel_maps_p{pi}", out_dir)
        print(f"  slice f={f}: sig px WT-SH={p12.sum()}, WT-GT={p13.sum()}, SH-GT={p23.sum()}")

    return figs


if __name__ == "__main__":
    main()
