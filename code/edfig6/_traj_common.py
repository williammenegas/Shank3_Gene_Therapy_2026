"""Shared implementation for the example trajectory and ethogram panels.

Extended Data Figure 5 panels c and h are both 2 x 3 grids of per-animal
spatial-density maps produced by the same routine. That routine lives here;
each panel script supplies only its two source files, the example animal
indices and the row labels.

Layout. Six sub-panels on a 9.0 x 6.2 in canvas. The top row comes from the
first source file and the bottom row from the second; within each row the
three panels are the same three example animals in the order WT, SH, GT.

Per panel. One animal's position trace is drawn as a faint gray
moving-average trajectory line, then five semi-transparent color layers
(blue, purple, red, orange, yellow), one per group of behavioral states, each
a smoothed 2-D occupancy density. A black 700 x 700 square marking the arena
boundary is drawn last, and is drawn even when the animal contributes no valid
samples.

Data selection. For the requested condition N (1 = WT, 2 = SH, 3 = GT), the
rows of `ID_condition_N` matching the requested animal index select the
corresponding rows of `u_condition_N` and `v_condition_N` (position),
`Bmat_condition_N` (per-sample behavioral-state code 1-80) and
`Dist_condition_N` (per-sample distance moved, used as a validity mask). Rows
are concatenated in order. Only samples with Dist > 0 are kept, capped at the
first 500,000 such samples.

Color layers. The eight state groups INDS_0 to INDS_7 (alone, near other
climbing, near other moving, move to or from other, large movements, climbing
alone, alone looking at other, near other resting) are pooled into five
layers, drawn in this order:

    blue    INDS_5                    climbing alone
    purple  INDS_4, INDS_3, INDS_1    large movement, move to or from other,
                                      near other climbing
    red     INDS_0, INDS_6            alone; alone looking at other
    orange  INDS_2                    near other moving
    yellow  INDS_7                    near other resting

INDS_0 and INDS_6 are not disjoint: INDS_6 is INDS_0 plus codes 60, 61, 63, 64
and 65. Both sets contribute to the red layer independently, so a sample
carrying one of the 16 shared codes is counted twice in that layer's density.

Orientation. The color layers are accumulated with the y coordinate on the
first histogram axis and x on the second, while the gray trajectory line and
the boundary square are drawn in the usual (x, y) order; the five density
layers are therefore transposed relative to the trajectory and the square.

Density image. A 100 x 100 histogram over edges 1 to 801 in steps of 8 is
upsampled with cubic spline interpolation, clipped at zero, normalized by
mean + 2 x SD of the upsampled field and clipped to [0, 1]. The result is used
as a per-pixel opacity for one flat color rather than as a colormap, and is
drawn without transposition using origin='upper'.

Transparency is genuine alpha. EPS cannot store alpha, so the transparency in
these panels is correct only in the PDF output.

Memory. Each `xy_*.mat` file is 500-680 MB and holds u, v, ID, Bmat and Dist
for all three conditions, each sized (animals in condition, 500000).
`_load_animal_track` reads one variable at a time, slices immediately to the
matching animal rows and releases the full array before reading the next, so
peak memory stays near the size of a single variable rather than the file.

Rendering parameters. `_ZOOM_FACTOR` (default 2.5) sets the upsampling of the
density histogram, with matplotlib's bilinear interpolation completing the
smoothing at render time; the alpha image is written as uint8 RGBA.
`_TRAJ_MAX_POINTS` (default 15000) caps the number of vertices handed to the
renderer for the background trajectory line by taking every Nth sample, while
the moving average itself is computed on the full sequence. Both affect
rendering time and output file size only.

The panels are descriptive; no statistical test is applied.
"""

import gc
import os
import sys

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import zoom as _zoom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))

# Rendering parameters. See the module docstring.
_ZOOM_FACTOR = 2.5
_TRAJ_MAX_POINTS = 15000

# ---- behavioral-state groupings (1-based state codes) -------------------
INDS_0 = [62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80]
INDS_1 = [30, 31, 32, 33, 35, 40, 41, 42]
INDS_2 = [36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59]
INDS_3 = [15, 19, 21, 22, 23, 24, 27, 29, 34]
INDS_4 = [6, 7, 8, 9, 10, 11, 12, 28]
INDS_5 = [1, 2, 3, 4, 5, 13, 14, 16, 17, 18, 20, 25, 26]
INDS_6 = [62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80,
          60, 61, 63, 64, 65]
INDS_7 = [43, 53, 54, 55, 56, 57]

COL_BLUE = (102 / 255, 98 / 255, 215 / 255)
COL_PURPLE = (168 / 255, 95 / 255, 186 / 255)
COL_RED = (240 / 255, 92 / 255, 155 / 255)
COL_ORANGE = (251 / 255, 141 / 255, 123 / 255)
COL_YELLOW = (254 / 255, 193 / 255, 93 / 255)


# ---------------------------------------------------------------------------
# Data loading, one variable at a time. See the module docstring.
# ---------------------------------------------------------------------------
def _load_animal_track(mat_path, id_type, ind_to_try, n_cols=500000):
    """Load one animal's track from one condition of an `xy_*.mat` file.

    id_type selects the condition (1 = WT, 2 = SH, 3 = GT) and ind_to_try the
    animal ID. Returns (Red, rB, dist_total_red): Red is (n_samples, 2) float64
    [u, v] position, rB is the (n_samples,) behavioral-state code and
    dist_total_red the (n_samples,) distance moved used as a validity mask.
    n_samples is 0 when no row matches ind_to_try.
    """
    cond = id_type
    id_key = f"ID_condition_{cond}"
    d = sio.loadmat(mat_path, variable_names=[id_key])
    ID = np.asarray(d[id_key], dtype=float).ravel()
    mask = ID == ind_to_try
    del d
    gc.collect()

    if not np.any(mask):
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,))

    u_key = f"u_condition_{cond}"
    d = sio.loadmat(mat_path, variable_names=[u_key])
    R1 = np.asarray(d[u_key], dtype=float)[mask, :]
    del d
    gc.collect()

    v_key = f"v_condition_{cond}"
    d = sio.loadmat(mat_path, variable_names=[v_key])
    R2 = np.asarray(d[v_key], dtype=float)[mask, :]
    del d
    gc.collect()

    # Row-major flatten: selected rows are concatenated in their original order.
    R1f = R1.ravel()
    R2f = R2.ravel()
    Red = np.column_stack([R1f, R2f])
    del R1, R2
    gc.collect()

    b_key = f"Bmat_condition_{cond}"
    d = sio.loadmat(mat_path, variable_names=[b_key])
    Bsel = np.asarray(d[b_key], dtype=float)[mask, :n_cols]
    del d
    gc.collect()
    rB = Bsel.ravel()
    del Bsel
    gc.collect()

    dist_key = f"Dist_condition_{cond}"
    d = sio.loadmat(mat_path, variable_names=[dist_key])
    Dsel = np.asarray(d[dist_key], dtype=float)[mask, :n_cols]
    del d
    gc.collect()
    dist_total_red = Dsel.ravel()
    del Dsel
    gc.collect()

    return Red, rB, dist_total_red


# ---------------------------------------------------------------------------
# Moving average and translucent line drawing
# ---------------------------------------------------------------------------
def _movmean_omitnan(x, window=5):
    """Centered moving mean with NaNs excluded from both sum and count.

    The averaging window shrinks at the array edges rather than being padded.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    half = window // 2
    isnan = np.isnan(x)
    xz = np.where(isnan, 0.0, x)
    ones = (~isnan).astype(float)
    csum = np.concatenate([[0.0], np.cumsum(xz)])
    cnt = np.concatenate([[0.0], np.cumsum(ones)])
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    s = csum[hi] - csum[lo]
    c = cnt[hi] - cnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c > 0, s / np.where(c > 0, c, 1.0), np.nan)
    return out


def _patchline(ax, x, y, color, lw, alpha, zorder=1):
    """Draw x/y as one or more contiguous non-NaN segments, so that a
    genuinely translucent line survives PDF export.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = ~(np.isnan(x) | np.isnan(y))
    if not np.any(good):
        return
    idx = np.flatnonzero(good)
    breaks = np.concatenate([[0], np.flatnonzero(np.diff(idx) > 1) + 1, [idx.size]])
    for s in range(breaks.size - 1):
        seg = idx[breaks[s]:breaks[s + 1]]
        if seg.size < 2:
            continue
        ax.plot(x[seg], y[seg], "-", color=color, linewidth=lw, alpha=alpha,
                 zorder=zorder, solid_capstyle="butt")


# ---------------------------------------------------------------------------
# Density image
# ---------------------------------------------------------------------------
def _create_density_image(ax, x_in, y_in, color, extent=(0, 800, 0, 800), zorder=2,
                            zoom_factor=_ZOOM_FACTOR):
    x_in = np.asarray(x_in, dtype=float)
    y_in = np.asarray(y_in, dtype=float)
    if x_in.size == 0:
        return
    edges = np.arange(1, 802, 8)   # 1 to 801 in steps of 8 -> 101 edges, 100 bins
    H, _, _ = np.histogram2d(x_in, y_in, bins=[edges, edges])

    # Cubic-spline upsampling of the density; matplotlib's bilinear
    # interpolation completes the smoothing at render time. `zoom_factor`
    # trades output file size and rendering time against raster resolution.
    Hf = _zoom(H, zoom_factor, order=3)
    Hf[Hf < 0] = 0.0
    Hf = Hf / 1000.0

    mean_c = float(np.nanmean(Hf))
    std_c = float(np.nanstd(Hf, ddof=1))
    denom = mean_c + 2.0 * std_c
    norm = Hf / denom if denom > 0 else np.zeros_like(Hf)
    alpha = np.clip(norm, 0.0, 1.0)

    # uint8 RGBA: the PS/PDF image writers encode a uint8 raster far more
    # compactly than a float64 one of the same pixel count. Alpha values are
    # unchanged apart from quantization to 1/255.
    rgba = np.empty(alpha.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = int(round(color[0] * 255))
    rgba[..., 1] = int(round(color[1] * 255))
    rgba[..., 2] = int(round(color[2] * 255))
    rgba[..., 3] = np.round(alpha * 255).astype(np.uint8)

    # H[i, j] indexes [first-axis bin, second-axis bin] and is drawn without
    # transposition; see the module docstring on orientation.
    ax.imshow(rgba, extent=extent, origin="upper", interpolation="bilinear",
              zorder=zorder)


# ---------------------------------------------------------------------------
# Per-panel drawing routine
# ---------------------------------------------------------------------------
def draw_example_panel(ax, Red, rB, dist_total_red, label=""):
    """Draw one example animal's trajectory and behavioral-state densities.

    Data loading is left to the caller (`_load_animal_track`) so that a single
    variable read can be reused.
    """
    STY = gt_style()
    n_valid = int(np.sum(dist_total_red > 0)) if dist_total_red.size else 0

    if n_valid > 0:
        keep = dist_total_red > 0
        cap = min(n_valid, 500000)

        Red_v = Red[keep][:cap]
        rB_v = rB[keep][:cap]
        # dist_total_red is not used again beyond the mask and cap.

        hlabel_r = rB_v

        x1 = _movmean_omitnan(Red_v[:, 0], 5) + 400
        y1 = _movmean_omitnan(Red_v[:, 1], 5) + 400
        # Subsample the smoothed trace for drawing only: the moving average is
        # computed on the full sequence above, and only the number of vertices
        # handed to the renderer is reduced.
        if x1.size > _TRAJ_MAX_POINTS:
            step = int(np.ceil(x1.size / _TRAJ_MAX_POINTS))
            x1 = x1[::step]
            y1 = y1[::step]
        _patchline(ax, x1, y1, color=(0.75, 0.75, 0.75), lw=STY.lw.trace, alpha=0.33, zorder=1)

        def _select(codes):
            m = np.isin(hlabel_r, codes)
            return Red_v[m, 0], Red_v[m, 1]

        def _select_multi(code_lists):
            xs, ys = [], []
            for codes in code_lists:
                x, y = _select(codes)
                xs.append(x)
                ys.append(y)
            return np.concatenate(xs), np.concatenate(ys)

        # blue = INDS_5
        bx, by = _select(INDS_5)
        _create_density_image(ax, by + 400, bx + 400, COL_BLUE, zorder=2)

        # purple = INDS_4 + INDS_3 + INDS_1
        px, py = _select_multi([INDS_4, INDS_3, INDS_1])
        _create_density_image(ax, py + 400, px + 400, COL_PURPLE, zorder=3)

        # red = INDS_0 + INDS_6; the sets overlap, so shared codes count twice
        rx, ry = _select_multi([INDS_0, INDS_6])
        _create_density_image(ax, ry + 400, rx + 400, COL_RED, zorder=4)

        # orange = INDS_2
        ox, oy = _select(INDS_2)
        _create_density_image(ax, oy + 400, ox + 400, COL_ORANGE, zorder=5)

        # yellow = INDS_7
        yx, yy = _select(INDS_7)
        _create_density_image(ax, yy + 400, yx + 400, COL_YELLOW, zorder=6)

        ax.set_aspect("equal")
        ax.set_xlim(0, 800)
        ax.set_ylim(0, 800)
        ax.axis("off")

    print(f"  {label}: n_valid={n_valid}"
          + (f" (capped to {min(n_valid, 500000)})" if n_valid > 500000 else ""))

    # Arena boundary square, drawn last and drawn even with zero valid samples.
    ax.plot([50, 750, 750, 50, 50], [50, 50, 750, 750, 50], "k", linewidth=0.5, zorder=10)
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 800)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Figure assembly, shared by the panel c and panel h scripts
# ---------------------------------------------------------------------------
def build_figure(save_stem, data_dir, out_dir,
                  block1_file, block2_file, wt_ind,
                  sh_ind=110, gt_ind=115,
                  block1_label="Condition 1", block2_label="Condition 2"):
    """Build the 2 x 3 example-animal grid and save it as `save_stem`.

    Row 1 is drawn from block1_file and row 2 from block2_file, each using
    conditions 1, 2 and 3 (WT, SH, GT) with that group's example index.
    """
    STY = gt_init()

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.2), facecolor="w")

    specs = [
        (0, 0, block1_file, 1, wt_ind, f"WT ({block1_label})"),
        (0, 1, block1_file, 2, sh_ind, f"SH ({block1_label})"),
        (0, 2, block1_file, 3, gt_ind, f"GT ({block1_label})"),
        (1, 0, block2_file, 1, wt_ind, f"WT ({block2_label})"),
        (1, 1, block2_file, 2, sh_ind, f"SH ({block2_label})"),
        (1, 2, block2_file, 3, gt_ind, f"GT ({block2_label})"),
    ]

    print(f"{save_stem}: building 2x3 example grid "
          f"({block1_label}={os.path.basename(block1_file)}, "
          f"{block2_label}={os.path.basename(block2_file)})")

    for r, c, fname, id_type, ind, label in specs:
        ax = axes[r, c]
        mat_path = os.path.join(data_dir, fname)
        Red, rB, dist_total_red = _load_animal_track(mat_path, id_type, ind)
        draw_example_panel(ax, Red, rB, dist_total_red, label=f"{label} (ID={ind})")
        ax.set_title(label, fontname=STY.font.name, fontsize=STY.font.title)
        del Red, rB, dist_total_red
        gc.collect()

    gt_figStyle(fig, STY)
    gt_saveFig(fig, save_stem, out_dir)
    return fig
