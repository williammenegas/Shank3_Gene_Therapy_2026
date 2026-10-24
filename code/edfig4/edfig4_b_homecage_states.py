"""Extended Data Figure 4, panel b: behavioral-state usage maps, WT/SH/GT.

Data: `Figure_S4b_data.mat`, holding `wt_state_usage`, `sh_state_usage` and
`gt_state_usage`, each 1000 sessions x 80 behavioral states. Averaging over
sessions (ignoring NaN) gives the per-state usage fraction for each group.

This script produces two figures per group, six in total.

  1. State map. Example t-SNE coordinates
     (`hcoord_Red_sparky_1_control_2_12months.csv`, 660000 x 2) and their raw
     state labels (`hlabel_Red_sparky_1_control_2_12months.csv`, codes 1-80)
     are relabeled through the fixed permutation LIST_5. Each state is drawn
     as one dot at the median (x, y) of its points, with area 250 * alpha,
     where alpha = clip(density * 20, 0.2, 0.8) and density is that state's
     share of the example points. Color comes from a 17-anchor RGB colormap
     interpolated to 80 states in CIELAB (D65 sRGB <-> CIELAB conversion is
     implemented locally in rgb2lab/lab2rgb); the colormap is presentational
     and feeds no computed quantity.

     The map is built from the same example coordinates and labels for every
     group, so the three state maps are identical; the group-specific usage
     values appear in the bar chart below.

  2. Bar chart of the group's mean per-state usage, one bar per state 1-80 in
     the same colormap, y axis fixed to [0, 0.075].

The panel is descriptive; no statistical test is applied.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_a_state_usage")
DATA_HLABEL = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_d_state_maps")
DATA_HCOORD = os.path.join(HERE, "..", "..", "data", "fig3", "fig3_g_approach_avoid")

# ---- fixed 80-element behavioral-state relabeling permutation ----------
LIST_5 = np.array([
    1, 6, 29, 13, 36, 44, 2, 37, 60, 43, 62, 63, 18, 38, 14, 22, 3, 67, 7, 45,
    68, 8, 40, 69, 30, 9, 19, 70, 53, 25, 16, 71, 23, 72, 28, 73, 74, 32, 54,
    46, 4, 64, 26, 47, 65, 55, 27, 10, 48, 20, 41, 75, 33, 24, 56, 42, 39, 5,
    17, 34, 66, 49, 50, 76, 21, 61, 11, 58, 77, 51, 78, 15, 52, 35, 31, 79,
    12, 59, 57, 80,
]) - 1  # -> 0-based

# ---- 17 RGB anchor colors (0-255, scaled to 0-1) interpolated to 80 states
CM_INPUT = np.array([
    [101, 98, 214], [118, 97, 209], [155, 95, 192], [166, 95, 187],
    [184, 101, 176], [202, 110, 163], [230, 127, 141], [245, 137, 128],
    [251, 147, 119], [252, 164, 111], [253, 177, 103], [253, 189, 95],
    [250, 151, 124], [247, 131, 136], [244, 115, 144], [241, 98, 152],
    [240, 91, 154],
]) / 255.0

# D65 2-degree white point (standard CIE values, normalized to Y=1)
_XN, _YN, _ZN = 0.95047, 1.00000, 1.08883
_SRGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ2SRGB = np.linalg.inv(_SRGB2XYZ)


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=float), 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb2lab(rgb):
    """Convert sRGB in [0, 1] with trailing axis of length 3 to CIELAB (D65),
    using the standard CIE formulas."""
    rgb = np.asarray(rgb, dtype=float)
    lin = _srgb_to_linear(rgb)
    xyz = lin @ _SRGB2XYZ.T
    x, y, z = xyz[..., 0] / _XN, xyz[..., 1] / _YN, xyz[..., 2] / _ZN

    d = 6.0 / 29.0
    def f(t):
        return np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4.0 / 29.0)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab2rgb(lab):
    """Convert CIELAB (D65) to sRGB in [0, 1]; inverse of rgb2lab above."""
    lab = np.asarray(lab, dtype=float)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    d = 6.0 / 29.0
    def finv(t):
        return np.where(t > d, t ** 3, 3 * d ** 2 * (t - 4.0 / 29.0))
    x, y, z = _XN * finv(fx), _YN * finv(fy), _ZN * finv(fz)
    xyz = np.stack([x, y, z], axis=-1)
    lin = xyz @ _XYZ2SRGB.T
    return _linear_to_srgb(lin)


def _make_cmap(n=80):
    """Interpolate the CM_INPUT anchor colors to `n` colors in CIELAB."""
    p = CM_INPUT.shape[0]
    lab = rgb2lab(CM_INPUT)
    idx_src = np.arange(1, p + 1)
    idx_tgt = np.linspace(1, p, n)
    lab_i = np.stack([np.interp(idx_tgt, idx_src, lab[:, k]) for k in range(3)], axis=-1)
    cm = lab2rgb(lab_i)
    return np.clip(cm, 0, 1)


def _tsne_watershed_scatter(ax, x, y, labels, cmap, num_labels=80, title=""):
    """Draw one marker per behavioral state at the median (x, y) of that
    state's points, sized by the state's share of all points.

    Coordinates are the native t-SNE values, which span about -92..96,
    the range the published panel uses. Marker area is
    250 * clip(density * 20, 0.2, 0.8).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels, dtype=int)

    density_per_label = np.array([(labels == i).sum() for i in range(1, num_labels + 1)], dtype=float)
    density_per_label = density_per_label / density_per_label.sum()
    alpha_values = np.clip(density_per_label * 20, 0.2, 0.8)

    for f in range(1, num_labels + 1):
        m = labels == f
        xf = np.nanmedian(x[m]) if m.any() else np.nan
        yf = np.nanmedian(y[m]) if m.any() else np.nan
        ax.scatter(xf, yf, s=250 * alpha_values[f - 1], c=[cmap[f - 1]],
                   edgecolors="k", linewidths=0.5, zorder=2)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_aspect("equal")
    # The published panel titles each map with its group.
    ax.set_title(title, fontsize=STY_TITLE_FS)
    ax.set_xlabel("tSNE-1")
    ax.set_ylabel("tSNE-2")
    return alpha_values


STY_TITLE_FS = 8  # module default; replaced from gt_style() inside main()


def main(data_dir=DATA, out_dir=HERE):
    global STY_TITLE_FS
    STY = gt_init()
    STY_TITLE_FS = STY.font.title

    S = load_struct(os.path.join(data_dir, "Figure_2a_data.mat"))

    rT = np.loadtxt(os.path.join(DATA_HCOORD, "hcoord_Red_sparky_1_control_2_12months.csv"),
                     delimiter=",")
    rB = np.loadtxt(os.path.join(DATA_HLABEL, "hlabel_Red_sparky_1_control_2_12months.csv"))
    hlabel_r = LIST_5[rB.astype(int) - 1] + 1   # relabeled state codes, 1-based

    cmap = _make_cmap(80)

    groups = [
        ("wt", np.asarray(S.wt_state_usage, dtype=float)),
        ("sh", np.asarray(S.sh_state_usage, dtype=float)),
        ("gt", np.asarray(S.gt_state_usage, dtype=float)),
    ]

    figs = []
    for name, usage in groups:
        usage_to_plot = np.nanmean(usage, axis=0)
        print(f"  {name}: usage_to_plot sum={usage_to_plot.sum():.4f}, "
              f"min={usage_to_plot.min():.5f}, max={usage_to_plot.max():.5f}")

        # ---- state map (built from the shared example coordinates and
        # labels, so it is the same for all three groups) ----
        fig_w, ax_w = plt.subplots(figsize=(4.5, 4.5), facecolor="white")
        _tsne_watershed_scatter(ax_w, rT[:, 0], rT[:, 1], hlabel_r, cmap,
                                title=name.upper())
        gt_figStyle(fig_w, STY)
        figs.append((f"edfig4_b_homecage_states_watershed_{name}", fig_w))

        # ---- bar chart of this group's mean per-state usage ---------------
        fig_b, ax_b = plt.subplots(figsize=(6.0, 2.0), facecolor="w")
        for f in range(80):
            # Stored as fractions; the published axis is per cent.
            ax_b.bar(f + 1, 100.0 * usage_to_plot[f], color=cmap[f], edgecolor="none")
        ax_b.set_ylim(0, 7.5)
        ax_b.set_xlabel("Behavioral state (ID)", fontname=STY.font.name,
                        fontsize=STY.font.lab)
        ax_b.set_ylabel("State usage (%)", fontname=STY.font.name,
                        fontsize=STY.font.lab)
        gt_axes(ax_b, STY)
        gt_figStyle(fig_b, STY)
        figs.append((f"edfig4_b_homecage_states_bar_{name}", fig_b))

    for stem, fig in figs:
        gt_saveFig(fig, stem, out_dir)

    return [f for _, f in figs]


if __name__ == "__main__":
    main()
