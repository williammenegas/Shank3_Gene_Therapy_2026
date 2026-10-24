"""
Figure 2, panel a: behavioral-state usage, one plot per genotype
(WT, SH, GT).

Marmoset behavior is discretized into 80 clustered states. State identity
1-80 uses the same numbering in every panel of Figures 2 and 3 that
references `hlabel`. The state-usage matrices are (sessions x 80) fractions
of frames each session spent in each state; the panel plots the
across-session mean usage of each state as a bar, with the y axis fixed to
[0, 0.075].

Each state keeps the same color in every panel built on this ethogram
(2a, 2d, 2g, 3g). The palette is 17 RGB anchors interpolated to 80 colors
through CIELAB rather than RGB, then clipped to [0, 1]
(gtstyle.gt_lab_colormap). Interpolating through CIELAB keeps the perceptual
step between adjacent states roughly uniform.

No statistical test is applied; the panel is descriptive.

Data: Figure_2a_data.mat
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig2", "fig2_a_state_usage")

# 17 RGB anchors (0-255) from which the 80-state palette is interpolated.
CM_INPUT = np.array([
    [101, 98, 214], [118, 97, 209], [155, 95, 192], [166, 95, 187],
    [184, 101, 176], [202, 110, 163], [230, 127, 141], [245, 137, 128],
    [251, 147, 119], [252, 164, 111], [253, 177, 103], [253, 189, 95],
    [250, 151, 124], [247, 131, 136], [244, 115, 144], [241, 98, 152],
    [240, 91, 154],
], dtype=float) / 255.0

N_STATES = 80


def _state_usage_panel(usage, cmap, STY, outname, out_dir):
    fig = plt.figure(figsize=(1.30, 0.98), facecolor="w")
    ax = fig.add_axes([0.20, 0.24, 0.76, 0.62])
    for f in range(N_STATES):
        # Stored as fractions; the published axis is per cent.
        ax.bar(f + 1, 100.0 * usage[f], width=0.8, color=cmap[f], edgecolor="none")
    ax.set_ylim(0, 7.5)
    gt_axes(ax, STY)
    ax.set_xlabel("Behavioral state", fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_ylabel("State usage (%)", fontname=STY.font.name, fontsize=STY.font.lab)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, outname, out_dir)
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    S = load_struct(os.path.join(data_dir, "Figure_2a_data.mat"))

    cmap = gt_lab_colormap(CM_INPUT, N_STATES)
    print(f"colormap[0]  = {cmap[0]}  (anchor 0 = {CM_INPUT[0]})")
    print(f"colormap[-1] = {cmap[-1]} (anchor -1 = {CM_INPUT[-1]})")

    figs = []
    for name, mat, outname in (
        ("WT", S.wt_state_usage, "fig2_a_state_usage_p1"),
        ("SH", S.sh_state_usage, "fig2_a_state_usage_p2"),
        ("GT", S.gt_state_usage, "fig2_a_state_usage_p3"),
    ):
        usage = np.nanmean(np.asarray(mat, dtype=float), axis=0)
        print(f"{name}: usage.shape={usage.shape} sum={usage.sum():.4f} "
              f"max={usage.max():.4f} (state {int(np.argmax(usage))+1})")
        figs.append(_state_usage_panel(usage, cmap, STY, outname, out_dir))

    return figs


if __name__ == "__main__":
    main()
