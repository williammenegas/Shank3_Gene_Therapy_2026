"""Extended Data Figure 8, panel c: head motion and physiology during fMRI.

Four plots documenting that the functional runs were acquired under stable
conditions: two example framewise-displacement traces showing the exclusion
criterion, and the heart rate and oxygen saturation of every animal scanned.

Data. data/edfig8:

  * motion_fd_example_clean.csv   -- a run with no frame over the threshold
  * motion_fd_example_flagged.csv -- a run with five frames over it
  * hr_data.mat, spo2_data.mat    -- per-animal means, WT n = 11, SH n = 5,
                                     GT n = 7

The two motion files are columns FD_mm and DVARS_pct over 600 frames; only
FD_mm is drawn. They are two of the 57 per-run files produced by the motion
QC, chosen to match the published examples: exactly two runs have five
frames above the threshold, and the clean run is one of the 51 with none,
picked for a peak displacement typical of that group. DVARS is carried in
the files but is not part of this panel.

Display. The motion traces share a y axis of 0 to 0.2 mm with a dashed rule
at the FD_THRESHOLD, the frames above it ringed, and the count of such
frames written on the panel, exactly as the published figure annotates them.
The two physiology plots show each animal as a point with the group mean and
a standard-deviation whisker.

Groups are colored by the shared palette. No statistical test is applied;
the panel reports acquisition quality rather than a comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import gt_init, gt_axes, gt_figStyle, gt_saveFig, load_struct  # noqa: E402
from gtstyle.plotting import gt_beeswarm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig8")

FD_THRESHOLD = 0.1          # mm; frames above this are flagged for exclusion
FD_YMAX = 0.2
GROUPS = ["WT", "SH", "GT"]
C_FLAG = (0.91, 0.12, 0.79)  # the magenta the published panel rings flagged frames in


def _motion_axis(ax, fd, STY):
    """One framewise-displacement trace with its threshold and flagged frames."""
    frames = np.arange(fd.size)
    over = fd > FD_THRESHOLD
    ax.plot(frames, fd, "-", color=(0.62, 0.62, 0.62), lw=0.5, zorder=1)
    ax.axhline(FD_THRESHOLD, ls=(0, (4, 3)), color=(0.45, 0.45, 0.45), lw=0.6, zorder=2)
    if over.any():
        ax.scatter(frames[over], fd[over], s=8, c=[C_FLAG], edgecolors="none", zorder=3)
    ax.set_xlim(0, fd.size)
    ax.set_ylim(0, FD_YMAX)
    ax.set_xticks([0, 200, 400, 600])
    ax.set_yticks([0.0, 0.1, 0.2])
    ax.set_xlabel("Frame (#)", fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_ylabel("Framewise displacement (mm)",
                  fontname=STY.font.name, fontsize=STY.font.lab)
    ax.text(0.5, 0.94, f"{int(over.sum())} frame(s) > {FD_THRESHOLD:g}",
            transform=ax.transAxes, ha="center", va="top",
            color=(0.45, 0.45, 0.45), fontname=STY.font.name, fontsize=STY.font.leg)
    return int(over.sum())


def _physiology_axis(ax, vals, title, ylabel, ylim, yticks, STY):
    """Per-animal points with the group mean and a standard-deviation whisker."""
    for gi, g in enumerate(GROUPS):
        v = np.asarray(vals[g], dtype=float).ravel()
        v = v[np.isfinite(v)]
        col = STY.cmean[gi]
        ax.scatter(gt_beeswarm(v, gi, 0.10), v, s=STY.mk.pt, c=[col],
                   edgecolors="none", zorder=2)
        m, sd = v.mean(), v.std(ddof=1)
        ax.plot([gi, gi], [m - sd, m + sd], "-", color=col, lw=STY.lw.se, zorder=3)
        ax.plot([gi - 0.14, gi + 0.14], [m, m], "-", color=col, lw=STY.lw.mean, zorder=3)
        print(f"    {title:18} {g}: n={v.size:2}  mean={m:7.2f}  sd={sd:5.2f}")
    ax.set_xlim(-0.55, len(GROUPS) - 0.45)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(GROUPS, fontname=STY.font.name, fontsize=STY.font.tick)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_title(title, fontname=STY.font.name, fontsize=STY.font.title)


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    clean = pd.read_csv(os.path.join(data_dir, "motion_fd_example_clean.csv"))["FD_mm"]
    flagged = pd.read_csv(os.path.join(data_dir, "motion_fd_example_flagged.csv"))["FD_mm"]
    hr = load_struct(os.path.join(data_dir, "hr_data.mat"))
    spo2 = load_struct(os.path.join(data_dir, "spo2_data.mat"))

    AX_H, MB, MT = 1.10, 0.46, 0.30
    widths = [1.30, 1.30, 0.80, 0.80]
    gaps = [0.46, 0.52, 0.46]
    ML, MR = 0.60, 0.14
    FW = ML + sum(widths) + sum(gaps) + MR
    FH = MB + AX_H + MT
    fig = plt.figure(figsize=(FW, FH), facecolor="w")
    lefts, x = [], ML
    for i, wid in enumerate(widths):
        lefts.append(x)
        x += wid + (gaps[i] if i < len(gaps) else 0.0)
    axes = [fig.add_axes([l / FW, MB / FH, wid / FW, AX_H / FH])
            for l, wid in zip(lefts, widths)]

    print("fMRI head motion and physiology")
    n_clean = _motion_axis(axes[0], clean.to_numpy(float), STY)
    n_flag = _motion_axis(axes[1], flagged.to_numpy(float), STY)
    print(f"    motion examples: {n_clean} and {n_flag} frames over {FD_THRESHOLD:g} mm")
    _physiology_axis(axes[2], hr, "Heart rate", "Heart rate (bpm)",
                     (150, 350), [150, 200, 250, 300, 350], STY)
    _physiology_axis(axes[3], spo2, "Oxygen saturation", "SpO2 (%)",
                     (90, 100), [90, 92, 94, 96, 98, 100], STY)
    for ax in axes:
        gt_axes(ax, STY, matlab_ticks=False)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig8_c_motion_physiology", out_dir)
    return fig


if __name__ == "__main__":
    main()
