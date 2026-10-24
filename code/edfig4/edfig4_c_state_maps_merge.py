"""Extended Data Figure 4, panel c: home-cage occupancy, all categories.

A top-down view of the home cage for each group, with the five behavioral
categories drawn over one another. Each category is a flat color whose
transparency carries how much of that group's time was spent at that spot in
that category, so the picture shows where in the cage each kind of behavior
happens: climbing around the walls, social resting in a corner, and so on.

Groups. WT, SH and GT, one figure each.

Data. data/edfig4/edfig4_c_state_maps_merge/state_density_by_group.npz,
fifteen 100 x 100 histograms, one per group and category. Panel d reads the
same file and draws the categories separately. See _state_density.py for
where the histograms come from and how they are normalized.

Display. This script draws each category as a constant-color image whose
alpha is the smoothed, normalized density, over an 800 x 800 arena with its
boundary square. Colors are the originals: climbing blue, active purple,
alone pink, social active orange, social resting yellow. The published panel
adds a dashed inner square and a marked bed position at assembly; those are
not drawn here, since the code carries no bed coordinates.

Nothing is tested; the panel is a description of where time was spent.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import gt_init, gt_figStyle, gt_saveFig  # noqa: E402
import _state_density as SD  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig4",
                    "edfig4_c_state_maps_merge")
NPZ = "state_density_by_group.npz"


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    cache = np.load(os.path.join(data_dir, NPZ))

    figs = []
    print("Home-cage occupancy density, all categories")
    for group, label in SD.GROUPS:
        fig, ax = plt.subplots(figsize=(1.70, 1.70), facecolor="w")
        counts = {}
        for (name, rgb), key in zip(SD.COLOR_LAYERS, SD.CACHE_KEYS):
            hist = cache[f"{group}_{key}"]
            counts[name] = int(hist.sum())
            ax.imshow(SD.layer_rgba(hist, rgb), origin="lower",
                      extent=SD.EXTENT, interpolation="bilinear", zorder=2)
        SD.draw_arena(ax)
        ax.set_title(label, fontname=STY.font.name, fontsize=STY.font.title)
        gt_figStyle(fig, STY)
        figs.append(fig)
        total = sum(counts.values())
        shares = {k: f"{100.0 * v / total:.1f}%" for k, v in counts.items()}
        print(f"  {label}: {total:,} samples  {shares}")

    for fig, (group, _) in zip(figs, SD.GROUPS):
        gt_saveFig(fig, f"edfig4_c_state_maps_merge_{group}", out_dir)
    return figs


if __name__ == "__main__":
    main()
