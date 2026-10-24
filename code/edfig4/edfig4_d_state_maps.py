"""Extended Data Figure 4, panel d: home-cage occupancy, by category.

This script draws the same densities panel c overlays, one category at a
time so that each can be read on its own. Climbing hugs the walls, social
resting sits in one corner, and the differences between groups are easier to
see apart than stacked.

Groups. WT, SH and GT, one figure of five maps each.

Data. data/edfig4/edfig4_c_state_maps_merge/state_density_by_group.npz,
which panel c owns. See _state_density.py for the reduction behind it.

Display. Five maps in a row, in the published order: Climbing, Active,
Alone, Social active, Social resting. Each is a constant-color image whose
alpha is the smoothed, normalized density over the 800 x 800 arena, with the
boundary square drawn. Normalizing each category against its own mean and
spread lets a rare category read as clearly as a common one; it also means
intensity cannot be compared between maps.

Nothing is tested; the panel describes where time was spent.
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
    print("Home-cage occupancy density, by category")
    for group, label in SD.GROUPS:
        fig, axes = plt.subplots(1, 5, figsize=(7.4, 1.70), facecolor="w")
        line = []
        for ax, (name, rgb), key in zip(axes, SD.COLOR_LAYERS, SD.CACHE_KEYS):
            hist = cache[f"{group}_{key}"]
            ax.imshow(SD.layer_rgba(hist, rgb), origin="lower",
                      extent=SD.EXTENT, interpolation="bilinear", zorder=2)
            SD.draw_arena(ax)
            ax.set_title(name, fontname=STY.font.name, fontsize=STY.font.tick)
            line.append(f"{name} {int(hist.sum()):,}")
        axes[0].text(-0.06, 0.5, label, transform=axes[0].transAxes,
                     ha="right", va="center", rotation=90,
                     fontname=STY.font.name, fontsize=STY.font.lab)
        gt_figStyle(fig, STY)
        figs.append(fig)
        print(f"  {label}: " + "  ".join(line))

    for fig, (group, _) in zip(figs, SD.GROUPS):
        gt_saveFig(fig, f"edfig4_d_state_maps_{group}", out_dir)
    return figs


if __name__ == "__main__":
    main()
