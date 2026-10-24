"""Extended Data Figure 1, panel a: AAV levels by brain region, 3 vs 9 months.

Five brain regions (frontal cortex, visual cortex, striatum, thalamus, CA3),
each measured in three animals at two ages (3 and 9 months). This script
reads the 5 x 2 x 3 values from data/edfig1/aav_by_region.csv, one row
per measurement with columns Region, Age_months, Replicate and AAV_level.
Replicate is the position within each region and age as recorded, and the
regions are drawn in the order they first appear in the file.

The script draws each region and age as one bar at the group mean with the
three individual animal values scattered on top, at x positions 1,2
(frontal), 5,6 (visual), 9,10 (striatum), 13,14 (thalamus) and 17,18 (CA3).

Colors encode age rather than genotype: gray bars with dark gray points for
3 months, violet bars with purple points for 9 months.

The y axis is fixed to [0, 60]. The panel is descriptive; no statistical test
is applied.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig1")

X_PAIRS = [(1, 2), (5, 6), (9, 10), (13, 14), (17, 18)]

COL_BAR_3MO = (0.75, 0.75, 0.75)
COL_BAR_9MO = (0.77, 0.63, 0.93)
COL_PT_3MO = (0.25, 0.25, 0.25)
COL_PT_9MO = (0.43, 0.13, 0.77)


def load_regions(data_dir=DATA):
    """Return region -> (3-month values, 9-month values) from the source CSV."""
    df = pd.read_csv(os.path.join(data_dir, "aav_by_region.csv"))
    return {r: (g[g.Age_months == 3].AAV_level.to_numpy(),
                g[g.Age_months == 9].AAV_level.to_numpy())
            for r, g in df.groupby('Region', sort=False)}


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    REGIONS = load_regions(data_dir)

    fig, ax = plt.subplots(figsize=(5.83, 4.375), facecolor="w")

    for (region, (v3, v9)), (x3, x9) in zip(REGIONS.items(), X_PAIRS):
        ax.bar(x3, np.nanmean(v3), color=COL_BAR_3MO, edgecolor="none")
        ax.bar(x9, np.nanmean(v9), color=COL_BAR_9MO, edgecolor="none")
        ax.scatter(np.full(v3.size, x3), v3, s=STY.mk.pt, c=[COL_PT_3MO], edgecolors="none")
        ax.scatter(np.full(v9.size, x9), v9, s=STY.mk.pt, c=[COL_PT_9MO], edgecolors="none")
        print(f"  {region}: 3mo mean={np.nanmean(v3):.2f} (n={v3.size}), "
              f"9mo mean={np.nanmean(v9):.2f} (n={v9.size})")

    ax.set_ylim(0, 60)
    gt_axes(ax, STY)
    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig1_a_aav_by_region", out_dir)
    return fig


if __name__ == "__main__":
    main()
