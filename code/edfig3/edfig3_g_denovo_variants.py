"""Extended Data Figure 3, panel g: de novo variants absent from both parents.

Every variant called in a founder and in neither of its parents, drawn as a
tick at its genomic position on its chromosome. The two founders share each
chromosome row and are told apart by color, so the panel shows both the
genome-wide spread of the calls and how the two founders compare.

Data. data/edfig3, which the off-target panels own:

  * Founder_K_variants.txt: 9,530 variants
  * Founder_S_variants.txt: 5,277 variants

Both are tab separated with a CHROM accession and a 0-based POS0, among the
per-sample genotype columns this panel does not read. The counts above are
the file lengths less their header, and are the ones quoted in the legend.

Chromosomes. The assembly names its sequences NC_133502.1 upward rather than
by chromosome, so the script maps the accessions to 1-22, X and Y in
accession order. That ordering is the assembly's own: the first twenty-two
run 220.7 Mb down to 51.3 Mb, which is the usual numbering by descending
length, and the last two are a 143.6 Mb sequence carrying very few calls and
a 15.4 Mb sequence, which is what X and Y look like in a male.

Display. Chromosome 1 at the top, position in megabases along x. Each
variant is a short vertical tick, half a row high, so that dense regions
read as solid blocks without any smoothing. The ticks are drawn as one line
collection per founder rather than one line each, which matters at this
count.

No statistical test is applied; the panel is a census of the calls.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import gt_init, gt_axes, gt_figStyle, gt_saveFig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "edfig3")

FOUNDERS = [("Founder K", "Founder_K_variants.txt"),
            ("Founder S", "Founder_S_variants.txt")]

# Sampled from the published panel.
C_K = (0.251, 0.471, 0.345)     # green
C_S = (0.376, 0.282, 0.471)     # purple

CHROM_LABELS = [str(i) for i in range(1, 23)] + ["X", "Y"]
TICK_HALF = 0.25                # half tick height, in row units
X_MAX_MB = 225.0


def _accession_order(frames):
    """Accessions sorted as the assembly names them, which is chromosome order."""
    acc = set()
    for d in frames:
        acc |= set(d["CHROM"].unique())
    return sorted(acc)


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    frames = {}
    for name, fname in FOUNDERS:
        frames[name] = pd.read_csv(os.path.join(data_dir, fname), sep="\t",
                                   usecols=["CHROM", "POS0"])
    acc = _accession_order(frames.values())
    if len(acc) != len(CHROM_LABELS):
        raise ValueError(f"expected {len(CHROM_LABELS)} sequences, found {len(acc)}")
    row_of = {a: i for i, a in enumerate(acc)}

    AX_W, AX_H = 4.20, 2.30
    ML, MR, MB, MT = 0.52, 0.18, 0.46, 0.34
    FW, FH = ML + AX_W + MR, MB + AX_H + MT
    fig = plt.figure(figsize=(FW, FH), facecolor="w")
    ax = fig.add_axes([ML / FW, MB / FH, AX_W / FW, AX_H / FH])

    print("De novo variants absent from parents")
    for (name, _), color in zip(FOUNDERS, (C_K, C_S)):
        d = frames[name]
        rows = d["CHROM"].map(row_of).to_numpy()
        xmb = d["POS0"].to_numpy(float) / 1e6
        segs = np.empty((xmb.size, 2, 2), dtype=float)
        segs[:, 0, 0] = xmb
        segs[:, 1, 0] = xmb
        segs[:, 0, 1] = rows - TICK_HALF
        segs[:, 1, 1] = rows + TICK_HALF
        ax.add_collection(LineCollection(segs, colors=[color], linewidths=0.35))
        print(f"  {name:10} {xmb.size:6,} variants over {d['CHROM'].nunique()} sequences")

    ax.set_xlim(0, X_MAX_MB)
    ax.set_ylim(len(acc) - 0.5, -0.5)          # chromosome 1 at the top
    ax.set_yticks(range(len(acc)))
    ax.set_yticklabels(CHROM_LABELS, fontname=STY.font.name, fontsize=STY.font.tick)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Position (Mb)", fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_ylabel("Chromosome", fontname=STY.font.name, fontsize=STY.font.lab)
    ax.set_title("De novo variants absent from parents",
                 fontname=STY.font.name, fontsize=STY.font.title)
    gt_axes(ax, STY, matlab_ticks=False)

    handles = [ax.plot(np.nan, np.nan, "-", color=c, lw=2)[0] for c in (C_K, C_S)]
    labels = [f"{name} ({frames[name].shape[0]:,} novel alleles)"
              for name, _ in FOUNDERS]
    ax.legend(handles, labels, frameon=False, fontsize=STY.font.leg,
              loc="lower right", handlelength=1.2)

    gt_figStyle(fig, STY)
    gt_saveFig(fig, "edfig3_g_denovo_variants", out_dir)
    return fig


if __name__ == "__main__":
    main()
