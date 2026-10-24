"""Extended Data Figure 3, panels g and h: CRISPR off-target analysis layouts.

Renders three presentations of the whole-genome-sequencing off-target analysis:

  edfig3_gh_offtarget_options_opt1_schematic
      Flow diagram in three stages: (a) sequenced founders and relatives,
      (b) de-novo variant detection, (c) intersection with CRISPOR off-target
      predictions. The counts drawn are the constants defined below.
  edfig3_gh_offtarget_options_opt2_distance
      Distribution of the distance from each observed variant to the nearest
      CRISPOR-predicted off-target site, one curve per variant table, with a
      dashed reference line at THRESH_BP.
  edfig3_gh_offtarget_options_opt3_genome
      Observed variants binned along concatenated chromosomes, with the
      guideA and guideE predictions marked as ticks above.

Data: the CRISPOR prediction tables (guideA, guideE) and the founder variant
tables in new_data/WGS. Options 2 and 3 are computed from those tables;
option 1 uses the summary constants below.

THRESH_BP is the proximity threshold in base pairs used to call an observed
variant coincident with a predicted off-target site.

The figures report counts and nearest-neighbor distances; no statistical test
is applied.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import central_output_dir  # noqa: E402
from gtstyle import gt_init, gt_style  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data", "edfig3")
OUT = central_output_dir(HERE)

# ---- Summary counts drawn in the schematic -------------------------------
KELLE_N, SPOCK_N = 40354, 33336
BCFTOOLS_K = 63720
GA_N, GE_N = 179, 534
KELLE_HITS, SPOCK_HITS = 0, 4          # within 10 bp of a CRISPOR prediction
THRESH_BP = 10
C_A = np.array([0.75, 0.22, 0.22])     # guideA red
C_E = np.array([0.24, 0.35, 0.65])     # guideE blue
C_VAR = np.array([0.45, 0.45, 0.45])


# ---- Option 1: analysis-flow schematic (stages a, b, c) ------------------
def option1_schematic(STY):
    fig, ax = plt.subplots(figsize=(4.6, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="k", fs=None, bold=False):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=1.2,rounding_size=2",
            facecolor=fc, edgecolor=ec, linewidth=0.8))
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fs or STY.font.tick, fontweight="bold" if bold else "normal",
                color="k")

    def arrow(y0, y1, x=50):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))

    # (a) samples
    ax.text(1, 98.5, "a", fontsize=STY.font.letter, fontweight="bold")
    box(50, 92, 92, 9,
        "WGS of founders + relatives\nKelle-Zoroark-Debra  |  Spock-Zoroark-Miles-Tildy\n"
        "(Spock's mother N/A -> maternal grandparents; 37-60x)",
        fc=(0.93, 0.93, 0.96), fs=6.2)
    arrow(87.5, 83)

    # (b) detection
    ax.text(1, 82, "b", fontsize=STY.font.letter, fontweight="bold")
    box(50, 76, 92, 8, "bwa-mem + GATK  ->  14.1M SNVs, 3.6M indels",
        fc=(0.93, 0.93, 0.96), fs=6.5)
    arrow(72, 66)
    box(50, 61, 92, 9,
        f"Variants NOT in parents\nKelle {KELLE_N:,}   |   Spock {SPOCK_N:,}\n"
        f"(bcftools trio-dnm2, Kelle: {BCFTOOLS_K:,})",
        fc=(0.90, 0.94, 0.90), fs=6.2)

    # (c) CRISPOR + intersection
    ax.text(1, 54, "c", fontsize=STY.font.letter, fontweight="bold")
    box(50, 48, 92, 8,
        f"CRISPOR off-target predictions (<=4 mm)\nguideA {GA_N}   |   guideE {GE_N}",
        fc=(0.96, 0.92, 0.92), fs=6.2)
    arrow(56.5, 40)
    arrow(44, 40)   # visual merge
    box(50, 35, 92, 8,
        f"Observed variants within {THRESH_BP} bp of a prediction\n"
        f"Kelle {KELLE_HITS}    |    Spock {SPOCK_HITS} (all guideE, 4-mismatch)",
        fc=(0.97, 0.97, 0.90), fs=6.2)
    arrow(31, 23)
    box(50, 18, 92, 9,
        "Spock's 4 -> mapping artifacts\n(homozygous non-ref; alleles also in parents)\n-> excluded",
        fc=(0.97, 0.97, 0.90), fs=6.0)
    arrow(13.5, 7)
    box(50, 3.5, 92, 6, "0 confirmed off-targets in Kelle & Spock",
        fc=(0.20, 0.55, 0.30), ec=(0.12, 0.35, 0.20), fs=8, bold=True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"edfig3_gh_offtarget_options_opt1_schematic.{ext}"),
                    dpi=200, facecolor="w", bbox_inches="tight")
    plt.close(fig)
    print("saved edfig3_gh_offtarget_options_opt1_schematic")


# ---- Options 2 and 3: computed from the variant tables in new_data/WGS ---
VARFILES = {"Kelle": "Founder_K_variants.txt", "Spock": "Founder_S_variants.txt",
            "Kelle (bcftools)": "Founder_K_variants_from_bcftoolstriodnm2.txt"}
VC = {"Kelle": C_A, "Spock": C_E, "Kelle (bcftools)": np.array([0.55, 0.30, 0.80])}


def _load_offtargets():
    dfs = []
    for g, f, c in [("guideA", "CRISPOR_offtarget_predictions_guideA.tsv", "A"),
                    ("guideE", "CRISPOR_offtarget_predictions_guideE.tsv", "E")]:
        d = pd.read_csv(os.path.join(DATA, f), sep="\t")
        dfs.append(d[["chrom", "start", "end"]].assign(guide=g))
    return pd.concat(dfs, ignore_index=True)


def _nearest(V, O):
    dist = np.full(len(V), np.inf)
    for chrom, sv in V.groupby("CHROM"):
        so = O[O["chrom"] == chrom]
        if so.empty:
            continue
        vlo = sv["POS0"].to_numpy()[:, None]; vhi = sv["END"].to_numpy()[:, None]
        olo = so["start"].to_numpy()[None, :]; ohi = so["end"].to_numpy()[None, :]
        gap = np.maximum(0, np.maximum(vlo - ohi, olo - vhi)).min(axis=1)
        dist[V.index.get_indexer(sv.index)] = gap
    return dist


def option2_distance(STY):
    O = _load_offtargets()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for name, f in VARFILES.items():
        V = pd.read_csv(os.path.join(DATA, f), sep="\t")
        d = _nearest(V, O); d = d[np.isfinite(d)]
        ax.hist(np.log10(d + 1), bins=40, histtype="step", color=VC[name], lw=1.3,
                density=True, label=f"{name} (n={len(V):,})")
    ax.axvline(np.log10(THRESH_BP + 1), ls="--", color=(0.3, 0.3, 0.3), lw=0.6)
    ax.text(np.log10(THRESH_BP + 1), ax.get_ylim()[1] * 0.96, f" {THRESH_BP} bp",
            fontsize=6, va="top", color=(0.3, 0.3, 0.3))
    ax.set_xlabel("log10(distance to nearest CRISPOR prediction + 1 bp)", fontsize=STY.font.lab)
    ax.set_ylabel("Density", fontsize=STY.font.lab)
    ax.set_title("Distance of observed variants to predictions",
                 fontsize=STY.font.title)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="out", labelsize=STY.font.tick, length=2.7, pad=1.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"edfig3_gh_offtarget_options_opt2_distance.{ext}"),
                    dpi=200, facecolor="w", bbox_inches="tight")
    plt.close(fig)
    print("saved edfig3_gh_offtarget_options_opt2_distance")


def option3_genome(STY):
    O = _load_offtargets()
    # chromosome ordering + cumulative offsets from union of variant + off-target chroms
    V_all = pd.read_csv(os.path.join(DATA, VARFILES["Kelle (bcftools)"]), sep="\t")
    chroms = sorted(set(O["chrom"]) | set(V_all["CHROM"]),
                    key=lambda c: (len(c), c))
    maxpos = {}
    for c in chroms:
        m = 0
        vc = V_all[V_all["CHROM"] == c]["END"]
        oc = O[O["chrom"] == c]["end"]
        if len(vc): m = max(m, vc.max())
        if len(oc): m = max(m, oc.max())
        maxpos[c] = m + 1
    offset = {}; run = 0
    for c in chroms:
        offset[c] = run; run += maxpos[c]

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    # observed variants (density rug, subsample)
    xg = (V_all["CHROM"].map(offset).to_numpy() + V_all["POS0"].to_numpy()) / 1e6
    xg = xg[np.isfinite(xg)]
    ax.hist(xg, bins=400, color=C_VAR, alpha=0.5, label=f"observed variants (n={len(V_all):,})")
    # predicted off-targets as ticks on top
    for g, c in [("guideA", C_A), ("guideE", C_E)]:
        sub = O[O["guide"] == g]
        xo = (sub["chrom"].map(offset).to_numpy() + sub["start"].to_numpy()) / 1e6
        y = ax.get_ylim()[1]
        ax.plot(xo, np.full(xo.size, y * 0.98), "|", color=c, ms=7, mew=1.0, label=f"{g} predicted")
    ax.set_xlabel("Genomic position (Mb, chromosomes concatenated)", fontsize=STY.font.lab)
    ax.set_ylabel("Observed variant\ncount / bin", fontsize=STY.font.lab)
    ax.set_title("Predicted off-targets vs observed variants across the genome",
                 fontsize=STY.font.title)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="out", labelsize=STY.font.tick, length=2.7, pad=1.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"edfig3_gh_offtarget_options_opt3_genome.{ext}"),
                    dpi=200, facecolor="w", bbox_inches="tight")
    plt.close(fig)
    print("saved edfig3_gh_offtarget_options_opt3_genome")


def main():
    gt_init()
    STY = gt_style()
    option1_schematic(STY)
    option2_distance(STY)
    option3_genome(STY)


if __name__ == "__main__":
    main()
