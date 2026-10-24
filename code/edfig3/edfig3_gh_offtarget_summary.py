"""Extended Data Figure 3, panels g and h: CRISPR off-target summary.

Two marmoset founders were whole-genome sequenced together with close
relatives: Kelle (43 bp deletion; trio Kelle / Zoroark / 17-B121) and Spock
(G insertion; father Zoroark plus maternal grandparents Miles and Tildy).
GATK-called heterozygous sites absent from the relevant relatives form a
permissive candidate de-novo / off-target set per founder:

    Founder_K_variants.txt                        Kelle, GATK
    Founder_S_variants.txt                        Spock, GATK
    Founder_K_variants_from_bcftoolstriodnm2.txt  Kelle, bcftools trio-dnm2,
        filtered on VAF > 0 only, as a second caller

CRISPOR (Concordet & Haeussler 2018) was run for the two guide RNAs used
(guideA, guideE) against the marmoset T2T assembly, allowing up to four
mismatches. The script reads the CRISPOR predictions from
CRISPOR_offtarget_predictions_guideA.tsv and _guideE.tsv.

This script recomputes the site counts from those files and prints them, and
computes the distance from every candidate site in each founder variant set
to the nearest CRISPOR-predicted off-target locus on the same chromosome.
Every founder set is compared against the union of the guideA and guideE
predictions. Candidate sites on chromosomes carrying no predicted off-target
site are assigned an infinite distance and are excluded from the
within-threshold count.

THRESH_BP (10 bp) is the proximity threshold used to call a candidate site
coincident with a predicted off-target locus, and is drawn as a dashed
reference line on the distance histograms.

Figures produced:
  edfig3_gh_offtarget_summary_predictions  Predicted off-target site counts by
      mismatch count and CFD off-target score distributions, per guide.
  edfig3_gh_offtarget_summary_distance_to_variants  One histogram per founder
      variant set of log10(distance to the nearest predicted site + 1 bp).
  edfig3_gh_offtarget_summary_extended     CFD score ranked descending, genomic
      context of the predictions, the three distance distributions overlaid,
      and read depth at the called sites.
  edfig3_gh_offtarget_summary_top_sites.csv  The ten highest-CFD predicted
      off-target sites with their genomic context and distance to the nearest
      founder variant.

Statistics: counting and nearest-neighbor distance only; no hypothesis test
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

GUIDE_FILES = {"guideA": "CRISPOR_offtarget_predictions_guideA.tsv",
              "guideE": "CRISPOR_offtarget_predictions_guideE.tsv"}
VARIANT_FILES = {
    "Kelle (GATK)": "Founder_K_variants.txt",
    "Spock (GATK)": "Founder_S_variants.txt",
    "Kelle (bcftools trio-dnm2)": "Founder_K_variants_from_bcftoolstriodnm2.txt",
}
GUIDE_COLORS = {"guideA": np.array([0.75, 0.22, 0.22]), "guideE": np.array([0.24, 0.35, 0.65])}
VARIANT_COLORS = {
    "Kelle (GATK)": np.array([0.75, 0.22, 0.22]),
    "Spock (GATK)": np.array([0.24, 0.35, 0.65]),
    "Kelle (bcftools trio-dnm2)": np.array([0.55, 0.30, 0.80]),
}
THRESH_BP = 10


def _nearest_distance(variants, offtargets):
    """Nearest-neighbor distance in base pairs from each variant interval to
    any off-target interval on the same chromosome.

    Distance is 0 for overlapping intervals, otherwise the gap between the
    nearer edges. Variants on chromosomes carrying no predicted off-target
    site are assigned +inf and are excluded from the within-threshold count.
    """
    dist = np.full(len(variants), np.inf)
    for chrom, sub_v in variants.groupby("CHROM"):
        sub_o = offtargets[offtargets["chrom"] == chrom]
        if sub_o.empty:
            continue
        v_lo = sub_v["POS0"].to_numpy()[:, None]
        v_hi = sub_v["END"].to_numpy()[:, None]
        o_lo = sub_o["start"].to_numpy()[None, :]
        o_hi = sub_o["end"].to_numpy()[None, :]
        # gap between intervals, zero if they overlap
        gap = np.maximum(0, np.maximum(v_lo - o_hi, o_lo - v_hi))
        dmin = gap.min(axis=1)
        dist[variants.index.get_indexer(sub_v.index)] = dmin
    return dist


def _genic_category(desc):
    """Leading token of the CRISPOR locusDesc field: exon, intron, intergenic."""
    return str(desc).split(":", 1)[0].strip().lower()


def _founder_depth(V):
    """Per-site read depth (DP) for the founder sample (column index 5),
    parsed from the trailing ';DP' field, e.g. 'Kelle;0|1;99;37,5;42'."""
    col = V.columns[5]
    dp = []
    for s in V[col].astype(str):
        try:
            dp.append(int(s.split(";")[-1]))
        except (ValueError, IndexError):
            pass
    return np.array(dp, dtype=float)


def _min_dist_to_variants(chrom, pos, var_by_chrom):
    """Nearest distance in base pairs from one locus to any founder variant on
    the same chromosome, by sorted-array search; +inf if there are none."""
    arr = var_by_chrom.get(chrom)
    if arr is None or arr.size == 0:
        return np.inf
    j = np.searchsorted(arr, pos)
    cand = []
    if j < arr.size:
        cand.append(abs(arr[j] - pos))
    if j > 0:
        cand.append(abs(pos - arr[j - 1]))
    return float(min(cand)) if cand else np.inf


CAT_ORDER = ["exon", "intron", "intergenic"]
CAT_COL = {"exon": np.array([0.80, 0.20, 0.20]),
           "intron": np.array([0.35, 0.55, 0.80]),
           "intergenic": np.array([0.70, 0.70, 0.70]),
           "other": np.array([0.50, 0.40, 0.65])}


def _extended_panels(guides, offtargets_union, data_dir, out_dir, STY):
    """CFD ranking, genomic context, distance overlay across callers, read
    depth at called sites, and the top-predicted-off-target table (CSV)."""
    # Union of founder variant positions per chromosome, for site-to-variant distance
    var_by_chrom = {}
    depths = {}
    for name, fname in VARIANT_FILES.items():
        V = pd.read_csv(os.path.join(data_dir, fname), sep="\t")
        depths[name] = _founder_depth(V)
        for chrom, sub in V.groupby("CHROM"):
            arr = np.sort(sub["POS0"].to_numpy())
            var_by_chrom[chrom] = (np.concatenate([var_by_chrom[chrom], arr])
                                   if chrom in var_by_chrom else arr)
    for chrom in var_by_chrom:
        var_by_chrom[chrom] = np.sort(var_by_chrom[chrom])

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))

    # (1) CFD score, ranked descending -------------------------------------
    ax = axes[0, 0]
    for name, df in guides.items():
        s = np.sort(df["cfdOfftargetScore"].to_numpy())[::-1]
        ax.plot(np.arange(1, s.size + 1), s, "-", color=GUIDE_COLORS[name],
                lw=1.2, label=f"{name} (max {s[0]:.2f})")
    ax.set_xlabel("Predicted off-target rank", fontsize=STY.font.lab)
    ax.set_ylabel("CFD off-target score", fontsize=STY.font.lab)
    ax.set_title("Off-target likelihood (CFD), ranked", fontsize=STY.font.title)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)

    # (2) genic context ------------------------------------------------------
    ax = axes[0, 1]
    for gi, (name, df) in enumerate(guides.items()):
        cats = df["locusDesc"].map(_genic_category)
        cats = cats.where(cats.isin(CAT_ORDER), "other")
        counts = cats.value_counts()
        bottom = 0.0
        for cat in CAT_ORDER + ["other"]:
            c = int(counts.get(cat, 0))
            if c == 0:
                continue
            ax.bar(gi, c, bottom=bottom, color=CAT_COL[cat], edgecolor="w",
                   lw=0.5, width=0.6, label=cat if gi == 0 else None)
            if cat == "exon":
                ax.text(gi, bottom + c / 2, f"exon={c}", ha="center", va="center",
                        fontsize=6, color="w", fontweight="bold")
            bottom += c
    ax.set_xticks(range(len(guides)))
    ax.set_xticklabels(list(guides.keys()))
    ax.set_ylabel("Predicted off-target sites", fontsize=STY.font.lab)
    ax.set_title("Genomic context of predictions", fontsize=STY.font.title)
    handles = [mpatches.Rectangle((0, 0), 1, 1, color=CAT_COL[c]) for c in CAT_ORDER]
    ax.legend(handles, CAT_ORDER, frameon=False, fontsize=7)

    # (3) cross-caller distance overlay -------------------------------------
    ax = axes[1, 0]
    for name, fname in VARIANT_FILES.items():
        V = pd.read_csv(os.path.join(data_dir, fname), sep="\t")
        d = _nearest_distance(V, offtargets_union)
        d = d[np.isfinite(d)]
        ax.hist(np.log10(d + 1), bins=40, histtype="step",
                color=VARIANT_COLORS[name], lw=1.3, density=True, label=name)
    ax.axvline(np.log10(THRESH_BP + 1), ls="--", color=(0.3, 0.3, 0.3),
               lw=STY.lw.hair)
    ax.text(np.log10(THRESH_BP + 1), ax.get_ylim()[1] * 0.95, f"  {THRESH_BP} bp",
            fontsize=6, color=(0.3, 0.3, 0.3), va="top")
    ax.set_xlabel("log10(distance to nearest\npredicted off-target + 1 bp)", fontsize=STY.font.lab)
    ax.set_ylabel("Density", fontsize=STY.font.lab)
    ax.set_title("Distance to predictions (all callers)", fontsize=STY.font.title)
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    # (4) WGS depth QC -------------------------------------------------------
    ax = axes[1, 1]
    for i, (name, dp) in enumerate(depths.items()):
        if dp.size == 0:
            continue
        rng = np.random.default_rng(3)
        parts = ax.violinplot(dp, positions=[i], widths=0.7, showmedians=True,
                              showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(VARIANT_COLORS[name] * 0.4 + 0.6)
            b.set_edgecolor(VARIANT_COLORS[name])
            b.set_alpha(0.8)
        ax.text(i, np.median(dp), f" med {np.median(dp):.0f}x", va="center",
                fontsize=6, color="k")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([n.split(" (")[0] + "\n(" + n.split("(")[1] if "(" in n else n
                        for n in depths], fontsize=6)
    ax.set_ylabel("Read depth at called sites (x)", fontsize=STY.font.lab)
    ax.set_title("WGS depth (detection power)", fontsize=STY.font.title)

    for ax in axes.ravel():
        ax.tick_params(direction="out", labelsize=STY.font.tick, length=2.7, pad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(STY.lw.axis)

    fig.suptitle("Off-target analysis, extended",
                 fontsize=STY.font.title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out_dir, "edfig3_gh_offtarget_summary_extended.png")
    fig.savefig(p, dpi=200, facecolor="w")
    fig.savefig(p.replace(".png", ".pdf"), facecolor="w")
    print(f"saved {p}")

    # (5) top predicted off-targets table -----------------------------------
    allsites = pd.concat([guides["guideA"].assign(guide="guideA"),
                          guides["guideE"].assign(guide="guideE")], ignore_index=True)
    allsites["context"] = allsites["locusDesc"].map(_genic_category)
    top = allsites.sort_values("cfdOfftargetScore", ascending=False).head(10).copy()
    top["nearest_variant_bp"] = [
        _min_dist_to_variants(r.chrom, (r.start + r.end) // 2, var_by_chrom)
        for r in top.itertuples()]
    tbl = top[["guide", "chrom", "start", "mismatchCount", "cfdOfftargetScore",
               "context", "nearest_variant_bp"]].rename(
        columns={"cfdOfftargetScore": "CFD"})
    tbl["CFD"] = tbl["CFD"].round(3)
    tbl["nearest_variant_bp"] = tbl["nearest_variant_bp"].replace(np.inf, -1).astype(int)
    csvp = os.path.join(out_dir, "edfig3_gh_offtarget_summary_top_sites.csv")
    tbl.to_csv(csvp, index=False)
    print(f"saved {csvp}")
    print("\nTop 10 predicted off-targets by CFD (nearest_variant_bp = -1 means "
          "no founder variant on that chromosome):")
    print(tbl.to_string(index=False))
    return fig


def main(data_dir=DATA, out_dir=central_output_dir(HERE), show=False):
    gt_init()
    STY = gt_style()
    os.makedirs(out_dir, exist_ok=True)

    guides = {}
    for name, fname in GUIDE_FILES.items():
        df = pd.read_csv(os.path.join(data_dir, fname), sep="\t")
        guides[name] = df
        n_by_mm = df["mismatchCount"].value_counts().sort_index()
        print(f"{name}: {len(df)} predicted off-target sites "
              f"({', '.join(f'{v} at {k} MM' for k, v in n_by_mm.items())})")

    offtargets_union = pd.concat([
        guides["guideA"][["chrom", "start", "end"]].assign(guide="guideA"),
        guides["guideE"][["chrom", "start", "end"]].assign(guide="guideE"),
    ], ignore_index=True)

    # ---- Figure 1: CRISPOR prediction summary ------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes1[0]
    mms = sorted(set(guides["guideA"]["mismatchCount"]) | set(guides["guideE"]["mismatchCount"]))
    width = 0.35
    for gi, (name, df) in enumerate(guides.items()):
        counts = df["mismatchCount"].value_counts().reindex(mms, fill_value=0)
        x = np.arange(len(mms)) + (gi - 0.5) * width
        ax.bar(x, counts.to_numpy(), width=width, color=GUIDE_COLORS[name],
              label=f"{name} (n={len(df)})")
    ax.set_xticks(np.arange(len(mms)))
    ax.set_xticklabels([str(m) for m in mms])
    ax.set_xlabel("Mismatch count")
    ax.set_ylabel("Predicted off-target sites")
    ax.set_title("CRISPOR off-target sites by mismatch count")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes1[1]
    bp = ax.boxplot([guides["guideA"]["cfdOfftargetScore"], guides["guideE"]["cfdOfftargetScore"]],
                    labels=["guideA", "guideE"], patch_artist=True, widths=0.5, showfliers=True)
    for patch, name in zip(bp["boxes"], ["guideA", "guideE"]):
        patch.set_facecolor(GUIDE_COLORS[name] * 0.3 + 0.7)
        patch.set_edgecolor(GUIDE_COLORS[name])
    for gi, name in enumerate(["guideA", "guideE"]):
        v = guides[name]["cfdOfftargetScore"].to_numpy()
        rng = np.random.default_rng(7)
        xj = gi + 1 + (rng.random(v.size) - 0.5) * 0.25
        ax.scatter(xj, v, s=STY.mk.pt, c=[GUIDE_COLORS[name]], alpha=0.4, edgecolors="none")
    ax.set_ylabel("CFD off-target score")
    ax.set_title("Predicted off-target CFD score")
    ax.tick_params(direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig1.suptitle("CRISPOR off-target predictions, guideA & guideE",
                  fontsize=STY.font.title, fontweight="bold")
    fig1.tight_layout(rect=(0, 0, 1, 0.93))
    p1 = os.path.join(out_dir, "edfig3_gh_offtarget_summary_predictions.png")
    fig1.savefig(p1, dpi=200, facecolor="w")
    fig1.savefig(p1.replace(".png", ".pdf"), facecolor="w")
    print(f"\nsaved {p1}")

    # ---- WGS variant sets, distance to nearest predicted off-target -------
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    print(f"\n{'Variant set':<32s} {'n sites':>10s} {'min dist (bp)':>15s} {f'n within {THRESH_BP}bp':>15s}")
    print("-" * 76)
    for i, (name, fname) in enumerate(VARIANT_FILES.items()):
        V = pd.read_csv(os.path.join(data_dir, fname), sep="\t")
        dist = _nearest_distance(V, offtargets_union)
        finite = dist[np.isfinite(dist)]
        n_within = int(np.sum(dist <= THRESH_BP))
        min_dist = finite.min() if finite.size else np.nan
        print(f"{name:<32s} {len(V):>10d} {min_dist:>15,.0f} {n_within:>15d}")

        ax = axes2[i]
        logd = np.log10(finite + 1)
        ax.hist(logd, bins=40, color=VARIANT_COLORS[name] * 0.3 + 0.7,
               edgecolor=VARIANT_COLORS[name], linewidth=0.6)
        ax.axvline(np.log10(THRESH_BP + 1), ls="--", color=(0.3, 0.3, 0.3), lw=STY.lw.hair,
                  label=f"{THRESH_BP} bp threshold")
        ax.set_xlabel("log10(distance to nearest\npredicted off-target site + 1 bp)", fontsize=STY.font.lab)
        if i == 0:
            ax.set_ylabel("Number of candidate\nde-novo/off-target sites")
        ax.set_title(f"{name}\n(n={len(V)}, closest = {min_dist:,.0f} bp)", fontsize=STY.font.title)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.tick_params(direction="out")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig2.suptitle("Distance from candidate de-novo/off-target sites to nearest CRISPOR "
                 "prediction", fontsize=STY.font.title, fontweight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.90))
    p2 = os.path.join(out_dir, "edfig3_gh_offtarget_summary_distance_to_variants.png")
    fig2.savefig(p2, dpi=200, facecolor="w")
    fig2.savefig(p2.replace(".png", ".pdf"), facecolor="w")
    print(f"\nsaved {p2}")

    print("\n0 sites within 10 bp of a CRISPOR-predicted off-target locus, in "
          "all three variant sets" if all(
              np.sum(_nearest_distance(pd.read_csv(os.path.join(data_dir, f), sep="\t"),
                                       offtargets_union) <= THRESH_BP) == 0
              for f in VARIANT_FILES.values())
          else "At least one variant set has sites within the threshold distance")

    # ---- Extended panels (CFD ranking, genomic context, distance overlay,
    #      read depth) and the top-sites table --------------------------------
    fig3 = _extended_panels(guides, offtargets_union, data_dir, out_dir, STY)

    if not show:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
    return fig1, fig2, fig3


if __name__ == "__main__":
    main()
