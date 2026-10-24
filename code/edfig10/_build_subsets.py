"""
_build_subsets.py -- write two endpoint datasets keyed by animal sex. Not a
panel; run before the panels that read them.

Row i of every endpoint array is the animal whose unique identifier is i + 1.
Sex is read from animal_lines.csv, which codes 0 as female and 1 as male,
for animals of any genotype. Values are already z-scored to the wild-type
distribution within each condition.

  data_mf/    wild-type males in the "wt" slot, wild-type females in the "sh"
              slot, "gt" empty: a two-group comparison of sex within wild
              type.
  data_male/  wild-type, mutant and treated males in the "wt", "sh" and "gt"
              slots: the three-group comparison restricted to males.

Writing the subsets into the standard group slots lets every _f6common helper
run unchanged. Animals with no recorded sex are dropped.
"""

import os, csv, numpy as np, scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "..", "..", "data", "fig6", "fig6_a1_radar_control")
IDENT = os.path.join(HERE, "..", "..", "data", "edfig10", "animal_lines.csv")
DOMS = ["beh", "soc", "cog", "fmri"]

GENO, SEX = {}, {}
with open(IDENT, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        uid = int(row["Unique_ID"])
        GENO[uid] = float(row["Genotype"]) if row["Genotype"] else None
        SEX[uid] = float(row["Sex"]) if row["Sex"] else None

def sex_of(i):
    """Return 'M', 'F' or None for row i (identifier i + 1), for an animal of
    any genotype."""
    s = SEX.get(i + 1)
    return "M" if s == 1 else ("F" if s == 0 else None)

def build(dataset, slot_rule):
    """Write one dataset to the directory named by `dataset`.

    slot_rule is a list of (source group, destination slot, wanted sex); each
    entry copies the rows of the source group whose sex matches into the
    destination slot, leaving every other row NaN. Returns the per-domain
    animal counts of each slot in the test condition."""
    out_dir = os.path.join(HERE, dataset); os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for dom in DOMS:
        M = sio.loadmat(os.path.join(SRC, f"marm_{dom}_all.mat")); out = {}
        for cond in ("c", "t"):
            src = {g: np.asarray(M[f"{dom}_{g}_{cond}"], dtype=float) for g in ("wt", "sh", "gt")}
            dst = {g: np.full_like(src["wt"], np.nan) for g in ("wt", "sh", "gt")}
            for srcg, dstg, want in slot_rule:
                arr = src[srcg]
                for i in range(arr.shape[0]):
                    if np.all(np.isnan(arr[i])):
                        continue
                    if sex_of(i) == want:
                        dst[dstg][i] = arr[i]
            for g in ("wt", "sh", "gt"):
                out[f"{dom}_{g}_{cond}"] = dst[g]
            if cond == "t":
                counts[dom] = {g: int(np.sum(~np.all(np.isnan(dst[g]), axis=1))) for g in ("wt","sh","gt")}
        sio.savemat(os.path.join(out_dir, f"marm_{dom}_all.mat"), out)
    return counts

# wild-type males against wild-type females
c1 = build("data_mf", [("wt", "wt", "M"), ("wt", "sh", "F")])
# the three groups, restricted to males
c2 = build("data_male", [("wt", "wt", "M"), ("sh", "sh", "M"), ("gt", "gt", "M")])

print("data_mf  (wt=WTmale, sh=WTfemale):")
for d in DOMS: print(f"  {d:<5} M={c1[d]['wt']:>2} F={c1[d]['sh']:>2}")
print("data_male (wt=WTmale, sh=SHmale, gt=GTmale):")
for d in DOMS: print(f"  {d:<5} WT={c2[d]['wt']:>2} SH={c2[d]['sh']:>2} GT={c2[d]['gt']:>2}")
