#!/usr/bin/env python3
"""
build_data.py -- populate python/data/ with the data each panel
loads.

Run from a checkout that also holds new_code/ and new_data/:

    cd <repo>/python
    python tools/build_data.py                # copy everything
    python tools/build_data.py --max-mb 200   # leave large files out
    python tools/build_data.py --dry-run      # report without copying

The tool reads data/manifest.json, which records for every panel the
files that panel opened during a traced run, and copies each one to the folder
tools/_ownership_map.json gives for it. That map turns a panel's logical claim
on a file,

    <figure>/<panel>/<basename>

into the one place the file is actually kept. Usually the two are the same,
but where several panels read the same content the map sends them all to the
owning panel's folder, so no duplicate is written. Where a figure has only
one panel folder, its files are kept directly in

    data/<figure>/<basename>

Each panel names its folder explicitly, so the bundle this writes is the only
thing the panels read; nothing is resolved by basename at run time.

--max-mb leaves files above the given size at their original location and
records them in data/external_files.json. The largest files are raw
position tracking and audio recordings, which back only the example-trace
panels.

A manifest may also carry a _trim section declaring, per file, the
spreadsheet columns a panel actually reads. The tool copies those files with
only the declared columns retained.

Two files in data are computed rather than copied: the GLM-HMM fits
read by Figure 4i and Extended Data Fig. 7g. They are listed in the manifest's
_derived section, which records the panel that produces each one and the
command that rebuilds it. This tool leaves them alone.
"""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root: tools/ now sits under code/
SD = os.path.join(PY_ROOT, "data")


def repo_root():
    """Return the directory holding new_code/ and new_data/, searching
    upwards from python/; the parent of python/ if neither is found."""
    d = PY_ROOT
    for _ in range(6):
        if any(os.path.isdir(os.path.join(d, x)) for x in ("new_code", "new_data")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(PY_ROOT)



def _trim_rule(man, base, fig, panel):
    """Return the manifest's column filter for basename `base`, or None.

    A rule whose `panel` field is set applies only to that figure/panel; a
    rule without one applies wherever the file appears."""
    r = (man.get("_trim") or {}).get(base)
    if not r:
        return None
    if r.get("panel") not in (None, f"{fig}/{panel}"):
        return None
    return r


def _copy_trimmed(src, dst, rule):
    """Copy a table, keeping only the columns the rule declares.

    Reads a spreadsheet or a CSV, dispatching on the source extension.
    Exits with an error if any declared column is absent from the source.
    Returns (columns kept, columns in the source)."""
    import pandas as pd
    if src.lower().endswith((".csv", ".tsv")):
        df = pd.read_csv(src, header=rule.get("header_row", 0),
                         sep="	" if src.lower().endswith(".tsv") else ",")
    else:
        df = pd.read_excel(src, header=rule.get("header_row", 0))
    keep = [c for c in rule["keep_columns"] if c in df.columns]
    missing = [c for c in rule["keep_columns"] if c not in df.columns]
    if missing:
        raise SystemExit(f"  columns not found in {os.path.basename(src)}: {missing}")
    out = df[keep]
    if dst.lower().endswith((".xlsx", ".xlsm")):
        with pd.ExcelWriter(dst, engine="openpyxl") as w:
            pd.DataFrame([[""] * len(keep)]).to_excel(
                w, index=False, header=False, startrow=0)
            out.to_excel(w, index=False, startrow=1)
    else:
        out.to_csv(dst, index=False)
    return len(keep), len(df.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mb", type=float, default=None,
                    help="skip files larger than this (they stay external)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man = json.load(open(os.path.join(SD, "manifest.json")))
    own = json.load(open(os.path.join(HERE, "_ownership_map.json")))
    root = repo_root()
    copied = skipped = missing = 0
    bytes_copied = 0
    external = {}

    for fig in sorted(k for k in man if not k.startswith("_")):
        for panel in sorted(man[fig]):
            for base, src in sorted(man[fig][panel].items()):
                src_abs = src.replace("<REPO>", root)
                if not os.path.isfile(src_abs):
                    print("  MISSING", src_abs); missing += 1; continue
                size = os.path.getsize(src_abs)
                if a.max_mb is not None and size > a.max_mb * 1e6:
                    external.setdefault(fig, {}).setdefault(panel, {})[base] = src
                    skipped += 1; continue
                claim = f"{fig}/{panel}/{base}"
                dst = os.path.join(SD, *own.get(claim, claim).split("/"))
                dst_dir = os.path.dirname(dst)
                rule = _trim_rule(man, base, fig, panel)
                if os.path.isfile(dst) and not rule and \
                        os.path.getsize(dst) == size:
                    continue
                if not a.dry_run:
                    os.makedirs(dst_dir, exist_ok=True)
                    if rule:
                        kept, total = _copy_trimmed(src_abs, dst, rule)
                        print(f"  trimmed {base}: kept {kept} of {total} columns "
                              f"({rule['reason'].split('.')[0]}.)")
                    else:
                        shutil.copy2(src_abs, dst)
                copied += 1; bytes_copied += size
        print(f"  {fig}: {len(man[fig])} panels")

    print(f"\ncopied {copied} files ({bytes_copied/1e9:.2f} GB), "
          f"skipped {skipped} oversize, {missing} missing")
    if external and not a.dry_run:
        with open(os.path.join(SD, "external_files.json"), "w") as fh:
            json.dump(external, fh, indent=1, sort_keys=True)
        print("oversize files left external and listed in "
              "data/external_files.json")


if __name__ == "__main__":
    main()
