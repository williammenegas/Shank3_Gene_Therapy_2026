#!/usr/bin/env python3
"""
make_reduced_inputs.py -- build reduced stand-ins for the large panel inputs.

The panel scripts read from `data/<figure>/<panel>/<basename>`, or from
`data/<figure>/<basename>` where a figure has only one panel folder,
named explicitly in each script. This tool writes a reduced stand-in to exactly
that location, under the same basename and in the same file format as the
original, so no panel script has to change. `tools/_ownership_map.json` gives
the folder for each panel's claim on a file, which is also how a file shared by
several panels is written once, into the owning panel's folder.

Each extractor below carries a docstring stating what its panel actually
consumes from the file and why the reduction is lossless *for that panel*
(that is, why the panel's printed numbers and rendered figure are
unchanged). The reductions were derived from
`audit_v6/REDUCED_INPUTS_SPEC.md`, which in turn was derived by reading the
panel scripts and their helpers.

Usage
-----
    python tools/make_reduced_inputs.py                  # build everything
    python tools/make_reduced_inputs.py --list           # show the extractors
    python tools/make_reduced_inputs.py --dry-run        # report, write nothing
    python tools/make_reduced_inputs.py --only fig2_b_traces
    python tools/make_reduced_inputs.py --source /path/to/repo

`--source` is the directory holding `new_code/` and `new_data/`; it defaults
to the repository that encloses this `python/` tree.

Precision (flag F8 of the spec)
-------------------------------
This tool uses float32 only where the panel's use of the values is a render
or a comparison with a wide margin, and keeps float64 wherever a printed
number could move (the velocity-bias bout gate sits exactly on 300 px/s,
and the EEG spike-wave ranking is a global argsort over near-ties). Integer
state codes become uint8, and a distance array used only through `> 0`
becomes a uint8 0/1 mask. The tool checks every narrowing cast against the
float64 original for the exact predicate the panel applies, and falls back
to float64 if the check fails.

All MAT output is written with `do_compression=True`.

Identity guard
--------------
Every file written is scanned for the colony animals' names and litter
identifiers before the build finishes, and a hit fails the run. See the
identity guard section below for what is covered and how an extractor
declares a string it genuinely needs.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root: tools/ now sits under code/
SD = os.path.join(PY_ROOT, "data")
OWN = json.load(open(os.path.join(HERE, "_ownership_map.json")))


# ---------------------------------------------------------------------------
# framework
# ---------------------------------------------------------------------------
_REGISTRY = []          # (figure, panel, function) in declaration order


def extractor(figure, panel):
    """Register one panel's extractor. `panel` is also its --only key."""
    def deco(fn):
        _REGISTRY.append((figure, panel, fn))
        return fn
    return deco


def default_source():
    """The directory holding new_code/ and new_data/, searching upward from
    python/; the parent of python/ if neither is found."""
    d = PY_ROOT
    for _ in range(6):
        if any(os.path.isdir(os.path.join(d, x)) for x in ("new_code", "new_data")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(PY_ROOT)


def _fmt(n):
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


class Build:
    """Collects the per-file size report and owns the output paths."""

    def __init__(self, source, dry_run):
        self.source = source
        self.dry_run = dry_run
        self.rows = []          # (panel, basename, src_bytes, dst_bytes)
        self.missing = []       # (panel, path, why)
        self.leaks = []         # (panel, basename, [colony strings found])
        self.allowed = {}       # basename -> lowercased strings it may carry
        self.ident_error = None # why the identity guard could not run
        self._ident_rx = None   # (loose, exact) regexes, built on first use

    # -- paths ------------------------------------------------------------
    def src(self, *parts):
        return os.path.join(self.source, *parts)

    def dst(self, figure, panel, basename):
        """The file's place in data, as the ownership map gives it."""
        claim = f"{figure}/{panel}/{basename}"
        path = os.path.join(SD, *OWN.get(claim, claim).split("/"))
        if not self.dry_run:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    # -- reporting --------------------------------------------------------
    def record(self, panel, src_path, dst_path, src_bytes=None):
        sb = src_bytes if src_bytes is not None else (
            os.path.getsize(src_path) if os.path.isfile(src_path) else 0)
        db = os.path.getsize(dst_path) if os.path.isfile(dst_path) else 0
        self.rows.append((panel, os.path.basename(dst_path), sb, db))
        fac = f"{sb / db:>6.1f}x" if db else "     --"
        print(f"    {os.path.basename(dst_path):<50} {_fmt(sb):>9} -> {_fmt(db):>9}  {fac}")
        if os.path.isfile(dst_path):
            self.check_identities(panel, dst_path)

    def note_missing(self, panel, path, why):
        self.missing.append((panel, path, why))
        print(f"    !! {os.path.basename(path)}: {why}")

    # -- identity guard ---------------------------------------------------
    def allow(self, basename, tokens, reason):
        """Declare that `basename` may carry these colony strings.

        The reason is printed with the build, so an exception to the guard
        is stated where it is taken rather than inferred from its silence.
        """
        self.allowed.setdefault(basename, set()).update(t.lower() for t in tokens)
        print(f"    allowed in {basename}: {', '.join(tokens)} ({reason})")

    def _ident_rules(self):
        """The two identity regexes, built once from the identity workbook.

        The workbook is looked for at the source root and then in
        revision_output/, which is where it sits in the tree these reductions
        were built from. Not finding it is fatal rather than quiet: a run that
        could not check what it wrote has not shown the output is safe.
        """
        if self._ident_rx is None:
            cand = [self.src("animal_identity.xlsx"),
                    self.src("revision_output", "animal_identity.xlsx")]
            ident = next((c for c in cand if os.path.isfile(c)), cand[0])
            founders = [self.src("new_data", "Toxicity", n) for n in
                        ("SHANK3_marmosets.xlsx", "SHANK3 marmosets.xlsx")]
            founder = next((c for c in founders if os.path.isfile(c)), None)
            try:
                loose, exact = _identity_tokens(ident, founder)
                self._ident_rx = (_boundary_rx(loose, re.I), _boundary_rx(exact))
            except Exception as e:
                self.ident_error = f"{type(e).__name__}: {e} (reading {ident})"
                self._ident_rx = (None, None)
        return self._ident_rx

    def check_identities(self, panel, path):
        """Record a leak if a written file carries a colony identity.

        A file that cannot be read is reported as a leak rather than passed
        over: an unreadable file has not been shown to be clean, and a guard
        that stays quiet on what it could not open is worse than none.
        """
        rules = [rx for rx in self._ident_rules() if rx is not None]
        if not rules:
            return
        allowed = self.allowed.get(os.path.basename(path), set())
        found = set()
        try:
            for text in _strings_in(path):
                for rx in rules:
                    for m in rx.finditer(text):
                        if m.group(0).lower() not in allowed:
                            found.add(m.group(0))
        except Exception as e:
            self.leaks.append((panel, os.path.basename(path),
                               ["<unscannable: %s: %s>" % (type(e).__name__, e)]))
            print("    !! %s could not be scanned for colony identities: %s"
                  % (os.path.basename(path), type(e).__name__))
            return
        if found:
            self.leaks.append((panel, os.path.basename(path), sorted(found)))
            shown = ', '.join(sorted(found)[:8])
            more = '' if len(found) <= 8 else f' (+{len(found) - 8} more)'
            print(f"    !! {os.path.basename(path)} carries colony "
                  f"identities: {shown}{more}")

    def need(self, panel, path):
        """True when `path` exists; otherwise record it as missing."""
        if os.path.isfile(path):
            return True
        self.note_missing(panel, path, f"not found at {path}")
        return False


# ---------------------------------------------------------------------------
# identity guard
# ---------------------------------------------------------------------------
# Every file this tool writes is deposited under data/ and published, so none
# of them may carry a colony animal's identity. Getting the numbers right is
# not enough on its own: the founder-line join used to label each animal with
# the founder it was sequenced beside, which quietly wrote a founder's name
# onto fifteen wild-type relatives that inherited no mutant allele. The guard
# below reads the identity workbook once and scans every written file, so that
# class of mistake fails the build instead of reaching data/.
#
# Scanned for: the animal names in column A and the litter identifiers in
# column C. The collar colours in column D are deliberately left out -- they
# are Blue, Red, White and Yellow, which occur throughout ordinary plotting
# text, so scanning for them would report nothing but noise. Filenames and
# constants inside the panel scripts are out of scope here; this guard covers
# what the tool itself writes.
#
# An extractor that must emit one of these strings declares it with
# Build.allow, so the exception is visible at its call site rather than implied
# by the guard staying silent.

_TEXT_EXT = (".csv", ".tsv", ".txt", ".json", ".md")
_ASCII_RUN = re.compile(rb"[ -~]{4,}")   # bytes: applied to raw file data


def _identity_tokens(ident_path, founder_path=None):
    """Colony names and litter identifiers, from the identity workbook.

    Names of four or more characters are matched without regard to case.
    Three-letter names are matched only in the capitalisation the workbook
    gives them, because lowercased several of them are ordinary English words
    and would fire on almost any text. Nothing shorter is matched at all.

    The names themselves are deliberately not written out here. A guard
    against publishing colony names should not publish eight of them in its
    own docstring.

    `founder_path` adds the two founders, whose names head the worksheets of
    the colony workbook. They are needed separately because a founder is not
    necessarily in the identity workbook: Kelle is, Spock is not, so reading
    only that sheet would leave one of the two unguarded.
    """
    import openpyxl
    wb = openpyxl.load_workbook(ident_path, read_only=True, data_only=True)
    loose, exact = set(), set()
    for r in list(wb["Sheet1"].iter_rows(values_only=True))[1:]:
        if r[4] is None:
            continue
        for cell in (r[0], r[2]):
            if not cell:
                continue
            tok = str(cell).strip()
            if len(tok) >= 4:
                loose.add(tok)
            elif len(tok) >= 3:
                exact.add(tok)
    wb.close()
    if founder_path and os.path.isfile(founder_path):
        wf = openpyxl.load_workbook(founder_path, read_only=True, data_only=True)
        loose.update(t.split()[0] for t in wf.sheetnames if t.split())
        wf.close()
    return loose, exact


def _boundary_rx(tokens, flags=0):
    """Alternation of `tokens`, each needing a non-letter on both sides.

    A plain \b is not enough: an underscore counts as a word character, so
    \bhelios\b never matches inside `_helios_`.
    """
    if not tokens:
        return None
    alt = "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
    return re.compile(r"(?<![A-Za-z])(?:" + alt + r")(?![A-Za-z])", flags)


def _strings_in(path):
    """Every string a written file carries, whatever its format."""
    ext = os.path.splitext(path)[1].lower()

    if ext in _TEXT_EXT:
        with open(path, "rb") as fh:
            yield fh.read().decode("utf-8", "replace")
        return

    if ext == ".mat":
        import scipy.io as sio
        try:
            m = sio.loadmat(path, squeeze_me=True)
        except Exception:                 # v7.3/HDF5, or unreadable: fall back
            yield from _ascii_runs(path)
            return
        stack = [v for k, v in m.items() if not k.startswith("__")]
        while stack:
            o = stack.pop()
            if isinstance(o, np.ndarray):
                if o.dtype.names:
                    stack.extend(np.ravel(o[n]).tolist() for n in o.dtype.names)
                elif o.dtype.kind in "USO":
                    stack.extend(np.ravel(o).tolist())
            elif isinstance(o, (list, tuple)):
                stack.extend(o)
            elif isinstance(o, str):
                yield o
        return

    if ext == ".npz":
        with np.load(path, allow_pickle=False) as z:
            for k in z.files:
                yield k
                a = z[k]
                if a.dtype.kind in "US":
                    for x in np.ravel(a):
                        yield str(x)
        return

    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic == b"PK" + bytes([3, 4]):
        # A zip container -- .xlsx and friends. Its raw bytes are a deflate
        # stream, in which a short name turns up by chance: scanning a
        # workbook that way matched a three-letter name inside a compressed
        # block, while none of its 126,234 string cells held that value.
        # Read the members instead.
        import zipfile
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                yield info.filename
                with z.open(info) as fh:
                    yield from _ascii_runs(fh)
        return

    yield from _ascii_runs(path)


def _ascii_runs(src, chunk=1 << 23):
    """Printable-ASCII runs out of a binary file, like strings(1).

    `src` is a path or an already-open binary file. Read in blocks with a
    short overlap so a run split across a block boundary is still seen whole.
    """
    fh = open(src, "rb") if isinstance(src, (str, bytes, os.PathLike)) else src
    try:
        tail = b""
        while True:
            blk = fh.read(chunk)
            if not blk:
                return
            buf = tail + blk
            for m in _ASCII_RUN.finditer(buf):
                yield m.group(0).decode("ascii")
            tail = buf[-64:]
    finally:
        if fh is not src:
            fh.close()


def _first_existing(b, candidates):
    """The first of `candidates` that exists under the source, else the first.

    Returning the first candidate when none exists keeps b.need()'s "not found"
    message pointed at the expected name rather than the last one tried.
    """
    paths = [b.src(*parts) for parts in candidates]
    return next((p for p in paths if os.path.isfile(p)), paths[0])


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _loadvar(path, name):
    """Read one variable out of a v7 .mat file.

    scipy seeks past the variables it was not asked for, so peak memory stays
    near the size of the single variable rather than the size of the file. That
    is what makes the 3.5 GB workspace dump tractable.
    """
    import scipy.io as sio
    d = sio.loadmat(path, variable_names=[name])
    if name not in d:
        raise KeyError(f"{name} not in {path}")
    return d[name]


def _savemat(path, mdict, dry_run):
    import scipy.io as sio
    if dry_run:
        return
    sio.savemat(path, mdict, do_compression=True)


def _rows_for(path, cond, animal_id, first_only):
    """Row indices the panel selects, derived the same way the panel does:
    `np.flatnonzero(ID_condition_N == animal_id)`, optionally its first
    element only."""
    ID = np.asarray(_loadvar(path, f"ID_condition_{cond}"), dtype=float).ravel()
    rows = np.flatnonzero(ID == animal_id)
    if rows.size == 0:
        raise ValueError(f"no rows with ID_condition_{cond} == {animal_id} in {path}")
    return rows[:1] if first_only else rows


def _as_uint8_mask(a):
    """A Dist array reduced to a uint8 0/1 validity mask.

    Lossless wherever the only use is `dist > 0`; NaN maps to 0, which is what
    `np.nan > 0` already evaluates to.
    """
    out = np.zeros(a.shape, dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        out[np.asarray(a, dtype=float) > 0] = 1
    return out


def _as_uint8_codes(a):
    """A Bmat array reduced to uint8 state codes.

    The codes are integers 1..80 and are only ever tested with `np.isin`
    against those codes, so NaN can map to 0, which is in none of the lists.
    """
    a = np.asarray(a, dtype=float)
    out = np.zeros(a.shape, dtype=np.uint8)
    ok = np.isfinite(a)
    out[ok] = np.rint(a[ok]).astype(np.uint8)
    bad = ok & (np.abs(a - np.rint(a)) > 1e-9)
    if bad.any():
        raise ValueError("Bmat holds non-integer codes; uint8 would not be lossless")
    return out


def _float32_if_safe(a, predicates, label):
    """Narrow to float32 only when every predicate the panel applies gives the
    same answer as on the float64 original; otherwise keep float64."""
    a64 = np.asarray(a, dtype=np.float64)
    a32 = a64.astype(np.float32)
    for name, fn in predicates:
        if not np.array_equal(fn(a32), fn(a64)):
            print(f"      (float32 would change `{name}` on {label}; keeping float64)")
            return a64
    return a32


# ===========================================================================
# fig2
# ===========================================================================
@extractor("fig2", "fig2_b_traces")
def fig2_b_traces(b):
    """xy_control.mat, xy_test.mat -> fig2/fig2_b_traces.py

    Consumed: `_select_data` asks scipy for u/v/Bmat/Dist/ID of conditions 1,
    2 and 3 but reads only U, V, D and ID. `Bmat_condition_N` is named and
    never touched, and scipy ignores names absent from the file, so it is
    dropped. Row selection is `np.flatnonzero(ID == ind)[0]`, i.e. the first
    matching row only, for (condition, ID) = (1, 245), (2, 107), (3, 115).
    All 500 000 columns are kept: the `Dist > 0` mask is applied before the
    108 000-frame cap, so the number of leading columns needed is not known
    in advance.

    Lossless because: one row per condition is all the panel indexes; u and v
    reach only `(x + 400) / 1000`, a plot and a scatter, so float32 is below
    the rendering resolution and preserves the NaN pattern the `isfinite`
    mask uses; Dist is used solely as `dist > 0`, so a uint8 0/1 mask is
    exact. The printed `n=<frames>` is a count over those two masks and is
    checked to be unchanged.
    """
    plan = [("xy_control.mat", [(1, 245), (2, 107), (3, 115)]),
            ("xy_test.mat", [(1, 245), (2, 107), (3, 115)])]
    for base, examples in plan:
        src = b.src("new_code", "fig2_new", base)
        if not b.need("fig2_b_traces", src):
            continue
        dst = b.dst("fig2", "fig2_b_traces", base)
        if b.dry_run:
            b.record("fig2_b_traces", src, dst)
            continue
        out = {}
        for cond, ind in examples:
            rows = _rows_for(src, cond, ind, first_only=True)
            for var in ("u", "v"):
                A = np.asarray(_loadvar(src, f"{var}_condition_{cond}"), float)[rows, :]
                out[f"{var}_condition_{cond}"] = _float32_if_safe(
                    A, [("isfinite", np.isfinite)], f"{var}_condition_{cond}")
                del A
            D = np.asarray(_loadvar(src, f"Dist_condition_{cond}"), float)[rows, :]
            out[f"Dist_condition_{cond}"] = _as_uint8_mask(D)
            del D
            out[f"ID_condition_{cond}"] = np.full((1, rows.size), float(ind))
            print(f"      cond {cond} ID {ind}: row {rows.tolist()}")
        _savemat(dst, out, b.dry_run)
        b.record("fig2_b_traces", src, dst)


@extractor("fig2", "fig2_d_state_maps")
def fig2_d_state_maps(b):
    """mRed_tracks_sparky_1_control_2_12months.csv -> fig2/fig2_d_state_maps.py

    Consumed: `pd.read_csv(header=None)` over a 661 217 x 42 comma-delimited
    file, then `tracks[1200:, :]` and only columns 0 and 1. Columns 2..41 are
    the other tracked keypoints and animals and are never indexed.

    Lossless because: only the first two columns are read, and they are
    copied verbatim as text, so every value, its 2-decimal spelling and the
    literal `NaN` for untracked frames survive exactly. All 661 217 rows are
    kept. The panel prints `tracks rows (post-trim)=660017`, so truncating to
    the 660 000 rows the label array can reach would change a printed number
    even though it would not change the figure.
    """
    base = "mRed_tracks_sparky_1_control_2_12months.csv"
    src = b.src("new_code", "fig2_new", base)
    if not b.need("fig2_d_state_maps", src):
        return
    dst = b.dst("fig2", "fig2_d_state_maps", base)
    if not b.dry_run:
        n = 0
        with open(src, "r") as fi, open(dst, "w", newline="") as fo:
            for line in fi:
                x, y, _rest = line.split(",", 2)
                fo.write(x + "," + y + "\n")
                n += 1
        print(f"      kept {n} rows x 2 of 42 columns")
    b.record("fig2_d_state_maps", src, dst)


@extractor("fig2", "fig2_e_stimulus_scatter")
def fig2_e_stimulus_scatter(b):
    """xy_object.mat (the new_code/fig2_new copy) -> fig2/fig2_e_stimulus_scatter.py

    Consumed: this file is a 3.5 GB .mat workspace dump holding ~200
    variables. The panel reads exactly three per condition
    (`Bmat_condition_N`, `Dist_condition_N`, `ID_condition_N`) and every row
    matching the example ID: (1, 238), (2, 119), (3, 115). `Bmat` is sliced to
    500 000 columns and `Dist` is not, but both are already that wide, so the
    width is kept.

    Lossless because: the other ~190 variables are never named in any
    `loadmat` call; the retained rows are exactly the ones `flatnonzero`
    selects, in order, so the row-major ravel that concatenates them is
    identical. `Bmat` reaches the figure only through `np.isin` against
    integer state codes, so uint8 is exact. `Dist` is tested both as `> 0`
    and as `< 150`, so it keeps its value: float32 is used only after
    checking that both comparisons give bit-identical masks.

    Column width matters here: the SH example is a single row with 218 057
    valid samples against the 216 002 the 60-window loop needs, so there is
    no headroom to truncate columns.
    """
    src = b.src("new_code", "fig2_new", "xy_object.mat")
    if not b.need("fig2_e_stimulus_scatter", src):
        return
    dst = b.dst("fig2", "fig2_e_stimulus_scatter", "xy_object.mat")
    if b.dry_run:
        b.record("fig2_e_stimulus_scatter", src, dst)
        return
    out = {}
    for cond, ind in [(1, 238), (2, 119), (3, 115)]:
        rows = _rows_for(src, cond, ind, first_only=False)
        B = np.asarray(_loadvar(src, f"Bmat_condition_{cond}"), float)[rows, :]
        out[f"Bmat_condition_{cond}"] = _as_uint8_codes(B)
        del B
        D = np.asarray(_loadvar(src, f"Dist_condition_{cond}"), float)[rows, :]
        out[f"Dist_condition_{cond}"] = _float32_if_safe(
            D, [("dist > 0", lambda x: x > 0), ("dist < 150", lambda x: x < 150)],
            f"Dist_condition_{cond}")
        del D
        out[f"ID_condition_{cond}"] = np.full((1, rows.size), float(ind))
        print(f"      cond {cond} ID {ind}: rows {rows.tolist()}")
    _savemat(dst, out, b.dry_run)
    b.record("fig2_e_stimulus_scatter", src, dst)


@extractor("fig2", "fig2_h_cagemate_scatter")
def fig2_h_cagemate_scatter(b):
    """Figure_2h_data.mat -> fig2/fig2_h_cagemate_scatter.py

    Consumed: the original is a saved MATLAB workspace of 141 variables. The
    panel reads two of them, `Bmat_red_running` and `Bmat_yellow_running`,
    each a 1x104 cell of 1x80 behavioural-state usage vectors, and draws
    cells 21, 58 and 93 of each as its three examples.

    Lossless because: no other variable is reached as a field or as a quoted
    key by the only reader, and both cell arrays are kept whole rather than
    subset, so any example index stays available. The six vectors the panel
    draws compare equal through the repo's own loader, and all six rendered
    sub-panels come out byte-identical.

    The 139 discarded include four 640 kB scratch matrices, eight unread
    332,726-sample label vectors under two naming schemes, the working
    directory, and about twenty loop counters. One of them, a 112-entry dir()
    listing, was also the only thing in this file that named colony animals.
    """
    src = b.src("new_code", "fig2_new", "Figure_2h_data.mat")
    if not b.need("fig2_h_cagemate_scatter", src):
        return
    dst = b.dst("fig2", "fig2_h_cagemate_scatter", "Figure_2h_data.mat")
    if b.dry_run:
        b.record("fig2_h_cagemate_scatter", src, dst)
        return
    import scipy.io as sio
    m = sio.loadmat(src, squeeze_me=False, struct_as_record=True)
    _savemat(dst, {k: m[k] for k in ("Bmat_red_running", "Bmat_yellow_running")},
             b.dry_run)
    b.record("fig2_h_cagemate_scatter", src, dst)


# ===========================================================================
# fig3
# ===========================================================================
def _reduce_tracks(b, panel, figure, src_dir, base, variables, n_frames,
                   n_cols, dtype):
    """Shared reducer for the four-animal `tracks_<session>.mat` files.

    Each file holds Blue, Red, White and Yellow, all (frames, 42). A panel
    reads at most Blue and Red, uses keypoint column pairs 0 and 3 (that is,
    columns 0, 1, 6 and 7) and caps the session at 108 000 frames
    (60 fps x 30 min), a cap the real row counts (760 000 to 1 250 000)
    always bind against. So the reduction is: drop the unread animals, keep
    the first 108 000 rows and the first 8 columns.
    """
    src = os.path.join(src_dir, base)
    if not b.need(panel, src):
        return
    dst = b.dst(figure, panel, base)
    if b.dry_run:
        b.record(panel, src, dst)
        return
    out = {}
    for v in variables:
        A = np.asarray(_loadvar(src, v), float)[:n_frames, :n_cols]
        if A.shape[0] < n_frames:
            raise ValueError(f"{base}:{v} has only {A.shape[0]} rows")
        out[v] = A.astype(dtype)
        del A
    print(f"      {', '.join(variables)}: ({n_frames}, {n_cols}) {np.dtype(dtype).name}")
    _savemat(dst, out, b.dry_run)
    b.record(panel, src, dst)


@extractor("fig3", "fig3_e_velocity_bias")
def fig3_e_velocity_bias(b):
    """tracks_<session>.mat x3 -> fig3/fig3_e_velocity_bias.py

    Consumed: `sio.loadmat(variable_names=["Blue", "Red"])`; `White` and
    `Yellow` are present in the file and never read. `_keypoints_xy` with
    0-based KP_HEADCTR=3 and KP_FRONT=0 indexes columns 6, 7 (used) and
    0, 1 (computed, then discarded), so the array has to stay at least 8
    columns wide. The frame cap is `min(rows, 60 * 60 * 30) = 108 000`.

    Lossless because: two of the four animals and 34 of the 42 columns are
    never indexed, and every retained value is copied verbatim. float64 is
    kept deliberately (F8): the bout gate is `speed > 300 px/s` on a
    `diff * 60` quantity, so a float32 rounding at the threshold could add or
    drop a bout and move the printed `subj bouts=` / `cagemate bouts=` counts.
    """
    d = b.src("new_code", "fig3_new")
    for s in ("tahini_pairtrial_dusty_FA_3", "onion_pairtrial_taquito_FA_3",
              "lemon_pairtrial_filo_FA_3"):
        _reduce_tracks(b, "fig3_e_velocity_bias", "fig3", d, f"tracks_{s}.mat",
                       ["Blue", "Red"], 108000, 8, np.float64)


@extractor("fig3", "fig3_h_bout_trajectories")
def fig3_h_bout_trajectories(b):
    """tracks_<session>.mat x3 -> fig3/fig3_h_bout_trajectories.py

    Consumed: `sio.loadmat(variable_names=["Blue"])`. `Red`, `White` and
    `Yellow` are never read, a fourfold saving on its own. Columns 6, 7 give
    the head center; columns 0, 1 are indexed and discarded. The frame cap is
    `min(Blue.rows, lab.size, 108 000)`, and the label CSVs in the bundle hold
    1.06-1.25 M rows, so 108 000 always binds.

    Lossless because: only the retained rows and columns are indexed, and the
    values reach the figure as a translated path with start and end markers.
    float32 is safe here (F8): the printed numbers are approach/avoid/neutral
    bout counts, which come from the label CSV, not from the coordinates.
    """
    d = b.src("new_code", "fig3_new")
    for s in ("pairtrial_sir_FA_3", "pairtrial_taquito_FA_3", "pairtrial_helios_FA_3"):
        _reduce_tracks(b, "fig3_h_bout_trajectories", "fig3", d, f"tracks_{s}.mat",
                       ["Blue"], 108000, 8, np.float32)


@extractor("fig3", "fig3_j_vocalisation_examples")
def fig3_j_vocalisation_examples(b):
    """new_shenandoah_pairtrial_flan_FA_1_b1.wav -> fig3/fig3_j_vocalisation_examples.py

    Consumed: a mono 16-bit 48 kHz recording of 772 547 074 frames (4 h 28
    min). The panel reads it with `wavfile.read(mmap=True)` and touches two
    absolute sample ranges: the waveform excerpt
    [round(285500 * 175.0012), round(287500 * 175.0012)) = [49 962 843,
    50 312 845), and the spectrogram crop [max(0, (285500-1)*175 - 700),
    287500*175 + 700) = [49 961 625, 50 313 200). Their union is 351 575
    samples, 7.32 s.

    Lossless because: the 351 575 samples that are read are copied byte for
    byte, and nothing outside that range is read. The stand-in holds that
    range alone, 0.7 MiB rather than the 96 MiB a file padded with silence to
    the original offsets would take, and the panel's WAV_FIRST_SAMPLE
    constant carries the shift: it must equal keep_from below, so the
    absolute indices above stay as the published panel computed them. The
    format stays mono int16 48 kHz because the panel asserts the rate,
    derives its full-scale divisor from `np.iinfo(raw.dtype).max`, and would
    draw two lines from a stereo array.
    """
    base = "new_shenandoah_pairtrial_flan_FA_1_b1.wav"
    src = b.src("new_code", "fig3_new", base)
    if not b.need("fig3_j_vocalisation_examples", src):
        return
    dst = b.dst("fig3", "fig3_j_vocalisation_examples", base)
    keep_from, keep_to = 49961625, 50313200
    if not b.dry_run:
        with wave.open(src, "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != 48000:
                raise ValueError(f"{base} is not mono 16-bit 48 kHz")
            if w.getnframes() < keep_to:
                raise ValueError(f"{base} has only {w.getnframes()} frames")
            w.setpos(keep_from)
            payload = w.readframes(keep_to - keep_from)
        with wave.open(dst, "wb") as o:
            o.setnchannels(1)
            o.setsampwidth(2)
            o.setframerate(48000)
            o.writeframes(payload)
        print(f"      {keep_to - keep_from} frames, {keep_from}..{keep_to - 1} "
              f"verbatim; the panel's WAV_FIRST_SAMPLE must be {keep_from}")
    b.record("fig3_j_vocalisation_examples", src, dst)


@extractor("fig3", "fig3_k_call_rasters")
def fig3_k_call_rasters(b):
    """<session>_b1_binary.csv x3 -> fig3/fig3_k_call_rasters.py

    Consumed: the whole trace. `pd.read_csv(header=None, dtype=float64)
    .to_numpy().ravel() > 0.5` gives a per-sample call indicator at 274.2387
    Hz; the 5-minute block count is `n_samples // chunk - 1`, the raster row
    filter compares each call against `n_samples`, and calls are detected end
    to end. So this is an aggregate, not a slice (flag F1): the row count is
    load-bearing and no row may be dropped.

    Lossless because: the array is used only through `> 0.5`, so each of the
    5 151 488 values is rewritten as `0` or `1`. The boolean pattern is
    preserved exactly, and with it every call start and stop, every
    inter-call interval, every raster row and the pie fraction. The saving is
    representational only: the originals spell continuous probabilities at 18
    decimal places, about 25 bytes a row.
    """
    import pandas as pd
    d = b.src("new_code", "fig3_new")
    for base in ("new_unicorn_pairtrial_horatio_FA_6_b1_binary.csv",
                 "new_salt_pairtrial_nacho_FA_6_b1_binary.csv",
                 "new_hooper_pairtrial_helios_FA_6_b1_binary.csv"):
        src = os.path.join(d, base)
        if not b.need("fig3_k_call_rasters", src):
            continue
        dst = b.dst("fig3", "fig3_k_call_rasters", base)
        if not b.dry_run:
            v = pd.read_csv(src, header=None, dtype=np.float64).to_numpy().ravel()
            bits = (v > 0.5).astype(np.uint8)
            raw = np.empty(bits.size * 2, dtype=np.uint8)
            raw[0::2] = 48 + bits          # ASCII '0' / '1'
            raw[1::2] = 10                 # newline
            with open(dst, "wb") as fo:
                fo.write(raw.tobytes())
            print(f"      {bits.size} rows, {int(bits.sum())} above threshold")
        b.record("fig3_k_call_rasters", src, dst)


# ===========================================================================
# fig5
# ===========================================================================
@extractor("fig5", "fig5_a_voxel_maps")
def fig5_a_voxel_maps(b):
    """global_fconn.<ID>.maskdump.txt x26 -> fig5/fig5_a_voxel_maps.py

    Consumed: each dump is 254 880 whitespace-delimited rows of
    (running index, grid x, grid y, slice, connectivity). `a[a == 0] = nan`
    is applied to the whole array on load. A row reaches the figure only if
    its slice is one of the twelve rendered ones (22, 24, ..., 44), its grid
    x is in [1, 57] (the range `_lr_average` mirrors) and its grid y is in
    [1, 80]. A row whose value is 0 becomes NaN, and a grid cell that
    receives only NaN is indistinguishable from a cell never written, so
    zero-valued rows can go too. That leaves about 10 per cent of the rows.

    One number must survive from outside that set: the per-animal baseline
    shift `if np.nanmin(a[:, 4]) < 0: x_score += abs(...)` is a whole-file
    aggregate. It is carried by a single appended sentinel row with slice
    index 61, which `slice_loop` (1..60) never matches and which the grid
    bounds test would exclude anyway. Its value is written as the original
    text token, so it parses back to exactly the same float.

    All 26 animals are kept (flag F2): every pixel carries a between-group
    mean, SD and t-test over WT n=13, SH n=6, GT n=7, so dropping one animal
    would change every Cohen's d and every p-value.
    """
    import glob
    d = b.src("new_code", "fig5_new")
    files = sorted(glob.glob(os.path.join(d, "*.maskdump.txt")))
    if not files:
        b.note_missing("fig5_a_voxel_maps", os.path.join(d, "*.maskdump.txt"),
                       "no mask dumps found")
        return
    keep_slices = set(range(22, 45, 2))
    tot_in = tot_out = 0
    for src in files:
        base = os.path.basename(src)
        dst = b.dst("fig5", "fig5_a_voxel_maps", base)
        if b.dry_run:
            b.record("fig5_a_voxel_maps", src, dst)
            continue
        kept, n_in, min_v, min_tok = [], 0, None, None
        with open(src, "r") as fi:
            for line in fi:
                p = line.split()
                if len(p) < 5:
                    continue
                n_in += 1
                v = float(p[4])
                if v != 0.0 and (min_v is None or v < min_v):
                    min_v, min_tok = v, p[4]
                if v == 0.0:
                    continue
                if int(float(p[3])) not in keep_slices:
                    continue
                x1, x2 = float(p[1]), float(p[2])
                if not (1.0 <= x1 <= 57.0 and 1.0 <= x2 <= 80.0):
                    continue
                kept.append(" ".join(p[:5]))
        if min_tok is None:
            min_tok = "0"
        kept.append(f"0 0 0 61 {min_tok}")
        with open(dst, "w", newline="") as fo:
            fo.write("\n".join(kept) + "\n")
        tot_in += n_in
        tot_out += len(kept)
        b.record("fig5_a_voxel_maps", src, dst)
    if not b.dry_run:
        print(f"      {len(files)} animals, {tot_in} rows -> {tot_out} rows "
              f"({100.0 * tot_out / max(tot_in, 1):.1f}%), sentinel carries nanmin")
    # the colormap is 2.6 kB and is copied through untouched
    cm = os.path.join(d, "fmri_colormap_gp.mat")
    if os.path.isfile(cm):
        cdst = b.dst("fig5", "fig5_a_voxel_maps", "fmri_colormap_gp.mat")
        if not b.dry_run and not os.path.isfile(cdst):
            shutil.copy2(cm, cdst)


# ===========================================================================
# edfig2
# ===========================================================================

@extractor("edfig5", "edfig5_b_stimulus_schematic")
def edfig5_b_stimulus_schematic(b):
    """stim_dist_red_sparky_1_control_2_12months.csv -> edfig5/edfig5_b_stimulus_schematic.py

    Consumed: one single line of 661 217 comma-separated distances, read with
    `np.loadtxt(delimiter=",")`. The panel drops the first 1201 values and
    then truncates to the 660 000 rows of the companion label file, so values
    661 201 and beyond are dead. Everything before that is binned into
    36 000-sample windows and averaged, so every retained value contributes
    to one of the 18 window means: an aggregate, not a slice (flag F1).

    Lossless because: the values reach the figure only through
    `dist < BED_THRESH (100)`, so each is rewritten as `0` (near) or `200`
    (far), preserving that comparison exactly, and the trailing 16 dead
    values are dropped. The printed per-category n stays 18 and the printed
    slopes and intercepts depend only on the near/far pattern.
    """
    base = "stim_dist_red_sparky_1_control_2_12months.csv"
    src = b.src("new_code", "sup_new", base)
    if not b.need("edfig5_b_stimulus_schematic", src):
        return
    dst = b.dst("edfig5", "edfig5_b_stimulus_schematic", base)
    if not b.dry_run:
        with open(src, "r") as fi:
            toks = fi.read().replace("\n", "").split(",")
        toks = toks[:661201]
        vals = np.array([float(t) for t in toks])
        out = np.where(vals < 100.0, "0", "200")
        with open(dst, "w", newline="") as fo:
            fo.write(",".join(out.tolist()) + "\n")
        print(f"      {vals.size} values, {int((vals < 100).sum())} below the "
              f"100-unit bed threshold")
    b.record("edfig5_b_stimulus_schematic", src, dst)


def _reduce_xy_traj(b, panel, figure, src_dir, base, examples):
    """Shared reducer for the `xy_*.mat` files read by `edfig5/_traj_common.py`.

    `_load_animal_track` reads ID, u, v, Bmat and Dist for conditions 1, 2
    and 3, one variable per `loadmat` call, keeps every row where
    `ID == ind` and slices Bmat and Dist to 500 000 columns. All 500 000
    columns are needed: one condition of the sup_new `xy_object.mat` yields
    only 499 564 valid samples against the 500 000 cap, so there is no
    headroom (flag F1).
    """
    src = os.path.join(src_dir, base)
    if not b.need(panel, src):
        return
    dst = b.dst(figure, panel, base)
    if b.dry_run:
        b.record(panel, src, dst)
        return
    out = {}
    for cond, ind in examples:
        rows = _rows_for(src, cond, ind, first_only=False)
        for v in ("u", "v"):
            A = np.asarray(_loadvar(src, f"{v}_condition_{cond}"), float)[rows, :]
            out[f"{v}_condition_{cond}"] = _float32_if_safe(
                A, [("isfinite", np.isfinite)], f"{v}_condition_{cond}")
            del A
        B = np.asarray(_loadvar(src, f"Bmat_condition_{cond}"), float)[rows, :]
        out[f"Bmat_condition_{cond}"] = _as_uint8_codes(B)
        del B
        D = np.asarray(_loadvar(src, f"Dist_condition_{cond}"), float)[rows, :]
        out[f"Dist_condition_{cond}"] = _as_uint8_mask(D)
        del D
        out[f"ID_condition_{cond}"] = np.full((1, rows.size), float(ind))
        print(f"      cond {cond} ID {ind}: rows {rows.tolist()}")
    _savemat(dst, out, b.dry_run)
    b.record(panel, src, dst)


@extractor("edfig5", "edfig5_c_example_traces")
def edfig5_c_example_traces(b):
    """xy_treat.mat and the sup_new xy_object.mat -> edfig5/edfig5_c_example_traces.py

    Consumed: via `_traj_common._load_animal_track`, the u, v, Bmat, Dist and
    ID arrays of conditions 1, 2 and 3, restricted to the rows of animals
    242 (WT), 110 (SH) and 115 (GT). That is 9 rows per file. Everything else
    in the file belongs to the other 130-odd animals.

    Lossless because: the retained rows are exactly the ones the ID mask
    selects, in order, so the ravel that concatenates them is unchanged. u
    and v become float32 (positions of about 0..800 px, binned in 8 px bins
    and drawn as a moving-average line, so float32 resolution there is about
    6e-5 px); Bmat becomes uint8 integer codes used only via `np.isin`; and
    Dist becomes a uint8 0/1 mask because `_traj_common` uses it only as
    `dist_total_red > 0`. The printed `n_valid=` per panel is a count over
    that mask and is preserved.

    Note the contrast with fig2_e, which also tests `Dist < 150` and so
    keeps its numeric values.
    """
    d = b.src("new_code", "sup_new")
    ex = [(1, 242), (2, 110), (3, 115)]
    for base in ("xy_treat.mat", "xy_object.mat"):
        _reduce_xy_traj(b, "edfig5_c_example_traces", "edfig5", d, base, ex)


@extractor("edfig5", "edfig5_h_example_traces_social")
def edfig5_h_example_traces_social(b):
    """xy_box.mat, xy_face.mat -> edfig5/edfig5_h_example_traces_social.py

    Identical mechanism to edfig5_c (the same `_traj_common` loader, the same
    five variables per condition and the same 500 000-column width), with
    example animals 212 (WT), 110 (SH) and 115 (GT), giving 9 rows from
    `xy_box.mat` and 8 from `xy_face.mat`.
    """
    d = b.src("new_code", "sup_new")
    ex = [(1, 212), (2, 110), (3, 115)]
    for base in ("xy_box.mat", "xy_face.mat"):
        _reduce_xy_traj(b, "edfig5_h_example_traces_social", "edfig5", d, base, ex)


# ===========================================================================
# edfig6
# ===========================================================================
@extractor("edfig6", "edfig6_e_velocity_bias")
def edfig6_e_velocity_bias(b):
    """tracks_<session>_FA_6.mat x3 -> edfig6/edfig6_e_velocity_bias.py

    Consumed: `gtstyle.load_struct` reads every top-level variable, so
    `White` and `Yellow` are loaded even though only `S["Blue"]` and
    `S["Red"]` are used. The stand-in does not contain them. The keypoint
    indices are 1-based here (KP_HEADCTR=4, KP_FRONT=1) but `xi(k) = 2*(k-1)`
    maps them to the same columns 6, 7 and 0, 1. The frame cap is again
    108 000.

    Lossless because: only the retained rows and columns are indexed, and
    both animals' values genuinely matter (`Rpos = Sctr - Octr`, and each
    animal's own speed gates its own bouts), so `Red` is copied for real.
    float64 is kept (F8) so that the printed bout counts cannot move at the
    300 px/s gate. The 8-column, 108 000-row shape also avoids
    `load_struct._clean`, which would ravel an array with a singleton
    dimension.
    """
    d = b.src("new_code", "sup_new")
    for s in ("unicorn_pairtrial_horatio_FA_6", "yertle_pairtrial_poseidon_FA_6",
              "lemon_pairtrial_filo_FA_6"):
        _reduce_tracks(b, "edfig6_e_velocity_bias", "edfig6", d, f"tracks_{s}.mat",
                       ["Blue", "Red"], 108000, 8, np.float64)


@extractor("edfig6", "edfig6_h_bout_trajectories")
def edfig6_h_bout_trajectories(b):
    """tracks_pairtrial_<session>_FA_6.mat x3 -> edfig6/edfig6_h_bout_trajectories.py

    Consumed: `load_struct` again, then `S["Blue"]` and `S["Red"]`. Only
    Blue's head center is drawn: `Octr` is computed from Red and never
    referenced, so Red enters the result only through
    `N = min(XYs.shape[0], XYo.shape[0], lab.size, 108000)`. Red is kept
    anyway, at its real values, because it costs almost nothing once the
    array is float32 and MAT-compressed, and it removes any doubt about the
    unused-variable argument.

    Lossless because: the plotted bouts are chosen by the label CSV, which is
    already in the bundle, and every drawn coordinate is one of the retained
    rows. float32 is safe here (F8): the panel prints bout counts, which come
    from the labels, not the coordinates.
    """
    d = b.src("new_code", "sup_new")
    for s in ("pairtrial_mandarin_FA_6", "pairtrial_nathanial_FA_6",
              "pairtrial_lars_FA_6"):
        _reduce_tracks(b, "edfig6_h_bout_trajectories", "edfig6", d, f"tracks_{s}.mat",
                       ["Blue", "Red"], 108000, 8, np.float32)


# ===========================================================================
# edfig10
# ===========================================================================
@extractor("edfig10", "edfig10_c_founder_lines")
def edfig10_c_founder_lines(b):
    """SHANK3_marmosets.xlsx + animal_identity.xlsx -> edfig10/animal_lines.csv

    The two workbooks are joined here and written as one small table, so the
    deposited data carries no animal names, litter identifiers, collar colours
    or colony identifiers. Columns: Unique_ID, Genotype, Sex, Founder_line.

    Consumed from SHANK3_marmosets.xlsx: the worksheet a colony UID appears on
    gives the founder line, Kelle or Spock, and only column A is read. The
    line is written only for animals whose genotype is not wild type, since
    an animal that inherited no mutant allele belongs to no founder line.
    Consumed from animal_identity.xlsx: column B the colony ID, column E the
    unique identifier that indexes the endpoint arrays, column F the genotype
    and column H the sex. Columns A, C, D and G, the name, litter, colour and
    gene therapy flag, are never read by any panel and are dropped here.

    Lossless because: the panel and _build_subsets.py need only the four
    columns written, and the join from colony ID to founder line is resolved
    at build time rather than at render time. The tool verifies that the
    uid -> line, uid -> genotype and uid -> sex mappings rebuilt from the CSV
    match those built from the two workbooks.
    """
    import csv
    import openpyxl
    # Both workbooks are looked for under either name the tree has used. The
    # colony workbook was renamed to an underscore inside data/ but still
    # carries a space upstream, and the identity workbook sits either at the
    # source root or in revision_output/.
    src = _first_existing(b, [("new_data", "Toxicity", "SHANK3_marmosets.xlsx"),
                              ("new_data", "Toxicity", "SHANK3 marmosets.xlsx")])
    ident = _first_existing(b, [("animal_identity.xlsx",),
                                ("revision_output", "animal_identity.xlsx")])
    if not (b.need("edfig10_c_founder_lines", src)
            and b.need("edfig10_c_founder_lines", ident)):
        return
    # The founder-line labels are the two founders' own names, and are the
    # one colony string that belongs in a deposited file, because the panel
    # groups animals by line. The figure prints them as Founder-K and
    # Founder-S, so spelling them "K" and "S" here would retire this
    # exception altogether.
    b.allow("animal_lines.csv", ("Kelle", "Spock"),
            "founder-line labels, drawn as Founder-K and Founder-S")
    dst = b.dst("edfig10", "edfig10_c_founder_lines", "animal_lines.csv")
    if b.dry_run:
        return
    wf = openpyxl.load_workbook(src, read_only=True, data_only=True)
    founder = {}
    for ws in wf.worksheets:
        line = "Kelle" if "Kelle" in ws.title else "Spock"
        for r in list(ws.iter_rows(values_only=True))[1:]:
            if r[0]:
                founder[str(r[0]).strip()] = line
    wf.close()

    wi = openpyxl.load_workbook(ident, read_only=True, data_only=True)
    rows, expect = [], {}
    for r in list(wi["Sheet1"].iter_rows(values_only=True))[1:]:
        if r[4] is None:
            continue
        uid = int(r[4])
        idn = str(r[1]).strip() if r[1] else None
        line = founder.get(idn)
        # An animal with a wild-type genotype inherited the mutant allele
        # from neither parent, so it belongs to no founder line; the colony
        # worksheets list it only as a relative. Dropping the line here
        # keeps a founder's name off the animals that do not carry its
        # mutation, and matches the panel, which joins a line for mutant
        # and treated animals only.
        if not r[5]:
            line = None
        rows.append((uid, r[5], r[7], line or ""))
        expect[uid] = (line, r[5], r[7])
    wi.close()
    rows.sort(key=lambda t: t[0])
    # Newline-terminated, matching the committed file, so a rebuild shows a
    # diff only where a value actually changed rather than on every line.
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["Unique_ID", "Genotype", "Sex", "Founder_line"])
        for uid, g, sx, line in rows:
            w.writerow([uid, "" if g is None else int(g),
                        "" if sx is None else int(sx), line])

    got = {}
    with open(dst, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            got[int(row["Unique_ID"])] = (
                row["Founder_line"] or None,
                float(row["Genotype"]) if row["Genotype"] else None,
                float(row["Sex"]) if row["Sex"] else None)
    if got != expect:
        raise ValueError("animal_lines.csv does not rebuild the source mappings")
    n_line = sum(1 for v in got.values() if v[0])
    print(f"      {len(got)} animals, {n_line} with a founder line; mappings verified")
    b.record("edfig10_c_founder_lines", src, dst)


# ===========================================================================
# edfig12
# ===========================================================================
@extractor("edfig12", "edfig12_a_example_eeg")
def edfig12_a_example_eeg(b):
    """25998_1_11321147.txt -> edfig12/edfig12_a_example_eeg.py

    Consumed: `pd.read_csv(sep=r"\\s+", header=None, usecols=[5])` takes
    field 5 (EEG2A-B) of a 900 100-row, 7-field recording, then
    `col[380 * 1000 : 450 * 1000]`. Nothing past sample 449 999 is touched,
    and everything before 380 000 is touched only as padding that keeps those
    two offsets right.

    Lossless because: the stand-in keeps 450 000 rows, so the slice lands on
    the same samples, and copies field 5 for rows 380 000..449 999 as the
    original text token, so the extracted channel is bit-identical. Fields
    0-4 become `0` (they are never indexed) and field 5 becomes `0.0` before
    row 380 000; `usecols=[5]` requires at least six fields per line, so the
    width cannot go below six. Most of the original file is trailing space
    padding, which is why this is a 27-fold saving.
    """
    base = "25998_1_11321147.txt"
    src = b.src("new_data", "EEG", base)
    if not b.need("edfig12_a_example_eeg", src):
        return
    dst = b.dst("edfig12", "edfig12_a_example_eeg", base)
    if not b.dry_run:
        n_rows, real_from = 450000, 380000
        pad = "0 0 0 0 0 0.0\n"
        with open(src, "r") as fi, open(dst, "w", newline="") as fo:
            for i, line in enumerate(fi):
                if i >= n_rows:
                    break
                if i < real_from:
                    fo.write(pad)
                else:
                    fo.write("0 0 0 0 0 " + line.split()[5] + "\n")
        print(f"      {n_rows} rows x 6 fields; field 5 real for rows "
              f"{real_from}..{n_rows - 1}")
    b.record("edfig12_a_example_eeg", src, dst)


@extractor("edfig12", "edfig12_a_swd_examples")
def edfig12_a_swd_examples(b):
    """25998_1_11321147.txt -> edfig12/edfig12_a_swd_examples.py

    Consumed: the whole recording. `pd.read_csv(sep=r"\\s+", header=None,
    usecols=[4])` takes field 4 (EEG1A-B), and every selection made from it
    is a global ranking: spike detection over all samples, candidate windows
    scored and taken through `np.argsort(score)[::-1]`, quiet windows through
    `np.argsort(ptp)`, an `np.argmax` over an 8 s cluster count and a
    `np.cumsum(x ** 2)`. So this is an aggregate, not a slice (flag F1): no
    row may be dropped, and no value may lose precision, since rounding could
    reorder near-ties in argsort and change which 50 windows are drawn.

    Lossless because: all 900 100 rows are kept and field 4 is copied as its
    original text token, so the parsed channel is bit-identical. The saving
    comes only from dropping the six unread fields and the trailing space
    padding, which is most of the 182 MB. `usecols=[4]` needs at least five
    fields per line, so fields 0-3 are written as `0`.

    A separate stand-in is written for edfig12_a_example_eeg because that
    panel reads a different field; the two share a basename but not their
    content, so each panel keeps its own copy and names its own folder
    (flag F7).
    """
    base = "25998_1_11321147.txt"
    src = b.src("new_data", "EEG", base)
    if not b.need("edfig12_a_swd_examples", src):
        return
    dst = b.dst("edfig12", "edfig12_a_swd_examples", base)
    if not b.dry_run:
        n = 0
        with open(src, "r") as fi, open(dst, "w", newline="") as fo:
            for line in fi:
                p = line.split()
                if len(p) < 5:
                    continue
                fo.write("0 0 0 0 " + p[4] + "\n")
                n += 1
        print(f"      {n} rows x 5 fields; field 4 real at full precision")
    b.record("edfig12_a_swd_examples", src, dst)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build reduced stand-ins for the large panel inputs.")
    ap.add_argument("--source", default=None,
                    help="directory holding new_code/ and new_data/ "
                         "(default: the enclosing repository)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be built, write nothing")
    ap.add_argument("--only", action="append", default=None, metavar="PANEL",
                    help="rebuild one panel only; repeatable")
    ap.add_argument("--list", action="store_true",
                    help="list the extractor keys and exit")
    a = ap.parse_args(argv)

    if a.list:
        for fig, panel, fn in _REGISTRY:
            first = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"{panel:<34} {fig:<8} {first}")
        return 0

    source = os.path.abspath(a.source or default_source())
    wanted = None
    if a.only:
        wanted = {p.split("/")[-1] for p in a.only}
        known = {p for _f, p, _fn in _REGISTRY}
        bad = wanted - known
        if bad:
            ap.error(f"unknown panel(s): {', '.join(sorted(bad))}; "
                     f"use --list to see the keys")

    print(f"source     : {source}")
    print(f"destination: {SD}")
    if a.dry_run:
        print("mode       : dry run (nothing is written)")
    print()

    b = Build(source, a.dry_run)
    t0 = time.time()
    for fig, panel, fn in _REGISTRY:
        if wanted and panel not in wanted:
            continue
        print(f"[{fig}/{panel}]")
        try:
            fn(b)
        except Exception as e:            # keep going; report at the end
            b.note_missing(panel, panel, f"{type(e).__name__}: {e}")
        print()

    print("=" * 92)
    print(f"{'panel':<32}{'file':<44}{'before':>10}{'after':>10}")
    print("-" * 92)
    tot_s = tot_d = 0
    for panel, base, sb, db in b.rows:
        print(f"{panel:<32}{base[:43]:<44}{_fmt(sb):>10}{_fmt(db):>10}")
        tot_s += sb
        tot_d += db
    print("-" * 92)
    fac = f"{tot_s / tot_d:.1f}x" if tot_d else "--"
    print(f"{'TOTAL':<32}{len(b.rows):<44}{_fmt(tot_s):>10}{_fmt(tot_d):>10}   {fac}")
    print(f"elapsed: {time.time() - t0:.0f} s")
    if b.missing:
        print("\nnot produced:")
        for panel, path, why in b.missing:
            print(f"  {panel}: {why}")
    if b.ident_error:
        print()
        print(f"IDENTITY GUARD DID NOT RUN: {b.ident_error}")
        print("  Nothing this run wrote was checked for colony identities,"
              " so none of it is safe to publish.")
        return 1
    if b.leaks:
        print()
        print("COLONY IDENTITIES IN WRITTEN FILES -- do not publish these:")
        for panel, base, toks in b.leaks:
            print(f"  {panel}: {base} -> " + ", ".join(toks))
        print("  Fix the extractor so it does not write the string. If the"
              " string genuinely belongs in the deposited file, declare it"
              " with Build.allow.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
