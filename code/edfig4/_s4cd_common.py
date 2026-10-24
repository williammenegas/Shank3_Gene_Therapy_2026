"""Shared data reduction for Extended Data Figure 4, panels c and d.

Both panels are built from the same source file, `xy_control_data.mat`, so the
reduction is performed once here, cached, and reused. This module is not a
panel script and produces no figure; it builds the per-condition,
per-behavior-color 2-D occupancy histograms the panels display.

`xy_control_data.mat` is a 6.2 GB HDF5 (v7.3 `.mat`) file whose per-condition
arrays (u, v, Dist and Bmat for conditions 1-3) reach 500000 x 475 float64.
The file is therefore streamed in row-aligned blocks rather than loaded whole:
its chunk layout spans a few hundred trials and all columns, so block reads
along the row axis are inexpensive. Every sample with Dist > 0 and finite
position is accumulated into a running 100 x 100 histogram, one per behavior
color layer, for each condition (1 = WT, 2 = SH, 3 = GT).

Because the source file is large, the build is checkpointed: each call
processes as much as fits in its time budget, writes its state to disk, and
can be resumed. Once all three conditions are complete the per-condition
checkpoints are merged into `_s4cd_density_cache.npz` and removed.

No statistic is computed here; the output is a density image only.
"""

import os
import numpy as np

# HDF5 file locking is disabled before h5py/libhdf5 opens any file, so that a
# stale lock left by an interrupted read cannot block later opens. The source
# file is opened read-only throughout.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

CACHE_NAME = "_s4cd_density_cache.npz"

# ---- fixed 80-element behavioral-state relabeling permutation ----------
LIST_5 = np.array([
    1, 6, 29, 13, 36, 44, 2, 37, 60, 43, 62, 63, 18, 38, 14, 22, 3, 67, 7, 45,
    68, 8, 40, 69, 30, 9, 19, 70, 53, 25, 16, 71, 23, 72, 28, 73, 74, 32, 54,
    46, 4, 64, 26, 47, 65, 55, 27, 10, 48, 20, 41, 75, 33, 24, 56, 42, 39, 5,
    17, 34, 66, 49, 50, 76, 21, 61, 11, 58, 77, 51, 78, 15, 52, 35, 31, 79,
    12, 59, 57, 80,
]) - 1  # 0-based. Not used here: these panels take state codes from Bmat
         # directly. Retained for reference alongside the panel that applies it.

# ---- behavioral-state groupings mapped onto 5 color layers -------------
INDS_0 = [62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80]  # alone
INDS_1 = [30, 31, 32, 33, 35, 40, 41, 42]                                  # near other climbing
INDS_2 = [36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59]      # near other moving
INDS_3 = [15, 19, 21, 22, 23, 24, 27, 29, 34]                              # move to/from other
INDS_4 = [6, 7, 8, 9, 10, 11, 12, 28]                                      # large movements
INDS_5 = [1, 2, 3, 4, 5, 13, 14, 16, 17, 18, 20, 25, 26]                   # climbing alone
INDS_6 = [62, 67, 69, 71, 77, 66, 68, 70, 72, 73, 74, 75, 76, 78, 79, 80,
          60, 61, 63, 64, 65]                                             # looking at other alone
INDS_7 = [43, 53, 54, 55, 56, 57]                                         # near other resting

COLOR_LAYERS = [
    # (name, contributing state codes, RGB), in draw order
    ("blue",   set(INDS_5),                         np.array([102, 98, 215]) / 255.0),
    ("purple", set(INDS_4) | set(INDS_3) | set(INDS_1), np.array([168, 95, 186]) / 255.0),
    ("red",    set(INDS_0) | set(INDS_6),            np.array([240, 92, 155]) / 255.0),
    ("orange", set(INDS_2),                          np.array([251, 141, 123]) / 255.0),
    ("yellow", set(INDS_7),                          np.array([254, 193, 93]) / 255.0),
]

EDGES = np.arange(1, 802, 8, dtype=float)  # 1 to 801 in steps of 8 -> 100 bins


_NAMES = [name for name, *_ in COLOR_LAYERS]
_LUT = np.full(81, -1, dtype=int)
for _ci, (_name, _lset, _rgb) in enumerate(COLOR_LAYERS):
    for _lab in _lset:
        _LUT[_lab] = _ci


def _ckpt_path(cache_dir, group):
    return os.path.join(cache_dir, f"_s4cd_ckpt_{group}.npz")


def process_condition_checkpointed(data_dir, cache_dir, cond_idx, group,
                                    block_size=2000, time_budget=35.0, log=print,
                                    block_stride=1):
    """Stream one condition, accumulating a 100 x 100 histogram per color layer.

    cond_idx: 1 = WT, 2 = SH, 3 = GT. `group` names the checkpoint file.
    block_size: rows read per block. time_budget: seconds of work per call;
    when it is exhausted the function writes the accumulated state to disk
    and returns done=False so the caller can resume.

    block_stride: process one row block in every `block_stride`, a systematic
    subsample along the trial axis. The default of 1 processes every row.

    Returns (done, n_valid_total, n_rows).
    """
    import h5py
    import time

    ck_path = _ckpt_path(cache_dir, group)
    if os.path.exists(ck_path):
        d = np.load(ck_path)
        start_row = int(d["start_row"])
        hists = {name: d[name] for name in _NAMES}
        n_valid_total = int(d["n_valid_total"])
        if bool(d["done"]):
            return True, n_valid_total, int(d["n_rows"])
    else:
        start_row = 0
        hists = {name: np.zeros((len(EDGES) - 1, len(EDGES) - 1)) for name in _NAMES}
        n_valid_total = 0

    t0 = time.time()
    with h5py.File(os.path.join(data_dir, "xy_control_data.mat"), "r") as f:
        u = f[f"u_condition_{cond_idx}"]
        v = f[f"v_condition_{cond_idx}"]
        d_ = f[f"Dist_condition_{cond_idx}"]
        b = f[f"Bmat_condition_{cond_idx}"]
        n_rows = u.shape[0]

        stride_span = block_size * block_stride
        s = start_row
        while s < n_rows:
            if time.time() - t0 > time_budget:
                break
            e = min(s + block_size, n_rows)
            u_blk = np.asarray(u[s:e, :])
            v_blk = np.asarray(v[s:e, :])
            d_blk = np.asarray(d_[s:e, :])
            b_blk = np.asarray(b[s:e, :])

            mask = (d_blk > 0) & np.isfinite(u_blk) & np.isfinite(v_blk)
            if mask.any():
                uf = u_blk[mask]
                vf = v_blk[mask]
                bf = b_blk[mask]
                good_lab = np.isfinite(bf) & (bf >= 1) & (bf <= 80)
                uf, vf, bf = uf[good_lab], vf[good_lab], bf[good_lab]
                n_valid_total += uf.size
                if uf.size:
                    ci = _LUT[bf.astype(int)]
                    for idx, name in enumerate(_NAMES):
                        sel = ci == idx
                        if not sel.any():
                            continue
                        # Histogram axes are (v + 400, u + 400): the v-derived
                        # coordinate occupies the first histogram axis.
                        x_in = vf[sel] + 400.0
                        y_in = uf[sel] + 400.0
                        hh, _, _ = np.histogram2d(x_in, y_in, bins=[EDGES, EDGES])
                        hists[name] += hh
            s += stride_span

        done = s >= n_rows
        save_kw = dict(hists)
        save_kw.update(start_row=s, n_valid_total=n_valid_total, n_rows=n_rows, done=done)
        np.savez(ck_path, **save_kw)
        log(f"  condition_{cond_idx} ({group}): processed rows {start_row}->{s} of {n_rows} "
            f"(done={done}, valid so far={n_valid_total})")
        return done, n_valid_total, n_rows


def build_cache_step(data_dir, cache_dir=None, time_budget=35.0, block_size=2000, log=print,
                      block_stride=1):
    """Advance the checkpointed build by up to `time_budget` seconds of work,
    across whichever conditions are not yet finished.

    Call repeatedly until it returns True, at which point all three conditions
    are complete and the merged cache file has been written.
    """
    if cache_dir is None:
        cache_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(cache_dir, CACHE_NAME)
    if os.path.exists(cache_path):
        return True

    import time
    t0 = time.time()
    conditions = [(1, "wt"), (2, "sh"), (3, "gt")]
    all_done = True
    for cond_idx, group in conditions:
        remaining = time_budget - (time.time() - t0)
        if remaining <= 1.0:
            all_done = False
            break
        done, n_valid, n_rows = process_condition_checkpointed(
            data_dir, cache_dir, cond_idx, group,
            block_size=block_size, time_budget=remaining, log=log,
            block_stride=block_stride)
        if not done:
            all_done = False
            break

    if all_done:
        out = {}
        for cond_idx, group in conditions:
            d = np.load(_ckpt_path(cache_dir, group))
            for name in _NAMES:
                out[f"{group}_{name}"] = d[name]
        np.savez_compressed(cache_path, **out)
        log(f"  ALL DONE -- saved final cache -> {cache_path}")
        for cond_idx, group in conditions:
            os.remove(_ckpt_path(cache_dir, group))
        return True
    return False


def load_or_build_cache(data_dir, cache_dir=None, log=print):
    if cache_dir is None:
        cache_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(cache_dir, CACHE_NAME)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"density histogram cache not found: {cache_path}")
    log(f"  using cached density histograms: {cache_path}")
    return dict(np.load(cache_path))


def smoothed_alpha(hist_coarse, zoom=8):
    """Upsample a coarse density histogram and normalize it to an alpha channel.

    `zoom` is the upsampling factor; the default of 8 turns the 100 x 100
    histogram into the 800 x 800 image drawn over the 800-unit axis. Cubic
    spline interpolation is used. Negative values introduced by interpolation
    are clipped to zero, and the result is divided by mean + 3 * SD and
    clipped to [0, 1]. The output is used as opacity only.
    """
    from scipy.ndimage import zoom as _zoom

    fine = _zoom(hist_coarse, zoom, order=3)
    fine[fine < 0] = 0
    fine = fine / 1000.0
    mean_c = np.nanmean(fine)
    std_c = np.nanstd(fine, ddof=1) if fine.size > 1 else 0.0
    denom = mean_c + 3 * std_c
    if denom <= 0:
        return np.zeros_like(fine)
    norm = fine / denom
    return np.clip(norm, 0, 1)
