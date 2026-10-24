"""
Figure 3, panel k: inter-call-interval structure of vocalisations, one
example session per genotype (WT, SH, GT).

The input is a per-sample binary call-detected trace at FS_VOCAL = 274.2387
Hz over an approximately 6 h session. Calls are runs of that trace, discarded
below 5 samples in length, then de-duplicated so that consecutive call starts
are at least RESP_MIN_GAP_SEC apart. Each session gives one figure with two
rows:

  Upper: the inter-call-interval probability curve, mean +/- s.e.m., over
     0-10 s in bins of 0.15 s, computed within CHUNK_MIN-minute blocks and
     then averaged across blocks. The post-call window [0, NEIGHBOR_WINDOW_S]
     is shaded green and the remainder purple, and an inset pie gives the
     fraction of intervals falling inside that window against the fraction
     outside it.

  Lower: a raster with one row per call, marking every call start within
     RASTER_WINDOW_S after that call's end, in RASTER_NCOLS columns. Rows are
     sorted in ascending order of the call's own end-to-next-start interval.
     A dashed reference curve gives the corresponding quantiles for a
     memoryless Poisson process of rate EXP_LAMBDA.

Call starts and stops are detected directly from the edges of the binary
trace. Threshold interpolation would place every crossing exactly half a
sample later, about 1.8 ms; that constant offset applies equally to starts
and stops and cancels in every quantity derived here, all of which are
differences of two crossing times.

For display, the script thickens each raster mark horizontally by
RASTER_MARK_WIDTH_COLS columns. For a 0/1 input this is a binary dilation,
computed as a one-dimensional maximum filter per row. Rows are stretched
vertically by nearest-neighbor resampling in imshow rather than by
materialising an array repeated RASTER_MARK_HEIGHT_ROWS times. The result is
the same blocky per-row rectangles. Row centers are therefore in logical-row
units, and the axes extent is set to match.

No statistical test is applied; the panel is descriptive.

Data: <session>_binary.csv for the sessions named in EXAMPLES
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig3", "fig3_k_call_rasters")

CHUNK_MIN = 5                  # block length for the interval curve, minutes
FS_VOCAL = 274.2387            # sampling rate of the binary call trace, Hz
RESP_MIN_GAP_SEC = 0.15        # minimum gap between retained call starts, s
ICI_EDGES_TOP = np.arange(0, 10 + 1e-9, 0.15)   # interval histogram edges, s
RASTER_WINDOW_S = 10           # raster window after each call end, s
RASTER_NCOLS = 1000            # raster columns spanning that window
EXP_LAMBDA = 1.0               # rate of the Poisson reference curve, per s
RASTER_MARK_HEIGHT_ROWS = 50   # nominal display height of one raster row
RASTER_MARK_WIDTH_COLS = 10    # horizontal thickening of raster marks, columns
NEIGHBOR_WINDOW_S = 2.5        # post-call window used for the shading and pie, s
MIN_GAP_AFTER_END_S = 0.15     # lower bound of that window, s

# (genotype label, session file base name)
EXAMPLES = [
    ("WT", "new_unicorn_pairtrial_horatio_FA_6_b1_binary"),
    ("SH", "new_salt_pairtrial_nacho_FA_6_b1_binary"),
    ("GT", "new_hooper_pairtrial_helios_FA_6_b1_binary"),
]


def _detect_calls(bin_trace):
    """Detect call starts and stops from the edges of the binary trace."""
    b = bin_trace.astype(np.int8)
    d = np.diff(b)
    starts = np.flatnonzero(d == 1) + 1     # first sample of the call (0-based)
    stops = np.flatnonzero(d == -1)         # last sample of the call (0-based)
    # Pair rising and falling edges in order, truncating any unmatched tail.
    n = min(starts.size, stops.size)
    return starts[:n], stops[:n]


def _filter_calls_by_gap_pair(starts, stops, fs, min_gap_s):
    if starts.size <= 1:
        return starts, stops
    min_gap_samples = max(1, round(min_gap_s * fs))
    keep = np.zeros(starts.size, dtype=bool)
    keep[0] = True
    last_kept = starts[0]
    for i in range(1, starts.size):
        if starts[i] - last_kept >= min_gap_samples:
            keep[i] = True
            last_kept = starts[i]
    return starts[keep], stops[keep]


def _one_session(name, base, data_dir, out_dir, STY):
    path = os.path.join(data_dir, f"{base}.csv")
    bin_trace = (pd.read_csv(path, header=None, dtype=np.float64).to_numpy().ravel() > 0.5)
    bin_trace[0] = False
    bin_trace[-1] = False
    n_samples = bin_trace.size

    starts_raw, stops_raw = _detect_calls(bin_trace)
    lens = stops_raw - starts_raw
    keep = lens >= 5
    starts_raw, stops_raw = starts_raw[keep], stops_raw[keep]

    call_starts, call_stops = _filter_calls_by_gap_pair(starts_raw, stops_raw, FS_VOCAL, RESP_MIN_GAP_SEC)
    print(f"  {name} ({base}): {n_samples} samples, {call_starts.size} calls after gap filter")

    # Upper row: inter-call-interval probability curve, computed per block.
    chunk_vocal = round(FS_VOCAL * 60 * CHUNK_MIN)
    num_blocks = max(n_samples // chunk_vocal - 1, 0)
    nbins_top = ICI_EDGES_TOP.size - 1
    ici_mat = np.full((num_blocks, nbins_top), np.nan)
    for b in range(num_blocks):
        st, en = b * chunk_vocal, (b + 1) * chunk_vocal
        s_in = call_starts[(call_starts > st) & (call_starts < en)]
        if s_in.size >= 2:
            ici = np.diff(s_in) / FS_VOCAL
            ici = ici[(ici > 0) & np.isfinite(ici)]
            if ici.size:
                counts, _ = np.histogram(ici, ICI_EDGES_TOP)
                ici_mat[b, :] = counts / max(counts.sum(), 1)

    with np.errstate(invalid="ignore"):
        m_curve = np.nanmean(ici_mat, axis=0) if num_blocks else np.zeros(nbins_top)
        n_valid = np.sum(~np.isnan(ici_mat), axis=0) if num_blocks else np.ones(nbins_top)
        sem_curve = np.nanstd(ici_mat, axis=0, ddof=0) / np.sqrt(np.maximum(n_valid, 1))
    ici_centers = (ICI_EDGES_TOP[:-1] + ICI_EDGES_TOP[1:]) / 2

    # Lower row: end-to-next-start intervals and the raster sorted by them.
    calls_1_all = call_stops.astype(float)
    if call_starts.size >= 2:
        n_pairs = call_starts.size - 1
        calls_1 = calls_1_all[:n_pairs]
        next_st = call_starts[1:].astype(float)
        iti_1 = (next_st - calls_1) / FS_VOCAL
    else:
        calls_1, iti_1 = np.array([]), np.array([])

    margin_samp = round(RASTER_WINDOW_S * FS_VOCAL)
    if calls_1.size:
        valid = np.isfinite(iti_1) & (iti_1 >= 0) & (calls_1 >= 0) & (calls_1 + margin_samp <= n_samples)
        calls_1, iti_1 = calls_1[valid], iti_1[valid]

    edges_raster = np.linspace(0, RASTER_WINDOW_S, RASTER_NCOLS + 1)
    n_rows = calls_1.size
    raster_mat = np.zeros((n_rows, RASTER_NCOLS), dtype=np.float32)
    # Call starts are monotonically increasing, so each row's window is
    # located with searchsorted rather than a boolean mask over all calls.
    lo_idx = np.searchsorted(call_starts, calls_1, side="left")
    hi_idx = np.searchsorted(call_starts, calls_1 + margin_samp, side="right")
    for r in range(n_rows):
        s_in = call_starts[lo_idx[r]:hi_idx[r]]
        if s_in.size == 0:
            continue
        rel_s = (s_in - calls_1[r]) / FS_VOCAL
        rel_s = rel_s[(rel_s >= 0) & (rel_s <= RASTER_WINDOW_S)]
        if rel_s.size:
            c, _ = np.histogram(rel_s, edges_raster)
            raster_mat[r, :] = (c > 0).astype(np.float32)

    if iti_1.size:
        order = np.argsort(iti_1, kind="stable")
        iti_sorted = iti_1[order]
        raster_sorted = raster_mat[order, :]
    else:
        iti_sorted, raster_sorted = iti_1, raster_mat

    if raster_sorted.shape[0] == 0:
        raster_display = np.zeros((0, RASTER_NCOLS), dtype=np.float32)
        y_centers = np.zeros(0)
    else:
        n_logic_rows, ncols = raster_sorted.shape
        if RASTER_MARK_WIDTH_COLS > 1:
            # Horizontal thickening is a binary dilation by the kernel width,
            # computed as a one-dimensional maximum filter per row.
            from scipy.ndimage import maximum_filter1d
            thick_h = maximum_filter1d((raster_sorted > 0).astype(np.float32),
                                        size=RASTER_MARK_WIDTH_COLS, axis=1, mode="constant")
        else:
            thick_h = (raster_sorted > 0).astype(float)

        # Vertical stretching is left to imshow's nearest-neighbor
        # resampling rather than repeating each logical row
        # RASTER_MARK_HEIGHT_ROWS times, which for a session of this length
        # would materialise an array of several hundred thousand rows. The
        # result is the same blocky per-row rectangles. Row centers are
        # therefore in logical-row units, and the axes extent below is set to
        # [0.5, n_logic_rows + 0.5] to match.
        raster_display = np.asarray(thick_h)
        y_centers = np.arange(1, n_logic_rows + 1).astype(float)

    # Fraction of intervals falling inside the post-call window, for the pie.
    iti_vec = iti_1 if iti_1.size else np.array([])
    iti_vec = iti_vec[np.isfinite(iti_vec) & (iti_vec >= 0)]
    n_total = iti_vec.size
    is_inside = (iti_vec > MIN_GAP_AFTER_END_S) & (iti_vec <= NEIGHBOR_WINDOW_S)
    n_inside = int(is_inside.sum())
    n_out = n_total - n_inside
    frac_inside = n_inside / max(n_total, 1)

    print(f"    ICI blocks={num_blocks}, raster rows={n_rows}, "
          f"frac_inside(0-{NEIGHBOR_WINDOW_S}s)={frac_inside:.3f} (n_in={n_inside}/{n_total})")

    # Plotting
    COL_GREEN, COL_PURPLE = STY.clr.toward, STY.clr.away
    COL_GREENBG = 1 - 0.18 * (1 - COL_GREEN)
    COL_PURPBG = 1 - 0.18 * (1 - COL_PURPLE)
    PCW = NEIGHBOR_WINDOW_S

    fig = plt.figure(figsize=(1.75, 1.75), facecolor="w")
    ax1 = fig.add_axes([0.33, 0.58, 0.61, 0.34])   # left margin sized for the two-line y label
    ax3 = fig.add_axes([0.33, 0.14, 0.61, 0.34])

    YMAXP = 9
    ax1.fill_between([0, PCW], 0, YMAXP, color=COL_GREENBG, zorder=0)
    ax1.fill_between([PCW, 10], 0, YMAXP, color=COL_PURPBG, zorder=0)
    strip_h = 0.06 * YMAXP
    strip_y = YMAXP - strip_h
    ax1.fill_between([0, PCW], strip_y, strip_y + strip_h, color=COL_GREEN, zorder=1)
    ax1.fill_between([PCW, 10], strip_y, strip_y + strip_h, color=COL_PURPLE, zorder=1)
    ax1.plot(ici_centers, 100 * (m_curve - sem_curve), "-", lw=STY.lw.thin, color=STY.clr.gray_ref)
    ax1.plot(ici_centers, 100 * (m_curve + sem_curve), "-", lw=STY.lw.thin, color=STY.clr.gray_ref)
    ax1.plot(ici_centers, 100 * m_curve, "k-", lw=STY.lw.mean)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, YMAXP)
    ax1.set_xticks(np.arange(0, 11, 2))
    ax1.set_yticks([0, 3, 6, 9])
    gt_axes(ax1, STY)
    ax1.set_ylabel("Prob. (%)", fontname=STY.font.name, fontsize=STY.font.lab)
    ax1.text(0.12, YMAXP * 1.02, "Post-call window", va="bottom",
              fontname=STY.font.name, fontsize=STY.font.lab, color=STY.clr.axis)
    # The published panel truncates this fraction rather than rounding it
    # (0.689 -> 68%, 0.669 -> 66%), so int() is used to match it.
    ax1.text(7.0, YMAXP * 0.70, f"{int(100*frac_inside)}%", fontname=STY.font.name,
              fontsize=STY.font.lab, fontweight="bold", color=COL_GREEN, ha="right")

    n_rows_sorted = raster_display.shape[0]
    if n_rows_sorted > 0:
        ax3.imshow(raster_display, cmap="gray_r", vmin=0, vmax=1, aspect="auto",
                   extent=[0, RASTER_WINDOW_S, 0.5, n_rows_sorted + 0.5], origin="lower",
                   interpolation="nearest", rasterized=True)
        strip_y0, strip_y1 = n_rows_sorted * 0.985, n_rows_sorted * 1.0
        ax3.fill_between([0, PCW], strip_y0, strip_y1, color=COL_GREEN)
        ax3.fill_between([PCW, RASTER_WINDOW_S], strip_y0, strip_y1, color=COL_PURPLE)
        ax3.plot([0, 0], [0, n_rows_sorted + 1], "-", color=(0.5, 0.5, 0.5), lw=STY.lw.hair)

        n_logic_rows = y_centers.size
        if EXP_LAMBDA > 0 and n_logic_rows > 0:
            q = (np.arange(1, n_logic_rows + 1) - 0.5) / n_logic_rows
            x_exp = -np.log(np.maximum(1 - q, np.finfo(float).eps)) / EXP_LAMBDA
            in_line = np.isfinite(x_exp) & (x_exp >= 0) & (x_exp <= RASTER_WINDOW_S)
            ax3.plot(x_exp[in_line], y_centers[in_line], "--", color=COL_PURPLE, lw=STY.lw.fit)

        ax3.set_xlim(0, RASTER_WINDOW_S)
        ax3.set_ylim(0.5, n_rows_sorted + 0.5)
        ax3.set_yticks([])
    else:
        ax3.set_xlim(0, RASTER_WINDOW_S)
        ax3.set_ylim(0.5, 1.5)
        ax3.text(RASTER_WINDOW_S * 0.35, 1, "No valid rows for raster.", color=(0.4, 0.4, 0.4))

    ax3.set_xticks(np.arange(0, 11, 2))
    gt_axes(ax3, STY)
    ax3.set_xlabel("Time (seconds)", fontname=STY.font.name, fontsize=STY.font.lab)
    ax3.set_ylabel("All calls\n(sorted by ICI)", fontname=STY.font.name, fontsize=STY.font.lab)

    # Inset pie of intervals inside against outside the post-call window.
    axPie = fig.add_axes([0.33 + 0.74 * 0.61, 0.58 + 0.48 * 0.34, 0.16 * 0.61, 0.42 * 0.34])
    if (n_inside + n_out) > 0:
        axPie.pie([max(n_inside, 1e-9), max(n_out, 1e-9)],
                   colors=[COL_GREEN, COL_PURPLE], wedgeprops={"edgecolor": "w", "linewidth": STY.lw.thin})
    axPie.set_aspect("equal")

    gt_figStyle(fig, STY)
    gt_saveFig(fig, f"fig3_k_call_rasters_{name}", out_dir)
    return fig


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()
    figs = []
    for name, base in EXAMPLES:
        figs.append(_one_session(name, base, data_dir, out_dir, STY))
    return figs


if __name__ == "__main__":
    main()
