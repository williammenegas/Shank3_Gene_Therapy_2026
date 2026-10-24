"""
edfig12_a_swd_examples.py -- Extended Data Fig. 12, panel a: example EEG
traces for the seizure assay.

WHAT IS PLOTTED. Two figures:
  edfig12_a_swd_examples_raw_annotated
                     two 20 s raw traces from the same recording, the window
                     in which spike-wave discharges are densest, with each
                     discharge boxed, above the window in which they are
                     sparsest. Scale bars mark 1 s and 300 uV.
  edfig12_a_swd_examples_bank
                     two aligned stacked columns of 2 s windows: up to 50
                     spike-wave discharges, aligned on the first spike of
                     each train, beside the 50 quietest windows of the same
                     recording. Both columns share one amplitude scale, and
                     scale bars mark 0.5 s and 300 uV.

DATA. 25998_1_11321147.txt, the SCN2A positive-control recording at 1 kHz;
column 5 of the file, channel EEG1A-B. This is the only recording in the
dataset, so the quiet column is the inter-ictal baseline of the same animal
rather than a separate wild-type recording.

EVENT DETECTION. A spike is a sample whose absolute amplitude exceeds 180 uV,
with a 40 ms refractory period so that one spike-wave complex is counted once.
A discharge is a run of at least six such complexes separated by no more than
0.4 s, whose peak-to-peak amplitude is at least 500 uV; the six-complex
criterion selects rhythmic trains rather than isolated high transients. The
quiet windows are those of lowest peak-to-peak amplitude, taken at least two
window lengths apart.

The script caches extracted windows in _swd_bank_cache.npz and reuses them
on later runs.

STATISTICS. None; these are example traces.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import gt_init, gt_style, gt_saveFig, central_output_dir  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data", "edfig12", "edfig12_a_swd_examples")
RAW = os.path.join(DATA, "25998_1_11321147.txt")
CACHE = os.path.join(HERE, "_swd_bank_cache.npz")

FS = 1000
WIN = 2000                   # window length, samples (2.0 s)
HALF = WIN // 2
N_EACH = 50                  # windows drawn per column
SPIKE_THR = 180.0            # spike detection threshold, uV
PRE = 300                    # pre-onset baseline shown, samples (0.3 s)


def _extract():
    """Return the spike-wave-discharge and quiet window banks, each an array
    of WIN-sample traces. Reads the cache if present, otherwise extracts the
    windows and writes it."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["swd"], z["flat"]
    x = pd.read_csv(RAW, sep=r"\s+", header=None, usecols=[4]).to_numpy().ravel()

    # Count supra-threshold spikes in each candidate window, with a 40 ms
    # refractory period so that one spike-wave complex is counted once. A
    # discharge is a rhythmic train of complexes, so windows holding many
    # well-separated spikes are the discharges rather than isolated high
    # transients.
    over = np.abs(x) > SPIKE_THR
    spike_idx = []
    last = -10 ** 9
    for i in np.where(over)[0]:
        if i - last > 40:                  # refractory period, samples
            spike_idx.append(i); last = i
    spike_idx = np.array(spike_idx)

    step = HALF // 2                        # candidate window hop, samples
    starts_all = np.arange(0, x.size - WIN, step)
    n_spk = np.array([np.sum((spike_idx >= s) & (spike_idx < s + WIN))
                      for s in starts_all])
    ptp_all = np.array([np.ptp(x[s:s + WIN]) for s in starts_all])

    # Discharge windows, ranked by spike count and then by amplitude, keeping
    # windows at least half a window apart so the same event is not drawn
    # twice.
    score = n_spk * 1000 + ptp_all
    swd_sel, used = [], []
    for j in np.argsort(score)[::-1]:
        if n_spk[j] < 6:
            break                           # rhythmic train: >= 6 complexes
        if ptp_all[j] < 500:
            continue                        # amplitude criterion, uV
        s = starts_all[j]
        if all(abs(s - u) > HALF for u in used):
            swd_sel.append(s); used.append(s)
        if len(swd_sel) >= N_EACH:
            break
    # Onset alignment: the first complex of each discharge is placed at a
    # fixed position, with PRE samples of baseline to its left.
    swd = []
    for s in sorted(swd_sel):
        in_win = spike_idx[(spike_idx >= s) & (spike_idx < s + WIN)]
        onset = int(in_win.min()) if in_win.size else s
        st = min(max(onset - PRE, 0), x.size - WIN)
        swd.append(x[st:st + WIN])
    swd = np.stack(swd)

    # Quiet windows: lowest peak-to-peak amplitude, at least two window
    # lengths apart.
    starts = np.arange(0, x.size - WIN, WIN)
    ptp = np.array([np.ptp(x[s:s + WIN]) for s in starts])
    flat_sel, used = [], []
    for s in starts[np.argsort(ptp)]:
        if all(abs(s - u) > 2 * WIN for u in used):
            flat_sel.append(s); used.append(s)
        if len(flat_sel) >= N_EACH:
            break
    flat = np.stack([x[s:s + WIN] for s in sorted(flat_sel)])
    np.savez(CACHE, swd=swd, flat=flat)
    print(f"  flat ptp: {ptp.min():.0f}-{np.sort(ptp)[N_EACH]:.0f} uV  |  "
          f"SWD windows: {len(swd_sel)} (>= 6 complexes, ptp >= 500 uV)")
    return swd, flat


def _column(ax, traces, color, title, off, t):
    """Draw one stacked column of traces, spaced `off` microvolts apart, with
    time axis t."""
    n = len(traces)
    for i, y in enumerate(traces):
        base = (n - 1 - i) * off
        ax.plot(t, y + base, "-", color=color, lw=0.3, zorder=2)
    ax.set_title(title, fontname="DejaVu Sans", fontsize=8, fontweight="bold")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-off, n * off)
    ax.axis("off")


def plot_raw_annotated(out_dir=HERE, seg_sec=20):
    """Draw two raw traces of seg_sec seconds each: the window in which
    spike-wave discharges are densest, with each discharge boxed, above the
    window in which they are sparsest.

    The dense window is centered on the centroid of the tightest cluster of
    discharges, found by sliding an 8 s window; the sparse window is the one
    with the fewest discharges, ties broken by lower signal energy. The traces
    are rasterized and the boxes remain vector, keeping the PDF small and
    editable."""
    STY = gt_style()
    x = pd.read_csv(RAW, sep=r"\s+", header=None, usecols=[4]).to_numpy().ravel()

    # Detect spikes, then cluster them into discharges: at least six
    # complexes no more than 0.4 s apart, peak-to-peak at least 500 uV.
    over = np.abs(x) > SPIKE_THR
    spk, last = [], -10 ** 9
    for i in np.where(over)[0]:
        if i - last > 40:
            spk.append(i); last = i
    spk = np.array(spk)
    events = []
    if spk.size:
        cur = [spk[0]]
        for i in spk[1:]:
            if i - cur[-1] <= int(0.4 * FS):
                cur.append(i)
            else:
                events.append(cur); cur = [i]
        events.append(cur)
    events = [(e[0], e[-1]) for e in events
              if len(e) >= 6 and np.ptp(x[e[0]:e[-1] + 1]) >= 500]

    # Slide a window of the display length in 1 s steps, counting discharges
    # by their midpoint, and record the signal energy of each window.
    seg = seg_sec * FS
    starts = np.arange(0, x.size - seg, FS)
    mids = np.array([(a + b) / 2 for a, b in events]) if events else np.array([])
    n_ev = np.array([int(np.sum((mids >= s) & (mids < s + seg))) for s in starts])
    csum = np.concatenate([[0.0], np.cumsum(x.astype(float) ** 2)])
    energy = np.sqrt((csum[starts + seg] - csum[starts]) / seg)

    # Find the tightest cluster of discharges, the 8 s window holding the
    # most, and center the displayed window on that cluster's centroid.
    clus = 8 * FS
    cstarts = np.arange(0, x.size - clus, FS)
    cn = np.array([int(np.sum((mids >= s) & (mids < s + clus))) for s in cstarts])
    cbest = int(cstarts[np.argmax(cn)])
    cde = mids[(mids >= cbest) & (mids < cbest + clus)]
    center = cde.mean() if cde.size else cbest + clus / 2
    dense_s = int(min(max(center - seg / 2, 0), x.size - seg))
    quiet_s = int(starts[np.lexsort((energy, n_ev))[0]])  # fewest, then lowest energy

    def evs(s):
        """Discharges falling wholly within the window starting at s."""
        return [(a, b) for a, b in events if s <= a and b < s + seg]

    rows = [("High SWD density", dense_s, evs(dense_s), (0.60, 0.42, 0.0)),
            ("Low SWD density (baseline)", quiet_s, evs(quiet_s), None)]

    # Two stacked example traces.
    row_off = 1500.0
    half = 520.0
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    for r, (label, s, es, boxcol) in enumerate(rows):
        seg_x = x[s:s + seg]
        lt = np.arange(seg_x.size) / FS
        base = (len(rows) - 1 - r) * row_off
        ax.plot(lt, seg_x + base, "-", color=(0.12, 0.12, 0.12), lw=0.3,
                rasterized=True, zorder=2)   # dense trace, rasterized
        for a, b in es:
            if boxcol is None:
                continue
            x0 = (a - s) / FS - 0.1
            x1 = min((b - s) / FS + 0.1, seg_sec)
            ax.add_patch(mpatches.Rectangle(
                (x0, base - half), x1 - x0, 2 * half, fill=False,
                edgecolor=boxcol, lw=1.0, zorder=4))
        ax.text(-0.25, base + half + 120, f"{label}   ({len(es)} SWD"
                f"{'s' if len(es) != 1 else ''} / {seg_sec}s)", ha="left",
                va="bottom", fontsize=8, fontweight="bold",
                color=(0.60, 0.42, 0.0) if boxcol else (0.4, 0.4, 0.4))

    # scale bars (bottom-left)
    x0, y0 = 0.2, -row_off * 0.7
    ax.plot([x0, x0 + 1.0], [y0, y0], "-", color="k", lw=1.2)
    ax.plot([x0, x0], [y0, y0 + 300], "-", color="k", lw=1.2)
    ax.text(x0 + 0.5, y0 - row_off * 0.08, "1 s", ha="center", va="top", fontsize=6)
    ax.text(x0 - 0.05, y0 + 150, "300 uV", ha="right", va="center", fontsize=6)

    ax.set_xlim(-0.3, seg_sec + 0.2)
    ax.set_ylim(-row_off, (len(rows) - 1) * row_off + row_off * 0.75)
    ax.axis("off")
    ax.set_title("Raw EEG (SCN2A, EEG1A-B): most vs least SWD-dense windows",
                 fontname=STY.font.name, fontsize=STY.font.title, fontweight="bold")
    fig.tight_layout()
    gt_saveFig(fig, "edfig12_a_swd_examples_raw_annotated", out_dir)
    fig.savefig(os.path.join(central_output_dir(out_dir), "edfig12_a_swd_examples_raw_annotated.png"), dpi=200, facecolor="w")
    print(f"  saved edfig12_a_swd_examples_raw_annotated.pdf  dense@{dense_s/FS:.0f}s "
          f"({rows[0][2].__len__()} SWDs)  quiet@{quiet_s/FS:.0f}s "
          f"({rows[1][2].__len__()} SWDs)")
    plt.close(fig)


def plot_bank(out_dir=HERE, show=False):
    """The aligned example bank: discharges beside the quietest windows."""
    gt_init()
    STY = gt_style()
    swd, flat = _extract()
    t = (np.arange(WIN) - PRE) / FS         # discharge onset at t = 0
    off = 950.0                            # trace spacing, uV, shared by both columns

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 16.0))
    _column(axes[0], flat, (0.30, 0.30, 0.30), f"Flat / no SWD  (n={len(flat)})", off, t)
    _column(axes[1], swd, (0.55, 0.40, 0.00), f"SWD, onset-aligned  (n={len(swd)})", off, t)
    # onset marker on the SWD column
    axes[1].axvline(0, color=(0.6, 0.6, 0.6), lw=0.5, ls="--", zorder=1)
    axes[1].text(0, len(swd) * off, "onset", ha="center", va="bottom",
                 fontsize=6, color=(0.5, 0.5, 0.5))

    # scale bars on the discharge column, shared by both columns
    ax = axes[1]
    x0, y0 = 0.45, -off * 0.7
    ax.plot([x0, x0 + 0.5], [y0, y0], "-", color="k", lw=1.2)
    ax.plot([x0, x0], [y0, y0 + 300], "-", color="k", lw=1.2)
    ax.text(x0 + 0.25, y0 - off * 0.12, "0.5 s", ha="center", va="top", fontsize=6)
    ax.text(x0 - 0.02, y0 + 150, "300 uV", ha="right", va="center", fontsize=6)

    fig.suptitle("EEG example bank - seizure assay (SCN2A, EEG1A-B)",
                 fontweight="bold", fontsize=STY.font.title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    gt_saveFig(fig, "edfig12_a_swd_examples_bank", out_dir)
    fig.savefig(os.path.join(central_output_dir(out_dir), "edfig12_a_swd_examples_bank.png"), dpi=200, facecolor="w")
    print("  saved edfig12_a_swd_examples_bank.pdf/.png")
    if not show:
        plt.close(fig)


def main(out_dir=HERE, show=False):
    """Both figures named in the module docstring, in that order."""
    plot_raw_annotated(out_dir)
    plot_bank(out_dir, show=show)


if __name__ == "__main__":
    main()
