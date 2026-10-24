"""
edfig12_a_example_eeg.py -- Extended Data Fig. 12, panel a: example EEG
frequency content.

WHAT IS PLOTTED. Three stacked panels over one 70 s window (380 to 450 s) of
a single recording:
  1. the raw trace in microvolts, shaded by gamma-band (30-50 Hz) power;
  2. the spectrogram from 0 to 50 Hz, in decibels;
  3. gamma band power over time, normalized to its own range.

DATA. 25998_1_11321147.txt, a 1 kHz multi-channel recording. This script
plots column 6 of the file.

SIGNAL PROCESSING. Band power is the mean power spectral density between 30
and 50 Hz, from Welch periodograms on 1 s windows with 50 per cent overlap
(256-sample Hann window, 128-sample overlap, 256-point transform). The
spectrogram is a short-time Fourier transform with a 1024-sample Hann window,
512-sample overlap and 1024-point transform, displayed as
10*log10 of the squared magnitude and scaled between the 5th and 99th
percentiles of that surface.

STATISTICS. None; this panel is an example trace.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import welch, spectrogram, get_window
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gtstyle import gt_init, gt_saveFig, central_output_dir  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data", "edfig12", "edfig12_a_example_eeg")
RAW = os.path.join(DATA, "25998_1_11321147.txt")
FS = 1000
T0, T1 = 380, 450                       # display window (s)
GAMMA = (30, 50)
GAMMA_COL = (0.75, 0.38, 0.42)


def main(out_dir=HERE, show=False):
    gt_init()
    # Channel: column 6 of the recording (zero-based index 5).
    col = pd.read_csv(RAW, sep=r"\s+", header=None, usecols=[5]).to_numpy().ravel()
    i0, i1 = T0 * FS, T1 * FS
    eeg = col[i0:i1].astype(float)
    N = eeg.size
    t = T0 + np.arange(N) / FS

    # Gamma band power: Welch periodogram on 1 s windows, 50% overlap.
    win_samp, hop = FS, FS // 2
    n_wins = (N - win_samp) // hop + 1
    t_bp = np.empty(n_wins)
    bp = np.empty(n_wins)
    for w in range(n_wins):
        a = w * hop
        seg = eeg[a:a + win_samp]
        t_bp[w] = t[a:a + win_samp].mean()
        f_psd, pxx = welch(seg, fs=FS, window=get_window("hann", 256),
                           noverlap=128, nfft=256)
        m = (f_psd >= GAMMA[0]) & (f_psd <= GAMMA[1])
        bp[w] = pxx[m].mean()
    bp_norm = (bp - bp.min()) / (bp.max() - bp.min() + 1e-12)

    # Spectrogram: short-time Fourier transform, displayed in decibels.
    f, tt, Z = spectrogram(eeg, fs=FS, window=get_window("hann", 1024),
                           noverlap=512, nfft=1024, mode="complex")
    fmask = f <= 50
    f, Z = f[fmask], Z[fmask]
    S_db = 10 * np.log10(np.abs(Z) ** 2 + 1e-12)
    tt = tt + T0
    vmin, vmax = np.percentile(S_db, 5), np.percentile(S_db, 99)

    fig = plt.figure(figsize=(9.0, 6.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1:3], sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)

    # panel 1: raw trace + gamma shading
    ylim = (eeg.min() * 1.1, eeg.max() * 1.1)
    for w in range(n_wins - 1):
        ax1.axvspan(t_bp[w], t_bp[w + 1], color=GAMMA_COL,
                    alpha=bp_norm[w] * 0.4, lw=0, zorder=1)
    ax1.plot(t, eeg, "-", color=(0.1, 0.1, 0.1), lw=0.35, rasterized=True, zorder=2)
    ax1.set_ylim(*ylim)
    ax1.set_ylabel("Amplitude (uV)")
    ax1.set_title("EEG1 - Raw Trace with Gamma (30-50 Hz) Power Shading",
                  fontweight="bold")
    ax1.tick_params(labelbottom=False)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # panel 2: spectrogram
    im = ax2.pcolormesh(tt, f, S_db, cmap="inferno", vmin=vmin, vmax=vmax,
                        shading="auto", rasterized=True)
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Spectrogram (0-50 Hz)", fontweight="bold")
    ax2.set_ylim(0, 50)
    ax2.tick_params(labelbottom=False)
    cax = ax2.inset_axes([1.012, 0.0, 0.015, 1.0])    # colorbar outside the axes
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Power (dB)")

    # panel 3: gamma band power over time
    ax3.plot(t_bp, bp_norm, "-", color=GAMMA_COL, lw=1.3,
             label="Gamma (30-50 Hz)")
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Norm. Power")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Band Power Over Time", fontweight="bold")
    ax3.legend(loc="upper right", frameon=False, fontsize=7)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    ax1.set_xlim(T0, T1)
    gt_saveFig(fig, "edfig12_a_example_eeg_frequency_380_450", out_dir)
    fig.savefig(os.path.join(central_output_dir(out_dir), "edfig12_a_example_eeg_frequency_380_450.png"), dpi=200, facecolor="w")
    print(f"  saved edfig12_a_example_eeg_frequency_380_450.pdf  ({T0}-{T1}s, channel col6/EEG2A-B)")
    if not show:
        plt.close(fig)


if __name__ == "__main__":
    main()
