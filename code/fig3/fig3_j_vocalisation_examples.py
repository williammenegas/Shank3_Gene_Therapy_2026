"""
Figure 3, panel j: example vocalisation, raw waveform and spectrogram
excerpt.

The script draws two plots from the same stretch of one example recording
(48 kHz): the raw audio waveform, and the log-compressed power spectrogram.

The spectrogram uses a Hamming window of WINDOW samples (about 14.6 ms) with
NOVERLAP samples of overlap and an NFFT-point FFT, giving a column stride of
WINDOW - NOVERLAP = 175 samples. Frequency rows FREQ_ROW_LO to FREQ_ROW_HI of
1001 are retained, 3.0 to 11.4 kHz at FS/NFFT = 24 Hz a row, covering the
marmoset vocal range; power is clipped to [1e-10, 10] and rows SUBROW_LO to
SUBROW_HI of that crop are displayed, 4.2 to 11.4 kHz. Both plots cover
spectrogram columns COL_LO to COL_HI; CONVERSION_FACTOR converts that column
range back to a raw-sample range for the waveform plot.

The script computes the spectrogram from the raw audio cropped to the sample
range covering those columns, with one window of padding on each side, rather
than from the whole recording. Short-time Fourier transform columns depend
only on a WINDOW-sample slice of the input, so the retained columns are the
same as they would be if taken from the full-signal spectrogram.

The bundled recording is the 7.32 s the panel reads, not the 4 h 28 min
original. Every sample index above is an absolute offset into that original,
and WAV_FIRST_SAMPLE is where the excerpt's first sample sits in it, so the
two reads subtract that offset. Point DATA at the full recording and set
WAV_FIRST_SAMPLE to 0 to read it directly: the samples the panel touches are
the same bytes either way.

Both x axes use a three-bin locator rather than the seven-bin house locator,
whose labels merge visually at this panel width. Both axes are data-driven,
with no fixed limits.

No statistical test is applied; the panel is descriptive.

Data: new_shenandoah_pairtrial_flan_FA_1_b1.wav
"""

import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib import ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig3", "fig3_j_vocalisation_examples")

WAV_NAME = "new_shenandoah_pairtrial_flan_FA_1_b1.wav"
# Absolute offset, in the full recording, of the bundled excerpt's first
# sample. 0 reads a full recording in place of the excerpt.
WAV_FIRST_SAMPLE = 49961625
WINDOW = 700          # Hamming window length, in samples
NOVERLAP = 525        # overlap between windows, in samples
NFFT = 2000           # FFT length
FS = 48000            # sampling rate, Hz
FREQ_ROW_LO, FREQ_ROW_HI = 126, 475    # frequency rows kept, 1-based, of the 1001-row PSD
CONVERSION_FACTOR = 175.0012           # samples per spectrogram column
COL_LO, COL_HI = 285500, 287500        # spectrogram columns displayed, 1-based
SUBROW_LO, SUBROW_HI = 50, 350         # rows displayed, 1-based, within the cropped PSD


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    wav_path = os.path.join(data_dir, WAV_NAME)
    fs_file, raw = wavfile.read(wav_path, mmap=True)
    assert fs_file == FS, f"expected {FS} Hz, file is {fs_file} Hz"

    # Normalize integer samples to full scale [-1, 1].
    scale = np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype, np.integer) else 1.0

    step = WINDOW - NOVERLAP     # 175 samples per spectrogram column

    def read_samples(first, last):
        """Samples [first, last) of the full recording, as full scale."""
        i, j = first - WAV_FIRST_SAMPLE, last - WAV_FIRST_SAMPLE
        if not 0 <= i < j <= raw.size:
            raise IndexError(f"samples [{first}:{last}) are outside {WAV_NAME}")
        return raw[i:j].astype(np.float64) / scale

    # Raw waveform excerpt.
    wav_start = int(round(COL_LO * CONVERSION_FACTOR))
    wav_end = int(round(COL_HI * CONVERSION_FACTOR))
    d1_excerpt = read_samples(wav_start, wav_end)

    fig1 = plt.figure(figsize=(1.15, 0.83), facecolor="w")
    ax1 = fig1.add_axes([0.16, 0.20, 0.80, 0.68])
    # Sample index converted to seconds, as the published panel is labeled.
    t_wave = np.arange(d1_excerpt.size) / FS
    ax1.plot(t_wave, d1_excerpt, "-", color="k", lw=STY.lw.trace)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amp.")
    gt_axes(ax1, STY)
    # A three-bin locator on x, whose sample-count labels would otherwise
    # merge at this panel width. The axis is data-driven, so a locator is
    # used rather than fixed tick positions; matlab_ticks is False on the
    # second call so the house locator is not re-applied. The y axis is left
    # as gt_axes set it.
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    gt_axes(ax1, STY, matlab_ticks=False)
    gt_figStyle(fig1, STY)
    gt_saveFig(fig1, "fig3_j_vocalisation_examples_p1", out_dir)

    # Spectrogram excerpt, computed from a local crop of the audio (see
    # module docstring). The first sample of 0-based column c is c * step, so
    # the crop spans the required column range with one window of padding on
    # each side.
    crop_start = max(0, (COL_LO - 1) * step - WINDOW)
    crop_end = COL_HI * step + WINDOW
    x = read_samples(crop_start, crop_end)

    f, t, Sxx = sig.spectrogram(x, fs=FS, window="hamming", nperseg=WINDOW,
                                 noverlap=NOVERLAP, nfft=NFFT, mode="psd",
                                 scaling="density", detrend=False)
    # Global 0-based column index of each locally computed column.
    col0_local = crop_start // step
    global_cols = col0_local + np.arange(Sxx.shape[1])

    sel = (global_cols >= COL_LO - 1) & (global_cols < COL_HI)
    ps1_excerpt = Sxx[FREQ_ROW_LO - 1:FREQ_ROW_HI, :][:, sel]
    ps1_excerpt = np.clip(ps1_excerpt, 0, 10)
    ps1_excerpt[ps1_excerpt < 1e-10] = 0

    sub = ps1_excerpt[SUBROW_LO - 1:SUBROW_HI, :]
    img = 1 - np.log(1e-7 + np.flipud(sub))

    fig2 = plt.figure(figsize=(1.15, 0.83), facecolor="w")
    ax2 = fig2.add_axes([0.16, 0.20, 0.80, 0.68])
    # Columns are a fixed number of samples apart and rows are FS/NFFT Hz
    # apart, so the image carries a seconds-by-kilohertz extent rather than
    # column and row indices.
    f_disp = f[FREQ_ROW_LO - 1:FREQ_ROW_HI][SUBROW_LO - 1:SUBROW_HI]
    t_span = img.shape[1] * step / FS
    ax2.imshow(img, cmap="gray", aspect="auto", interpolation="nearest",
               rasterized=True,
               extent=(0.0, t_span, f_disp[0] / 1000.0, f_disp[-1] / 1000.0))
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("kHz")
    gt_axes(ax2, STY)
    # A three-bin locator on x, whose column-count labels would otherwise
    # merge at this panel width. The y axis, in frequency-row index, is left
    # as gt_axes set it.
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    gt_axes(ax2, STY, matlab_ticks=False)
    gt_figStyle(fig2, STY)
    gt_saveFig(fig2, "fig3_j_vocalisation_examples_p2", out_dir)

    print(f"waveform excerpt: samples [{wav_start}:{wav_end}] "
          f"({(wav_end-wav_start)/FS:.2f} s), amplitude range=[{d1_excerpt.min():.3f}, {d1_excerpt.max():.3f}]")
    print(f"spectrogram excerpt: shape={sub.shape}, matched columns={sel.sum()}/{COL_HI-COL_LO}, "
          f"power range=[{ps1_excerpt.min():.3g}, {ps1_excerpt.max():.3g}]")

    return fig1, fig2


if __name__ == "__main__":
    main()
