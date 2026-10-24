"""
fig4_g_touch_distributions.py

Figure 4, panel g: touch-position distributions and touch-by-touch state
sequences for one example animal per group (WT, SH, GT).

Data. xy_type1.mat through xy_type5.mat, fields x_touches_* and y_touches_*.
The example animals are rows 269 (WT), 112 (SH) and 110 (GT), the same
animals shown in panel e; all five session types are concatenated. Two
coordinates are derived per touch: X = |0.5 - x_touches| (horizontal
distance from the screen center) and Y = 1 - y_touches (vertical touch
position, flipped). NaNs are dropped independently for each coordinate.

The script writes fifteen sub-panels, in rows of three (WT, SH, GT):

  p1-p3    Histogram of Y in bins of width 0.02 over [0, 1]: the density of
           vertical touch positions.
  p4-p6    Twenty-bin histogram of d = 1 - Y, with exact zeros in Y excluded.
  p7-p9    Centrality histogram: for f = 1, 3, ..., 99 a bar at x = f whose
           height is the number of touches with ||d| - f/100| < 0.01, i.e. a
           width-0.02 histogram of |d| expressed on a 1-100 index scale. The
           y axis is capped at one tenth of the number of valid touches.
  p10-p12  Raster of the binarized state over touches 321-420: d is set to 0
           below STATE_THRESH (0.7) and to 1 above it, and 1 - |d| is drawn
           against touch index. The two samples at the window edges (touch
           321 and touch 420) are set to 1 for display.
  p13-p15  Maximum a posteriori GLM-HMM state per touch over the same touch
           window as the raster. States come from the GLM-HMM fit stored in
           data as fig4_i_glmhmm_fit.npz and are relabeled 1..K in
           ascending order of pooled center bias, so a higher line means a
           more center-biased state and the labeling is comparable across the
           three examples. The decode uses the same 0.7 binarization boundary
           as the raster, so state and observation are aligned touch for
           touch. These three sub-panels are skipped, with a message, if that
           file is absent.

The panel is descriptive; no statistical test is applied. The state line is
a decode of the fitted model, not a test.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_e_touch_positions")

WT_SEL, SH_SEL, GT_SEL = 269 - 1, 112 - 1, 110 - 1   # example animal rows, 0-based
STATE_THRESH = 1 - 0.3

# fig4_i_glmhmm.py fits the GLM-HMM model; importing that module makes its
# decode functions and cache path available without running its main(). It
# binarizes at the same boundary used here (y_touches at 0.7, equivalently
# Y = 1 - y_touches at 0.3), so decoded states align index for index with the
# raster below.
import fig4_i_glmhmm as I4   # noqa: E402
GLMHMM_THRESH = 0.3            # Y > 0.3 is coded as a center touch


def _decode_map_state(Y, model):
    """Maximum a posteriori GLM-HMM state per touch for one example animal,
    aligned index for index with Y. Returns an array of len(Y); the first L
    touches carry no lag history and are left NaN.

    States are relabeled by their rank in `global_order` (ascending pooled
    center bias), so a higher value is a more center-biased state and the
    labeling is comparable across the three examples.
    """
    beta, A, pi0 = model["beta"], model["A"], model["pi0"]
    L = int(model["L"])
    order = np.asarray(model["global_order"]).ravel()

    Y = np.asarray(Y, dtype=float).ravel()
    sym01 = (Y > GLMHMM_THRESH).astype(float)     # 0 = edge, 1 = center
    X, y_switch = I4.make_lag_design(sym01, L)
    state_full = np.full(Y.size, np.nan)
    if X.shape[0] == 0:
        return state_full

    Gamma, _ = I4.posterior_glm_hmm_sequence(X, y_switch, beta, A, pi0)
    map_raw = np.argmax(Gamma, axis=0)            # raw state per touch >= L
    rank = np.empty(order.size, dtype=int)
    rank[order] = np.arange(1, order.size + 1)    # raw state -> 1..K by bias
    for t in range(map_raw.size):
        state_full[L + t] = rank[map_raw[t]]
    return state_full


def _state_line_panel(name, Y, model, STY, color):
    """MAP-state step line for one example animal, over the raster window."""
    state = _decode_map_state(Y, model)
    K = int(model["N_STATES"])
    x = np.arange(1, state.size + 1)
    fig, ax = plt.subplots(figsize=(1.020, 0.610), facecolor="w")
    ax.step(x, state, where="post", color=color, lw=STY.lw.fit)
    ax.set_xlim(321, 420)                          # same window as the raster
    ax.set_ylim(0.5, K + 0.5)
    # Three y ticks (first, middle, last state), matching the tick density of
    # the other sub-panels; the panel is too short for K integer labels.
    yt = sorted({1, (K + 1) // 2, K})
    _finish_axes(fig, ax, STY, xticks=[340, 380, 420], yticks=yt)
    return fig


def _finish_axes(fig, ax, STY, xticks=None, yticks=None, xnbins=None, ynbins=None):
    """Set the tick locations and the subplot margins for one sub-panel.

    Each sub-panel is 1.02 x 0.61 in and is exported at exactly that page
    size, so matplotlib's default margins leave too little room for a 7 pt
    tick label. The margins set here reserve space for the tick labels; data,
    axis limits, colors and marker sizes are unaffected.
    """
    fig.subplots_adjust(left=0.34, right=0.88, bottom=0.36, top=0.86)
    if xticks is not None:
        ax.set_xticks(xticks)
    elif xnbins is not None:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=xnbins, steps=[1, 2, 5, 10]))
    if yticks is not None:
        ax.set_yticks(yticks)
    elif ynbins is not None:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ynbins, steps=[1, 2, 5, 10]))
    gt_axes(ax, STY, matlab_ticks=False)


def _load_group_xy(data_dir, row, xkey, ykey):
    X, Y = [], []
    for t in range(1, 6):
        S = load_struct(os.path.join(data_dir, f"xy_type{t}.mat"))
        x = np.asarray(S[xkey], dtype=float)[row]
        y = np.asarray(S[ykey], dtype=float)[row]
        X.append(np.abs(0.5 - x))
        Y.append(1 - y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X[~np.isnan(X)], Y[~np.isnan(Y)]


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    _, Yw = _load_group_xy(data_dir, WT_SEL, "x_touches_wt", "y_touches_wt")
    _, Ys = _load_group_xy(data_dir, SH_SEL, "x_touches_sh", "y_touches_sh")
    _, Yg = _load_group_xy(data_dir, GT_SEL, "x_touches_gt", "y_touches_gt")

    figs = []

    # p1-p3: touch-position histogram, bins of width 0.02 over [0, 1].
    edges = np.arange(0, 1 + 1e-9, 0.02)
    for name, Y, color in (("wt", Yw, STY.cmean[0]), ("sh", Ys, STY.cmean[1]), ("gt", Yg, STY.cmean[2])):
        counts = np.array([np.sum((Y > f) & (Y < f + 0.02)) for f in edges])
        fig, ax = plt.subplots(figsize=(1.020, 0.610), facecolor="w")
        ax.bar(np.arange(1, counts.size + 1), counts, color=color, edgecolor="none")
        _finish_axes(fig, ax, STY, xnbins=3, ynbins=3)
        gt_figStyle(fig, STY)
        figs.append((f"fig4_g_touch_distributions_touchpos_{name}", fig))

    # p4-p6: 20-bin histogram of d = 1 - Y, with exact zeros in Y excluded.
    d_store = {}
    for name, Y in (("wt", Yw), ("sh", Ys), ("gt", Yg)):
        y_masked = Y.copy()
        y_masked[y_masked == 0] = np.nan
        d = 1 - y_masked
        d_store[name] = d
        fig, ax = plt.subplots(figsize=(1.020, 0.610), facecolor="w")
        ax.hist(d[~np.isnan(d)], bins=20)
        _finish_axes(fig, ax, STY, xnbins=3, ynbins=3)
        gt_figStyle(fig, STY)
        figs.append((f"fig4_g_touch_distributions_dhist_{name}", fig))

    # p7-p9: centrality histogram of |d| on the 1-100 index scale.
    for name, color in (("wt", STY.cmean[0]), ("sh", STY.cmean[1]), ("gt", STY.cmean[2])):
        d = d_store[name]
        dC = np.abs(d)
        n_valid = np.sum(~np.isnan(dC))
        f_vals = np.arange(1, 100, 2)
        counts = np.array([np.sum(np.abs(dC - f / 100.0) < 0.01) for f in f_vals])
        fig, ax = plt.subplots(figsize=(1.020, 0.610), facecolor="w")
        ax.bar(f_vals, counts, width=0.8, color=color, edgecolor="none")
        ax.set_ylim(0, n_valid / 10)
        _finish_axes(fig, ax, STY, xticks=[0, 50, 100], ynbins=3)
        gt_figStyle(fig, STY)
        figs.append((f"fig4_g_touch_distributions_centrality_{name}", fig))
        print(f"  {name}: n_valid={n_valid}, centrality ylim=[0, {n_valid/10:.2f}]")

    # p10-p12: binarized-state raster over touches 321-420.
    for name, color in (("wt", STY.cmean[0]), ("sh", STY.cmean[1]), ("gt", STY.cmean[2])):
        d = d_store[name].copy()
        below = d < STATE_THRESH
        above = d > STATE_THRESH
        d[below] = 0
        d[above] = 1
        # Display convention: the two samples at the edges of the plotted
        # window are set to the center state.
        if d.size > 419:
            d[320] = 1   # touch 321
            d[419] = 1   # touch 420
        window = d[:1800]
        fig, ax = plt.subplots(figsize=(1.020, 0.610), facecolor="w")
        ax.scatter(np.arange(1, window.size + 1), 1 - np.abs(window), s=STY.mk.pt,
                   c=[color], marker="s", edgecolors="none")
        ax.set_xlim(321, 420)
        ax.set_ylim(-0.5, 1.5)
        _finish_axes(fig, ax, STY, xticks=[340, 380, 420], yticks=[-0.5, 0.5, 1.5])
        gt_figStyle(fig, STY)
        figs.append((f"fig4_g_touch_distributions_raster_{name}", fig))

    # p13-p15: GLM-HMM MAP-state step line, aligned under each raster.
    # Reads the GLM-HMM fit that fig4_i_glmhmm.py stores in data. If that
    # fit is absent, these sub-panels are skipped; the model is not refitted here.
    if os.path.exists(I4.MODEL_CACHE):
        model = dict(np.load(I4.MODEL_CACHE))
        for name, Y, color in (("wt", Yw, STY.cmean[0]),
                                ("sh", Ys, STY.cmean[1]),
                                ("gt", Yg, STY.cmean[2])):
            fig = _state_line_panel(name, Y, model, STY, color)
            gt_figStyle(fig, STY)
            figs.append((f"fig4_g_touch_distributions_stateline_{name}", fig))
        print("  added GLM-HMM MAP-state step lines (p13-p15) from the 4i fit.")
    else:
        print("  [state line] no GLM-HMM fit found at "
              f"{I4.MODEL_CACHE};\n    run fig4_i_glmhmm.py with GT_REFIT=1 to "
              "rebuild it, then re-run this script to add the MAP-state "
              "panels (p13-p15).")

    for i, (stem, fig) in enumerate(figs, start=1):
        gt_saveFig(fig, f"fig4_g_touch_distributions_p{i}_{stem[len('fig4_g_touch_distributions_'):]}", out_dir)

    return [f for _, f in figs]


if __name__ == "__main__":
    main()
