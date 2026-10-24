"""Extended Data Figure 7, panel e: touch-position distributions for one
example animal per group.

Data: `xy_type2.mat` and `xy_type3.mat`, variable `y_touches_*`, for one
example animal per group (rows 242 for WT, 106 for SH and 103 for GT, 1-based)
concatenated across session types 2 and 3 only. The plotted quantity is
Y = 1 - y_touches, the vertical touch position with the axis flipped, with NaN
entries dropped.

The script produces the figures in this order.

  p1-p3    Bar histogram of Y in bins of width 0.02 spanning [0, 1], one per
           group.
  p4-p6    Twenty-bin histogram of d = 1 - Y, where Y values of exactly zero
           are first set to NaN. One per group.
  p7-p9    Centrality histogram: for f = 1, 3, 5, ..., 99, a bar at x = f whose
           height is the number of touches with abs(abs(d) - f / 100) < 0.01.
           The y axis is capped at n_valid / 10.
  p10-p12  Raster of d binarized at STATE_THRESH = 0.7: values strictly below
           become 0, values strictly above become 1, and values exactly at the
           threshold or NaN are left unchanged. Two samples at 0-based indices
           1000 and 1099 are forced to 1 so the raster reads clearly. The full
           trace is scattered as 1 - abs(d) and the view is restricted to
           x in RASTER_WINDOW = [1001, 1100] by the axis limits alone; no data
           are truncated.
  p13-p15  Maximum a posteriori GLM-HMM state as a step line over the same
           raster window, one per group. These use the GLM-HMM fit of panel g
           (`edfig7_g_glmhmm`), stored in data as
           edfig7_g_glmhmm_fit.npz and fitted on the same session types 2 and
           3, with touches binarized at GLMHMM_THRESH = 0.3. States are
           relabeled 1..K in ascending order of pooled center bias. The first
           L touches have no state because the lag features are undefined
           there. If that file is absent these three figures are skipped and a
           message is printed; the model is not refitted here.

The panel is descriptive; no statistical test is applied.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_e_touch_positions")

WT_SEL, SH_SEL, GT_SEL = 242 - 1, 106 - 1, 103 - 1   # example animal rows, 0-based
TYPES = (2, 3)
STATE_THRESH = 1 - 0.3             # binarization threshold for d = 1 - Y
FORCE_IDX = (1001 - 1, 1100 - 1)   # samples forced to 1 in the raster, 0-based
RASTER_WINDOW = (1001, 1100)

# GLM-HMM model fitted by panel g on the same session types (2, 3). Importing
# the module does not run its main().
import edfig7_g_glmhmm as S7g   # noqa: E402
GLMHMM_THRESH = 0.3            # Y > 0.3 counts as a center touch, as in panel g


def _decode_map_state(Y, model):
    """Maximum a posteriori GLM-HMM state per touch, aligned index for index
    with Y.

    The first L touches have no state (NaN) because the lag features are
    undefined there. Raw states are relabeled 1..K in ascending order of
    pooled center bias.
    """
    beta, A, pi0 = model["beta"], model["A"], model["pi0"]
    L = int(model["L"])
    order = np.asarray(model["global_order"]).ravel()

    Y = np.asarray(Y, dtype=float).ravel()
    sym01 = (Y > GLMHMM_THRESH).astype(float)
    X, y_switch = S7g.make_lag_design(sym01, L)
    state_full = np.full(Y.size, np.nan)
    if X.shape[0] == 0:
        return state_full

    Gamma, _ = S7g.posterior_glm_hmm_sequence(X, y_switch, beta, A, pi0)
    map_raw = np.argmax(Gamma, axis=0)
    rank = np.empty(order.size, dtype=int)
    rank[order] = np.arange(1, order.size + 1)
    for t in range(map_raw.size):
        state_full[L + t] = rank[map_raw[t]]
    return state_full


def _state_line_panel(Y, model, STY, color):
    """Step line of the MAP state for one example, aligned to the raster window."""
    state = _decode_map_state(Y, model)
    K = int(model["N_STATES"])
    x = np.arange(1, state.size + 1)
    fig, ax = plt.subplots(figsize=(1.020, 0.610), facecolor="w")
    ax.step(x, state, where="post", color=color, lw=STY.lw.fit)
    ax.set_xlim(*RASTER_WINDOW)
    ax.set_ylim(0.5, K + 0.5)
    lo, hi = RASTER_WINDOW
    ax.set_xticks([lo + 20, lo + 60, hi])
    ax.set_yticks(sorted({1, (K + 1) // 2, K}))
    # matlab_ticks=False keeps the sparse ticks set above; the default
    # re-derives ticks via MaxNLocator and would restore all five y labels.
    gt_axes(ax, STY, matlab_ticks=False)
    return fig


def _load_group_y(data_dir, row, ykey):
    Y = []
    for t in TYPES:
        S = load_struct(os.path.join(data_dir, f"xy_type{t}.mat"))
        y = np.asarray(S[ykey], dtype=float)[row]
        Y.append(1 - y)
    Y = np.concatenate(Y)
    return Y[~np.isnan(Y)]


def main(data_dir=DATA, out_dir=HERE):
    STY = gt_init()

    Yw = _load_group_y(data_dir, WT_SEL, "y_touches_wt")
    Ys = _load_group_y(data_dir, SH_SEL, "y_touches_sh")
    Yg = _load_group_y(data_dir, GT_SEL, "y_touches_gt")

    figs = []

    # ---- p1-p3: Y-position bar histogram, bins of width 0.02 over [0, 1] --
    edges = np.arange(0, 1 + 1e-9, 0.02)
    for name, Y, color in (("wt", Yw, STY.cmean[0]), ("sh", Ys, STY.cmean[1]), ("gt", Yg, STY.cmean[2])):
        counts = np.array([np.sum((Y > f) & (Y < f + 0.02)) for f in edges])
        fig, ax = plt.subplots(figsize=(1.020, 0.610), facecolor="w")
        ax.bar(np.arange(1, counts.size + 1), counts, color=color, edgecolor="none")
        gt_axes(ax, STY)
        gt_figStyle(fig, STY)
        figs.append((f"touchpos_{name}", fig))

    # ---- p4-p6: 20-bin histogram of d = 1 - Y, zero-valued Y set to NaN ----
    d_store = {}
    for name, Y in (("wt", Yw), ("sh", Ys), ("gt", Yg)):
        y_masked = Y.copy()
        y_masked[y_masked == 0] = np.nan
        d = 1 - y_masked
        d_store[name] = d
        fig, ax = plt.subplots(figsize=(1.020, 0.610), facecolor="w")
        ax.hist(d[~np.isnan(d)], bins=20)
        gt_axes(ax, STY)
        gt_figStyle(fig, STY)
        figs.append((f"dhist_{name}", fig))

    # ---- p7-p9: centrality histogram on the 1-100 index scale -------------
    for name, color in (("wt", STY.cmean[0]), ("sh", STY.cmean[1]), ("gt", STY.cmean[2])):
        d = d_store[name]
        dC = np.abs(d)
        n_valid = np.sum(~np.isnan(dC))
        f_vals = np.arange(1, 100, 2)
        counts = np.array([np.sum(np.abs(dC - f / 100.0) < 0.01) for f in f_vals])
        fig, ax = plt.subplots(figsize=(1.020, 0.610), facecolor="w")
        ax.bar(f_vals, counts, width=0.8, color=color, edgecolor="none")
        ax.set_ylim(0, n_valid / 10)
        gt_axes(ax, STY)
        gt_figStyle(fig, STY)
        figs.append((f"centrality_{name}", fig))
        print(f"  {name}: n_valid={n_valid}, centrality ylim=[0, {n_valid/10:.2f}]")

    # ---- p10-p12: binarized-state raster window [1001, 1100] --------------
    for name, color in (("wt", STY.cmean[0]), ("sh", STY.cmean[1]), ("gt", STY.cmean[2])):
        d = d_store[name].copy()
        below = d < STATE_THRESH
        above = d > STATE_THRESH
        d[below] = 0
        d[above] = 1
        # force two samples on, so the example window reads clearly
        for idx in FORCE_IDX:
            if d.size <= idx:
                d = np.pad(d, (0, idx + 1 - d.size), constant_values=0.0)
            d[idx] = 1
        fig, ax = plt.subplots(figsize=(1.020, 0.610), facecolor="w")
        ax.scatter(np.arange(1, d.size + 1), 1 - np.abs(d), s=STY.mk.pt,
                   c=[color], marker="s", edgecolors="none")
        ax.set_xlim(1001, 1100)
        ax.set_ylim(-0.5, 1.5)
        gt_axes(ax, STY)
        gt_figStyle(fig, STY)
        figs.append((f"raster_{name}", fig))

    # ---- p13-p15: GLM-HMM MAP-state step line, aligned under each raster ---
    # Reads the GLM-HMM fit of panel g (session types 2 and 3), stored in
    # data. Skipped with a message if that file is absent.
    if os.path.exists(S7g.MODEL_CACHE):
        model = dict(np.load(S7g.MODEL_CACHE))
        for name, Y, color in (("wt", Yw, STY.cmean[0]),
                                ("sh", Ys, STY.cmean[1]),
                                ("gt", Yg, STY.cmean[2])):
            fig = _state_line_panel(Y, model, STY, color)
            gt_figStyle(fig, STY)
            figs.append((f"stateline_{name}", fig))
        print("  added GLM-HMM MAP-state step lines (p13-p15) from the edfig7_g fit.")
    else:
        print("  [state line] no GLM-HMM fit found at "
              f"{S7g.MODEL_CACHE};\n    run edfig7_g_glmhmm.py with GT_REFIT=1 to "
              "rebuild it, then re-run this script to add the MAP-state "
              "panels (p13-p15).")

    for i, (stem, fig) in enumerate(figs, start=1):
        gt_saveFig(fig, f"edfig7_e_touch_distributions_p{i}_{stem}", out_dir)

    return [f for _, f in figs]


if __name__ == "__main__":
    main()
