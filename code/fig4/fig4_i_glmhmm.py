"""
fig4_i_glmhmm.py

Figure 4, panel i: GLM-HMM decomposition of center/edge touch dynamics
(WT, SH, GT).

Data. xy_type1.mat, xy_type4.mat and xy_type5.mat, fields y_touches_*. Per
animal the three session types are concatenated into a binary symbol
sequence: d = 1 - y_touches is thresholded at STATE_THRESH (0.3), giving
symbol 1 ("edge") below and symbol 2 ("center") above, with touches exactly
at threshold dropped. Animals with at least MIN_SEQ_LEN (10) usable touches
are pooled across WT, SH and GT to train a single shared model.

Model. A K = 5 state GLM-HMM. Each state carries its own logistic regression
for the probability of switching symbol, with an intercept and the previous
L = 6 binary symbols as regressors (D = L + 1 = 7 coefficients per state).
Transitions use a Dirichlet prior with self-transition weight SELF_BIAS = 12.
Fitting is by expectation-maximization: a scaled forward-backward E step, a
ridge-penalized IRLS M step for the GLM weights (LAMBDA_RIDGE = 1e-4) and a
MAP M step under the Dirichlet prior for the transition matrix. EM stops at a
relative log-likelihood improvement below TOL_EM (1e-8) or after MAX_EM
(1000) iterations.

Every animal is then decoded to give posterior state probabilities, per-state
usage (mean posterior across that animal's touches) and posterior-weighted
2 x 2 observation-transition matrices. Per-state bias is the mean probability
of a center touch under that state, 0.5 * (P(center | edge) +
P(center | center)).

Figures produced:
  p1     Training log-likelihood, AIC and BIC against the number of states,
         from a separate K = 1..5 sweep with the same EM settings. The
         log-likelihood elbow and the AIC and BIC minima are marked.
  p2-p4  One panel per group. Upward bars are the median per-state bias,
         downward bars the median per-state usage, with per-animal jittered
         points, a black median marker and a black 25th-75th percentile bar.
         States are sorted by ascending median bias within each group; the x
         tick labels give the underlying state index.

Fitting the model and the K sweep together take several minutes, so the
result of that fit is stored in data as fig4_i_glmhmm_fit.npz and the
script plots from it by default. The file holds the fitted parameters, the
K-sweep log-likelihood, AIC and BIC, and the per-animal bias and usage arrays
the group panels draw. Set the environment variable GT_REFIT=1 to refit from
xy_type*.mat and overwrite it. The same file also carries a global ordering of
states by pooled median center bias, which fig4_g_touch_distributions.py reads
to decode its example animals.

No group comparison test is applied; the outputs are log-likelihood, AIC, BIC
and descriptive medians and quartiles. The EM weight initialization is seeded
(12345). Point jitter is unseeded, so horizontal offsets differ slightly
between runs.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_e_touch_positions")

STATE_THRESH = 0.3
MIN_SEQ_LEN = 10
N_STATES = 5
TYPES = [1, 4, 5]
GROUPS = ["wt", "sh", "gt"]

# Precomputed fit. Read by this script and by fig4_g_touch_distributions.py,
# which decodes the MAP state sequence of its example animals from it.
MODEL_CACHE = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_i_glmhmm",
                           "fig4_i_glmhmm_fit.npz")
# GT_REFIT=1 refits from xy_type*.mat and rewrites MODEL_CACHE.
REFIT = os.environ.get("GT_REFIT", "0") not in ("", "0")

L = 6
MAX_EM = int(os.environ.get("GT_MAX_EM", "1000"))   # EM iteration cap. Lower
# it with GT_MAX_EM, alongside GT_REFIT=1, for a quick layout preview, in which
# case the model is not converged and the numbers are not final.
TOL_EM = 1e-8
LAMBDA_RIDGE = 1e-4
SELF_BIAS = 12
K_VALUES = [1, 2, 3, 4, 5]


# Data assembly
def _build_animal_obs(data_dir):
    animal_obs = {g: [None] * 1000 for g in GROUPS}
    for t in TYPES:
        S = load_struct(os.path.join(data_dir, f"xy_type{t}.mat"))
        for g in GROUPS:
            y = np.asarray(S[f"y_touches_{g}"], dtype=float).copy()
            y[y == 0] = np.nan
            Y = 1 - y
            for a in range(Y.shape[0]):
                d = Y[a]
                b = np.full(d.shape, np.nan)
                b[d < STATE_THRESH] = 0
                b[d > STATE_THRESH] = 1
                obs = b[~np.isnan(b)]
                if obs.size == 0:
                    continue
                sym = obs + 1   # symbols {1, 2}
                if animal_obs[g][a] is None:
                    animal_obs[g][a] = sym
                else:
                    animal_obs[g][a] = np.concatenate([animal_obs[g][a], sym])
    return animal_obs


# GLM-HMM core
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def make_lag_design(y01, lag):
    y01 = np.asarray(y01, dtype=float).ravel()
    T = y01.size
    if T <= lag:
        return np.zeros((0, lag + 1)), np.zeros(0)
    X = np.ones((T - lag, lag + 1))
    for l_ in range(1, lag + 1):
        X[:, l_] = y01[lag - l_: T - l_]
    y_curr = y01[lag:T]
    y_prev = y01[lag - 1: T - 1]
    y_switch = (y_curr != y_prev).astype(float)
    return X, y_switch


def make_pooled_design(seqs, lag):
    X_all, y_all, slices = [], [], []
    row = 0
    for seq in seqs:
        y01 = seq - 1
        X, y = make_lag_design(y01, lag)
        n = X.shape[0]
        if n == 0:
            slices.append((row, row - 1))
            continue
        X_all.append(X)
        y_all.append(y)
        slices.append((row, row + n - 1))
        row += n
    X_all = np.vstack(X_all) if X_all else np.zeros((0, lag + 1))
    y_all = np.concatenate(y_all) if y_all else np.zeros(0)
    return X_all, y_all, slices


def fb_core(like, A, pi0):
    """Scaled forward-backward. like: (K,T), A: (K,K), pi0: (K,)."""
    K, T = like.shape
    alpha = np.zeros((K, T))
    c = np.zeros(T)
    alpha[:, 0] = pi0 * like[:, 0]
    c[0] = alpha[:, 0].sum()
    if c[0] == 0:
        c[0] = np.finfo(float).eps
    alpha[:, 0] /= c[0]
    for t in range(1, T):
        alpha[:, t] = (A.T @ alpha[:, t - 1]) * like[:, t]
        c[t] = alpha[:, t].sum()
        if c[t] == 0:
            c[t] = np.finfo(float).eps
        alpha[:, t] /= c[t]

    beta = np.ones((K, T))
    for t in range(T - 2, -1, -1):
        beta[:, t] = A @ (like[:, t + 1] * beta[:, t + 1])
        beta[:, t] /= c[t + 1]

    Gamma = alpha * beta
    Gamma = Gamma / np.maximum(Gamma.sum(axis=0), np.finfo(float).eps)

    Xi = np.zeros((K, K, max(T - 1, 0)))
    for t in range(T - 1):
        numer = alpha[:, t][:, None] * A * (like[:, t + 1] * beta[:, t + 1])[None, :]
        denom = numer.sum()
        if denom == 0:
            denom = np.finfo(float).eps
        Xi[:, :, t] = numer / denom

    ll = np.sum(np.log(c))
    return Gamma, Xi, ll


def e_step_glm_hmm(X_all, y_all, slices, beta, A, pi0):
    K = beta.shape[1]
    Gamma_cells, Xi_cells = [], []
    ll_sum = 0.0
    for (r0, r1) in slices:
        if r1 < r0:
            Gamma_cells.append(None)
            Xi_cells.append(None)
            continue
        X = X_all[r0:r1 + 1]
        y = y_all[r0:r1 + 1]
        P1 = sigmoid(X @ beta)
        P1 = np.clip(P1, np.finfo(float).eps, 1 - np.finfo(float).eps)
        like = (P1 * y[:, None] + (1 - P1) * (1 - y[:, None])).T   # (K, n)
        Gamma, Xi, ll = fb_core(like, A, pi0)
        Gamma_cells.append(Gamma)
        Xi_cells.append(Xi)
        ll_sum += ll
    return Gamma_cells, Xi_cells, ll_sum


def m_step_transitions(Gamma_cells, Xi_cells, self_bias):
    K = None
    for xi in Xi_cells:
        if xi is not None:
            K = xi.shape[0]
            break
    if K is None:
        raise ValueError("No valid sequences for transition update.")

    Cexp = np.zeros((K, K))
    pi_acc = np.zeros(K)
    for Gm, Xi in zip(Gamma_cells, Xi_cells):
        if Gm is None:
            continue
        Cexp += Xi.sum(axis=2)
        pi_acc += Gm[:, 0]

    A_prior = np.ones((K, K))
    np.fill_diagonal(A_prior, self_bias)
    A_new = Cexp + A_prior
    A_new = A_new / A_new.sum(axis=1, keepdims=True)
    pi0_new = pi_acc / pi_acc.sum() if pi_acc.sum() > 0 else np.ones(K) / K
    if not np.all(np.isfinite(pi0_new)):
        pi0_new = np.ones(K) / K
    return A_new, pi0_new


def irls_logreg(X, y, w, lam, max_it=50, tol=1e-6):
    D = X.shape[1]
    b = np.zeros(D)
    I = np.eye(D)
    for _ in range(max_it):
        eta = X @ b
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        W = w * p * (1 - p)
        z = eta + (y - p) / np.maximum(p * (1 - p), 1e-9)
        XtW = X.T * W
        H = XtW @ X + lam * I
        g = XtW @ z
        b_new = np.linalg.solve(H, g)
        if np.linalg.norm(b_new - b) < tol * (1 + np.linalg.norm(b)):
            b = b_new
            break
        b = b_new
    return b


def m_step_betas(X_all, y_all, Gamma_cells, slices, K, lam):
    D = X_all.shape[1]
    beta_new = np.zeros((D, K))
    for k in range(K):
        w_all = np.zeros(X_all.shape[0])
        for (r0, r1), Gm in zip(slices, Gamma_cells):
            if r1 < r0 or Gm is None:
                continue
            w_all[r0:r1 + 1] = Gm[k, :]
        beta_new[:, k] = irls_logreg(X_all, y_all, w_all, lam)
    return beta_new


def fit_glm_hmm(seqs_train, K, D, rng, max_em=MAX_EM, tol_em=TOL_EM,
                 self_bias=SELF_BIAS, lam=LAMBDA_RIDGE, verbose=False, label=""):
    X_all, y_all, slices = make_pooled_design(seqs_train, L)
    beta = 0.01 * rng.standard_normal((D, K))
    A = np.ones((K, K))
    np.fill_diagonal(A, self_bias)
    A = A / A.sum(axis=1, keepdims=True)
    pi0 = np.ones(K) / K

    prev_ll = -np.inf
    ll_sum = prev_ll
    for it in range(max_em):
        Gamma_cells, Xi_cells, ll_sum = e_step_glm_hmm(X_all, y_all, slices, beta, A, pi0)
        A_new, pi0_new = m_step_transitions(Gamma_cells, Xi_cells, self_bias)
        beta_new = m_step_betas(X_all, y_all, Gamma_cells, slices, K, lam)
        rel_improve = (ll_sum - prev_ll) / max(1, abs(prev_ll))
        if verbose:
            print(f"  [{label}] EM {it+1:3d} | LL={ll_sum:.6f} | rel={rel_improve:.3e}")
        beta, A, pi0 = beta_new, A_new, pi0_new
        if rel_improve < tol_em:
            break
        prev_ll = ll_sum

    return dict(beta=beta, A=A, pi0=pi0, ll=ll_sum, X_all=X_all, y_all=y_all, slices=slices)


def posterior_glm_hmm_sequence(X_seq, y_switch, beta, A, pi0):
    if X_seq.shape[0] == 0:
        K = A.shape[0]
        return np.zeros((K, 0)), np.full((K, K), np.nan)
    P1 = sigmoid(X_seq @ beta)
    P1 = np.clip(P1, np.finfo(float).eps, 1 - np.finfo(float).eps)
    like = (P1 * y_switch[:, None] + (1 - P1) * (1 - y_switch[:, None])).T
    Gamma, Xi, _ = fb_core(like, A, pi0)
    Cexp = Xi.sum(axis=2)
    rows = Cexp.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1
    Texp = Cexp / rows
    return Gamma, Texp


def _k_sweep(seqs_train, D, rng):
    """Refit the model at each K in K_VALUES and return LL, AIC and BIC."""
    LL_perK, AIC_perK, BIC_perK = [], [], []
    for K in K_VALUES:
        fit = fit_glm_hmm(seqs_train, K, D, rng, label=f"K={K} sweep")
        LL_perK.append(fit["ll"])
        Nobs = fit["y_all"].size
        p = D * K + K * (K - 1) + (K - 1)
        AIC_perK.append(2 * p - 2 * fit["ll"])
        BIC_perK.append(p * np.log(max(1, Nobs)) - 2 * fit["ll"])
    return {"K_values": np.array(K_VALUES, dtype=float),
            "LL_perK": np.array(LL_perK),
            "AIC_perK": np.array(AIC_perK),
            "BIC_perK": np.array(BIC_perK)}


# Panels
def _panel1_k_sweep(R, STY):
    """Log-likelihood, AIC and BIC against the number of states."""
    K_values = [int(k) for k in R["K_values"]]
    LL_perK = np.asarray(R["LL_perK"], dtype=float)
    AIC_perK = np.asarray(R["AIC_perK"], dtype=float)
    BIC_perK = np.asarray(R["BIC_perK"], dtype=float)
    Kv = np.asarray(K_values, dtype=float)

    x = (Kv - Kv[0]) / max(np.finfo(float).eps, Kv[-1] - Kv[0])
    yv = (LL_perK - LL_perK[0]) / max(np.finfo(float).eps, LL_perK[-1] - LL_perK[0])
    elbow_idx = int(np.argmax(np.abs(yv - x)))
    K_elbow = K_values[elbow_idx]
    iA = int(np.argmin(AIC_perK))
    iB = int(np.argmin(BIC_perK))
    print(f"  Elbow(LL): K={K_elbow} | min AIC at K={K_values[iA]} | min BIC at K={K_values[iB]}")

    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.0), facecolor="w")
    axes[0].plot(K_values, LL_perK, "-o", lw=STY.lw.fit)
    axes[0].axvline(K_elbow, ls="--", color="k")
    axes[0].set_xlabel("Number of states K")
    axes[0].set_ylabel("Log-likelihood (train)")
    axes[0].set_title("GLM-HMM fit vs K (LL)")

    axes[1].plot(K_values, AIC_perK, "-o", lw=STY.lw.fit)
    axes[1].axvline(K_values[iA], ls="--", color="k")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("AIC")
    axes[1].set_title("AIC vs K")

    axes[2].plot(K_values, BIC_perK, "-o", lw=STY.lw.fit)
    axes[2].axvline(K_values[iB], ls="--", color="k")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("BIC")
    axes[2].set_title("BIC vs K")

    for ax in axes:
        gt_axes(ax, STY)
    gt_figStyle(fig, STY)
    return fig


def _panel_group_bias_usage(group_label, bias, usage, STY, rng_plot):
    """Per-state bias and usage for one group, states sorted by ascending
    median bias. bias and usage are (nAnimals, K) arrays."""
    K = bias.shape[1]
    valid = np.any(~np.isnan(bias), axis=1) | np.any(~np.isnan(usage), axis=1)
    B = bias[valid]
    U = usage[valid]

    with np.errstate(invalid="ignore"):
        med_bias = np.nanmedian(B, axis=0)
        q25_bias = np.nanpercentile(B, 25, axis=0)
        q75_bias = np.nanpercentile(B, 75, axis=0)
        med_use = np.nanmedian(U, axis=0)
        q25_use = np.nanpercentile(U, 25, axis=0)
        q75_use = np.nanpercentile(U, 75, axis=0)

    order = np.argsort(med_bias)
    B, U = B[:, order], U[:, order]
    med_bias, q25_bias, q75_bias = med_bias[order], q25_bias[order], q75_bias[order]
    med_use, q25_use, q75_use = med_use[order], q25_use[order], q75_use[order]
    xlabs = [f"s{s+1}" for s in order]

    bias_color = STY.clr.bias
    usage_color = STY.clr.usage

    fig, ax = plt.subplots(figsize=(1.52, 1.43), facecolor="w")
    xs = np.arange(1, K + 1)
    ax.bar(xs, med_bias, width=0.8, color=bias_color, alpha=0.65, edgecolor="none",
           zorder=1)
    ax.bar(xs, -med_use, width=0.8, color=usage_color, alpha=0.65, edgecolor="none",
           zorder=1)

    # Jittered per-animal dots (behind the black median markers).
    for s in range(K):
        xj = xs[s] + 0.14 * (rng_plot.random(B.shape[0]) - 0.5)
        ax.scatter(xj, B[:, s], s=STY.mk.pt, c=[bias_color], alpha=0.60,
                   edgecolors="none", zorder=2)
        xj2 = xs[s] + 0.14 * (rng_plot.random(U.shape[0]) - 0.5)
        ax.scatter(xj2, -U[:, s], s=STY.mk.pt, c=[usage_color], alpha=0.60,
                   edgecolors="none", zorder=2)

    # Black median dot and black interquartile bar per state, so the summary
    # reads clearly against the colored bars behind it.
    for s in range(K):
        ax.plot([xs[s], xs[s]], [q25_bias[s], q75_bias[s]], "-", color="k",
                lw=STY.lw.fit, zorder=3)
        ax.plot([xs[s], xs[s]], [-q25_use[s], -q75_use[s]], "-", color="k",
                lw=STY.lw.fit, zorder=3)
        ax.scatter([xs[s]], [med_bias[s]], s=STY.mk.mean, c="k", zorder=4)
        ax.scatter([xs[s]], [-med_use[s]], s=STY.mk.mean, c="k", zorder=4)

    ax.set_xlim(0.5, K + 0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabs)
    ax.set_ylabel("Bias (up)  |  Usage (down)")
    # Group name only: the x tick order already shows the bias sorting, and a
    # longer title does not fit the 1.52 in panel.
    ax.set_title(group_label.upper())
    ax.axhline(0, color="k", lw=STY.lw.axis)          # solid 0 baseline
    ax.axhline(0.5, color="k", ls="--", lw=STY.lw.hair)  # dashed 0.5 marker
    gt_axes(ax, STY)
    gt_figStyle(fig, STY)
    return fig


def _fit_all(data_dir):
    """Fit the K = 5 model, run the K sweep and decode every animal.

    Returns the arrays the four sub-panels draw: the fitted parameters, the
    global state order, the K-sweep LL/AIC/BIC, the per-animal per-state bias
    and usage of each group, and the sequence counts printed by _report."""
    rng_em = np.random.default_rng(12345)   # seeds the EM weight initialization

    print("fig4_i_glmhmm: assembling per-animal binary state sequences (types 1,4,5)...")
    animal_obs = _build_animal_obs(data_dir)

    counts = {}
    for g in GROUPS:
        idx = [a for a in range(1000) if animal_obs[g][a] is not None]
        tot0 = sum(int(np.sum(animal_obs[g][a] - 1 == 0)) for a in idx)
        tot1 = sum(int(np.sum(animal_obs[g][a] - 1 == 1)) for a in idx)
        ratios = [np.mean(animal_obs[g][a] - 1) for a in idx]
        counts[g] = np.array([len(idx), tot0, tot1, float(np.mean(ratios))])

    seqs_train = []
    seq_owner = []   # (group, animal_idx) parallel to seqs_train
    for g in GROUPS:
        for a in range(1000):
            O = animal_obs[g][a]
            if O is not None and O.size >= MIN_SEQ_LEN:
                seqs_train.append(O)
                seq_owner.append((g, a))
    n_pooled = int(sum(s.size for s in seqs_train))
    print(f"Collected {len(seqs_train)} animals (WT+SH+GT combined) with >= {MIN_SEQ_LEN} samples for TRAINING.")

    D = L + 1

    print(f"Fitting main K={N_STATES} GLM-HMM ({n_pooled} pooled observations)...")
    fit = fit_glm_hmm(seqs_train, N_STATES, D, rng_em, label=f"K={N_STATES} main")
    beta, A, pi0 = fit["beta"], fit["A"], fit["pi0"]
    print(f"  converged: LL={fit['ll']:.3f}")

    # Decode every animal: per-state usage and 0/1 observation-transition matrices.
    usage_by_group = {g: np.full((1000, N_STATES), np.nan) for g in GROUPS}
    obs01_trans_by_state = {g: [None] * 1000 for g in GROUPS}

    for g, a in seq_owner:
        O = animal_obs[g][a]
        y_seq = O - 1
        X_seq, y_switch = make_lag_design(y_seq, L)
        Gamma, _ = posterior_glm_hmm_sequence(X_seq, y_switch, beta, A, pi0)
        if Gamma.shape[1] == 0:
            continue
        usage_by_group[g][a, :] = Gamma.mean(axis=1)

        T = y_seq.size
        mats = [None] * N_STATES
        if T >= L + 2:
            for k in range(N_STATES):
                C = np.zeros((2, 2))
                for t_g in range(1, Gamma.shape[1]):
                    y_prev = y_seq[L + t_g - 1]
                    y_curr = y_seq[L + t_g]
                    if np.isnan(y_prev) or np.isnan(y_curr):
                        continue
                    w = 0.5 * (Gamma[k, t_g - 1] + Gamma[k, t_g])
                    C[int(y_prev), int(y_curr)] += w
                rows = C.sum(axis=1, keepdims=True)
                rows[rows == 0] = 1
                mats[k] = C / rows
        obs01_trans_by_state[g][a] = mats

    # Per-animal per-state bias, drawn by sub-panels p2-p4.
    R = {}
    for g in GROUPS:
        bias = np.full((1000, N_STATES), np.nan)
        for a in range(1000):
            mats = obs01_trans_by_state[g][a]
            if mats is None:
                continue
            for s in range(N_STATES):
                M = mats[s]
                if M is None or not np.all(np.isfinite(M)):
                    continue
                bias[a, s] = 0.5 * (M[0, 1] + M[1, 1])
        R[f"bias_{g}"] = bias
        R[f"usage_{g}"] = usage_by_group[g]

    # Global state ordering, by pooled (all-group) median center bias, so that
    # a higher relabeled state index is more center-biased and the labeling
    # is comparable across the three example animals of panel g.
    # fig4_g_touch_distributions.py reads it to draw their MAP state over time.
    # The per-group panels below sort independently; this ordering is used only
    # for the state lines in panel g.
    all_bias = np.vstack([R[f"bias_{g}"] for g in GROUPS])
    all_bias = all_bias[np.any(~np.isnan(all_bias), axis=1)]
    global_med_bias = (np.nanmedian(all_bias, axis=0) if all_bias.size
                       else np.full(N_STATES, np.nan))

    # Sub-panel p1: refit at each K for the LL/AIC/BIC curves.
    print("K-sweep (LL/AIC/BIC vs K)...")
    R.update(_k_sweep(seqs_train, D, rng_em))

    R.update(beta=beta, A=A, pi0=pi0, L=L, N_STATES=N_STATES,
             global_order=np.argsort(global_med_bias),
             global_med_bias=global_med_bias, ll=fit["ll"],
             n_train=len(seqs_train), n_pooled=n_pooled,
             **{f"counts_{g}": counts[g] for g in GROUPS})
    return R


def _report(R):
    """Print the sequence counts, the fitted log-likelihood and the K sweep."""
    for g in GROUPS:
        n, tot0, tot1, ratio1 = R[f"counts_{g}"]
        print(f"  {g.upper()}: n={int(n)} animals, total frames={int(tot0 + tot1)} "
              f"[0]={int(tot0)}, [1]={int(tot1)}, mean ratio1={ratio1:.3f}")
    print(f"  {int(R['n_train'])} animals pooled for training, "
          f"{int(R['n_pooled'])} observations; K={int(R['N_STATES'])} LL={float(R['ll']):.3f}")
    for K, ll, aic, bic in zip(R["K_values"], R["LL_perK"], R["AIC_perK"], R["BIC_perK"]):
        print(f"  K={int(K):2d} | LL={ll:.3f} | AIC={aic:.3f} | BIC={bic:.3f}")


def main(data_dir=DATA, out_dir=HERE, refit=REFIT):
    STY = gt_init()
    rng_plot = np.random.default_rng()

    if refit or not os.path.exists(MODEL_CACHE):
        R = _fit_all(data_dir)
        os.makedirs(os.path.dirname(MODEL_CACHE), exist_ok=True)
        np.savez(MODEL_CACHE, **R)
        print(f"  wrote fit -> {MODEL_CACHE}")
    else:
        R = dict(np.load(MODEL_CACHE))
        print(f"fig4_i_glmhmm: plotting the precomputed fit in {os.path.basename(MODEL_CACHE)}"
              "\n  (set GT_REFIT=1 to refit from xy_type*.mat, which takes several minutes)")

    _report(R)

    figs = [_panel1_k_sweep(R, STY)]
    for g in GROUPS:
        figs.append(_panel_group_bias_usage(g, R[f"bias_{g}"], R[f"usage_{g}"],
                                            STY, rng_plot))

    for i, fig in enumerate(figs, start=1):
        gt_saveFig(fig, f"fig4_i_glmhmm_p{i}", out_dir)

    return figs


if __name__ == "__main__":
    main()
