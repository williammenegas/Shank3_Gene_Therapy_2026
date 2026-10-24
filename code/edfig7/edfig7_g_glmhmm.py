"""Extended Data Figure 7, panel g: GLM-HMM decomposition of touchscreen
behavior into hidden states, per-state bias and usage.

Data: `xy_type2.mat` and `xy_type3.mat`, variable `y_touches_*`, session types
2 and 3. Each animal's touch position is taken as Y = 1 - y_touches and
binarized at STATE_THRESH = 0.3: Y < 0.3 gives 0, Y > 0.3 gives 1, and touches
exactly at the threshold are left out, breaking the series at that point. The
two session types are concatenated per animal.

Model: a K = N_STATES = 5 state GLM-HMM (generalised linear model with hidden
Markov dynamics), fitted by expectation-maximization on every eligible animal
from all three groups pooled together. The fitted model is therefore shared
across groups; only the posterior state usage differs by group. Sequences
shorter than MIN_SEQ_LEN = 10 observations are excluded from training.

Each state's emission model is a logistic regression, sigmoid(X @ beta_k),
giving the probability of a switch (y_t differing from y_{t-1}) from an
intercept plus L = 6 lags of the binarized series. The transition matrix
carries a sticky Dirichlet prior with SELF_BIAS = 12 on the diagonal, so states
persist.

EM alternates:
  E step  Per-sequence scaled forward-backward (`fb_core`) over the per-state
          Bernoulli likelihood of the observed switch labels, returning the
          state posteriors Gamma and Xi and the total log-likelihood.
  M step  The transition matrix and initial distribution from the expected
          transition and initial counts plus the sticky prior; each state's
          beta_k from a responsibility-weighted ridge logistic regression
          solved by iteratively reweighted least squares (`irls_logreg`, up to
          50 Newton steps, ridge penalty LAMBDA_RIDGE = 1e-4).

The loop runs to MAX_EM = 1000 iterations or until the relative improvement in
log-likelihood falls below TOL_EM = 1e-8. GT_MAX_EM overrides MAX_EM.
Alongside GT_REFIT=1, it gives a quick preview from an unconverged model.

Posterior decoding: every animal with at least MIN_SEQ_LEN + L observations
receives posterior state probabilities (the first L frames have no lag features
and are left undefined), a per-state mean usage, and a
responsibility-weighted 2 x 2 observation transition matrix per state, formed
by adding 0.5 * (Gamma[k, t-1] + Gamma[k, t]) to the (y_prev, y_curr) cell and
normalizing each row. From each of those matrices, the state's center bias is
0.5 * (P(y_t = 1 | y_{t-1} = 0) + P(y_t = 1 | y_{t-1} = 1)).

Plot: one figure per group. The five states are sorted by ascending median
bias. Median bias is drawn as upward bars and median usage as downward bars,
each with a 25th-to-75th-percentile bar and jittered per-animal dots. The x
tick labels give each state's index before sorting. The y axis spans
[-1.05, 1.05], with reference lines at 0 and 0.5.

Fitting takes several minutes, so the result is stored in data as
edfig7_g_glmhmm_fit.npz and the script plots from it by default. The file
holds the fitted parameters, the pooled state ordering and the per-animal bias
and usage arrays the three panels draw. Set the environment variable
GT_REFIT=1 to refit from xy_type*.mat and overwrite it. Panel e reads the same
file to draw its example animals' states.

Random number generation: one generator seeded 12345 initializes the emission
weights, and a second seeded 0 produces the per-animal dot jitter in the plots.
EM is a non-convex fit, so the converged parameters depend on the
initialization.

The panel is descriptive; the GLM-HMM is a model fit rather than a hypothesis
test, and no significance test is applied.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gtstyle import *  # noqa: E402,F403

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "fig4", "fig4_e_touch_positions")

STATE_THRESH = 0.3
MIN_SEQ_LEN = 10
N_STATES = 5
TYPES = (2, 3)
GROUPS = ("wt", "sh", "gt")

# Precomputed fit, trained on session types (2, 3). This script reads it, and
# so does edfig7_e_touch_distributions.py. That script draws its example
# animals' MAP state over time from the fit.
MODEL_CACHE = os.path.join(HERE, "..", "..", "data", "edfig7",
                           "edfig7_g_glmhmm", "edfig7_g_glmhmm_fit.npz")
# GT_REFIT=1 refits from xy_type*.mat and rewrites MODEL_CACHE.
REFIT = os.environ.get("GT_REFIT", "0") not in ("", "0")

L = 6                                               # number of lag features
MAX_EM = int(os.environ.get("GT_MAX_EM", "1000"))   # EM iteration cap; a lower
# GT_MAX_EM, alongside GT_REFIT=1, gives a fast preview from a model that has
# not converged.
TOL_EM = 1e-8
LAMBDA_RIDGE = 1e-4
SELF_BIAS = 12
VERBOSE = True

EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def make_lag_design(y01, Lval):
    """Build the lag design matrix for one sequence.

    y01: 1-D 0/1 array. Returns (X, y_switch), where X holds an intercept and
    `Lval` lags of y01 and y_switch marks where the series changes value.
    """
    y01 = np.asarray(y01, dtype=float).ravel()
    T = y01.size
    if T <= Lval:
        return np.zeros((0, Lval + 1)), np.zeros((0,))
    X = np.ones((T - Lval, Lval + 1))
    for l in range(1, Lval + 1):
        X[:, l] = y01[Lval - l:T - l]
    y_curr = y01[Lval:T]
    y_prev = y01[Lval - 1:T - 1]
    y_switch = (y_curr != y_prev).astype(float)
    return X, y_switch


def make_pooled_design(seqs, Lval):
    """Stack the lag designs of many sequences into one matrix.

    Returns (X_all, y_all, slices), where slices holds one (start, n) pair per
    input sequence indexing into X_all and y_all; n = 0 marks a sequence too
    short to contribute any rows.
    """
    X_chunks, y_chunks, slices = [], [], []
    row = 0
    for seq in seqs:
        y01 = np.asarray(seq, dtype=float).ravel() - 1
        X, y = make_lag_design(y01, Lval)
        n = X.shape[0]
        if n == 0:
            slices.append((row, 0))
            continue
        X_chunks.append(X)
        y_chunks.append(y)
        slices.append((row, n))
        row += n
    X_all = np.concatenate(X_chunks, axis=0) if X_chunks else np.zeros((0, Lval + 1))
    y_all = np.concatenate(y_chunks, axis=0) if y_chunks else np.zeros((0,))
    return X_all, y_all, slices


def fb_core(like, A, pi0):
    """Scaled forward-backward pass over one sequence.

    like: K x T per-state observation likelihoods. Returns the state posteriors
    Gamma (K x T), the pairwise posteriors Xi (K x K x T-1) and the sequence
    log-likelihood.
    """
    K, T = like.shape
    alpha = np.zeros((K, T))
    c = np.zeros(T)
    alpha[:, 0] = pi0 * like[:, 0]
    c[0] = alpha[:, 0].sum()
    if c[0] == 0:
        c[0] = EPS
    alpha[:, 0] /= c[0]

    AT = A.T
    for t in range(1, T):
        a = (AT @ alpha[:, t - 1]) * like[:, t]
        s = a.sum()
        if s == 0:
            s = EPS
        alpha[:, t] = a / s
        c[t] = s

    beta = np.ones((K, T))
    for t in range(T - 2, -1, -1):
        b = A @ (like[:, t + 1] * beta[:, t + 1])
        beta[:, t] = b / c[t + 1]

    Gamma = alpha * beta
    Gamma = Gamma / np.maximum(Gamma.sum(axis=0), EPS)

    Xi = np.zeros((K, K, T - 1))
    for t in range(T - 1):
        likexbeta = like[:, t + 1] * beta[:, t + 1]
        numer = np.outer(alpha[:, t], likexbeta) * A
        denom = numer.sum()
        if denom == 0:
            denom = EPS
        Xi[:, :, t] = numer / denom

    ll = np.sum(np.log(c))
    return Gamma, Xi, ll


def e_step_glm_hmm(X_all, y_all, slices, beta, A, pi0):
    """EM E step: per-sequence posteriors and the total log-likelihood."""
    Gamma_cells = [None] * len(slices)
    Xi_cells = [None] * len(slices)
    ll_sum = 0.0
    for i, (start, n) in enumerate(slices):
        if n == 0:
            continue
        X = X_all[start:start + n]
        y = y_all[start:start + n]
        P1 = sigmoid(X @ beta)                    # n x K
        P1 = np.clip(P1, EPS, 1 - EPS)
        like = (P1 * y[:, None] + (1 - P1) * (1 - y[:, None])).T   # K x n
        Gamma, Xi, ll = fb_core(like, A, pi0)
        Gamma_cells[i] = Gamma
        Xi_cells[i] = Xi
        ll_sum += ll
    return Gamma_cells, Xi_cells, ll_sum


def m_step_transitions(Gamma_cells, Xi_cells, self_bias):
    """EM M step for the transition matrix and initial distribution.

    Expected transition counts are combined with a Dirichlet prior carrying
    `self_bias` on the diagonal, then row-normalized.
    """
    K = None
    for Xi in Xi_cells:
        if Xi is not None:
            K = Xi.shape[0]
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
    denom = pi_acc.sum()
    pi0_new = pi_acc / denom if denom != 0 else np.full(K, np.nan)
    if not np.all(np.isfinite(pi0_new)):
        pi0_new = np.ones(K) / K
    return A_new, pi0_new


def irls_logreg(X, y, w, lam, max_it=50, tol=1e-6):
    """Weighted ridge logistic regression by iteratively reweighted least
    squares. `w` holds the per-observation weights and `lam` the ridge
    penalty."""
    D = X.shape[1]
    b = np.zeros(D)
    I = np.eye(D)
    for _ in range(max_it):
        eta = X @ b
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        W = -(w * p * (p - 1))
        z = eta + (y - p) / np.maximum(p * (1 - p), 1e-9)
        XTW = X.T * W
        H = XTW @ X + lam * I
        g = XTW @ z
        b_new = np.linalg.solve(H, g)
        if np.linalg.norm(b_new - b) < tol * (1 + np.linalg.norm(b)):
            b = b_new
            break
        b = b_new
    return b


def m_step_betas(X_all, y_all, Gamma_cells, slices, K, lambda_ridge):
    """EM M step for the emission weights: one responsibility-weighted ridge
    logistic regression per state."""
    D = X_all.shape[1]
    beta_new = np.zeros((D, K))
    N = X_all.shape[0]
    for k in range(K):
        w_all = np.zeros(N)
        for Gm, (start, n) in zip(Gamma_cells, slices):
            if n == 0:
                continue
            w_all[start:start + n] = Gm[k, :]
        beta_new[:, k] = irls_logreg(X_all, y_all, w_all, lambda_ridge)
    return beta_new


def fit_glm_hmm(seqs_train, K, D, rng, max_em=MAX_EM, tol_em=TOL_EM,
                self_bias=SELF_BIAS, lam=LAMBDA_RIDGE, verbose=VERBOSE,
                time_budget_s=None, t_start=None):
    """EM fit of a K-state GLM-HMM to the pooled sequences.

    Emission weights start from 0.01 * standard normal draws, the transition
    matrix from a self-biased row-stochastic matrix and the initial state
    distribution from the uniform. EM stops at max_em iterations, at a relative
    log-likelihood improvement below tol_em, or at the optional wall-clock
    budget time_budget_s measured from t_start."""
    X_all, y_all, seq_slices = make_pooled_design(seqs_train, L)

    beta = 0.01 * rng.standard_normal((D, K))
    A = np.ones((K, K))
    np.fill_diagonal(A, self_bias)
    A = A / A.sum(axis=1, keepdims=True)
    pi0 = np.ones(K) / K

    prev_ll = -np.inf
    ll_sum = prev_ll
    n_iters_run = 0
    for it in range(1, max_em + 1):
        Gamma_cells, Xi_cells, ll_sum = e_step_glm_hmm(X_all, y_all, seq_slices, beta, A, pi0)
        A_new, pi0_new = m_step_transitions(Gamma_cells, Xi_cells, self_bias)
        beta_new = m_step_betas(X_all, y_all, Gamma_cells, seq_slices, K, lam)

        with np.errstate(invalid="ignore"):
            # On the first iteration prev_ll is -inf, so this evaluates to NaN.
            # NaN < tol_em is false, so the loop does not stop there.
            rel_improve = (ll_sum - prev_ll) / max(1.0, abs(prev_ll))
        if verbose:
            print(f"EM iter {it:3d} | LL = {ll_sum:.6f} | rel improve = {rel_improve:.3e}")

        beta, A, pi0 = beta_new, A_new, pi0_new
        n_iters_run = it
        if rel_improve < tol_em:
            if verbose:
                print(f"Converged (tol = {tol_em:.2e}).")
            break
        prev_ll = ll_sum
        if time_budget_s is not None and (time.time() - t_start) > time_budget_s:
            print(f"[time_budget_s={time_budget_s}] wall-clock budget reached at iter {it} "
                  f"(rel_improve={rel_improve:.3e}, tolerance not yet met); "
                  f"EM stopped early.")
            break

    print(f"EM finished after {n_iters_run} iterations (of a max {max_em}).")
    return dict(beta=beta, A=A, pi0=pi0, ll=ll_sum, n_iters=n_iters_run)


def posterior_glm_hmm_sequence(X_seq, y_switch, beta, A, pi0):
    """Posterior state probabilities and expected transition matrix for one
    sequence under a fitted model."""
    K = A.shape[0]
    if X_seq.shape[0] == 0:
        return np.full((K, 0), np.nan), np.full((K, K), np.nan)
    P1 = sigmoid(X_seq @ beta)
    P1 = np.clip(P1, EPS, 1 - EPS)
    like = (P1 * y_switch[:, None] + (1 - P1) * (1 - y_switch[:, None])).T
    Gamma, Xi, _ = fb_core(like, A, pi0)
    Cexp = Xi.sum(axis=2)
    rows = Cexp.sum(axis=1)
    rows[rows == 0] = 1
    Texp = Cexp / rows[:, None]
    return Gamma, Texp


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _fit_all(data_dir, time_budget_s=None):
    """Fit the GLM-HMM and decode every animal.

    Returns the arrays the three panels draw: the fitted parameters, the
    pooled state ordering, and the per-animal per-state bias and usage of each
    group.

    time_budget_s is an optional wall-clock cutoff for the EM loop, intended
    for quick test runs. With the default of None, EM stops only at MAX_EM
    iterations or when the relative improvement falls below TOL_EM. If a budget
    is supplied and reached first, the last 'EM iter N' line in the console log
    records how far the fit progressed.
    """
    t_start = time.time()
    rng_em = np.random.default_rng(12345)     # seeds the emission-weight init only

    # ---- per-animal binarized observation sequences, pooled across types --
    animal_obs = {g: [None] * 1000 for g in GROUPS}

    for t in TYPES:
        S = load_struct(os.path.join(data_dir, f"xy_type{t}.mat"))
        for g in GROUPS:
            y = np.asarray(S[f"y_touches_{g}"], dtype=float).copy()
            y[y == 0] = np.nan
            Yg = 1 - y
            nAnimals = Yg.shape[0]
            for a in range(nAnimals):
                d = Yg[a]
                b = np.full(d.shape, np.nan)
                b[d < STATE_THRESH] = 0
                b[d > STATE_THRESH] = 1
                obs = b[~np.isnan(b)]
                if obs.size == 0:
                    continue
                if animal_obs[g][a] is None:
                    animal_obs[g][a] = obs + 1
                else:
                    animal_obs[g][a] = np.concatenate([animal_obs[g][a], obs + 1])

    # ---- per-animal 0/1 counts and ratios, summarized on the console ------
    counts01 = {g: [None] * 1000 for g in GROUPS}
    ratio1 = {g: np.full(1000, np.nan) for g in GROUPS}
    for g in GROUPS:
        for a in range(1000):
            O = animal_obs[g][a]
            if O is None:
                continue
            y01 = O - 1
            c0 = np.sum(y01 == 0)
            c1 = np.sum(y01 == 1)
            counts01[g][a] = (c0, c1)
            if c0 + c1 > 0:
                ratio1[g][a] = c1 / (c0 + c1)

    counts = {}
    for g in GROUPS:
        idx = [a for a in range(1000) if counts01[g][a] is not None]
        tot0 = sum(counts01[g][a][0] for a in idx)
        tot1 = sum(counts01[g][a][1] for a in idx)
        mean_ratio = np.nanmean(ratio1[g][idx]) if idx else np.nan
        counts[g] = np.array([len(idx), tot0, tot1, float(mean_ratio)])

    # ---- collect training sequences from all groups -----------------------
    seqs_train = []
    for g in GROUPS:
        for a in range(1000):
            O = animal_obs[g][a]
            if O is not None and O.size >= MIN_SEQ_LEN:
                seqs_train.append(O)
    print(f"Collected {len(seqs_train)} animals (WT+SH+GT combined) with >= "
          f"{MIN_SEQ_LEN} samples for TRAINING.")

    K = N_STATES
    D = L + 1
    F = fit_glm_hmm(seqs_train, K, D, rng_em,
                    time_budget_s=time_budget_s, t_start=t_start)
    beta, A, pi0 = F["beta"], F["A"], F["pi0"]
    ll_sum, n_iters_run = F["ll"], F["n_iters"]

    # ---- posterior decoding, all groups -------------------------------------
    usage_by_group = {g: np.full((1000, K), np.nan) for g in GROUPS}
    obs01_trans_by_state = {g: [None] * 1000 for g in GROUPS}

    for g in GROUPS:
        for a in range(1000):
            O = animal_obs[g][a]
            if O is None or O.size < (MIN_SEQ_LEN + L):
                continue
            y_seq = O - 1
            X_seq, y_switch = make_lag_design(y_seq, L)
            Gamma, _Texp_post = posterior_glm_hmm_sequence(X_seq, y_switch, beta, A, pi0)

            T = y_seq.size
            if X_seq.shape[0] == 0:
                continue

            usage_by_group[g][a, :] = Gamma.mean(axis=1)

            Kloc = Gamma.shape[0]
            Tg = Gamma.shape[1]
            mats = [None] * Kloc
            if T >= (L + 2):
                for k in range(Kloc):
                    C = np.zeros((2, 2))
                    for t_g in range(2, Tg + 1):    # 1-based index into Gamma
                        y_prev = y_seq[L + t_g - 1 - 1]
                        y_curr = y_seq[L + t_g - 1]
                        if np.isnan(y_prev) or np.isnan(y_curr):
                            continue
                        w = 0.5 * (Gamma[k, t_g - 2] + Gamma[k, t_g - 1])
                        C[int(y_prev), int(y_curr)] += w
                    rows = C.sum(axis=1)
                    rows[rows == 0] = 1
                    mats[k] = C / rows[:, None]
            else:
                mats = [np.full((2, 2), np.nan) for _ in range(Kloc)]
            obs01_trans_by_state[g][a] = mats

    # ---- per-animal, per-state bias/usage metrics ----------------------------
    metrics = {}
    for g in GROUPS:
        mats_all = obs01_trans_by_state[g]
        nAnimalsMax = len(mats_all)
        Kloc = N_STATES
        bias = np.full((nAnimalsMax, Kloc), np.nan)
        for a in range(nAnimalsMax):
            if mats_all[a] is None:
                continue
            for s in range(Kloc):
                M = mats_all[a][s]
                if M is None or not np.all(np.isfinite(M)):
                    continue
                p10 = M[0, 1]
                p11 = M[1, 1]
                bias[a, s] = 0.5 * (p10 + p11)
        metrics[g] = bias

    # ---- pooled state order, used by panel e -------------------------------
    # The order is by ascending pooled median center bias, so that panel e's
    # relabeled state line is consistent with the panels below.
    global_med_bias = np.nanmedian(np.vstack([metrics[g] for g in GROUPS]), axis=0)

    R = dict(beta=beta, A=A, pi0=pi0, L=L, N_STATES=N_STATES,
             global_order=np.argsort(global_med_bias),
             global_med_bias=global_med_bias, ll=ll_sum,
             n_train=len(seqs_train), n_iters=n_iters_run,
             **{f"counts_{g}": counts[g] for g in GROUPS})
    for g in GROUPS:
        R[f"bias_{g}"] = metrics[g]
        R[f"usage_{g}"] = usage_by_group[g]
    return R


def _report(R):
    """Print the per-group sequence counts and the fitted log-likelihood."""
    for g in GROUPS:
        n, tot0, tot1, ratio1 = R[f"counts_{g}"]
        print(f"  {g.upper()}: n={int(n)} animals, total frames={int(tot0 + tot1)}  "
              f"[0]={int(tot0)}, [1]={int(tot1)}, mean ratio1={ratio1:.3f}")
    print(f"  {int(R['n_train'])} animals pooled for training; "
          f"K={int(R['N_STATES'])} LL={float(R['ll']):.6f} "
          f"after {int(R['n_iters'])} EM iterations")


def main(data_dir=DATA, out_dir=HERE, refit=REFIT):
    """Draw one bias/usage figure per group from the fitted GLM-HMM."""
    STY = gt_init()

    if refit or not os.path.exists(MODEL_CACHE):
        R = _fit_all(data_dir)
        os.makedirs(os.path.dirname(MODEL_CACHE), exist_ok=True)
        np.savez(MODEL_CACHE, **R)
        print(f"  wrote fit -> {MODEL_CACHE}")
    else:
        R = dict(np.load(MODEL_CACHE))
        print(f"edfig7_g_glmhmm: plotting the precomputed fit in {os.path.basename(MODEL_CACHE)}"
              "\n  (set GT_REFIT=1 to refit from xy_type*.mat, which takes several minutes)")

    _report(R)

    metrics = {g: R[f"bias_{g}"] for g in GROUPS}
    usage_by_group = {g: R[f"usage_{g}"] for g in GROUPS}

    # ---- sorted bias-up / usage-down summary, one figure per group ----------
    rng_plot = np.random.default_rng(0)    # seeds the per-animal dot jitter only
    biasColor = tuple(STY.clr.bias)
    usageColor = tuple(STY.clr.usage)
    barsAlpha = 0.65
    dotAlpha = 0.60
    dotSize = STY.mk.pt

    figs = []
    for g in GROUPS:
        B = metrics[g]
        U = usage_by_group[g]

        valid = np.any(~np.isnan(B), axis=1) | np.any(~np.isnan(U), axis=1)
        B = B[valid]
        U = U[valid]

        if B.size == 0 and U.size == 0:
            print(f"WARNING: {g.upper()}: no data to plot.")
            continue

        with np.errstate(invalid="ignore"):
            medBias = np.nanmedian(B, axis=0)
            q25Bias = np.nanpercentile(B, 25, axis=0)
            q75Bias = np.nanpercentile(B, 75, axis=0)
            medUse = np.nanmedian(U, axis=0)
            q25Use = np.nanpercentile(U, 25, axis=0)
            q75Use = np.nanpercentile(U, 75, axis=0)

        order = np.argsort(medBias, kind="stable")   # ascending
        B = B[:, order]
        U = U[:, order]
        medBias = medBias[order]
        q25Bias = q25Bias[order]
        q75Bias = q75Bias[order]
        medUse = medUse[order]
        q25Use = q25Use[order]
        q75Use = q75Use[order]

        xlabs = [f"s{int(s)+1}" for s in order]   # state index before sorting, 1-based

        fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor="w")
        fig.subplots_adjust(left=0.12, right=0.72, bottom=0.12, top=0.90)
        ax.set_axisbelow(False)

        xpos = np.arange(1, N_STATES + 1)
        ax.bar(xpos, medBias, width=0.8, color=biasColor, alpha=barsAlpha, edgecolor="none")
        ax.bar(xpos, -medUse, width=0.8, color=usageColor, alpha=barsAlpha, edgecolor="none")

        for si in range(N_STATES):
            s = si + 1
            ax.plot([s, s], [q25Bias[si], q75Bias[si]], "-", color=biasColor, lw=2.5)
            ax.plot([s, s], [-q25Use[si], -q75Use[si]], "-", color=usageColor, lw=2.5)

        for si in range(N_STATES):
            s = si + 1
            nB = B.shape[0]
            xj = s + 0.14 * (rng_plot.random(nB) - 0.5)
            ax.scatter(xj, B[:, si], s=dotSize, c=[biasColor], alpha=dotAlpha,
                       edgecolors=[biasColor], linewidths=0.3)
            nU = U.shape[0]
            xj2 = s + 0.14 * (rng_plot.random(nU) - 0.5)
            ax.scatter(xj2, -U[:, si], s=dotSize, c=[usageColor], alpha=dotAlpha,
                       edgecolors=[usageColor], linewidths=0.3)

        ax.set_xlim(0.5, N_STATES + 0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabs, rotation=0)
        ax.set_ylabel("Bias (up)  |  Usage (down)", fontname=STY.font.name, fontsize=STY.font.tick)
        ax.set_title(f"{g.upper()}",
                     fontname=STY.font.name, fontsize=STY.font.title)

        ax.axhline(0, color="k", lw=STY.lw.hair, linestyle="-")
        ax.axhline(0.5, color="k", lw=STY.lw.hair, linestyle=":")

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_handles = [
            Patch(facecolor=biasColor, alpha=barsAlpha, label="Median bias"),
            Patch(facecolor=usageColor, alpha=barsAlpha, label="Median usage"),
            Line2D([0], [0], color="k", lw=2.5, label="IQR"),
        ]
        ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=STY.font.tick)

        # ---- axis styling: house tick and line settings, then all four
        # spines restored so the panel keeps its full frame.
        gt_axes(ax, STY)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        for side in ("top", "right"):
            ax.spines[side].set_linewidth(STY.lw.axis)
            ax.spines[side].set_color(tuple(STY.clr.axis))
        ax.title.set_fontsize(STY.font.tick)
        for txt in fig.findobj(match=plt.Text):
            txt.set_fontsize(STY.font.tick)
            txt.set_fontname(STY.font.name)

        gt_figStyle(fig, STY)
        gt_saveFig(fig, f"edfig7_g_glmhmm_biasusage_{g}", out_dir)
        figs.append(fig)

        n_report = B.shape[0]
        print(f"{g.upper()}: n={n_report} animals, states sorted (orig idx) = {xlabs}, "
              f"medBias={np.round(medBias,3).tolist()}, medUse={np.round(medUse,3).tolist()}")

    return figs


if __name__ == "__main__":
    main()
