# -*- coding: utf-8 -*-
"""
Helper routines for the Chan, Poon & Zhu (2024) mixed-frequency approach.

This module contains faithful Python/NumPy/SciPy ports of the MATLAB reference
routines used by the ``chan_poon_zhu`` estimation path (``_estimation_cpz.py``).
The reference implementation is the ``WeeklyVARcode`` provided by the maintainer
(Chan, Poon & Zhu, "High-dimensional conditionally Gaussian state space models
with missing data", Journal of Econometrics 236 (2024) 105468).

The following MATLAB routines are ported here:

* ``vec.m``                  -> :func:`vec`
* ``get_resid_var.m``        -> :func:`get_resid_var`
* ``construct_minnesota.m``  -> :func:`construct_minnesota`
* ``sample_CSV.m``           -> :func:`sample_CSV`
* ``Sample_latent_Y_approx.m`` (per-block adaptation) -> :func:`sample_latent_states`

In addition, the per-block selection-matrix / intertemporal-constraint builders
(:func:`build_selection_matrices` and :func:`build_intertemporal_constraint`)
adapt the CPZ missing-data machinery to a single bi-frequency block of MBFVAR's
sequential scheme.  They are parameterised by the block frequency ratio ``r`` so
they work for an arbitrary transition (Q->M, M->W, ...), never hardcoding a
specific ratio.

Unlike the original MATLAB code -- which stacks *all* frequencies into one large
conditionally-Gaussian system -- MBFVAR keeps its existing sequential
bi-frequency chaining and applies these CPZ building blocks to *each* block in
turn.  See :mod:`MBFVAR._estimation_cpz` for the driver.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky, solve_triangular


# Large precision weight used to enforce the intertemporal-aggregation
# equality constraints as a hard (near-exact) restriction in the latent-state
# Gaussian system (mirrors the diffuse-observation trick in the reference code).
CONSTRAINT_PRECISION = 1e10


def vec(A):
    """Stack the columns of a matrix into a single column vector.

    Port of the MATLAB ``vec.m`` helper (column-major / Fortran order flatten).

    Parameters
    ----------
    A : numpy.ndarray
        Two-dimensional array.

    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the columns of ``A`` stacked on top of
        one another.
    """
    return np.asarray(A).flatten(order="F")


def get_resid_var(tmpY, nlags=4):
    """Estimate univariate AR(``nlags``) residual variances.

    Port of ``get_resid_var.m``.  For every column of ``tmpY`` an AR(``nlags``)
    model with intercept is estimated by OLS and the residual variance is
    returned.  Rows containing NaNs are dropped first (equivalent to MATLAB's
    ``rmmissing``).

    Parameters
    ----------
    tmpY : numpy.ndarray
        Observations, shape ``(T, n)``.
    nlags : int, optional
        AR order (default 4, matching the reference implementation).

    Returns
    -------
    numpy.ndarray
        Residual variances, shape ``(n,)``.
    """
    tmpY = np.asarray(tmpY, dtype=float)
    # rmmissing: drop rows with any NaN
    tmpY = tmpY[~np.isnan(tmpY).any(axis=1), :]
    T, n = tmpY.shape
    sig2 = np.zeros(n)
    for i in range(n):
        # Z = [const, y_{t-1}, ..., y_{t-nlags}] over t = nlags..T-1
        cols = [np.ones(T - nlags)]
        for l in range(1, nlags + 1):
            cols.append(tmpY[nlags - l:T - l, i])
        Z = np.column_stack(cols)
        y = tmpY[nlags:, i]
        # OLS coefficients
        tmpb, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ tmpb
        sig2[i] = np.mean(resid ** 2)
    return sig2


def construct_minnesota(AR_s2, n, lag, theta=(0.2 ** 2, 0.5 ** 2, 100.0, 2.0),
                        variance_floor=1e-12):
    """Build the Minnesota prior precision diagonal ``invVbeta``.

    Port of ``construct_minnesota.m`` adapted to MBFVAR's coefficient layout.

    In the MATLAB reference the coefficient vector orders the intercept *first*
    for each equation.  MBFVAR instead stores VAR coefficients with the lag
    blocks first and the intercept as the *last* row (see ``Phi`` in
    :mod:`MBFVAR._estimation`).  The Minnesota formulas below define prior
    *variances*.  The CPZ beta update adds this object to the posterior
    precision matrix, so this function returns floored and inverted prior
    variances ordered to match ``vec(Phi)`` where ``Phi`` has shape
    ``(n * lag + 1, n)`` with rows ``[lag1 vars, lag2 vars, ..., lagp vars,
    const]`` and columns indexing the ``n`` equations (column-major /
    order='F').

    Parameters
    ----------
    AR_s2 : numpy.ndarray
        AR residual variances, shape ``(n,)`` (from :func:`get_resid_var`).
    n : int
        Number of variables in the block.
    lag : int
        Number of lags.
    theta : sequence of float, optional
        ``(theta1, theta2, theta3, theta4)`` prior tightness parameters.  The
        defaults ``(0.2**2, 0.5**2, 100, 2)`` match the reference.
    variance_floor : float, optional
        Small positive floor applied before inversion.

    Returns
    -------
    numpy.ndarray
        Diagonal of ``invVbeta``, shape ``(n * (n * lag + 1),)``.
    """
    AR_s2 = np.asarray(AR_s2, dtype=float).ravel()
    theta1, theta2, theta3, theta4 = theta
    k = n * lag + 1  # number of regressors per equation (incl. constant)
    var_diag = np.zeros(n * k)
    for i in range(n):            # equation i
        base = i * k
        for l in range(1, lag + 1):
            for j in range(n):    # regressor variable j
                pos = base + (l - 1) * n + j
                if i == j:
                    var_diag[pos] = theta1 / (l ** theta4)
                else:
                    var_diag[pos] = (AR_s2[i] / AR_s2[j]) * theta1 * theta2 / (l ** theta4)
        # constant term stored last for this equation
        var_diag[base + n * lag] = AR_s2[i] * theta3
    var_diag = np.maximum(var_diag, variance_floor)
    return 1.0 / var_diag


def sample_CSV(s2, rho, sigh2, h, n, is_forced_accept=True, tol=1e-3):
    """Sample the common stochastic-volatility log-variance path ``h``.

    Faithful port of ``sample_CSV.m`` -- an accept/reject Metropolis-Hastings
    step built around a Gaussian approximation obtained by Newton iteration.

    Parameters
    ----------
    s2 : numpy.ndarray
        Per-period sum of squared standardised residuals, shape ``(T,)`` where
        ``T`` is the effective sample (``Tnew`` in the reference).
    rho : float
        AR(1) persistence of the log-volatility.
    sigh2 : float
        Innovation variance of the log-volatility.
    h : numpy.ndarray
        Current log-volatility path, shape ``(T,)``.
    n : int
        Cross-sectional dimension (number of variables in the block).
    is_forced_accept : bool, optional
        If True the proposal is always accepted (mirrors the ``is_ForcedAccept``
        argument used by the reference driver).
    tol : float, optional
        Convergence tolerance for the Newton iteration.

    Returns
    -------
    h : numpy.ndarray
        Updated log-volatility path.
    is_accept : int
        1 if the MH proposal was accepted, 0 otherwise.
    """
    s2 = np.asarray(s2, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel().copy()
    T = s2.shape[0]
    is_accept = 0

    # Hrho = I - rho * subdiagonal
    Hrho = sp.eye(T, format="csc") - rho * sp.diags(
        np.ones(T - 1), -1, shape=(T, T), format="csc"
    )
    dvec = np.concatenate(([(1 - rho ** 2) / sigh2], np.ones(T - 1) / sigh2))
    HiSH = Hrho.T @ sp.diags(dvec) @ Hrho
    HiSH = HiSH.tocsc()

    # Newton iteration to find the mode ht of the conditional posterior
    errh = 1.0
    ht = h.copy()
    while errh > tol:
        eht = np.exp(ht)
        sieht = s2 / eht
        fh = -n / 2.0 + 0.5 * sieht
        Gh = 0.5 * sieht
        Kh = (HiSH + sp.diags(Gh)).tocsc()
        newht = spsolve(Kh, fh + Gh * ht)
        errh = np.max(np.abs(newht - ht))
        ht = newht

    Kh_dense = Kh.toarray()
    # Lower Cholesky factor of Kh
    CKh = cholesky(Kh_dense, lower=True)

    hstar = ht
    logc = (-0.5 * hstar @ (HiSH @ hstar) - n / 2.0 * np.sum(hstar)
            - 0.5 * np.exp(-hstar) @ s2 + np.log(3.0))

    # Accept/reject draw of the candidate
    flag = 0
    hc = ht.copy()
    while flag == 0:
        # hc = ht + CKh'\randn  ==  solve(CKh^T, randn)
        hc = ht + solve_triangular(CKh.T, np.random.standard_normal(T), lower=False)
        alpARc = (-0.5 * hc @ (HiSH @ hc) - n / 2.0 * np.sum(hc)
                  - 0.5 * np.exp(-hc) @ s2
                  + 0.5 * (hc - ht) @ (Kh_dense @ (hc - ht)) - logc)
        if alpARc > np.log(np.random.rand()):
            flag = 1

    alpAR = (-0.5 * h @ (HiSH @ h) - n / 2.0 * np.sum(h)
             - 0.5 * np.exp(-h) @ s2
             + 0.5 * (h - ht) @ (Kh_dense @ (h - ht)) - logc)

    if alpAR < 0:
        alpMH = 1.0
    elif alpARc < 0:
        alpMH = -alpAR
    else:
        alpMH = alpARc - alpAR

    if alpMH > np.log(np.random.rand()) or is_forced_accept:
        h = hc
        is_accept = 1

    return h, is_accept


def build_selection_matrices(n, Nm, T, Y_obs=None):
    """Build observed/unobserved selection matrices for one bi-frequency block.

    In a bi-frequency block the first ``Nm`` variables are the block's own
    high-frequency (HF) variables.  Finite HF entries are treated as observed;
    missing HF entries, including ragged-edge NaNs, are treated as latent.  The
    remaining ``Nq = n - Nm`` variables are low-frequency (LF) and are always
    latent at the HF sampling rate (they only enter through intertemporal
    aggregation constraints).  The full stacked latent vector is ordered
    time-slow / variable-fast: ``[Y_0; Y_1; ...; Y_{T-1}]`` with each ``Y_t`` of
    length ``n``.

    Parameters
    ----------
    n : int
        Total number of variables in the block (``Nm + Nq``).
    Nm : int
        Number of HF (observed) variables.
    T : int
        Number of HF periods.
    Y_obs : numpy.ndarray or None, optional
        HF observation matrix, shape ``(T, Nm)``.  If omitted, all HF entries
        are treated as observed for backward compatibility.

    Returns
    -------
    M_o : scipy.sparse.csc_matrix
        ``(n*T, No)`` selection matrix mapping observed values into the full
        stacked vector.
    M_u : scipy.sparse.csc_matrix
        ``(n*T, Nu)`` selection matrix mapping unobserved (latent LF) values
        into the full stacked vector.
    obs_idx : numpy.ndarray
        Indices (into the ``n*T`` stacked vector) of the observed entries.
    unobs_idx : numpy.ndarray
        Indices of the unobserved (latent LF) entries.
    """
    obs_mask_2d = np.zeros((T, n), dtype=bool)
    if Y_obs is None:
        obs_mask_2d[:, :Nm] = True
    else:
        Y_obs = np.asarray(Y_obs, dtype=float)
        if Y_obs.shape[0] != T or Y_obs.shape[1] < Nm:
            raise ValueError("Y_obs must have shape (T, Nm) or wider.")
        obs_mask_2d[:, :Nm] = np.isfinite(Y_obs[:, :Nm])

    # ind flattened time-slow/var-fast: entry (t, var) -> t*n + var
    obs_mask = obs_mask_2d.reshape(-1)
    full_idx = np.arange(n * T)
    obs_idx = full_idx[obs_mask]
    unobs_idx = full_idx[~obs_mask]

    No = obs_idx.size
    Nu = unobs_idx.size
    M_o = sp.csc_matrix((np.ones(No), (obs_idx, np.arange(No))), shape=(n * T, No))
    M_u = sp.csc_matrix((np.ones(Nu), (unobs_idx, np.arange(Nu))), shape=(n * T, Nu))
    return M_o, M_u, obs_idx, unobs_idx


def build_intertemporal_constraint(n, Nm, Nq, T, r, nQ, unobs_idx, temp_agg="mean"):
    """Build the intertemporal aggregation constraint matrix ``M_a``.

    For each low-frequency observation the (weighted) average of the ``r``
    latent high-frequency states within its aggregation window must equal the
    observed low-frequency value.  This mirrors MBFVAR's own measurement
    equation (simple mean over ``r`` periods for ``temp_agg='mean'`` or a plain
    sum for ``temp_agg='sum'``) so the CPZ path stays consistent with the rest
    of the package's aggregation logic.  The builder is parameterised by the
    block frequency ratio ``r`` and never hardcodes a specific transition.

    Window ``g`` (``g = 0 .. nQ-1``) covers HF periods ``[g*r, g*r + r - 1]``.

    Parameters
    ----------
    n, Nm, Nq, T, r, nQ : int
        Block dimensions: total vars, HF vars, LF vars, HF periods, frequency
        ratio and number of LF observations.
    unobs_idx : numpy.ndarray
        Indices of the unobserved entries in the full stacked vector (from
        :func:`build_selection_matrices`).
    temp_agg : {'mean', 'sum'}, optional
        Aggregation used by the measurement equation.

    Returns
    -------
    M_a : scipy.sparse.csc_matrix
        ``(Nu, Tq)`` constraint matrix, ``Nu = Nq*T`` and ``Tq = nQ*Nq``.
    con_index : list of tuple
        For every constraint column, the ``(window, lf_var)`` pair it encodes,
        so the driver can fill in the matching target values ``Y_con``.
    """
    weight = 1.0 / r if temp_agg == "mean" else 1.0
    Nu = unobs_idx.size
    # Map full-vector index -> compact unobserved index
    pos_of = {int(idx): u for u, idx in enumerate(unobs_idx)}

    rows = []
    cols = []
    vals = []
    con_index = []
    col = 0
    for g in range(nQ):
        for q in range(Nq):          # LF variable q (global var index Nm + q)
            var = Nm + q
            for t in range(g * r, g * r + r):
                if t >= T:
                    continue
                full = t * n + var
                rows.append(pos_of[full])
                cols.append(col)
                vals.append(weight)
            con_index.append((g, q))
            col += 1

    Tq = col
    M_a = sp.csc_matrix((vals, (rows, cols)), shape=(Nu, Tq))
    return M_a, con_index


def _build_companion_selector(T, lag):
    """Return the (T-lag) x T sparse selectors A_l for l = 0 .. lag.

    ``A_0`` selects ``Y_t`` for ``t = lag .. T-1``; ``A_l`` selects
    ``Y_{t-l}``.  Used to assemble the stacked VAR residual operator ``C``.
    """
    selectors = []
    rows = np.arange(T - lag)
    for l in range(lag + 1):
        cols = np.arange(lag - l, T - l)
        A = sp.csc_matrix((np.ones(T - lag), (rows, cols)), shape=(T - lag, T))
        selectors.append(A)
    return selectors


def _observed_vector(Y_obs, n, Nm, T, obs_idx):
    """Stack finite observed HF entries in the order selected by ``obs_idx``."""
    full_vec = np.zeros(n * T)
    for var in range(Nm):
        full_vec[var::n] = Y_obs[:, var]
    vecY = full_vec[obs_idx]
    if not np.isfinite(vecY).all():
        raise ValueError("Observed CPZ entries contain non-finite values.")
    return vecY


def _latent_precision_terms(Y_obs, beta, invSig, h, n, Nm, lag,
                            M_o, M_u, obs_idx, init_ridge):
    """Build the latent-state precision and linear term shared by CPZ steps."""
    T = Y_obs.shape[0]
    Tnew = T - lag

    vecY = _observed_vector(Y_obs, n, Nm, T, obs_idx)

    A_l = beta[:n * lag, :]
    const = beta[n * lag, :]
    selectors = _build_companion_selector(T, lag)
    I_n = sp.eye(n, format="csc")
    C = sp.kron(selectors[0], I_n, format="csc")
    for l in range(1, lag + 1):
        A_lag = A_l[(l - 1) * n:l * n, :].T
        C = C - sp.kron(selectors[l], sp.csc_matrix(A_lag), format="csc")

    Cu = (C @ M_u).tocsc()
    invSig_big = sp.kron(sp.diags(np.exp(-h)), sp.csc_matrix(invSig), format="csc")
    bigK = Cu.T @ invSig_big
    K = (bigK @ Cu).tolil()

    ridge_dim = min((n - Nm) * lag, K.shape[0])
    for d in range(ridge_dim):
        K[d, d] += init_ridge
    K = K.tocsc()

    const_stack = np.tile(const, Tnew)
    Kmu = bigK @ (const_stack - (C @ (M_o @ vecY)))
    return K, Kmu, vecY


def latent_constraint_loglik(Y_obs, beta, invSig, h, n, Nm, Nq, lag, r, nQ,
                             Y_con, M_o, M_u, M_a, obs_idx,
                             init_ridge=100.0):
    """Return the CPZ downstream log-likelihood kernel for aggregation targets.

    This integrates the block's latent unobserved entries out of the same
    Gaussian precision system used by :func:`sample_latent_states`.  Terms that
    are constant in ``Y_con`` are omitted, which is sufficient for adjacent-block
    MH ratios where the downstream block parameters and HF observations are
    fixed and only the upstream aggregation targets change.
    """
    K, Kmu, _ = _latent_precision_terms(
        Y_obs, beta, invSig, h, n, Nm, lag, M_o, M_u, obs_idx, init_ridge)

    Y_con = np.asarray(Y_con, dtype=float).ravel()
    if not np.isfinite(Y_con).all():
        return -np.inf

    Tq = M_a.shape[1]
    iW = sp.diags(CONSTRAINT_PRECISION * np.ones(Tq))
    Knew = (K + M_a @ iW @ M_a.T).tocsc()
    rhs = Kmu + M_a @ (iW @ Y_con)
    sol = spsolve(Knew, rhs)
    return float(0.5 * rhs @ sol - 0.5 * CONSTRAINT_PRECISION * (Y_con @ Y_con))


def sample_latent_states(Y_obs, beta, invSig, h, n, Nm, Nq, lag, r, nQ,
                         Y_con, M_o, M_u, M_a, obs_idx, temp_agg="mean",
                         init_ridge=100.0):
    """Draw the block's latent high-frequency states (CPZ approximation).

    Port of ``Sample_latent_Y_approx.m`` adapted to a single bi-frequency
    block.  Given the current VAR coefficients ``beta`` (MBFVAR layout), the
    error precision ``invSig`` and the common-SV path ``h``, this samples the
    full ``T x n`` matrix of latent HF states subject to the intertemporal
    aggregation constraints in ``M_a`` / ``Y_con``.

    Parameters
    ----------
    Y_obs : numpy.ndarray
        HF data for the block, shape ``(T, Nm)``.  Finite entries selected by
        ``obs_idx`` are conditioned on; missing entries are sampled as latent.
    beta : numpy.ndarray
        VAR coefficients, shape ``(n*lag+1, n)`` (rows ``[lag1..lagp, const]``).
    invSig : numpy.ndarray
        Error precision matrix, shape ``(n, n)``.
    h : numpy.ndarray
        Log-volatility path, shape ``(T-lag,)``.
    n, Nm, Nq, lag, r, nQ : int
        Block dimensions.
    Y_con : numpy.ndarray
        Constraint target values, shape ``(Tq,)`` ordered as ``con_index``.
    M_o, M_u, M_a : scipy.sparse matrices
        Selection / constraint matrices from the builders above.
    obs_idx : numpy.ndarray
        Observed indices in the full stacked vector.
    temp_agg : {'mean', 'sum'}, optional
        Aggregation kind (only used for documentation / consistency).
    init_ridge : float, optional
        Ridge added to the first few latent states as a diffuse initial-state
        prior (mirrors the ``K(1:4*lag,...) += 100`` regularisation of the
        reference).

    Returns
    -------
    Y_new : numpy.ndarray
        Sampled latent HF states, shape ``(T, n)``.
    """
    T = Y_obs.shape[0]

    # --- build stacked observed vector vecY (length No) ---
    vecY = _observed_vector(Y_obs, n, Nm, T, obs_idx)

    K, Kmu, _ = _latent_precision_terms(
        Y_obs, beta, invSig, h, n, Nm, lag, M_o, M_u, obs_idx, init_ridge)

    Tq = M_a.shape[1]
    iW = sp.diags(CONSTRAINT_PRECISION * np.ones(Tq))
    Knew = (K + M_a @ iW @ M_a.T).tocsc()
    rhs = Kmu + M_a @ (iW @ Y_con)
    munew = spsolve(Knew, rhs)

    # draw ~ N(munew, Knew^{-1}) via lower Cholesky
    Knew_dense = Knew.toarray()
    CK = cholesky(Knew_dense, lower=True)
    z = np.random.standard_normal(Knew_dense.shape[0])
    # chol(Knew,'lower')' \ randn  ==  solve(CK^T, z)
    draw = solve_triangular(CK.T, z, lower=False)
    vecY_u = munew + draw

    full_new = M_o @ vecY + M_u @ vecY_u
    Y_new = full_new.reshape(T, n)   # time-slow/var-fast -> (T, n)
    return Y_new
