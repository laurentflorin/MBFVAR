# -*- coding: utf-8 -*-
"""
Chan, Poon & Zhu (2024) mixed-frequency estimator for MBFVAR.

This module provides an *isolated*, opt-in estimation path that is selected via
``method="chan_poon_zhu"`` on :meth:`MBFVAR.MixedFrequencyBVAR.fit`.  It does not
touch the default Schorfheide--Song (SS) estimator in :mod:`MBFVAR._estimation`.

Design
------
MBFVAR keeps its existing sequential bi-frequency chaining: blocks ``m = 0 ..
M-1`` are processed in order, and each block disaggregates the current lower
frequency into the next higher frequency, feeding its disaggregated series to
the next block.  For *each* block, instead of the SS balanced/unbalanced Kalman
filter + smoother, this path uses the CPZ machinery ported in
:mod:`MBFVAR._cpz_funcs`:

1. sample the block's latent high-frequency states from the stacked precision
   system with intertemporal-aggregation constraints
   (:func:`~MBFVAR._cpz_funcs.sample_latent_states`);
2. sample the block VAR coefficients under a Minnesota prior
   (:func:`~MBFVAR._cpz_funcs.construct_minnesota`), with a stationarity
   (explosive-VAR) rejection step reusing
   :func:`~MBFVAR.mfbvar_funcs.is_explosive`;
3. sample the block error precision via a Wishart draw;
4. sample a common stochastic-volatility (SV) path for the block
   (:func:`~MBFVAR._cpz_funcs.sample_CSV`) together with its persistence and
   innovation-variance hyperparameters.

The block error covariance used by the latent sampler is
``kron(diag(exp(-h)), invSig)``, i.e. common SV enters every block.

Crucially, ``fit_cpz`` stores its posterior draws and latent states in the
**same** attribute names/shapes the existing ``forecast`` / ``aggregate`` /
``_plots`` / ``_save`` consume, so those downstream methods keep working with no
change.  An extra ``self.h_list`` holds the per-block SV paths.

Reference: Chan, Poon & Zhu, "High-dimensional conditionally Gaussian state
space models with missing data", J. Econometrics 236 (2024) 105468 (MATLAB
``WeeklyVARcode``).
"""

import sys
import math
import copy
from collections import deque

import numpy as np
from scipy.stats import wishart
from scipy.linalg import cholesky, solve_triangular

from tqdm import tqdm

from .mfbvar_funcs import is_explosive
from ._cpz_funcs import (
    get_resid_var,
    construct_minnesota,
    sample_CSV,
    build_selection_matrices,
    build_intertemporal_constraint,
    sample_latent_states,
)


def _draw_beta(Y_reg, X_reg, invSig, h, invVbeta_diag, n, p,
               check_explosive, max_it_stable):
    """Draw VAR coefficients under the Minnesota prior (CPZ conjugate step).

    Ports the ``beta`` update of ``MFVAR.m``:

        Kbeta = kron(invSig, X' diag(exp(-h)) X) + invVbeta
        mu    = Kbeta \\ vec(X' diag(exp(-h)) Y invSig)
        beta  = mu + chol(Kbeta)^{-1} randn

    Returned ``beta`` uses MBFVAR's layout ``(n*p+1, n)`` with rows
    ``[lag1..lagp, const]`` so it is directly usable as ``Phi`` downstream.
    """
    D = np.exp(-h)
    XtD = X_reg.T * D                    # (k, Tnew)
    XtDX = XtD @ X_reg                   # (k, k)
    XtDY = XtD @ Y_reg                   # (k, n)
    k = X_reg.shape[1]

    Kbeta = np.kron(invSig, XtDX)
    Kbeta[np.diag_indices_from(Kbeta)] += invVbeta_diag
    rhs = (XtDY @ invSig).flatten(order="F")

    L = cholesky(Kbeta, lower=True)
    # mu = Kbeta^{-1} rhs
    tmp = solve_triangular(L, rhs, lower=True)
    mu = solve_triangular(L.T, tmp, lower=False)

    beta = None
    if check_explosive:
        for _ in range(max_it_stable):
            z = np.random.standard_normal(k * n)
            draw = mu + solve_triangular(L.T, z, lower=False)
            cand = draw.reshape(k, n, order="F")
            if not is_explosive(cand, n, p):
                beta = cand
                break
        if beta is None:
            # fall back to the posterior mean (guaranteed finite) if we cannot
            # find a stationary draw within the attempt budget
            beta = mu.reshape(k, n, order="F")
            return beta, True
        return beta, False
    else:
        z = np.random.standard_normal(k * n)
        draw = mu + solve_triangular(L.T, z, lower=False)
        beta = draw.reshape(k, n, order="F")
        return beta, False


def fit_cpz(self, mbfvar_data, hyp, var_of_interest=None, temp_agg="mean",
            max_it_stable=1000, return_mdd=False, check_explosive=True, **kwargs):
    """Estimate the MBFVAR model with the Chan--Poon--Zhu (CPZ) approach.

    This is the ``method="chan_poon_zhu"`` estimation path.  See the module
    docstring for the algorithm.  The public signature mirrors the default
    :func:`MBFVAR._estimation.fit` so the two methods are interchangeable
    through the dispatcher.

    Parameters
    ----------
    mbfvar_data : mbfvar_data
        Prepared multi-frequency data object.
    hyp : list of list
        Minnesota hyperparameters per frequency step.  For the CPZ path the
        first two entries of each block's list are mapped to the CPZ prior
        tightness ``(theta1, theta2)`` (``theta1`` is squared internally to
        match the reference convention ``theta1 = lambda**2``); the remaining
        reference defaults ``(theta3=100, theta4=2)`` are used.
    var_of_interest : list of str or None, optional
        Restrict which variables are propagated to higher-frequency blocks.
    temp_agg : {'mean', 'sum'}, optional
        Temporal aggregation used by the intertemporal constraints.
    max_it_stable : int, optional
        Maximum attempts to draw a non-explosive VAR per block/draw.
    return_mdd : bool, optional
        Kept for signature compatibility with the SS path.  The CPZ path does
        not provide an MDD (hyperparameter tuning is RMSE-based); when True a
        NaN is returned.
    check_explosive : bool, optional
        If True, reject explosive VAR draws.

    Returns
    -------
    float or None
        ``numpy.nan`` if ``return_mdd`` is True, otherwise None.
    """
    assert temp_agg in ("mean", "sum"), f"Invalid temp_agg: {temp_agg}. Choose 'mean' or 'sum'."

    self.method = "chan_poon_zhu"
    self.nex = 1
    self.hyp = hyp
    self.temp_agg = temp_agg

    if var_of_interest is not None:
        if isinstance(var_of_interest, str):
            var_of_interest = [var_of_interest]
        else:
            var_of_interest = list(var_of_interest)

    # ---- read data structures (mirrors the SS estimator) ----
    YMX_list = copy.deepcopy(mbfvar_data.YMX_list)
    YM0_list = copy.deepcopy(mbfvar_data.YM0_list)
    select_m_list = copy.deepcopy(mbfvar_data.select_m_list)
    YMh_list = copy.deepcopy(mbfvar_data.YMh_list)
    index_list = copy.deepcopy(mbfvar_data.index_list)
    frequencies = copy.deepcopy(mbfvar_data.frequencies)
    self.frequencies = frequencies
    YQX_list = copy.deepcopy(mbfvar_data.YQX_list)
    YQ0_list = copy.deepcopy(mbfvar_data.YQ0_list)
    select_q = copy.deepcopy(mbfvar_data.select_q)
    self.input_data_Q = copy.deepcopy(mbfvar_data.input_data_Q)
    varlist_list = copy.deepcopy(mbfvar_data.varlist_list)
    select_list = copy.deepcopy(mbfvar_data.select_list)
    Nm_list = copy.deepcopy(mbfvar_data.Nm_list)
    nv_list = copy.deepcopy(mbfvar_data.nv_list)
    Nq_list = copy.deepcopy(mbfvar_data.Nq_list)
    freq_ratio_list = copy.deepcopy(mbfvar_data.freq_ratio_list)
    YM_list = copy.deepcopy(mbfvar_data.YM_list)
    self.input_data = copy.deepcopy(mbfvar_data.input_data)

    varstxt_list = deque()

    M = len(YMX_list)
    nlags = list(self.nlags)

    for i in range(len(freq_ratio_list)):
        if nlags[i] < freq_ratio_list[i]:
            sys.exit("The number of lags at each step must be at least as long as the corresponding frequency ratio")

    nburn = round(self.nburn_perc * math.ceil(self.nsim / self.thining))
    self.nburn = nburn
    n_store = math.ceil(self.nsim / self.thining)

    # indices of variables of interest within the base low-frequency block
    idx_voi_q = None
    idx_voi_m = [None] * M
    if var_of_interest is not None:
        cols_q = YQX_list[0].columns.tolist()
        idx_voi_q = [x for x in range(len(cols_q)) if cols_q[x] in var_of_interest]
        for m in range(M):
            cols_m = YMX_list[m].columns.tolist()
            idx_voi_m[m] = [x for x in range(len(cols_m)) if cols_m[x] in var_of_interest]

    # ------------------------------------------------------------------
    # Static per-block bookkeeping (independent of the MCMC draw).
    # Lengths reproduce the SS estimator's chaining exactly so downstream
    # forecast/aggregate history alignment is preserved.
    # ------------------------------------------------------------------
    blocks = []
    T0_list = deque()
    Tstar_list = deque()
    T_list = deque()
    Tnew_list = deque()
    Tnobs_list = deque()
    nlags_list = deque()

    prev_len = None       # length of the disaggregated series feeding this block
    for m in range(M):
        p = int(nlags[m])
        r = int(freq_ratio_list[m])
        Nm = int(Nm_list[m])

        if m == 0:
            YM_block = np.asarray(YM_list[0], dtype=float)
            Nq = int(Nq_list[0])
            nQ = YQ0_list[0].shape[0]
        else:
            trim = 2 * int(np.prod(np.array(nlags[:m + 1])))
            YM_block = np.asarray(YM_list[m], dtype=float)[trim:, :]
            nQ = prev_len
            if var_of_interest is None:
                Nq = int(nv_list[m - 1])
            else:
                Nq = len(idx_voi_m[m - 1]) + len(idx_voi_q)
            Nq_list[m] = Nq
            nv_list[m] = Nm + Nq

        nv = Nm + Nq
        Tstar = YM_block.shape[0]
        T_bal = nQ * r
        # ragged HF tail beyond the last complete low-frequency window
        Tnew = Tstar - T_bal
        Tnobs = Tstar - p

        # this block's disaggregated output length (feeds the next block),
        # matching the SS estimator: YYact length = Tnobs - p
        out_len = Tnobs - p

        # pre-build the (draw-invariant) selection and constraint matrices
        M_o, M_u, obs_idx, unobs_idx = build_selection_matrices(nv, Nm, Tstar)
        nQ_fit = min(nQ, Tstar // r)   # keep windows inside the sample
        M_a, con_index = build_intertemporal_constraint(
            nv, Nm, Nq, Tstar, r, nQ_fit, unobs_idx, temp_agg)

        # trim the forecast-with-history HF array exactly like the SS path
        if m == 0:
            YMh_block = np.asarray(YMh_list[0], dtype=float)[p:-r, :] if YMh_list[0].size else YMh_list[0]
            varstxt_list.append(np.hstack((YMX_list[0].columns, YQX_list[0].columns)))
        else:
            trim = 2 * int(np.prod(np.array(nlags[:m + 1])))
            YMh_block = np.asarray(YMh_list[m], dtype=float)[trim + p:-r, :] if YMh_list[m].size else YMh_list[m]
            varstxt_list.append(np.hstack((YMX_list[m].columns, YQX_list[0].columns)))
        YMh_list[m] = YMh_block

        T0_list.append(p)
        nlags_list.append(p)
        Tstar_list.append(Tstar)
        T_list.append(T_bal)
        Tnew_list.append(Tnew)
        Tnobs_list.append(Tnobs)

        blocks.append(dict(
            m=m, p=p, r=r, Nm=Nm, Nq=Nq, nv=nv, Tstar=Tstar, nQ=nQ_fit,
            T_bal=T_bal, Tnew=Tnew, Tnobs=Tnobs, out_len=out_len,
            YM_block=YM_block, M_o=M_o, M_u=M_u, obs_idx=obs_idx,
            unobs_idx=unobs_idx, M_a=M_a, con_index=con_index,
        ))
        prev_len = out_len

    # ------------------------------------------------------------------
    # Persistent per-block sampler state.
    # ------------------------------------------------------------------
    for b in blocks:
        n, p, Tstar = b["nv"], b["p"], b["Tstar"]
        b["beta"] = np.random.standard_normal((n * p + 1, n)) / (n * 10.0)
        b["invSig"] = np.eye(n)
        b["h"] = np.zeros(Tstar - p)
        b["rho"] = 0.9
        b["sigh2"] = 0.1
        b["invVbeta"] = None   # built lazily on the first draw

    # posterior-draw / latent-state storage for the LAST block
    last = blocks[-1]
    nv_L, p_L, r_L, Nm_L, Nq_L = last["nv"], last["p"], last["r"], last["Nm"], last["Nq"]
    Tnobs_L = last["Tnobs"]
    Phip_list = [np.zeros((n_store, b["nv"] * b["p"] + 1, b["nv"])) for b in blocks]
    Sigmap_list = [np.zeros((n_store, b["nv"], b["nv"])) for b in blocks]
    h_list = [np.zeros((n_store, b["Tstar"] - b["p"])) for b in blocks]
    lstate_list = [np.zeros((n_store, Nq_L, Tnobs_L))]
    YYactsim_list = [np.full((n_store, r_L + 1, nv_L), np.nan)]
    XXactsim_list = [np.full((n_store, r_L + 1, nv_L * p_L + 1), np.nan)]
    valid_draws = []

    print(" ", end="\n")
    print("Multi Frequency BVAR: Estimation (Chan-Poon-Zhu)", end="\n")
    print("Frequencies: ", self.frequencies, end="\n")
    print("Total Number of Draws: ", self.nsim)

    theta_defaults = (100.0, 2.0)   # theta3, theta4 from the reference

    for j in tqdm(range(self.nsim)):
        store = (j % self.thining == 0)
        j_temp = int(j / self.thining)
        prev_Ynew = None

        for bi, b in enumerate(blocks):
            n, p, r, Nm, Nq, nv = b["nv"], b["p"], b["r"], b["Nm"], b["Nq"], b["nv"]
            Tstar, nQ = b["Tstar"], b["nQ"]

            # --- assemble the low-frequency observations feeding this block ---
            if bi == 0:
                lf_obs = np.asarray(YQ0_list[0], dtype=float)
            else:
                lf_obs = prev_Ynew
            # build the constraint target vector Y_con in con_index order
            Y_con = np.empty(len(b["con_index"]))
            for c, (g, q) in enumerate(b["con_index"]):
                Y_con[c] = lf_obs[g, q]

            hyp_b = self.hyp[bi]
            theta = (float(hyp_b[0]) ** 2, float(hyp_b[1]), theta_defaults[0], theta_defaults[1])

            # --- 1. sample latent high-frequency states (CPZ) ---
            Y_new = sample_latent_states(
                b["YM_block"], b["beta"], b["invSig"], b["h"],
                nv, Nm, Nq, p, r, nQ, Y_con,
                b["M_o"], b["M_u"], b["M_a"], b["obs_idx"], temp_agg)

            # --- build regression matrices from the latent series ---
            Y_reg = Y_new[p:, :]                     # (Tnew_reg, n)
            Tnew_reg = Y_reg.shape[0]
            X_reg = np.ones((Tnew_reg, n * p + 1))
            for l in range(1, p + 1):
                X_reg[:, (l - 1) * n:l * n] = Y_new[p - l:Tstar - l, :]

            if b["invVbeta"] is None:
                AR_s2 = get_resid_var(Y_new)
                AR_s2 = np.where(AR_s2 <= 0, 1e-8, AR_s2)
                b["invVbeta"] = construct_minnesota(AR_s2, nv, p, theta)

            # --- 2. draw VAR coefficients (Minnesota, with stability check) ---
            beta, forced = _draw_beta(
                Y_reg, X_reg, b["invSig"], b["h"], b["invVbeta"],
                nv, p, check_explosive, max_it_stable)
            b["beta"] = beta

            # --- 3. draw error precision (Wishart) ---
            err = Y_reg - X_reg @ beta
            D = np.exp(-b["h"])
            S = 100.0 * np.eye(nv) + err.T @ (err * D[:, None])
            scale = np.linalg.inv(S)
            scale = 0.5 * (scale + scale.T)
            invSig = wishart.rvs(df=Tnew_reg + nv + 3, scale=scale)
            invSig = np.atleast_2d(invSig)
            b["invSig"] = invSig

            # --- 4. common stochastic volatility ---
            R = cholesky(invSig, lower=False)        # upper: R'R = invSig
            s2 = np.sum((err @ R) ** 2, axis=1)
            h_new, _ = sample_CSV(s2, b["rho"], b["sigh2"], b["h"], nv, True)
            b["h"] = h_new
            eh = h_new[1:] - b["rho"] * h_new[:-1]
            b["sigh2"] = 1.0 / np.random.gamma(10 + Tnew_reg / 2.0,
                                               1.0 / (0.004 + np.sum(eh ** 2) / 2.0))
            hlag = h_new[:-1]
            K_rho = hlag @ hlag / b["sigh2"] + 100.0
            b["rho"] = (hlag @ h_new[1:] / b["sigh2"]) / K_rho + np.random.randn() / math.sqrt(K_rho)

            # base error covariance scaled by the latest volatility level
            Sigma = np.linalg.inv(invSig) * math.exp(h_new[-1])
            Sigma = 0.5 * (Sigma + Sigma.T)

            if store:
                Phip_list[bi][j_temp, :, :] = beta
                Sigmap_list[bi][j_temp, :, :] = Sigma
                h_list[bi][j_temp, :] = h_new
                if bi == M - 1:
                    valid_draws.append(j_temp)
                    # latent LF states over the estimation sample
                    lstate_list[0][j_temp, :, :] = Y_new[p:, Nm:].T
                    # tail of the actual/regressor data for forward simulation
                    tail = r + 1
                    Y_tail = Y_new[Tstar - tail:Tstar, :]
                    YYactsim_list[0][j_temp, :, :] = Y_tail
                    for kk in range(tail):
                        t = Tstar - tail + kk
                        row = np.ones(n * p + 1)
                        for l in range(1, p + 1):
                            row[(l - 1) * n:l * n] = Y_new[t - l, :]
                        XXactsim_list[0][j_temp, kk, :] = row

            # --- feed the disaggregated series to the next block ---
            if bi < M - 1:
                out = Y_new[2 * p:, :]                # length out_len
                if var_of_interest is not None:
                    idx_vars = np.concatenate((
                        np.array(idx_voi_m[bi], dtype=int),
                        Nm + np.array(idx_voi_q, dtype=int)))
                    out = out[:, idx_vars]
                prev_Ynew = out

    # ------------------------------------------------------------------
    # store results under the SAME attribute names the SS path uses so that
    # forecast / aggregate / plots / save keep working unchanged.
    # ------------------------------------------------------------------
    if var_of_interest is not None:
        # mirror the SS estimator: restrict varlist/select to the variables of
        # interest so the last block's forecast/aggregate widths stay consistent
        for m in range(M):
            idx = [x for x in range(len(varlist_list[m]))
                   if varlist_list[m][x] in (YMX_list[m].columns.tolist() + var_of_interest)]
            varlist_list[m] = varlist_list[m][idx]
            select_list[m] = select_list[m][idx]
        for m in range(1, M):
            combo = YMX_list[m - 1].columns.tolist() + YQX_list[0].columns.tolist()
            idx_q = [x for x in range(len(combo)) if combo[x] in var_of_interest]
            select_q[m] = select_q[m][idx_q]

    self.YMh_list = YMh_list
    self.T0_list = T0_list
    self.freq_ratio_list = freq_ratio_list
    self.varstxt_list = varstxt_list
    self.Nm_list = Nm_list
    self.Nq_list = Nq_list
    self.nv_list = nv_list
    self.YYactsim_list = YYactsim_list
    self.XXactsim_list = XXactsim_list
    self.Phip_list = Phip_list
    self.Sigmap_list = Sigmap_list
    self.h_list = h_list
    self.select_list = select_list
    self.Tnew_list = Tnew_list
    self.Tnobs_list = Tnobs_list
    self.Tstar_list = Tstar_list
    self.T_list = T_list
    self.select_m_list = select_m_list
    self.select_q = select_q
    self.lstate_list = lstate_list
    self.nlags_list = nlags_list
    self.varlist_list = varlist_list
    self.YMX_list = YMX_list
    self.index_list = index_list
    self.var_of_interest = var_of_interest
    self.valid_draws = [d for d in valid_draws if d >= self.nburn / self.thining]

    # convenience scalars for the last block (mirrors the SS estimator)
    self.Nm = Nm_list[-1]
    self.nv = nv_list[-1]
    self.freq_ratio = freq_ratio_list[-1]
    self.select = select_list[-1]
    self.select_m = select_m_list[-1]
    self.varlist = varlist_list[-1]

    if return_mdd:
        return np.nan
    return None


def forecast_cpz(self, H, conditionals=None, **kwargs):
    """Forecast for the Chan--Poon--Zhu path.

    Because :func:`fit_cpz` populates exactly the same draw/state attributes as
    the default estimator, the forward simulation is identical to the SS path.
    This thin wrapper simply delegates to the shared forecasting implementation
    so behaviour and output shapes stay consistent across methods.
    """
    from ._estimation import _forecast_impl
    return _forecast_impl(self, H, conditionals=conditionals)
