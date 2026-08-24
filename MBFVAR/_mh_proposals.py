# -*- coding: utf-8 -*-
"""
Reversible proposal kernels for the Metropolis-within-Gibbs (MwG)
cross-block correction.

Why this module exists
----------------------
The MwG correction accepts a block proposal with probability
``min(1, exp(ll_prop - ll_cur))`` -- the ratio of the *downstream* block's
likelihood under the proposed vs. current interface data.  That acceptance
ratio is exactly valid only when the proposal kernel Q is REVERSIBLE with
respect to the block's local posterior.  The legacy proposal (the
``mh_proposal="systematic"`` path in ``_estimation.py``) is a
systematic-scan Gibbs sweep -- states, then parameters, in fixed order --
which is invariant but NOT reversible, so the cancellation that reduces
the acceptance ratio to the downstream likelihood ratio does not go
through for it.  See ``docs/proposal_kernel.md`` in the paper repository
for the full argument and code references.

This module provides the fix for the Schorfheide-Song path: a
PALINDROMIC proposal

    K_s  K_theta  K_s

(draw states given current parameters, draw parameters given those
states, draw states again given the new parameters, and propose the final
(states, parameters) pair).  A palindromic composition of reversible
component kernels is self-adjoint w.r.t. the local posterior, hence
reversible, so the downstream-ratio acceptance is exact for it.

Each ``K_s`` here re-runs the balanced and unbalanced Kalman filters
UNDER THE PARAMETERS IT CONDITIONS ON before simulation smoothing.  This
also removes a second defect of the legacy proposal, which recycled
filtered moments computed under the previous iteration's parameters
(``docs/proposal_kernel.md``, section 3.1).

The functions transcribe the forward-pass filter/smoother/parameter-draw
code of ``_estimation.py`` without behavioural changes; the legacy
systematic path in ``_estimation.py`` is left untouched so that
``mh_proposal="systematic"`` reproduces the historical sampler
bit-for-bit.
"""

import numpy as np
from scipy.stats import invwishart

from .cholcov.cholcov_module import cholcovOrEigendecomp
from .inverse.matrix_inversion import invert_matrix
from .mfbvar_funcs import calc_yyact, is_explosive


def build_block_matrices(Phi, sigma, Nm, Nq, p, freq_ratio, temp_agg):
    """Build all transition/measurement matrices of one bi-frequency block
    from its VAR parameters ``(Phi, sigma)``.

    Exact transcription of the forward-pass construction in
    ``_estimation.py`` (the ``GAMMA*``/``LAMBDA*`` updates after the
    parameter draw), so that a state draw under these matrices is the
    exact full conditional under ``(Phi, sigma)``.
    """
    phi_qm = np.zeros((Nm * p, Nq))
    phi_qq = np.zeros((Nq * p, Nq))
    phi_mm = np.zeros((Nm * p, Nm))
    phi_mq = np.zeros((Nq * p, Nm))
    for i in range(p):
        phi_qm[Nm*i:Nm*(i+1), :] = Phi[i*(Nm+Nq):i*(Nm+Nq)+Nm, Nm:]
        phi_qq[Nq*i:Nq*(i+1), :] = Phi[i*(Nm+Nq)+Nm:(i+1)*(Nm+Nq), Nm:]
        phi_mm[Nm*i:Nm*(i+1), :] = Phi[i*(Nm+Nq):i*(Nm+Nq)+Nm, :Nm]
        phi_mq[Nq*i:Nq*(i+1), :] = Phi[i*(Nm+Nq)+Nm:(i+1)*(Nm+Nq), :Nm]
    phi_qc = Phi[-1, Nm:, np.newaxis]
    phi_mc = Phi[-1, :Nm, np.newaxis]

    if Nm:
        sig_mm = sigma[:Nm, :Nm]
        sig_mq = 0.5 * (sigma[:Nm, Nm:] + np.transpose(sigma[Nm:, :Nm]))
        sig_qm = 0.5 * (sigma[Nm:, :Nm] + np.transpose(sigma[:Nm, Nm:]))
        sig_qq = sigma[Nm:, Nm:]
    else:
        sig_mm = np.zeros((0, 0))
        sig_mq = np.zeros((0, Nq))
        sig_qm = np.zeros((Nq, 0))
        sig_qq = np.atleast_2d(sigma)

    GAMMAs = np.vstack((
        np.hstack((np.transpose(phi_qq), np.zeros((Nq, Nq)))),
        np.hstack((np.eye(p*Nq), np.zeros((p*Nq, Nq)))),
    ))
    GAMMAz = np.vstack((np.transpose(phi_qm), np.zeros((p*Nq, p*Nm))))
    GAMMAc = np.vstack((phi_qc, np.zeros((p*Nq, 1))))
    GAMMAu = np.vstack((np.eye(Nq), np.zeros((p*Nq, Nq))))

    if temp_agg == "mean":
        LAMBDAs = np.vstack((
            np.hstack((np.zeros((Nm, Nq)), np.transpose(phi_mq))),
            1.0/freq_ratio * np.hstack((
                np.tile(np.eye(Nq), freq_ratio),
                np.zeros((Nq, Nq*(p-(freq_ratio-1)))),
            )),
        ))
    else:  # "sum"
        LAMBDAs = np.vstack((
            np.hstack((np.zeros((Nm, Nq)), np.transpose(phi_mq))),
            np.hstack((
                np.tile(np.eye(Nq), freq_ratio),
                np.zeros((Nq, Nq*(p-(freq_ratio-1)))),
            )),
        ))
    LAMBDAz = np.vstack((np.transpose(phi_mm), np.zeros((Nq, p*Nm))))
    LAMBDAc = np.vstack((phi_mc, np.zeros((Nq, 1))))
    LAMBDAu = np.vstack((np.eye(Nm), np.zeros((Nq, Nm))))

    W = np.hstack((np.eye(Nm), np.zeros((Nm, Nq))))

    return dict(
        GAMMAs=GAMMAs, GAMMAz=GAMMAz, GAMMAc=GAMMAc, GAMMAu=GAMMAu,
        LAMBDAs=LAMBDAs, LAMBDAz=LAMBDAz, LAMBDAc=LAMBDAc, LAMBDAu=LAMBDAu,
        LAMBDAs_t=W @ LAMBDAs, LAMBDAz_t=W @ LAMBDAz,
        LAMBDAc_t=W @ LAMBDAc, LAMBDAu_t=W @ LAMBDAu,
        W=W, sig_mm=sig_mm, sig_mq=sig_mq, sig_qm=sig_qm, sig_qq=sig_qq,
    )


def draw_states_block(Phi, sigma, At_init, Pt_init,
                      Zm, Ym, Yq, YDATA, index_NY,
                      nobs, Tnobs, Tnew, T0, freq_ratio,
                      Nm, Nq, nv, p, temp_agg):
    """One exact ``K_s`` update: simulation-smoother draw of the block's
    full latent trajectory from ``p(states | Phi, sigma, data)``.

    Runs the balanced Kalman filter, the unbalanced (ragged-edge) filter,
    the terminal draw, and both backward simulation smoothers, all under
    the SAME parameters ``(Phi, sigma)``.  Transcribed from the forward
    pass of ``_estimation.py``; the only difference from the legacy MwG
    state proposal is that the filter moments are recomputed here under
    the conditioning parameters instead of being recycled from a filter
    run under the previous iteration's parameters.

    Returns ``(At_draw, AT_draw, Pmean)``: the balanced-period state draw,
    the ragged-edge state draw, and the terminal smoothed covariance used
    to seed the next iteration's filter.
    """
    mats = build_block_matrices(Phi, sigma, Nm, Nq, p, freq_ratio, temp_agg)
    GAMMAs, GAMMAz, GAMMAc, GAMMAu = (mats["GAMMAs"], mats["GAMMAz"],
                                      mats["GAMMAc"], mats["GAMMAu"])
    LAMBDAs, LAMBDAz, LAMBDAc, LAMBDAu = (mats["LAMBDAs"], mats["LAMBDAz"],
                                          mats["LAMBDAc"], mats["LAMBDAu"])
    LAMBDAs_t, LAMBDAz_t, LAMBDAc_t, LAMBDAu_t = (
        mats["LAMBDAs_t"], mats["LAMBDAz_t"], mats["LAMBDAc_t"], mats["LAMBDAu_t"])
    sig_qq, sig_mm, sig_mq, sig_qm = (mats["sig_qq"], mats["sig_mm"],
                                      mats["sig_mq"], mats["sig_qm"])

    ns = Nq * (p + 1)
    kn = nv * (p + 1)
    has_hf = Ym.size > 0

    # ---- balanced Kalman filter under (Phi, sigma) ----
    At = At_init.copy()
    Pt = Pt_init.copy()
    At_mat = np.zeros((nobs, ns))
    Pt_mat = np.zeros((nobs, ns**2))
    for t in range(nobs):
        lf_step = ((t+1+T0)/freq_ratio - np.floor((t+T0+1)/freq_ratio) == 0)
        alphahat = GAMMAs @ At + GAMMAz @ Zm[t, :] + GAMMAc[:, 0]
        Phat = GAMMAs @ Pt @ GAMMAs.T + GAMMAu @ sig_qq @ GAMMAu.T
        Phat = 0.5 * (Phat + Phat.T)
        if has_hf and lf_step:
            obs = np.concatenate((Ym[t, :], Yq[t, :]))
            yhat = LAMBDAs @ alphahat + LAMBDAz @ Zm[t, :] + LAMBDAc[:, 0]
            nut = obs - yhat
            Ft = (LAMBDAs @ Phat @ LAMBDAs.T + LAMBDAu @ sig_mm @ LAMBDAu.T
                  + LAMBDAs @ GAMMAu @ sig_qm @ LAMBDAu.T
                  + LAMBDAu @ sig_mq @ GAMMAu.T @ LAMBDAs.T)
            Ft = 0.5 * (Ft + Ft.T)
            Xit = LAMBDAs @ Phat + LAMBDAu @ sig_mq @ GAMMAu.T
            sol = Xit.T @ invert_matrix(Ft)
            At = alphahat + sol @ nut
            Pt = Phat - sol @ Xit
        elif has_hf:
            yhat = LAMBDAs_t @ alphahat + LAMBDAz_t @ Zm[t, :] + LAMBDAc_t[:, 0]
            nut = Ym[t, :] - yhat
            Ft = (LAMBDAs_t @ Phat @ LAMBDAs_t.T + LAMBDAu_t @ sig_mm @ LAMBDAu_t.T
                  + LAMBDAs_t @ GAMMAu @ sig_qm @ LAMBDAu_t.T
                  + LAMBDAu_t @ sig_mq @ GAMMAu.T @ LAMBDAs_t.T)
            Ft = 0.5 * (Ft + Ft.T)
            Xit = LAMBDAs_t @ Phat + LAMBDAu_t @ sig_mq @ GAMMAu.T
            sol = Xit.T @ invert_matrix(Ft)
            At = alphahat + sol @ nut
            Pt = Phat - sol @ Xit
        elif lf_step:
            yhat = LAMBDAs @ alphahat + LAMBDAz @ Zm[t, :] + LAMBDAc[:, 0]
            nut = Yq[t, :] - yhat
            Ft = (LAMBDAs @ Phat @ LAMBDAs.T + LAMBDAu @ sig_mm @ LAMBDAu.T
                  + LAMBDAs @ GAMMAu @ sig_qm @ LAMBDAu.T
                  + LAMBDAu @ sig_mq @ GAMMAu.T @ LAMBDAs.T)
            Ft = 0.5 * (Ft + Ft.T)
            Xit = LAMBDAs @ Phat + LAMBDAu @ sig_mq @ GAMMAu.T
            sol = Xit.T @ invert_matrix(Ft)
            At = alphahat + sol @ nut
            Pt = Phat - sol @ Xit
        else:
            At = alphahat
            Pt = Phat
        At_mat[t, :] = At.T
        Pt_mat[t, :] = Pt.reshape((1, ns**2), order="F")

    # ---- unbalanced (ragged-edge) filter under (Phi, sigma) ----
    Atilde = At_mat[nobs-1, :]
    Ptilde = Pt_mat[nobs-1, :].reshape((ns, ns), order="F")

    Z1 = np.zeros((Nm, kn))
    Z1[:, :Nm] = np.eye(Nm)
    Z2 = np.zeros((Nq, kn))
    for bb in range(Nq):
        for ll in range(freq_ratio):
            if temp_agg == "mean":
                Z2[bb, (ll+1)*Nm + ll*Nq + bb] = 1.0 / freq_ratio
            else:
                Z2[bb, (ll+1)*Nm + ll*Nq + bb] = 1.0
    ZZ = np.vstack((Z1, Z2))

    if has_hf:
        BAt = np.concatenate((Ym[-1, :], np.atleast_1d(np.squeeze(Atilde[:Nq]))))
        for rr in range(1, p+1):
            BAt = np.concatenate((BAt, np.concatenate((
                Ym[-(rr+1), :],
                np.atleast_1d(np.squeeze(Atilde[rr*Nq:(rr+1)*Nq]))))))
    else:
        BAt = np.atleast_1d(np.squeeze(Atilde[:Nq]))
        for rr in range(1, p+1):
            BAt = np.concatenate((BAt, np.atleast_1d(
                np.squeeze(Atilde[rr*Nq:(rr+1)*Nq]))))

    BPt = np.zeros((kn, kn))
    for rr in range(p+1):
        for vv in range(p+1):
            BPt[(rr+1)*Nm + rr*Nq:(rr+1)*(Nm+Nq),
                (vv+1)*Nm + vv*Nq:(vv+1)*(Nm+Nq)] = \
                Ptilde[rr*Nq:(rr+1)*Nq, vv*Nq:(vv+1)*Nq]

    PHIF = np.zeros((kn, kn))
    IF = np.eye(nv)
    for i in range(p-1):
        PHIF[(i+1)*nv:(i+2)*nv, i*nv:(i+1)*nv] = IF
    PHIF[:nv, :nv*p] = Phi[:-1, :].T
    CONF = np.hstack((Phi[-1, :].T, np.zeros((nv*p,))))
    SIGF = np.zeros((kn, kn))
    SIGF[:nv, :nv] = sigma

    BAt_mat = np.zeros((Tnobs, kn))
    BPt_mat = np.zeros((Tnobs, kn**2))
    BAt_mat[nobs-1, :] = BAt
    BPt_mat[nobs-1, :] = BPt.reshape((1, kn**2), order="F")
    BAt_cur = BAt.copy()
    BPt_cur = BPt.copy()
    for t in range(nobs, Tnobs):
        kkk = t - nobs
        ND = YDATA[nobs+T0+kkk, :][~np.isnan(YDATA[nobs+T0+kkk, :])]
        NZ = ZZ[~index_NY[:, kkk], :]
        Balphahat = PHIF @ BAt_cur + CONF
        BPhat = PHIF @ BPt_cur @ PHIF.T + SIGF
        BPhat = 0.5 * (BPhat + BPhat.T)
        Bnut = ND - NZ @ Balphahat
        BFt = NZ @ BPhat @ NZ.T
        BFt = 0.5 * (BFt + BFt.T)
        sol = (BPhat @ NZ.T) @ invert_matrix(BFt)
        BAt_cur = Balphahat + sol @ Bnut
        BPt_cur = BPhat - sol @ (BPhat @ NZ.T).T
        BAt_mat[t, :] = BAt_cur
        BPt_mat[t, :] = BPt_cur.reshape((1, kn**2), order="F")

    # ---- terminal draw + backward unbalanced simulation smoother ----
    AT_draw = np.zeros((Tnew+1, kn))
    Pchol = cholcovOrEigendecomp(BPt_mat[Tnobs-1, :].reshape((kn, kn), order="F"))
    AT_draw[-1, :] = BAt_mat[Tnobs-1, :] + np.transpose(
        Pchol @ np.random.standard_normal(kn))
    for i in range(Tnew):
        BAtt = BAt_mat[Tnobs-(i+2), :]
        BPtt = BPt_mat[Tnobs-(i+2), :].reshape((kn, kn), order="F")
        BPhat = PHIF @ BPtt @ PHIF.T + SIGF
        BPhat = 0.5 * (BPhat + BPhat.T)
        inv_BPhat = invert_matrix(BPhat)
        Bnut = AT_draw[-(i+1), :] - PHIF @ BAtt - CONF
        Amean = BAtt + (BPtt @ PHIF.T) @ inv_BPhat @ Bnut
        Pmean_unb = BPtt - (BPtt @ PHIF.T) @ inv_BPhat @ np.transpose(BPtt @ PHIF.T)
        Pmchol = cholcovOrEigendecomp(Pmean_unb)
        AT_draw[-2-i, :] = np.transpose(
            Amean + Pmchol @ np.random.standard_normal(kn))

    # ---- backward balanced simulation smoother ----
    At_draw = np.zeros((nobs, ns))
    for kk in range(p+1):
        At_draw[nobs-1, kk*Nq:(kk+1)*Nq] = AT_draw[
            0, (kk+1)*Nm + kk*Nq:(kk+1)*(Nm+Nq)]

    Pmean = Pt_mat[nobs-1, :].reshape((ns, ns), order="F").copy()
    for i in range(nobs-1):
        Att = At_mat[nobs-(i+2), :]
        Ptt = Pt_mat[nobs-(i+2), :].reshape((ns, ns), order="F")
        Phat = GAMMAs @ Ptt @ GAMMAs.T + GAMMAu @ sig_qq @ GAMMAu.T
        Phat = 0.5 * (Phat + Phat.T)
        inv_Phat = invert_matrix(Phat)
        nut = (At_draw[nobs-(i+1), :] - GAMMAs @ Att
               - GAMMAz @ Zm[nobs-1-(i+1)] - GAMMAc[:, 0])
        temp = Ptt @ GAMMAs.T
        Amean = Att + temp @ inv_Phat @ nut
        Pmean = Ptt - temp @ inv_Phat @ np.transpose(temp)
        Pmchol = cholcovOrEigendecomp(Pmean)
        At_draw[nobs-1-(i+1), :] = np.transpose(
            Amean + Pmchol @ np.random.standard_normal(ns))

    return At_draw, AT_draw, Pmean


def build_completed_data(Ym, At_draw, AT_draw, Nm, Nq):
    """Assemble the completed (balanced + ragged-edge) block data ``YY``
    from a latent-state draw, as in the forward pass."""
    if Ym.size:
        return np.vstack((np.hstack((Ym, At_draw[:, :Nq])),
                          AT_draw[1:, :(Nm+Nq)]))
    return np.vstack((At_draw[:, :Nq], AT_draw[1:, :(Nm+Nq)]))


def draw_params_block(hyp_m, YY, spec, check_explosive, max_it_stable,
                      premom=None):
    """One ``K_theta`` update: joint draw of ``(sigma, Phi)`` from the
    NIW-type full conditional given completed data ``YY`` (inverse-Wishart
    marginal for sigma, conditional normal for Phi), truncated by the
    ``is_explosive`` screen.

    Returns ``(Phi, sigma, YYact, proposals, rejected)``; ``Phi`` is None
    when the update must be treated as an auto-reject (degenerate
    regression or explosive-cap exhaustion), with the stability counters
    still reported.
    """
    YYact, YYdum, XXact, XXdum = calc_yyact(hyp_m, YY, spec, premom=premom)
    Tdummy = YYdum.shape[0]
    Tobs = YYact.shape[0]
    if Tobs == 0:
        return None, None, None, 0, 0
    n = YYact.shape[1]
    p_spec = int(spec[0])
    X = np.vstack((XXact, XXdum))
    Y = np.vstack((YYact, YYdum))
    T = Tobs + Tdummy
    if (np.isnan(X).any() or np.isnan(Y).any()
            or np.isinf(X).any() or np.isinf(Y).any()):
        return None, None, None, 0, 0
    if T - n*p_spec - 1 <= 0:
        return None, None, None, 0, 0

    vl, d, vr = np.linalg.svd(X, full_matrices=False)
    vr = vr.T
    di = 1.0 / d
    B = vl.T @ Y
    xxi = vr * np.tile(di.T, (n*p_spec+1, 1))
    inv_x = xxi @ xxi.T
    Phi_tilde = xxi @ B
    Sigma = (Y - X @ Phi_tilde).T @ (Y - X @ Phi_tilde)
    sigma = invwishart.rvs(scale=Sigma, df=T - n*p_spec - 1)
    sigma = np.atleast_2d(sigma)
    sigma_chol = cholcovOrEigendecomp(np.kron(sigma, inv_x))

    proposals = 0
    rejected = 0
    if check_explosive:
        Phi = None
        for _ in range(max_it_stable):
            phi_new = (np.squeeze(Phi_tilde.reshape(n*(n*p_spec+1), 1, order="F"))
                       + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0]))
            cand = phi_new.reshape(n*p_spec+1, n, order="F")
            proposals += 1
            if not is_explosive(cand, n, p_spec):
                Phi = cand
                break
            rejected += 1
        if Phi is None:
            return None, None, None, proposals, rejected
    else:
        phi_new = (np.squeeze(Phi_tilde.reshape(n*(n*p_spec+1), 1, order="F"))
                   + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0]))
        Phi = phi_new.reshape(n*p_spec+1, n, order="F")
        proposals = 1

    return Phi, sigma, YYact, proposals, rejected


def palindromic_proposal_ss(Phi_cur, sigma_cur, hyp_m, nlags_m, nex,
                            At_init, Pt_init, Zm, Ym, Yq, YDATA, index_NY,
                            nobs, Tnobs, Tnew, T0, freq_ratio,
                            Nm, Nq, nv, p, temp_agg,
                            check_explosive, max_it_stable, premom=None):
    """The reversible MwG proposal ``K_s K_theta K_s`` for one SS block.

    1. ``s'  ~ p(states | theta_cur, data)``   (filters run under theta_cur)
    2. ``theta' ~ p(theta | s', data)``        (NIW draw, truncated)
    3. ``s'' ~ p(states | theta', data)``      (filters run under theta')

    and propose ``(s'', theta')``.  Reversibility: each step is a
    full-conditional (hence reversible) update of the local posterior and
    the component sequence is a palindrome, so the composition is
    self-adjoint w.r.t. the local posterior.  The MwG acceptance ratio may
    therefore be the bare downstream likelihood ratio.

    Returns a dict with the proposal (``Phi`` is None on auto-reject) and
    the stability-truncation counters.
    """
    out = dict(Phi=None, sigma=None, At_draw=None, AT_draw=None, Pmean=None,
               YYact=None, stab_proposals=0, stab_rejected=0)

    # K_s under theta_cur
    At1, AT1, _ = draw_states_block(
        Phi_cur, sigma_cur, At_init, Pt_init, Zm, Ym, Yq, YDATA, index_NY,
        nobs, Tnobs, Tnew, T0, freq_ratio, Nm, Nq, nv, p, temp_agg)

    # K_theta given s'
    YY1 = build_completed_data(Ym, At1, AT1, Nm, Nq)
    spec = np.hstack((nlags_m, T0, nex, nv, np.shape(YY1)[0] - T0))
    Phi_p, sigma_p, _, n_prop, n_rej = draw_params_block(
        hyp_m, YY1, spec, check_explosive, max_it_stable, premom=premom)
    out["stab_proposals"] += n_prop
    out["stab_rejected"] += n_rej
    if Phi_p is None:
        return out

    # K_s under theta'
    At2, AT2, Pmean2 = draw_states_block(
        Phi_p, sigma_p, At_init, Pt_init, Zm, Ym, Yq, YDATA, index_NY,
        nobs, Tnobs, Tnew, T0, freq_ratio, Nm, Nq, nv, p, temp_agg)

    YY2 = build_completed_data(Ym, At2, AT2, Nm, Nq)
    YYact2, _, _, _ = calc_yyact(hyp_m, YY2, spec, premom=premom)

    out.update(Phi=Phi_p, sigma=sigma_p, At_draw=At2, AT_draw=AT2,
               Pmean=Pmean2, YYact=YYact2)
    return out
