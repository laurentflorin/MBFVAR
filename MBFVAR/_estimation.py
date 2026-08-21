import os
import sys
import warnings

import numpy as np
import math

from collections import deque
from scipy.linalg import companion, solve_discrete_lyapunov
from scipy.stats import invwishart
import pandas as pd
from scipy.stats import multivariate_normal
from datetime import datetime
from pandas.tseries.offsets import Week , MonthBegin, QuarterBegin, Day

import itertools

#plotting
import matplotlib.pyplot as plt

#for progressbar
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position = 0, leave=True) # this line does the magic

# for plots

import matplotlib.backends.backend_pdf

import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default='browser'
import plotly.express as px

#to save objects
import pickle
import copy



#from MBFVAR.pseudo_inverse.pseudo_inverse import calculate_pseudo_inverse
from .cholcov.cholcov_module import cholcovOrEigendecomp
from .inverse.matrix_inversion import invert_matrix
from .mfbvar_funcs import calc_yyact, is_explosive, mdd_
# for hyperparameter tuning




def _kalman_filter_loglik_block(
    GAMMAs, GAMMAz, GAMMAc, GAMMAu,
    LAMBDAs, LAMBDAz, LAMBDAc, LAMBDAu,
    LAMBDAs_t, LAMBDAz_t, LAMBDAc_t, LAMBDAu_t,
    sig_qq, sig_mm, sig_mq, sig_qm,
    At_init, Pt_init,
    Zm, Ym, Yq,
    nobs, T0, freq_ratio,
):
    """
    Run the balanced Kalman filter for one frequency block and return the
    accumulated log-likelihood of the observations.

    The filter mirrors the balanced forward KF inside fit() so that it can be
    used for the Metropolis-within-Gibbs acceptance ratio.  The only difference
    from the forward pass is that innovation log-likelihoods are accumulated
    instead of being discarded.

    Parameters
    ----------
    GAMMAs, GAMMAz, GAMMAc, GAMMAu : ndarray
        Transition-equation matrices for the block.
    LAMBDAs, LAMBDAz, LAMBDAc, LAMBDAu : ndarray
        Full measurement matrices (used when both HF and LF are observed).
    LAMBDAs_t, LAMBDAz_t, LAMBDAc_t, LAMBDAu_t : ndarray
        HF-only measurement matrices (used when only HF is observed).
    sig_qq, sig_mm, sig_mq, sig_qm : ndarray
        Covariance submatrices for the block.
    At_init, Pt_init : ndarray
        Initial filter mean and covariance.
    Zm : ndarray, shape (nobs, p*Nm)
        Lagged HF observations.
    Ym : ndarray, shape (nobs, Nm) or empty
        HF observations.
    Yq : ndarray, shape (nobs, Nq)
        LF observations passed in from the upstream block.
    nobs : int
        Number of observations in the balanced period.
    T0 : int
        Lag burn-in length (used only for the LF-step condition).
    freq_ratio : int
        Frequency ratio for this block.

    Returns
    -------
    float
        Sum of Gaussian log-likelihood contributions over the balanced period.
    """
    At = At_init.copy()
    Pt = Pt_init.copy()
    has_hf = Ym.size > 0
    ll = 0.0

    for t in range(nobs):
        at_lf_step = (
            (t + 1 + T0) / freq_ratio
            - np.floor((t + T0 + 1) / freq_ratio) == 0
        )

        alphahat = GAMMAs @ At + GAMMAz @ Zm[t, :] + GAMMAc[:, 0]
        Phat = GAMMAs @ Pt @ GAMMAs.T + GAMMAu @ sig_qq @ GAMMAu.T
        Phat = 0.5 * (Phat + Phat.T)

        if has_hf and at_lf_step:
            # Both HF and LF observed
            obs = np.concatenate((Ym[t, :], Yq[t, :]))
            yhat = LAMBDAs @ alphahat + LAMBDAz @ Zm[t, :] + LAMBDAc[:, 0]
            nut = obs - yhat
            Ft = (
                LAMBDAs @ Phat @ LAMBDAs.T
                + LAMBDAu @ sig_mm @ LAMBDAu.T
                + LAMBDAs @ GAMMAu @ sig_qm @ LAMBDAu.T
                + LAMBDAu @ sig_mq @ GAMMAu.T @ LAMBDAs.T
            )
            Ft = 0.5 * (Ft + Ft.T)
            Xit = LAMBDAs @ Phat + LAMBDAu @ sig_mq @ GAMMAu.T
            sol = Xit.T @ invert_matrix(Ft)
            At = alphahat + sol @ nut
            Pt = Phat - sol @ Xit
        elif has_hf and not at_lf_step:
            # Only HF observed
            obs = Ym[t, :]
            yhat = LAMBDAs_t @ alphahat + LAMBDAz_t @ Zm[t, :] + LAMBDAc_t[:, 0]
            nut = obs - yhat
            Ft = (
                LAMBDAs_t @ Phat @ LAMBDAs_t.T
                + LAMBDAu_t @ sig_mm @ LAMBDAu_t.T
                + LAMBDAs_t @ GAMMAu @ sig_qm @ LAMBDAu_t.T
                + LAMBDAu_t @ sig_mq @ GAMMAu.T @ LAMBDAs_t.T
            )
            Ft = 0.5 * (Ft + Ft.T)
            Xit = LAMBDAs_t @ Phat + LAMBDAu_t @ sig_mq @ GAMMAu.T
            sol = Xit.T @ invert_matrix(Ft)
            At = alphahat + sol @ nut
            Pt = Phat - sol @ Xit
        elif (not has_hf) and at_lf_step:
            # Only LF observed (no HF variables in this block)
            obs = Yq[t, :]
            yhat = LAMBDAs @ alphahat + LAMBDAz @ Zm[t, :] + LAMBDAc[:, 0]
            nut = obs - yhat
            Ft = (
                LAMBDAs @ Phat @ LAMBDAs.T
                + LAMBDAu @ sig_mm @ LAMBDAu.T
                + LAMBDAs @ GAMMAu @ sig_qm @ LAMBDAu.T
                + LAMBDAu @ sig_mq @ GAMMAu.T @ LAMBDAs.T
            )
            Ft = 0.5 * (Ft + Ft.T)
            Xit = LAMBDAs @ Phat + LAMBDAu @ sig_mq @ GAMMAu.T
            sol = Xit.T @ invert_matrix(Ft)
            At = alphahat + sol @ nut
            Pt = Phat - sol @ Xit
        else:
            # No observation at this step
            At = alphahat
            Pt = Phat
            nut = None
            Ft = None

        if nut is not None and Ft is not None:
            k = len(nut)
            sign, logdet = np.linalg.slogdet(Ft)
            if sign > 0:
                ll += -0.5 * (logdet + nut @ invert_matrix(Ft) @ nut + k * np.log(2.0 * np.pi))

    return ll


def fit(self, mbfvar_data, hyp, var_of_interest = None, temp_agg = 'mean', max_it_stable = 1000, return_mdd = False, check_explosive = True, method = 'schorfheide_song', seed = None):

    '''
    Estimates the model using the model parameter specified in the initialization. \n
    And the data provided.

    Parameters
    ----------
    mbfvar_data : mbfvar_data class object
        data in the form of a mbfvar_data class object
    hyp : list of list
        list containing list of the hyperparameters for each frequency step\n
        1. overall tightness\n
        2. scaling down the variance for the coefficients of a distant lag\n
        3. number of observations used for obtaining the prior for the covariance matrix of error terms\n
        4. tuning parameter for coefficients for constant\n
        5. tuning parameter for the covariance between coefficients\n
    var_of_interest: list of names of variables that we are interested in or None
        Only the variables that are in this list get used in every bi frequency var.
        If None all variables get taken into each higher frequency bi frequency var.
    temp_agg : str
        `mean` or `sum` defines the measurement equation
    max_it_stable : int
        maximum number of attempts to draw non-explosive VAR coefficients
    return_mdd : bool
        if True, returns the marginal data density (used for hyperparameter optimization)
    check_explosive : bool
        if True (default), checks for explosive VAR coefficients and rejects them.
        if False, skips explosive VAR checks (useful for hyperparameter optimization)
    method : str
        Estimation approach to use. One of:\n
        - ``'schorfheide_song'`` (default): the Schorfheide--Song (2015) state-space
          estimator (unchanged default behaviour).\n
        - ``'chan_poon_zhu'``: the Chan, Poon & Zhu (2024) conditionally-Gaussian
          latent-state sampler with common stochastic volatility, adapted to each
          bi-frequency block.

    Returns
    -------
    float or None
        If return_mdd is True, returns the MDD for the last frequency block.
        Otherwise, returns None (standard behavior).
    '''

    valid_methods = ('schorfheide_song', 'chan_poon_zhu')
    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method!r}. Choose one of {valid_methods}.")
    self.method = method
    if seed is None:
        seed = getattr(self, 'seed', 0)
    self.seed = seed
    np.random.seed(seed)
    if method == 'chan_poon_zhu':
        from ._estimation_cpz import fit_cpz
        return fit_cpz(self, mbfvar_data, hyp, var_of_interest=var_of_interest,
                       temp_agg=temp_agg, max_it_stable=max_it_stable,
                       return_mdd=return_mdd, check_explosive=check_explosive)

    explosive_counter = 0
    # Stability-truncation accounting: Phi proposals drawn and rejections by
    # the is_explosive() check, split between the forward Gibbs draw and the
    # Metropolis-within-Gibbs backward pass (which auto-rejects explosive
    # proposals before the likelihood-ratio acceptance).
    stability_proposals = 0
    stability_rejected = 0
    stability_proposals_backward = 0
    stability_rejected_backward = 0
    valid_draws = []
    mdd_list = [np.nan] * (len(mbfvar_data.frequencies)-1) if return_mdd else None

    self.nex = 1
    self.hyp = hyp
    self.temp_agg = temp_agg

    # normalize var_of_interest to a list so single strings do not get treated as sequences of characters
    if var_of_interest is not None:
        if isinstance(var_of_interest, str):
            var_of_interest = [var_of_interest]
        else:
            var_of_interest = list(var_of_interest)
    
    assert self.temp_agg in ("mean", "sum"), f"Invalid temp_agg: {self.temp_agg}. Choose 'mean' or 'sum'."
    
    # data from mbfvar_data
    
    YMX_list = copy.deepcopy(mbfvar_data.YMX_list)
    YM0_list = copy.deepcopy(mbfvar_data.YM0_list)
    select_m_list = copy.deepcopy(mbfvar_data.select_m_list)
    vars_m_list = copy.deepcopy(mbfvar_data.vars_m_list)
    YMh_list = copy.deepcopy(mbfvar_data.YMh_list)
    index_list = copy.deepcopy(mbfvar_data.index_list)
    frequencies = copy.deepcopy(mbfvar_data.frequencies)
    self.frequencies = frequencies
    YQX_list = copy.deepcopy(mbfvar_data.YQX_list)
    YQ0_list = copy.deepcopy(mbfvar_data.YQ0_list)
    select_q = copy.deepcopy(mbfvar_data.select_q)
    input_data_Q =  copy.deepcopy(mbfvar_data.input_data_Q)
    self.input_data_Q = input_data_Q
    varlist_list = copy.deepcopy(mbfvar_data.varlist_list)
    select_list = copy.deepcopy(mbfvar_data.select_list)
    select_c_list = copy.deepcopy(mbfvar_data.select_c_list)
    Nm_list = copy.deepcopy(mbfvar_data.Nm_list)
    nv_list = copy.deepcopy(mbfvar_data.nv_list)
    Nq_list = copy.deepcopy(mbfvar_data.Nq_list)
    select_list_sep = copy.deepcopy(mbfvar_data.select_list_sep)
    freq_ratio_list = copy.deepcopy(mbfvar_data.freq_ratio_list)
    YQ_list = copy.deepcopy(mbfvar_data.YQ_list)
    Tstar_list = copy.deepcopy(mbfvar_data.Tstar_list)
    T_list = copy.deepcopy(mbfvar_data.T_list)
    YDATA_list = copy.deepcopy(mbfvar_data.YDATA_list)
    YM_list = copy.deepcopy(mbfvar_data.YM_list)
    input_data = copy.deepcopy(mbfvar_data.input_data)
    self.input_data = input_data
    
    if not(var_of_interest is None):
        idx_var_of_interest = list(filter(
            lambda x: YQX_list[0].columns.tolist()[x] in var_of_interest,
            range(len(YQX_list[0].columns.tolist()))))
    
    nburn = round((self.nburn_perc)*math.ceil(self.nsim/self.thining))
    self.nburn = nburn
    
    #test if nlags for each step is at least frequency ratio
    for i in range(len(freq_ratio_list)):
        if deque(self.nlags)[i] < freq_ratio_list[i]:
            sys.exit("The number of lags at each step must be at least as long as the corresponding frequency ratio")
    
    # fit the model
    ###############
    
    nlags_list_ = self.nlags
    
    # Initialize Matrices
    #########################
    
    # here we initialize all the matrixes for the sampling for all frequencies
    # we again create lists and append them, in the sample loop we can then
    # access these matrices 
    
    #create empty lists, so we can append to them
    T0_list = deque()
    p_list = deque()
    nlags_list = deque()
    kq_list = deque()
    nobs_list = deque()
    Tnew_list = deque()
    Tnobs_list = deque()
    
    varstxt_list = deque()
    smpltxt_list = deque()
    
    index_NY_list = deque()
    
    Sigmap_list    = deque()
    Phip_list      = deque()
    Cons_list      = deque()
    lstate_list    = deque()
    YYactsim_list  = deque()
    XXactsim_list  = deque()
    
    At_mat_list    = deque()
    Pt_mat_list    = deque()
    Atildemat_list = deque()
    Ptildemat_list = deque()
    loglh_list   = deque()
    counter_list = deque()
    
    phi_mm_list = deque()
    
    phi_mq_list = deque()
    phi_mc_list = deque()
    phi_qc_list = deque()
    
    Phi_list = deque()
    
    # Define Transition Equation Matrices in eq (10)
    GAMMAs_list = deque()
    
    
    GAMMAz_list = deque()
    GAMMAc_list = deque()
    GAMMAu_list = deque()

    LAMBDAs_list = deque()
    
    LAMBDAz_list = deque()
    LAMBDAc_list = deque()
    LAMBDAu_list = deque()
    
    # Define Covariance Terms sig_mm, sig_mq_sig_qm, siq_qq
    sigma_list = deque()
    sig_mm_list = deque()
    sig_mq_list = deque()
    sig_qm_list = deque()
    sig_qq_list = deque()
    
    # Define W matrix in eq (15) -- _t for tilde
    Wmatrix_list = deque()
    LAMBDAs_t_list = deque()
    LAMBDAz_t_list = deque()
    LAMBDAc_t_list = deque()
    LAMBDAu_t_list = deque()
    
    At_list = deque()
    At_draw_list = deque()
    Pt_list = deque()
    Pmean_list = deque()
    
    Zm_list = deque()
    
    Yq_list = deque()
    Ym_list = deque()
    
    Pt_last_list = deque()
    
    YY_list = deque()
    
    #for i in range(len(YMh_list)):
    T0_list.append(int(nlags_list_[0]))
    p_list.append(int(nlags_list_[0]))
    nlags_list.append(p_list[0])
    kq_list.append(Nq_list[0] * p_list[0])
    nobs_list.append(int(T_list[0]) - int(T0_list[0]))
    Tnew_list.append(Tstar_list[0]-T_list[0])
    Tnobs_list.append(Tstar_list[0]-T0_list[0])
    

    
    #for writing to a forecast w/ history file
    YMh_list[0] = YMh_list[0][int(T0_list[0]):-int(freq_ratio_list[0]),:]
    varstxt_list.append(np.hstack((YMX_list[0].columns, YQX_list[0].columns)))
    smpltxt_list.append(YMX_list[0].index[int(T0_list[0]):])
    
    index_NY_list.append(np.isnan(YDATA_list[0][nobs_list[0]+T0_list[0]:Tnobs_list[0]+T0_list[0],:]).T) # TODO CHECK
    
    # Parameter estimation
    # Matrices for collecting draws from Posterior Density
    Sigmap_list.append(np.zeros((math.ceil((self.nsim)/self.thining),nv_list[0],nv_list[0])))
    Phip_list.append(np.zeros((math.ceil((self.nsim)/self.thining),int(nv_list[0])*int(p_list[0])+1,int(nv_list[0]))))
    Cons_list.append(np.zeros((math.ceil((self.nsim)/self.thining),nv_list[0])))
    #lstate_list.append(np.zeros((round((self.nsim)/self.thining),Nq_list[0],int(Tnobs_list[0]))))
    #YYactsim_list.append(np.zeros((round((self.nsim)/self.thining),freq_ratio_list[0]+1,nv_list[0])))
    #XXactsim_list.append(np.zeros((round((self.nsim)/self.thining),int(freq_ratio_list[0])+1,int(nv_list[0])*int(p_list[0])+1)))
    
    At_mat_list.append(np.zeros((int(Tnobs_list[0]), Nq_list[0]*(int(p_list[0])+1))))
    Pt_mat_list.append(np.zeros((int(Tnobs_list[0]), (Nq_list[0]*(int(p_list[0])+1))**2)))
    Atildemat_list.append(np.zeros((self.nsim, Nq_list[0]*(int(p_list[0])+1))))
    Ptildemat_list.append(np.zeros((self.nsim, Nq_list[0]*(int(p_list[0])+1),Nq_list[0]*(int(p_list[0])+1))))
    loglh_list.append(0)
    counter_list.append(0)
    
    # Define phi, phi(mm), phi(mq), phi(mc) used in alt ss rep -- eq (9)
    phi_mm_list.append(np.zeros((int(Nm_list[0])*int(p_list[0]),int(Nm_list[0]))))
    phi_mm_list[0][:Nm_list[0],:Nm_list[0]] = np.eye(Nm_list[0])
    
    phi_mq_list.append(np.zeros((Nq_list[0]*int(p_list[0]),int(Nm_list[0]))))
    phi_mc_list.append(np.zeros((1,Nm_list[0])))
    phi_qc_list.append(np.zeros((1,Nq_list[0])))
    
    Phi_list.append(np.vstack((0.95 * np.eye(int(Nm_list[0])+Nq_list[0]), np.zeros(((int(Nm_list[0])+Nq_list[0])*(int(p_list[0])-1)+1, int(Nm_list[0])+Nq_list[0])))))
    
    # Define Transition Equation Matrices in eq (10)
    GAMMAs_list.append(np.zeros((Nq_list[0]*(int(p_list[0])+1), Nq_list[0] * (int(p_list[0])+1))))
    IQ = np.eye(Nq_list[0])
    
    for j in range(int(p_list[0])):
        GAMMAs_list[0][(j+1)*Nq_list[0]:(j+2)*Nq_list[0], j*Nq_list[0]:(j+1)*Nq_list[0]] = IQ
        
    GAMMAs_list[0][:Nq_list[0],:Nq_list[0]] = 0.95 * np.eye(Nq_list[0])
    
    GAMMAz_list.append(np.zeros((Nq_list[0]* (int(p_list[0])+1), int(Nm_list[0])*p_list[0])))
    GAMMAc_list.append(np.zeros((Nq_list[0]*(p_list[0]+1), 1)))
    GAMMAu_list.append(np.vstack((np.eye(Nq_list[0]), np.zeros((p_list[0]*Nq_list[0],Nq_list[0])))))

    if self.temp_agg == "mean":
        LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[0],Nq_list[0])), np.transpose(phi_mq_list[0]))),1/freq_ratio_list[0]*np.hstack((np.tile(np.eye(Nq_list[0]), freq_ratio_list[0]), np.zeros((Nq_list[0],Nq_list[0]*(p_list[0]-(freq_ratio_list[0]-1)))))))))
    
    if self.temp_agg == "sum":
        LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[0],Nq_list[0])), np.transpose(phi_mq_list[0]))),np.hstack((np.tile(np.eye(Nq_list[0]), freq_ratio_list[0]), np.zeros((Nq_list[0],Nq_list[0]*(p_list[0]-(freq_ratio_list[0]-1)))))))))
        
    
    LAMBDAz_list.append(np.vstack((np.transpose(phi_mm_list[0]), np.zeros((Nq_list[0], p_list[0]*Nm_list[0])))))
    LAMBDAc_list.append(np.vstack((np.transpose(phi_mc_list[0]), np.zeros((Nq_list[0],1)))))
    LAMBDAu_list.append(np.vstack((np.eye(Nm_list[0]), np.zeros((Nq_list[0],Nm_list[0])))))
    
    # Define Covariance Terms sig_mm, sig_mq_sig_qm, siq_qq
    sigma_list.append((1e-4)*np.eye(Nm_list[0]+Nq_list[0]))
    sig_mm_list.append(sigma_list[0][:Nm_list[0],:Nm_list[0]])
    sig_mq_list.append(sigma_list[0][:Nm_list[0], Nm_list[0]:])
    sig_qm_list.append(sigma_list[0][Nm_list[0]:, :Nm_list[0]])
    sig_qq_list.append(sigma_list[0][Nm_list[0]:,Nm_list[0]:])
    
    # Define W matrix in eq (15) -- _t for tilde
    
    Wmatrix_list.append(np.hstack((np.eye(Nm_list[0]), np.zeros((Nm_list[0],Nq_list[0])))))
    LAMBDAs_t_list.append(Wmatrix_list[0] @ LAMBDAs_list[0])
    LAMBDAz_t_list.append(Wmatrix_list[0] @ LAMBDAz_list[0])
    LAMBDAc_t_list.append(Wmatrix_list[0] @ LAMBDAc_list[0])
    LAMBDAu_t_list.append(Wmatrix_list[0] @ LAMBDAu_list[0])
    
    At_list.append(np.zeros((Nq_list[0]*(p_list[0]+1))))
    Pt_list.append(np.zeros((Nq_list[0]*(p_list[0]+1), Nq_list[0]*(p_list[0]+1))))

    # Solve the discrete Lyapunov equation exactly for the steady-state covariance:
    # Pt = GAMMAs @ Pt @ GAMMAs^T + GAMMAu @ sig_qq @ GAMMAu^T
    Q0 = GAMMAu_list[0] @ sig_qq_list[0] @ GAMMAu_list[0].T
    Pt_list[0] = solve_discrete_lyapunov(GAMMAs_list[0], Q0)
    
    # Lagged HF observations
    Zm_list.append(np.zeros((nobs_list[0], Nm_list[0]*p_list[0])))
    
    if Zm_list[0].size:
        for j in range(p_list[0]):
            Zm_list[0][:, j * Nm_list[0]:(j+1)*Nm_list[0]] = YM_list[0][T0_list[0]-(j+1):T0_list[0]+nobs_list[0]-(j+1),:]
            
    # Observations in HF
    
    Ym_list.append(YM_list[0][T0_list[0]:nobs_list[0]+T0_list[0],:])
    
    Yq_list.append(YQ_list[0][T0_list[0]:nobs_list[0]+T0_list[0],:])
    
    # Estimation
    #################
    print(" ", end = '\n')
    print("Multi Frequency BVAR: Estimation", end = '\n')
    print("Frequencies: ", self.frequencies, end = "\n")
    print("Total Number of Draws: ",self.nsim)
    
    #Here we start the sample loop, j is the current sample
    #inside the sample loop we need a loop for the MFBVARS: m

    # MH backward correction state
    M_blks = len(YMh_list)
    mh_accept_counts = [0] * (M_blks - 1)
    mh_total_counts = [0] * (M_blks - 1)
    _mh_At_init = [None] * M_blks
    _mh_Pt_init = [None] * M_blks

    for j in tqdm(range(self.nsim)):
        for m in range(len(YMh_list)):
            
            # initialization
            if j > 0:
                At_list[m] = At_draw_list[m][0,:].T 
                Pt_list[m] = Pmean_list[m]

            # Save KF initial conditions for the MH backward correction
            _mh_At_init[m] = At_list[m].copy()
            _mh_Pt_init[m] = Pt_list[m].copy()

            # Kalman Filter Loop
            #########################
            for t in range(nobs_list[m]):  
                
                if Ym_list[m].size:
                    
                    if (t+1+T0_list[m])/freq_ratio_list[m] - np.floor((t+T0_list[m]+1)/freq_ratio_list[m]) == 0:
                        At1 = At_list[m]
                        Pt1 = Pt_list[m]
                        # Forecasting
                        alphahat = GAMMAs_list[m] @ At1 + GAMMAz_list[m] @ Zm_list[m][t,:] + GAMMAc_list[m][:,0]
                        Phat = GAMMAs_list[m] @ Pt1 @ GAMMAs_list[m].T + GAMMAu_list[m] @ sig_qq_list[m] @ GAMMAu_list[m].T
                        Phat = 0.5*(Phat + Phat.T)
                        yhat = LAMBDAs_list[m] @ alphahat + LAMBDAz_list[m] @ Zm_list[m][t,:] + LAMBDAc_list[m][:,0]
                        nut = np.concatenate((Ym_list[m][t,:], Yq_list[m][t,:])) - yhat  
                        Ft = (LAMBDAs_list[m] @ Phat @ LAMBDAs_list[m].T + LAMBDAu_list[m] @ sig_mm_list[m] @ LAMBDAu_list[m].T
                            + LAMBDAs_list[m] @ GAMMAu_list[m] @ sig_qm_list[m] @ LAMBDAu_list[m].T
                            + LAMBDAu_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T @ LAMBDAs_list[m].T)
                        Ft = 0.5*(Ft+Ft.T)
                        Xit = LAMBDAs_list[m] @ Phat + LAMBDAu_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T
                        sol = Xit.T @ invert_matrix(Ft)
                        At_list[m] = alphahat + sol @ nut
                        Pt_list[m] = Phat - sol @ Xit
                        At_mat_list[m][t,:] = At_list[m].T
                        Pt_mat_list[m][t,:] = Pt_list[m].reshape((1, (Nq_list[m]*(p_list[m]+1))**2), order = "F")
                    else:
                        At1 = At_list[m]
                        Pt1 = Pt_list[m]
                        # Forecasting
                        alphahat = GAMMAs_list[m] @ At1 + GAMMAz_list[m] @ Zm_list[m][t,:] + GAMMAc_list[m][:,0]
                        Phat = GAMMAs_list[m] @ Pt1 @ GAMMAs_list[m].T + GAMMAu_list[m] @ sig_qq_list[m] @ GAMMAu_list[m].T
                        Phat = 0.5*(Phat + Phat.T)
                        yhat = LAMBDAs_t_list[m] @ alphahat + LAMBDAz_t_list[m] @ Zm_list[m][t,:] + LAMBDAc_t_list[m][:,0]
                        nut = Ym_list[m][t,:] - yhat
                        Ft = (LAMBDAs_t_list[m] @ Phat @ LAMBDAs_t_list[m].T + LAMBDAu_t_list[m] @ sig_mm_list[m] @ LAMBDAu_t_list[m].T
                            + LAMBDAs_t_list[m] @ GAMMAu_list[m] @ sig_qm_list[m] @ LAMBDAu_t_list[m].T
                            + LAMBDAu_t_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T @ LAMBDAs_t_list[m].T)
                        Ft = 0.5*(Ft+Ft.T)
                        Xit = LAMBDAs_t_list[m] @ Phat + LAMBDAu_t_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T
                        sol = Xit.T @ invert_matrix(Ft)
                        At_list[m] = alphahat + sol @ nut 
                        Pt_list[m] = Phat - sol @ Xit
                    
                        At_mat_list[m][t,:] = At_list[m].T
                        Pt_mat_list[m][t,:] = Pt_list[m].reshape((1, (Nq_list[m]*(p_list[m]+1))**2), order = "F")
                else:
                    if (t+1+T0_list[m])/freq_ratio_list[m] - np.floor((t+T0_list[m]+1)/freq_ratio_list[m]) == 0:
                        At1 = At_list[m]
                        Pt1 = Pt_list[m]
                        # Forecasting
                        alphahat = GAMMAs_list[m] @ At1 + GAMMAz_list[m] @ Zm_list[m][t,:] + GAMMAc_list[m][:,0]
                        Phat = GAMMAs_list[m] @ Pt1 @ GAMMAs_list[m].T + GAMMAu_list[m] @ sig_qq_list[m] @ GAMMAu_list[m].T
                        Phat = 0.5*(Phat + Phat.T)
                        yhat = LAMBDAs_list[m] @ alphahat + LAMBDAz_list[m] @ Zm_list[m][t,:] + LAMBDAc_list[m][:,0]
                        nut = Yq_list[m][t,:] - yhat  
                        Ft = (LAMBDAs_list[m] @ Phat @ LAMBDAs_list[m].T + LAMBDAu_list[m] @ sig_mm_list[m] @ LAMBDAu_list[m].T
                            + LAMBDAs_list[m] @ GAMMAu_list[m] @ sig_qm_list[m] @ LAMBDAu_list[m].T
                            + LAMBDAu_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T @ LAMBDAs_list[m].T)
                        Ft = 0.5*(Ft+Ft.T)
                        Xit = LAMBDAs_list[m] @ Phat + LAMBDAu_list[m] @ sig_mq_list[m] @ GAMMAu_list[m].T
                        sol = Xit.T @ invert_matrix(Ft)
                        At_list[m] = alphahat + sol @ nut
                        Pt_list[m] = Phat - sol @ Xit
                        At_mat_list[m][t,:] = At_list[m].T
                        Pt_mat_list[m][t,:] = Pt_list[m].reshape((1, (Nq_list[m]*(p_list[m]+1))**2), order = "F")
                    else:
                        At1 = At_list[m]
                        Pt1 = Pt_list[m]
                        # Forecasting
                        alphahat = GAMMAs_list[m] @ At1 + GAMMAz_list[m] @ Zm_list[m][t,:] + GAMMAc_list[m][:,0]
                        Phat = GAMMAs_list[m] @ Pt1 @ GAMMAs_list[m].T + GAMMAu_list[m] @ sig_qq_list[m] @ GAMMAu_list[m].T
                        Phat = 0.5*(Phat + Phat.T)
                        At_list[m] = alphahat 
                        Pt_list[m] = Phat 
                        At_mat_list[m][t,:] = At_list[m].T
                        Pt_mat_list[m][t,:] = Pt_list[m].reshape((1, (Nq_list[m]*(p_list[m]+1))**2), order = "F")
            Atildemat_list[m][j,:] = At_mat_list[m][nobs_list[m]-1,:]
            
            if j == 0:
                Pt_last_list.append(Pt_mat_list[m][nobs_list[m]-1,:].reshape((Nq_list[m]*(p_list[m]+1),Nq_list[m]*(p_list[m]+1)), order = "F"))
            else:
                Pt_last_list[m] = Pt_mat_list[m][nobs_list[m]-1,:].reshape((Nq_list[m]*(p_list[m]+1),Nq_list[m]*(p_list[m]+1)), order = "F")
            Ptildemat_list[m][j,:,:] = Pt_last_list[m]
            
            ########## Unbalanced Dataset ###########
            
            kn = nv_list[m]*(p_list[m]+1)
            
            # Measurement Equation
            
            Z1 = np.zeros((Nm_list[m], kn))
            Z1[:,:Nm_list[m]] = np.eye(Nm_list[m])
            
            Z2 = np.zeros((Nq_list[m], kn))
            
            for bb in range(Nq_list[m]):
                for ll in range(freq_ratio_list[m]):
                    if self.temp_agg == "mean":
                        Z2[bb, (ll+1)*Nm_list[m]+ll*Nq_list[m]+bb] = 1/freq_ratio_list[m]
                    if self.temp_agg == "sum":
                        Z2[bb, (ll+1)*Nm_list[m]+ll*Nq_list[m]+bb] = 1
            
            ZZ = np.vstack((Z1,Z2))
            
            if Ym_list[m].size:
                BAt = (np.concatenate((Ym_list[m][-1,:], np.atleast_1d(np.squeeze(Atildemat_list[m][j,:Nq_list[m]])))))
                for rr in range(1,p_list[m]+1):
                    BAt = np.concatenate((BAt,np.concatenate((Ym_list[m][-(rr+1),:], np.atleast_1d(np.squeeze(Atildemat_list[m][j,rr*Nq_list[m]:(rr+1)*Nq_list[m]]))))))
            
            else: 
                BAt = np.atleast_1d(np.squeeze(Atildemat_list[m][j,:Nq_list[m]]))
                for rr in range(1,p_list[m]+1):
                    BAt = np.concatenate((BAt,np.atleast_1d(np.squeeze(Atildemat_list[m][j,rr*Nq_list[m]:(rr+1)*Nq_list[m]]))))
            
            BPt = np.zeros((kn,kn))
            for rr in range(p_list[m]+1):
                for vv in range(p_list[m]+1):
                    BPt[(rr+1)*Nm_list[m]+rr*Nq_list[m]:(rr+1)*(Nm_list[m]+Nq_list[m]), (vv+1)*Nm_list[m]+vv*Nq_list[m]:(vv+1)*(Nm_list[m]+Nq_list[m])] = np.squeeze(
                        Ptildemat_list[m][j,rr*Nq_list[m]:(rr+1)*Nq_list[m],vv*Nq_list[m]:(vv+1)*Nq_list[m]])
                    
            BAt_mat = np.zeros((Tnobs_list[m], kn))
            BPt_mat = np.zeros((Tnobs_list[m], kn**2))
            
            
            BAt_mat[nobs_list[m]-1,:] = BAt
            BPt_mat[nobs_list[m]-1,:] = BPt.reshape((1,kn**2), order = "F")
            
            # Define companion form matrix PHIF
            PHIF = np.zeros((kn,kn))
            IF = np.eye(nv_list[m])
            for i in range(p_list[m]):
                PHIF[(i+1)*nv_list[m]:(i+2)*nv_list[m], i*nv_list[m]:(i+1)*nv_list[m]] = IF
                
            PHIF[:nv_list[m], :nv_list[m]*p_list[m]] = Phi_list[m][:-1,:].T
            
            # Define constant term CONF
            CONF = np.hstack((Phi_list[m][-1,:].T, np.zeros((nv_list[m]*p_list[m]))))
            
            # Define covariance term SIGF
            SIGF = np.zeros((kn,kn))
            SIGF[:nv_list[m],:nv_list[m]] = sigma_list[m]
            
            # Filter Loop
            ###################
            
            for t in range(nobs_list[m], Tnobs_list[m]):
                
                #new indicator
                kkk = t-nobs_list[m]
                
                # Define new data (ND) and new Z matrix (NZ)
                ND = YDATA_list[m][nobs_list[m]+T0_list[m]+kkk,:][~np.isnan(YDATA_list[m][nobs_list[m]+T0_list[m]+kkk,:])]
                NZ = ZZ[~index_NY_list[m][:,kkk],:]
                
                BAt1 = BAt
                BPt1 = BPt
                
                # Forecasting
                Balphahat = PHIF @ BAt1 + CONF
                BPhat = PHIF @ BPt1 @ PHIF.T + SIGF
                BPhat = 0.5 * (BPhat+BPhat.T)
                
                Byhat = NZ @ Balphahat
                Bnut = ND - Byhat
                
                BFt = NZ @ BPhat @ NZ.T
                BFt = 0.5*(BFt+BFt.T)
                
                #Updating
                sol_1 = (BPhat @ NZ.T) @ invert_matrix(BFt)
                BAt = Balphahat + sol_1 @ Bnut
                BPt = BPhat - sol_1 @ (BPhat @ NZ.T).T
                BAt_mat[t,:] = BAt
                BPt_mat[t,:] = BPt.reshape((1,kn**2), order = "F")
                
                
            AT_draw = np.zeros((Tnew_list[m]+1, kn))
            
            # Draw from multivariate normal
            Pchol = cholcovOrEigendecomp(BPt_mat[Tnobs_list[m]-1,:].reshape((kn, kn), order = "F"))
            AT_draw[-1, :] = BAt_mat[Tnobs_list[m]-1,:]+np.transpose(Pchol @ np.random.standard_normal(kn))
            
            # Kalman Smoother
            #####################
            
            for i in range(Tnew_list[m]):
                
                BAtt = BAt_mat[Tnobs_list[m]-(i+2),:]
                BPtt = BPt_mat[Tnobs_list[m]-(i+2),:].reshape((kn,kn), order = "F")
                
                BPhat = PHIF @ BPtt @ PHIF.T + SIGF
                BPhat = 0.5*(BPhat+BPhat.T)
                
                inv_BPhat = invert_matrix(BPhat)
                
                Bnut = AT_draw[-(i+1),:]- PHIF @ BAtt - CONF
                
                Amean = BAtt + (BPtt @ PHIF.T) @ inv_BPhat @ Bnut
                Pmean = BPtt - (BPtt @ PHIF.T) @ inv_BPhat @ np.transpose(BPtt @ PHIF.T)        
        
                #Draw from multivariate normal
                Pmchol = cholcovOrEigendecomp(Pmean)
                AT_draw[-2-i,:] = np.transpose(Amean + Pmchol @ np.random.standard_normal(kn))
                
            ########## Balanced Dataset ###########
            
            At_draw = np.zeros((nobs_list[m], Nq_list[m] * (p_list[m]+1)))
            for kk in range(p_list[m]+1):
                At_draw[nobs_list[m]-1, kk * Nq_list[m]:(kk+1)*+Nq_list[m]] = AT_draw[0,(kk+1)*Nm_list[m] + kk*Nq_list[m]:(kk+1)*(Nm_list[m]+Nq_list[m])]
                
        
            # Kalman Smoother
            #####################
            
            # Fallback: use filtered covariance in case smoother loop runs 0
            # iterations (nobs_list[m] == 1).  The smoother overwrites this
            # with the proper smoothed covariance when it does run.
            Pmean = Pt_list[m].copy()

            for i in range(nobs_list[m]-1):
                Att = At_mat_list[m][nobs_list[m]-(i+2),:]#[:, np.newaxis]
                Ptt = Pt_mat_list[m][nobs_list[m]-(i+2),:].reshape(Nq_list[m]*(p_list[m]+1), Nq_list[m]*(p_list[m]+1), order = "F")
                
                
                Phat = GAMMAs_list[m] @ Ptt @ GAMMAs_list[m].T + GAMMAu_list[m] @ sig_qq_list[m] @ GAMMAu_list[m].T
                
                Phat = 0.5*(Phat + Phat.T)
                
                inv_Phat = invert_matrix(Phat)
                
                nut = At_draw[nobs_list[m]-(i+1), :] - GAMMAs_list[m] @ Att - GAMMAz_list[m] @ Zm_list[m][nobs_list[m]-1-(i+1)] - GAMMAc_list[m][:,0]

                
                temp = Ptt @ GAMMAs_list[m].T
                Amean = Att + temp @ inv_Phat @ nut
                Pmean = Ptt - temp @ inv_Phat @ np.transpose(temp)
                
                Pmchol = cholcovOrEigendecomp(Pmean)
                At_draw[nobs_list[m]-1-(i+1), :] = np.transpose(Amean + Pmchol @ np.random.standard_normal(Nq_list[m]*(p_list[m]+1)))
            
        
        
            # Minesota Prior                            
            ########################
            if Ym_list[m].size:
                YY = np.vstack((np.hstack((Ym_list[m], At_draw[:,:Nq_list[m]])), AT_draw[1:,:(Nm_list[m]+Nq_list[m])]))
            else:
                YY = np.vstack((At_draw[:,:Nq_list[m]], AT_draw[1:,:(Nm_list[m]+Nq_list[m])]))
        
            
            #lstate = np.zeros((Nq_list[m], Tnobs_list[m]))
            #for hh in range(Nq_list[m]):
            #    lstate[hh, :nobs_list[m]] = At_draw[:, hh]
            #    lstate[hh, nobs_list[m]:] = AT_draw[1:, Nm_list[m]+hh]
        
            if (j%self.thining == 0 and m == (len(YMh_list)-1)):
                if lstate_list:
                    for hh in range(Nq_list[m]):
                        lstate_list[0][int(int((j)/self.thining)), hh, :nobs_list[m]] = At_draw[:, hh]
                        lstate_list[0][int(int((j)/self.thining)), hh, nobs_list[m]:] = AT_draw[1:, Nm_list[m]+hh]
                else:
                    lstate_list.append(np.zeros((math.ceil((self.nsim)/self.thining),Nq_list[0],int(Tnobs_list[0]))))
                    for hh in range(Nq_list[m]):
                        lstate_list[0][int(int((j)/self.thining)), hh, :nobs_list[m]] = At_draw[:, hh]
                        lstate_list[0][int(int((j)/self.thining)), hh, nobs_list[m]:] = AT_draw[1:, Nm_list[m]+hh]
            
            nobs_ = np.shape(YY)[0] - T0_list[m]
            spec = np.hstack((nlags_list_[m], T0_list[m], self.nex, nv_list[m], nobs_))
        
        
            # dummy observations and actual observations
            if return_mdd:
                mdd_list[m], YYact, YYdum, XXact, XXdum = mdd_(self.hyp[m], YY, spec)
            else:
                YYact, YYdum, XXact, XXdum = calc_yyact(self.hyp[m], YY, spec)
            
            if (j%self.thining == 0 and m == (len(YMh_list)-1)):
                # With ragged-edge masking, YYact may be shorter than expected
                tail_len = min(freq_ratio_list[m] + 1, YYact.shape[0])

                if YYactsim_list:
                    YYactsim_list[0][int(int((j)/self.thining)), :, :] = np.nan
                    XXactsim_list[0][int(int((j)/self.thining)), :, :] = np.nan
                    if tail_len > 0:
                        YYactsim_list[0][int(int((j)/self.thining)), -tail_len:, :] = YYact[-tail_len:, :]
                        XXactsim_list[0][int(int((j)/self.thining)), -tail_len:, :] = XXact[-tail_len:, :]
                else:
                    YYactsim_list.append(np.full((math.ceil((self.nsim)/self.thining), freq_ratio_list[0]+1, nv_list[0]), np.nan))
                    XXactsim_list.append(np.full((math.ceil((self.nsim)/self.thining), int(freq_ratio_list[0])+1, int(nv_list[0])*int(p_list[0])+1), np.nan))
                    if tail_len > 0:
                        YYactsim_list[0][int(int((j)/self.thining)), -tail_len:, :] = YYact[-tail_len:, :]
                        XXactsim_list[0][int(int((j)/self.thining)), -tail_len:, :] = XXact[-tail_len:, :]
            # Draws from posterior distribution

            Tdummy, n = YYdum.shape
            n = int(n)
            Tdummy = int(Tdummy)
            Tobs, n = YYact.shape

            if Tobs == 0:
                warnings.warn("Skipping iteration: no valid VAR rows after ragged-edge masking.")
                if j == 0:
                    At_draw_list.append(At_draw)
                    Pmean_list.append(Pmean)
                continue

            X = np.vstack((XXact, XXdum))
            Y = np.vstack((YYact, YYdum))
            p = int(spec[0])
            T = Tobs + Tdummy

            if np.isnan(X).any() or np.isnan(Y).any():
                warnings.warn("Skipping iteration: regression matrices contain NaNs after ragged-edge masking.")
                if j == 0:
                    At_draw_list.append(At_draw)
                    Pmean_list.append(Pmean)
                continue

            if np.isinf(X).any() or np.isinf(Y).any():
                warnings.warn("Skipping iteration: regression design contains inf values.")
                if j == 0:
                    At_draw_list.append(At_draw)
                    Pmean_list.append(Pmean)
                continue

            # Need positive degrees of freedom for inverse-Wishart draw.
            df_sigma = T - n * p - 1
            if df_sigma <= 0:
                warnings.warn(
                    f"Skipping iteration: insufficient effective sample after ragged-edge masking: "
                    f"T={T}, n={n}, p={p}, df={df_sigma}. Need T > n*p + 1."
                )
                if j == 0:
                    At_draw_list.append(At_draw)
                    Pmean_list.append(Pmean)
                continue

            F = np.zeros((int(n*p), int(n*p)))
            I = np.eye(n)

            for i in range(p-1):
                F[(i+1)*n:(i+2)*n, i*n:(i+1)*n] = I
    
            vl, d, vr = np.linalg.svd(X, full_matrices=False)
            vr = vr.T
            di = 1/d
            B = vl.T @ Y
            xxi = (vr * np.tile(di.T,(n*p+1,1))) 
            inv_x = xxi @ xxi.T
            Phi_tilde = xxi @ B
    
            Sigma = (Y - X @ Phi_tilde).T @ (Y - X @ Phi_tilde)
            
            sigma = invwishart.rvs(scale = Sigma, df = T-n*p-1)
            # Draws from the density Sigma | Y

            # IMPORTANT: Explosive VAR check is necessary and NOT redundant
            # This implements a stationarity constraint (truncated prior) that excludes
            # explosive parameter regions. The MH algorithm (backward pass) does NOT
            # automatically reject explosive VARs because:
            # 1. MH acceptance uses only the likelihood ratio (ll_prop - ll_cur)
            # 2. Explosive VARs can have high likelihood on observed data
            # 3. Without this check, unstable parameters could be accepted
            # This constraint ensures all sampled parameters satisfy stationarity
            # conditions required for valid VAR forecasting and inference.
            sigma_chol = cholcovOrEigendecomp(np.kron(sigma, inv_x))

            if check_explosive:
                attempts = 0
                while attempts < max_it_stable:
                    phi_new = np.squeeze(Phi_tilde.reshape(n*(n*p+1), 1, order="F")) + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0])
                    Phi = phi_new.reshape(n*p+1, n, order = "F")
                    if not is_explosive(Phi, n, p):
                        break
                    attempts += 1
                stability_rejected += attempts
                stability_proposals += attempts + (1 if attempts < max_it_stable else 0)
                if attempts == max_it_stable:
                    explosive_counter += 1
                    print(f"Explosive VAR detected {explosive_counter} times.")
                    m = 0
                    if j == 0:
                        j -= 1
                        At_draw_list.append(At_draw)
                        Pmean_list.append(Pmean)
                    continue
            else:
                # Skip explosive VAR check when check_explosive is False
                phi_new = np.squeeze(Phi_tilde.reshape(n*(n*p+1), 1, order="F")) + sigma_chol @ np.random.standard_normal(sigma_chol.shape[0])
                Phi = phi_new.reshape(n*p+1, n, order = "F")
                
            #while loop bis hier
            
            if j > 0:
                Phi_list[m] = Phi
            elif j == 0 and m == 0:
                Phi_list[m] = Phi
            
            if (j % self.thining == 0):
                j_temp = int(j/self.thining)
                Sigmap_list[m][j_temp,:,:] = sigma
                Phip_list[m][j_temp,:,:]   = Phi
                Cons_list[m][j_temp,:]     = Phi[-1,:]
                if (m == len(YMh_list)-1):
                    valid_draws.append(j_temp)
            
            # Define phi(qm), phi(qq), phi(qc)
            phi_qm = np.zeros((Nm_list[m]*p,Nq_list[m]))
            for i in range(p_list[m]):
                phi_qm[Nm_list[m]*i:Nm_list[m]*(i+1), :] = Phi[i*(Nm_list[m]+Nq_list[m]):i*(Nm_list[m]+Nq_list[m])+Nm_list[m],Nm_list[m]:]
                
            phi_qq = np.zeros((Nq_list[m]*p,Nq_list[m]))
            for i in range(p):
                phi_qq[Nq_list[m]*i:Nq_list[m]*(i+1), :] = Phi[i*(Nm_list[m]+Nq_list[m])+Nm_list[m]:(i+1)*(Nm_list[m]+Nq_list[m]), Nm_list[m]:]
                
            phi_qc = Phi[-1,Nm_list[m]:, np.newaxis]
            
            # Define  phi(mm), phi(mq), phi(mc)
            phi_mm = np.zeros((Nm_list[m]*p, Nm_list[m]))
            for i in range(p):
                phi_mm[Nm_list[m]*i:Nm_list[m]*(i+1),:] = Phi[i*(Nm_list[m]+Nq_list[m]):i*(Nm_list[m]+Nq_list[m])+Nm_list[m],:Nm_list[m]]
        
            phi_mq = np.zeros((Nq_list[m]*p, Nm_list[m]))
            for i in range(p):
                phi_mq[Nq_list[m]*i:Nq_list[m]*(i+1),:] = Phi[i*(Nm_list[m]+Nq_list[m])+Nm_list[m]:(i+1)*(Nm_list[m]+Nq_list[m]),:Nm_list[m]]      
            
            phi_mc = Phi[-1,:Nm_list[m], np.newaxis]
        
            # Define Covariance Term sig_mm, sig_mq, sig_qm, sig_qq
            if Nm_list[m]:
                sig_mm_list[m] = sigma[:Nm_list[m],:Nm_list[m]]  
                sig_mq_list[m] = 0.5*(sigma[:Nm_list[m],Nm_list[m]:]+np.transpose(sigma[Nm_list[m]:,:Nm_list[m]]))
                sig_qm_list[m] = 0.5*(sigma[Nm_list[m]:,:Nm_list[m]]+np.transpose(sigma[:Nm_list[m],Nm_list[m]:]))
                sig_qq_list[m] = sigma[Nm_list[m]:,Nm_list[m]:]
            else:
                sig_qq_list[m] =  np.atleast_2d(sigma)
            
            
            # Define Transition Equation Matrices
            GAMMAs_list[m] = np.vstack((np.hstack((np.transpose(phi_qq), np.zeros((Nq_list[m],Nq_list[m])))), np.hstack((np.eye(p*Nq_list[m]), np.zeros((p*Nq_list[m],Nq_list[m]))))))
            GAMMAz_list[m] = np.vstack((np.transpose(phi_qm), np.zeros((p*Nq_list[m], p*Nm_list[m]))))
            GAMMAc_list[m] = np.vstack((phi_qc, np.zeros((p*Nq_list[m],1))))
            GAMMAu_list[m] = np.vstack((np.eye(Nq_list[m]), np.zeros((p*Nq_list[m],Nq_list[m]))))
            
            # Define Measurment equation Matrices
            if self.temp_agg == "mean":
                LAMBDAs_list[m] = np.vstack((np.hstack((np.zeros((Nm_list[m],Nq_list[m])), np.transpose(phi_mq))),1/freq_ratio_list[m]*np.hstack((np.tile(np.eye(Nq_list[m]), freq_ratio_list[m]), np.zeros((Nq_list[m],Nq_list[m]*(p-(freq_ratio_list[m]-1))))))))
            if self.temp_agg == "sum":
                LAMBDAs_list[m] = np.vstack((np.hstack((np.zeros((Nm_list[m],Nq_list[m])), np.transpose(phi_mq))),np.hstack((np.tile(np.eye(Nq_list[m]), freq_ratio_list[m]), np.zeros((Nq_list[m],Nq_list[m]*(p-(freq_ratio_list[m]-1))))))))
                
            LAMBDAz_list[m] = np.vstack((np.transpose(phi_mm), np.zeros((Nq_list[m], p*Nm_list[m]))))
            LAMBDAc_list[m] = np.vstack((phi_mc, np.zeros((Nq_list[m],1))))
            LAMBDAu_list[m] = np.vstack((np.eye(Nm_list[m]), np.zeros((Nq_list[m],Nm_list[m]))))
            
            # Define W matrix
            Wmatrix_list[m] = np.hstack((np.eye(Nm_list[m]), np.zeros((Nm_list[m], Nq_list[m]))))
            LAMBDAs_t_list[m] = np.matmul(Wmatrix_list[m],LAMBDAs_list[m])
            LAMBDAz_t_list[m] = np.matmul(Wmatrix_list[m], LAMBDAz_list[m])
            LAMBDAc_t_list[m] = np.matmul(Wmatrix_list[m], LAMBDAc_list[m])
            LAMBDAu_t_list[m] = np.matmul(Wmatrix_list[m], LAMBDAu_list[m])
            # now we need to define the new low frequency data as the temporally disaggregated
            # low frequency data of the current iteration
            
            #get relevant high frequency variables, so that the get used in the next var
            
            if not(var_of_interest is None):
                idx_var_of_interest_m = list(filter(lambda x: YMX_list[m].columns.tolist()[x] in var_of_interest, range(len(YMX_list[m].columns.tolist()))))
            
            if m < (len(YMh_list)-1):
                if j == 0:
                    #Yq_list.append((np.kron(lstate, np.ones((1,freq_ratio_list[m+1])))).T)
                    #YQ0_list.append(YYact)#YQ0_list.append(YYact[:,-Nq_list[m+1]:])
                    if var_of_interest is None:
                        YQ0_list.append(YYact)
                    else:
                        idx_vars = np.concatenate((np.array(idx_var_of_interest_m) , (YM_list[m].shape[1]+np.array(idx_var_of_interest))))
                        YQ0_list.append(YYact[:,np.int_(idx_vars)].reshape(-1, len(idx_vars)))
                        #we also need to update nv_list and Nq_lsit
                        nv_list[m + 1] = len(idx_vars) + YM0_list[m+1].shape[1]
                        Nq_list[m + 1] = len(idx_vars)
                    YQ_list.append(np.kron(YQ0_list[m+1], np.ones((freq_ratio_list[m+1],1))))#[np.prod(np.array(nlags_list_[:(m+2)])):,:])
                    #Yq_list.append(YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                    if YM_list[m].size:
                        YM_list[m+1] = YM_list[m+1][2*np.prod(np.array(nlags_list_[:(m+2)])):,:]
                    else:
                        YM_list[m+1] = YM_list[m+1][2*nlags_list_[(m+1)]:,:]
                    
                    T_list.append(YQ_list[m+1].shape[0])
                    
                    Tstar_list.append(YM_list[m+1].shape[0])

                    Tnew_list.append(Tstar_list[m+1]-T_list[m+1])

                    
                    YDATA_list.append(np.full((Tstar_list[m+1],nv_list[m+1]), np.nan))
                    YDATA_list[m+1][:,:Nm_list[m+1]] = YM_list[m+1]#[-Tstar_list[m+1]:,:]
                    
                    T0_list.append(int(nlags_list_[m+1]))
                    p_list.append(int(nlags_list_[m+1]))
                    nlags_list.append(p_list[m+1])
                    kq_list.append(Nq_list[m+1] * p_list[m+1])
                    nobs_list.append(int(T_list[m+1]) - int(T0_list[m+1]))
                    Tnobs_list.append(Tstar_list[m+1]-T0_list[m+1])
                    
                
                    
                    #for writing to a forecast w/ history file
                    #YMh_list[m+1] = YMh_list[m+1][int(T0_list[m+1]):-int(freq_ratio_list[m+1]),:]
                    if YM_list[m].size:
                        YMh_list[m+1] = YMh_list[m+1][2*np.prod(np.array(nlags_list_[:(m+2)]))+int(T0_list[m+1]):-int(freq_ratio_list[m+1]),:]
                    varstxt_list.append(np.hstack((YMX_list[m+1].columns, YQX_list[0].columns)))
                    smpltxt_list.append(YMX_list[m+1].index[int(T0_list[m+1]):])
                    
                    index_NY_list.append(np.isnan(YDATA_list[m+1][nobs_list[m+1]+T0_list[m+1]:Tnobs_list[m+1]+T0_list[m+1],:]).T) # TODO CHECK
                    
                    # Parameter estimation
                    # Matrices for collecting draws from Posterior Density
                    Sigmap_list.append(np.zeros((math.ceil((self.nsim)/self.thining),nv_list[m+1],nv_list[m+1])))
                    Phip_list.append(np.zeros((math.ceil((self.nsim)/self.thining),int(nv_list[m+1])*int(p_list[m+1])+1,int(nv_list[m+1]))))
                    Cons_list.append(np.zeros((math.ceil((self.nsim)/self.thining),nv_list[m+1])))
                    if m == (len(YMh_list)-2):
                        lstate_list.append(np.zeros((math.ceil((self.nsim)/self.thining),Nq_list[m+1],int(Tnobs_list[m+1]))))
                        YYactsim_list.append(np.zeros((math.ceil((self.nsim)/self.thining),freq_ratio_list[m+1]+1,nv_list[m+1])))
                        XXactsim_list.append(np.zeros((math.ceil((self.nsim)/self.thining),int(freq_ratio_list[m+1])+1,int(nv_list[m+1])*int(p_list[m+1])+1)))
                    
                    At_mat_list.append(np.zeros((int(Tnobs_list[m+1]), Nq_list[m+1]*(int(p_list[m+1])+1))))
                    Pt_mat_list.append(np.zeros((int(Tnobs_list[m+1]), (Nq_list[m+1]*(int(p_list[m+1])+1))**2)))
                    Atildemat_list.append(np.zeros((self.nsim, Nq_list[m+1]*(int(p_list[m+1])+1))))
                    Ptildemat_list.append(np.zeros((self.nsim, Nq_list[m+1]*(int(p_list[m+1])+1),Nq_list[m+1]*(int(p_list[m+1])+1))))
                    loglh_list.append(0)
                    counter_list.append(0)
                    
                    # Define phi, phi(mm), phi(mq), phi(mc) used in alt ss rep -- eq (9)
                    phi_mm_list.append(np.zeros((int(Nm_list[m+1])*int(p_list[m+1]),int(Nm_list[m+1]))))
                    phi_mm_list[m+1][:Nm_list[m+1],:Nm_list[m+1]] = np.eye(Nm_list[m+1])
                    
                    phi_mq_list.append(np.zeros((Nq_list[m+1]*int(p_list[m+1]),int(Nm_list[m+1]))))
                    phi_mc_list.append(np.zeros((1,Nm_list[m+1])))
                    phi_qc_list.append(np.zeros((1,Nq_list[m+1])))
                    
                    Phi_list.append(np.vstack((0.95 * np.eye(int(Nm_list[m+1])+Nq_list[m+1]), np.zeros(((int(Nm_list[m+1])+Nq_list[m+1])*(int(p_list[m+1])-1)+1, int(Nm_list[m+1])+Nq_list[m+1])))))
                    
                    # Define Transition Equation Matrices in eq (10)
                    GAMMAs_list.append(np.zeros((Nq_list[m+1]*(int(p_list[m+1])+1), Nq_list[m+1] * (int(p_list[m+1])+1))))
                    IQ = np.eye(Nq_list[m+1])
                    
                    for k in range(int(p_list[m+1])):
                        GAMMAs_list[m+1][(k+1)*Nq_list[m+1]:(k+2)*Nq_list[m+1], k*Nq_list[m+1]:(k+1)*Nq_list[m+1]] = IQ
                        
                    GAMMAs_list[m+1][:Nq_list[m+1],:Nq_list[m+1]] = 0.95 * np.eye(Nq_list[m+1])
                    
                    GAMMAz_list.append(np.zeros((Nq_list[m+1]* (int(p_list[m+1])+1), int(Nm_list[m+1])*p_list[m+1])))
                    GAMMAc_list.append(np.zeros((Nq_list[m+1]*(p_list[m+1]+1), 1)))
                    GAMMAu_list.append(np.vstack((np.eye(Nq_list[m+1]), np.zeros((p_list[m+1]*Nq_list[m+1],Nq_list[m+1])))))

                    
                    if self.temp_agg == "mean":
                        LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[m+1],Nq_list[m+1])), np.transpose(phi_mq_list[m+1]))),1/freq_ratio_list[m+1]*np.hstack((np.tile(np.eye(Nq_list[m+1]), freq_ratio_list[m+1]), np.zeros((Nq_list[m+1],Nq_list[m+1]*(p_list[m+1]-(freq_ratio_list[m+1]-1)))))))))
                    if self.temp_agg == "sum":
                        LAMBDAs_list.append(np.vstack((np.hstack((np.zeros((Nm_list[m+1],Nq_list[m+1])), np.transpose(phi_mq_list[m+1]))),np.hstack((np.tile(np.eye(Nq_list[m+1]), freq_ratio_list[m+1]), np.zeros((Nq_list[m+1],Nq_list[m+1]*(p_list[m+1]-(freq_ratio_list[m+1]-1)))))))))
                        
                            
                    LAMBDAz_list.append(np.vstack((np.transpose(phi_mm_list[m+1]), np.zeros((Nq_list[m+1], p_list[m+1]*Nm_list[m+1])))))
                    LAMBDAc_list.append(np.vstack((np.transpose(phi_mc_list[m+1]), np.zeros((Nq_list[m+1],1)))))
                    LAMBDAu_list.append(np.vstack((np.eye(Nm_list[m+1]), np.zeros((Nq_list[m+1],Nm_list[m+1])))))
                    
                    # Define Covariance Terms sig_mm, sig_mq_sig_qm, siq_qq
                    sigma_list.append((1e-4)*np.eye(Nm_list[m+1]+Nq_list[m+1]))
                    sig_mm_list.append(sigma_list[m+1][:Nm_list[m+1],:Nm_list[m+1]])
                    sig_mq_list.append(sigma_list[m+1][:Nm_list[m+1], Nm_list[m+1]:])
                    sig_qm_list.append(sigma_list[m+1][Nm_list[m+1]:, :Nm_list[m+1]])
                    sig_qq_list.append(sigma_list[m+1][Nm_list[m+1]:,Nm_list[m+1]:])
                    
                    # Define W matrix in eq (15) -- _t for tilde
                    Wmatrix_list.append(np.hstack((np.eye(Nm_list[m+1]), np.zeros((Nm_list[m+1],Nq_list[m+1])))))
                    LAMBDAs_t_list.append(Wmatrix_list[m+1] @ LAMBDAs_list[m+1])
                    LAMBDAz_t_list.append(Wmatrix_list[m+1] @ LAMBDAz_list[m+1])
                    LAMBDAc_t_list.append(Wmatrix_list[m+1] @ LAMBDAc_list[m+1])
                    LAMBDAu_t_list.append(Wmatrix_list[m+1] @ LAMBDAu_list[m+1])
                    
                    At_list.append(np.zeros((Nq_list[m+1]*(p_list[m+1]+1))))
                    Pt_list.append(np.zeros((Nq_list[m+1]*(p_list[m+1]+1), Nq_list[m+1]*(p_list[m+1]+1))))

                    # Solve the discrete Lyapunov equation exactly
                    Qm = GAMMAu_list[m+1] @ sig_qq_list[m+1] @ GAMMAu_list[m+1].T
                    Pt_list[m+1] = solve_discrete_lyapunov(GAMMAs_list[m+1], Qm)
                    
                    #YM_short = YM_list[m+1][2*np.prod(np.array(nlags_list_[:(m+2)])):,:]
                    
                    # Lagged HF observations
                    Zm_list.append(np.zeros((nobs_list[m+1], Nm_list[m+1]*p_list[m+1])))
                    for k in range(p_list[m+1]):
                        Zm_list[m+1][:, k * Nm_list[m+1]:(k+1)*Nm_list[m+1]] = YM_list[m+1][T0_list[m+1]-(k+1):T0_list[m+1]+nobs_list[m+1]-(k+1),:]
                    # Observations in Monthly Freq
                    
                    Ym_list.append(YM_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                    Yq_list.append(YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:])
                    
                    
                else:
                    #Yq_list[m+1] = (np.kron(lstate, np.ones((1,freq_ratio_list[m+1])))).T
                    if var_of_interest is None:
                        YQ0_list[m+1] = YYact
                    else:
                        idx_vars = np.concatenate((np.array(idx_var_of_interest_m) , (YM_list[m].shape[1]+np.array(idx_var_of_interest))))
                        YQ0_list[m+1] = YYact[:,np.int_(idx_vars)].reshape(-1, len(idx_vars))
                        
                    YQ_list[m+1] =np.kron(YQ0_list[m+1], np.ones((freq_ratio_list[m+1],1)))#[np.prod(np.array(nlags_list_[:(m+2)])):,:]
                    
                    Yq_list[m+1] = YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:]#YQ_list[m+1][T0_list[m+1]:nobs_list[m+1]+T0_list[m+1],:]
                    
                    
                    T_list[m+1] = YQ_list[m+1].shape[0]
                    Tnew_list[m+1] = Tstar_list[m+1]-T_list[m+1]
                    
                #if self.temp_agg == "mean":    
                #    YDATA_list[m+1][:T_list[m+1],Nm_list[m+1]:] = YQ_list[m+1]
                #if sself.temp_agg == "sum":    
                #    YDATA_list[m+1][:T_list[m+1],Nm_list[m+1]:] = 1/freq_ratio_list[m+1]* YQ_list[m+1]
                YDATA_list[m+1][:T_list[m+1],Nm_list[m+1]:] = YQ_list[m+1]    
            
            if j == 0:
                At_draw_list.append(At_draw)
            else:
                At_draw_list[m] = At_draw
            
            if j == 0:
                Pmean_list.append(Pmean)
            else:
                Pmean_list[m] = Pmean

        # ----------------------------------------------------------------
        # Metropolis-within-Gibbs backward correction (Version 2)
        #
        # After the full forward sweep, run a backward pass over blocks
        # m = 0, 1, ..., M_blks-2.  For each block _m we propose fresh
        # latent-state and parameter draws (from the same forward-pass
        # prior) and accept/reject based on the log-likelihood of block
        # _m+1's balanced-period data under the proposed vs current
        # downstream input Yq.  This introduces the cross-block feedback
        # that is absent from the plain Gibbs forward sweep.
        # ----------------------------------------------------------------
        for _m in range(M_blks - 1):
            mh_total_counts[_m] += 1
            try:

                # Current log-likelihood of block (_m+1) given current Yq_{m+1}
                _ll_cur = _kalman_filter_loglik_block(
                    GAMMAs_list[_m+1], GAMMAz_list[_m+1], GAMMAc_list[_m+1], GAMMAu_list[_m+1],
                    LAMBDAs_list[_m+1], LAMBDAz_list[_m+1], LAMBDAc_list[_m+1], LAMBDAu_list[_m+1],
                    LAMBDAs_t_list[_m+1], LAMBDAz_t_list[_m+1], LAMBDAc_t_list[_m+1], LAMBDAu_t_list[_m+1],
                    sig_qq_list[_m+1], sig_mm_list[_m+1], sig_mq_list[_m+1], sig_qm_list[_m+1],
                    _mh_At_init[_m+1], _mh_Pt_init[_m+1],
                    Zm_list[_m+1], Ym_list[_m+1], Yq_list[_m+1],
                    nobs_list[_m+1], T0_list[_m+1], freq_ratio_list[_m+1],
                )

                # ------------------------------------------------------------------
                # Proposal: re-draw block _m latent states and parameters.
                # We re-run the same smoother and parameter-draw code used in the
                # forward pass (with new random draws) to produce a fresh proposal.
                # ------------------------------------------------------------------
                _p_m = int(p_list[_m])
                _kn_m = nv_list[_m] * (_p_m + 1)

                # Reconstruct unbalanced-KF initial state from stored quantities
                if Ym_list[_m].size:
                    _BAt = np.concatenate((
                        Ym_list[_m][-1, :],
                        np.atleast_1d(np.squeeze(Atildemat_list[_m][j, :Nq_list[_m]]))
                    ))
                    for _rr in range(1, _p_m + 1):
                        _BAt = np.concatenate((_BAt, np.concatenate((
                            Ym_list[_m][-(1 + _rr), :],
                            np.atleast_1d(np.squeeze(
                                Atildemat_list[_m][j, _rr*Nq_list[_m]:(_rr+1)*Nq_list[_m]]
                            ))
                        ))))
                else:
                    _BAt = np.atleast_1d(np.squeeze(
                        Atildemat_list[_m][j, :Nq_list[_m]]
                    ))
                    for _rr in range(1, _p_m + 1):
                        _BAt = np.concatenate((_BAt, np.atleast_1d(np.squeeze(
                            Atildemat_list[_m][j, _rr*Nq_list[_m]:(_rr+1)*Nq_list[_m]]
                        ))))

                _BPt = np.zeros((_kn_m, _kn_m))
                for _rr in range(_p_m + 1):
                    for _vv in range(_p_m + 1):
                        _BPt[
                            (_rr+1)*Nm_list[_m]+_rr*Nq_list[_m]:(_rr+1)*(Nm_list[_m]+Nq_list[_m]),
                            (_vv+1)*Nm_list[_m]+_vv*Nq_list[_m]:(_vv+1)*(Nm_list[_m]+Nq_list[_m])
                        ] = np.squeeze(Ptildemat_list[_m][j,
                            _rr*Nq_list[_m]:(_rr+1)*Nq_list[_m],
                            _vv*Nq_list[_m]:(_vv+1)*Nq_list[_m]
                        ])

                # Build companion-form matrices from current block _m parameters
                _PHIF_m = np.zeros((_kn_m, _kn_m))
                _IF_m = np.eye(nv_list[_m])
                for _i in range(_p_m):
                    _PHIF_m[(_i+1)*nv_list[_m]:(_i+2)*nv_list[_m],
                             _i*nv_list[_m]:(_i+1)*nv_list[_m]] = _IF_m
                _PHIF_m[:nv_list[_m], :nv_list[_m]*_p_m] = Phi_list[_m][:-1, :].T
                _CONF_m = np.hstack((Phi_list[_m][-1, :].T, np.zeros((nv_list[_m] * _p_m,))))
                _SIGF_m = np.zeros((_kn_m, _kn_m))
                _SIGF_m[:nv_list[_m], :nv_list[_m]] = sigma_list[_m]

                # Measurement matrix for the unbalanced filter
                _Z1_m = np.zeros((Nm_list[_m], _kn_m))
                _Z1_m[:, :Nm_list[_m]] = np.eye(Nm_list[_m])
                _Z2_m = np.zeros((Nq_list[_m], _kn_m))
                for _bb in range(Nq_list[_m]):
                    for _lll in range(freq_ratio_list[_m]):
                        if self.temp_agg == "mean":
                            _Z2_m[_bb, (_lll+1)*Nm_list[_m]+_lll*Nq_list[_m]+_bb] = (
                                1.0 / freq_ratio_list[_m]
                            )
                        else:
                            _Z2_m[_bb, (_lll+1)*Nm_list[_m]+_lll*Nq_list[_m]+_bb] = 1.0
                _ZZ_m = np.vstack((_Z1_m, _Z2_m))

                # Re-run unbalanced forward filter (deterministic given current state)
                _BAt_mat_m = np.zeros((Tnobs_list[_m], _kn_m))
                _BPt_mat_m = np.zeros((Tnobs_list[_m], _kn_m ** 2))
                _BAt_mat_m[nobs_list[_m] - 1, :] = _BAt
                _BPt_mat_m[nobs_list[_m] - 1, :] = _BPt.reshape((1, _kn_m ** 2), order="F")
                _BAt_cur = _BAt.copy()
                _BPt_cur = _BPt.copy()
                for _t in range(nobs_list[_m], Tnobs_list[_m]):
                    _kkk = _t - nobs_list[_m]
                    _ND_m = YDATA_list[_m][nobs_list[_m]+T0_list[_m]+_kkk, :][
                        ~np.isnan(YDATA_list[_m][nobs_list[_m]+T0_list[_m]+_kkk, :])
                    ]
                    _NZ_m = _ZZ_m[~index_NY_list[_m][:, _kkk], :]
                    _Balphahat_m = _PHIF_m @ _BAt_cur + _CONF_m
                    _BPhat_m = _PHIF_m @ _BPt_cur @ _PHIF_m.T + _SIGF_m
                    _BPhat_m = 0.5 * (_BPhat_m + _BPhat_m.T)
                    _Byhat_m = _NZ_m @ _Balphahat_m
                    _Bnut_m = _ND_m - _Byhat_m
                    _BFt_m = _NZ_m @ _BPhat_m @ _NZ_m.T
                    _BFt_m = 0.5 * (_BFt_m + _BFt_m.T)
                    _sol_m = (_BPhat_m @ _NZ_m.T) @ invert_matrix(_BFt_m)
                    _BAt_cur = _Balphahat_m + _sol_m @ _Bnut_m
                    _BPt_cur = _BPhat_m - _sol_m @ (_BPhat_m @ _NZ_m.T).T
                    _BAt_mat_m[_t, :] = _BAt_cur
                    _BPt_mat_m[_t, :] = _BPt_cur.reshape((1, _kn_m ** 2), order="F")

                # Draw new unbalanced backward smoother trajectory (proposal)
                _AT_draw_p = np.zeros((Tnew_list[_m] + 1, _kn_m))
                _Pchol_p = cholcovOrEigendecomp(
                    _BPt_mat_m[Tnobs_list[_m]-1, :].reshape((_kn_m, _kn_m), order="F")
                )
                _AT_draw_p[-1, :] = (
                    _BAt_mat_m[Tnobs_list[_m]-1, :]
                    + np.transpose(_Pchol_p @ np.random.standard_normal(_kn_m))
                )
                for _i in range(Tnew_list[_m]):
                    _BAtt_p = _BAt_mat_m[Tnobs_list[_m] - (_i+2), :]
                    _BPtt_p = _BPt_mat_m[Tnobs_list[_m] - (_i+2), :].reshape(
                        (_kn_m, _kn_m), order="F"
                    )
                    _BPhat_p = _PHIF_m @ _BPtt_p @ _PHIF_m.T + _SIGF_m
                    _BPhat_p = 0.5 * (_BPhat_p + _BPhat_p.T)
                    _inv_BPhat_p = invert_matrix(_BPhat_p)
                    _Bnut_p = _AT_draw_p[-(_i+1), :] - _PHIF_m @ _BAtt_p - _CONF_m
                    _Amean_p = _BAtt_p + (_BPtt_p @ _PHIF_m.T) @ _inv_BPhat_p @ _Bnut_p
                    _Pmean_unb_p = (
                        _BPtt_p
                        - (_BPtt_p @ _PHIF_m.T) @ _inv_BPhat_p @ np.transpose(_BPtt_p @ _PHIF_m.T)
                    )
                    _Pmchol_p = cholcovOrEigendecomp(_Pmean_unb_p)
                    _AT_draw_p[-2-_i, :] = np.transpose(
                        _Amean_p + _Pmchol_p @ np.random.standard_normal(_kn_m)
                    )

                # Terminal draw for balanced period from unbalanced smoother
                _At_draw_p = np.zeros((nobs_list[_m], Nq_list[_m] * (_p_m + 1)))
                for _kk in range(_p_m + 1):
                    _At_draw_p[nobs_list[_m]-1, _kk*Nq_list[_m]:(_kk+1)*Nq_list[_m]] = (
                        _AT_draw_p[0,
                            (_kk+1)*Nm_list[_m]+_kk*Nq_list[_m]:(_kk+1)*(Nm_list[_m]+Nq_list[_m])
                        ]
                    )

                # Draw balanced backward smoother (proposal)
                _Pmean_prop = None
                for _i in range(nobs_list[_m] - 1):
                    _Att_p = At_mat_list[_m][nobs_list[_m] - (_i+2), :]
                    _Ptt_p = Pt_mat_list[_m][nobs_list[_m] - (_i+2), :].reshape(
                        Nq_list[_m]*(_p_m+1), Nq_list[_m]*(_p_m+1), order="F"
                    )
                    _Phat_p = (
                        GAMMAs_list[_m] @ _Ptt_p @ GAMMAs_list[_m].T
                        + GAMMAu_list[_m] @ sig_qq_list[_m] @ GAMMAu_list[_m].T
                    )
                    _Phat_p = 0.5 * (_Phat_p + _Phat_p.T)
                    _inv_Phat_p = invert_matrix(_Phat_p)
                    _nut_p = (
                        _At_draw_p[nobs_list[_m]-(_i+1), :]
                        - GAMMAs_list[_m] @ _Att_p
                        - GAMMAz_list[_m] @ Zm_list[_m][nobs_list[_m]-1-(_i+1)]
                        - GAMMAc_list[_m][:, 0]
                    )
                    _temp_p = _Ptt_p @ GAMMAs_list[_m].T
                    _Amean_bal_p = _Att_p + _temp_p @ _inv_Phat_p @ _nut_p
                    _Pmean_prop = _Ptt_p - _temp_p @ _inv_Phat_p @ _temp_p.T
                    _Pmchol_bal_p = cholcovOrEigendecomp(_Pmean_prop)
                    _At_draw_p[nobs_list[_m]-1-(_i+1), :] = np.transpose(
                        _Amean_bal_p
                        + _Pmchol_bal_p @ np.random.standard_normal(Nq_list[_m]*(_p_m+1))
                    )

                if _Pmean_prop is None:
                    # nobs == 1: balanced smoother did not run; fall back to filtered covariance
                    _Pmean_prop = Pt_mat_list[_m][0, :].reshape(
                        Nq_list[_m]*(_p_m+1), Nq_list[_m]*(_p_m+1), order="F"
                    )

                # Draw proposal parameters (Minnesota prior / invwishart + normal)
                if Ym_list[_m].size:
                    _YY_p = np.vstack((
                        np.hstack((Ym_list[_m], _At_draw_p[:, :Nq_list[_m]])),
                        _AT_draw_p[1:, :(Nm_list[_m]+Nq_list[_m])]
                    ))
                else:
                    _YY_p = np.vstack((
                        _At_draw_p[:, :Nq_list[_m]],
                        _AT_draw_p[1:, :(Nm_list[_m]+Nq_list[_m])]
                    ))
                _nobs_p = np.shape(_YY_p)[0] - T0_list[_m]
                _spec_p = np.hstack((nlags_list_[_m], T0_list[_m], self.nex,
                                       nv_list[_m], _nobs_p))
                _YYact_p, _YYdum_p, _XXact_p, _XXdum_p = calc_yyact(
                    self.hyp[_m], _YY_p, _spec_p
                )
                _Tdummy_p = _YYdum_p.shape[0]
                _Tobs_p = _YYact_p.shape[0]
                _n_p = int(nv_list[_m])
                _p_spec = int(_spec_p[0])
                _X_p = np.vstack((_XXact_p, _XXdum_p))
                _Y_p = np.vstack((_YYact_p, _YYdum_p))
                _T_p = _Tobs_p + _Tdummy_p
                _vl_p, _d_p, _vr_p = np.linalg.svd(_X_p, full_matrices=False)
                _vr_p = _vr_p.T
                _di_p = 1.0 / _d_p
                _B_p = _vl_p.T @ _Y_p
                _xxi_p = _vr_p * np.tile(_di_p.T, (_n_p*_p_spec+1, 1))
                _inv_x_p = _xxi_p @ _xxi_p.T
                _Phi_tilde_p = _xxi_p @ _B_p
                _Sigma_p = (_Y_p - _X_p @ _Phi_tilde_p).T @ (_Y_p - _X_p @ _Phi_tilde_p)
                _sigma_prop = invwishart.rvs(scale=_Sigma_p, df=_T_p - _n_p*_p_spec - 1)

                # Draw Phi_prop; auto-reject if explosive
                # IMPORTANT: This explosive VAR check is necessary in the MH backward pass.
                # Although we use MH acceptance below (line ~1330), the acceptance ratio
                # is ONLY based on likelihood (log_alpha = ll_prop - ll_cur), NOT the full
                # posterior. An explosive VAR proposal could have high likelihood and be
                # accepted by the MH step if we don't enforce this stationarity constraint.
                # This check ensures the truncated prior (excluding explosive region) is
                # properly implemented in the Metropolis-within-Gibbs correction step.

                if check_explosive:
                    _explosive_prop = True
                    _Phi_prop = None
                    for _try in range(max_it_stable):
                        _sigma_chol_p = cholcovOrEigendecomp(np.kron(_sigma_prop, _inv_x_p))
                        _phi_new_p = (
                            np.squeeze(_Phi_tilde_p.reshape(_n_p*(_n_p*_p_spec+1), 1, order="F"))
                            + _sigma_chol_p @ np.random.standard_normal(_sigma_chol_p.shape[0])
                        )
                        _Phi_prop = _phi_new_p.reshape(_n_p*_p_spec+1, _n_p, order="F")
                        stability_proposals_backward += 1
                        if not is_explosive(_Phi_prop, _n_p, _p_spec):
                            _explosive_prop = False
                            break
                        stability_rejected_backward += 1

                    if _explosive_prop:
                        continue  # auto-reject explosive proposals
                else:
                    # Skip explosive VAR check when check_explosive is False
                    _sigma_chol_p = cholcovOrEigendecomp(np.kron(_sigma_prop, _inv_x_p))
                    _phi_new_p = (
                        np.squeeze(_Phi_tilde_p.reshape(_n_p*(_n_p*_p_spec+1), 1, order="F"))
                        + _sigma_chol_p @ np.random.standard_normal(_sigma_chol_p.shape[0])
                    )
                    _Phi_prop = _phi_new_p.reshape(_n_p*_p_spec+1, _n_p, order="F")

                # Compute proposed downstream input Yq for block _m+1
                if var_of_interest is None:
                    _YQ0_prop = _YYact_p
                else:
                    _idx_voi_m = list(filter(
                        lambda x: YMX_list[_m].columns.tolist()[x] in var_of_interest,
                        range(len(YMX_list[_m].columns.tolist()))
                    ))
                    _idx_vars_p = np.concatenate((
                        np.array(_idx_voi_m),
                        YM_list[_m].shape[1] + np.array(idx_var_of_interest)
                    ))
                    _YQ0_prop = _YYact_p[:, np.int_(_idx_vars_p)].reshape(
                        -1, len(_idx_vars_p)
                    )
                _YQ_prop = np.kron(_YQ0_prop, np.ones((freq_ratio_list[_m+1], 1)))
                _expected_rows = YQ_list[_m+1].shape[0]
                if _YQ_prop.shape[0] != _expected_rows:
                    if _YQ_prop.shape[0] > _expected_rows:
                        _YQ_prop = _YQ_prop[:_expected_rows, :]
                    else:
                        _pad = np.tile(_YQ_prop[-1:, :], (_expected_rows - _YQ_prop.shape[0], 1))
                        _YQ_prop = np.vstack((_YQ_prop, _pad))
                _Yq_prop = _YQ_prop[T0_list[_m+1]:nobs_list[_m+1]+T0_list[_m+1], :]

                # Proposal log-likelihood using proposed downstream input
                _ll_prop = _kalman_filter_loglik_block(
                    GAMMAs_list[_m+1], GAMMAz_list[_m+1], GAMMAc_list[_m+1], GAMMAu_list[_m+1],
                    LAMBDAs_list[_m+1], LAMBDAz_list[_m+1], LAMBDAc_list[_m+1], LAMBDAu_list[_m+1],
                    LAMBDAs_t_list[_m+1], LAMBDAz_t_list[_m+1], LAMBDAc_t_list[_m+1], LAMBDAu_t_list[_m+1],
                    sig_qq_list[_m+1], sig_mm_list[_m+1], sig_mq_list[_m+1], sig_qm_list[_m+1],
                    _mh_At_init[_m+1], _mh_Pt_init[_m+1],
                    Zm_list[_m+1], Ym_list[_m+1], _Yq_prop,
                    nobs_list[_m+1], T0_list[_m+1], freq_ratio_list[_m+1],
                )

                # MH acceptance step
                _log_alpha = _ll_prop - _ll_cur
                if np.isnan(_log_alpha) or not np.isfinite(_ll_prop):
                    _log_alpha = -np.inf

                if np.log(np.random.uniform()) < _log_alpha:
                    # ---- Accept proposal ----
                    mh_accept_counts[_m] += 1

                    At_draw_list[_m] = _At_draw_p
                    Pmean_list[_m] = _Pmean_prop
                    Phi_list[_m] = _Phi_prop

                    # Update covariance sub-blocks
                    if Nm_list[_m]:
                        sig_mm_list[_m] = _sigma_prop[:Nm_list[_m], :Nm_list[_m]]
                        sig_mq_list[_m] = 0.5 * (
                            _sigma_prop[:Nm_list[_m], Nm_list[_m]:]
                            + np.transpose(_sigma_prop[Nm_list[_m]:, :Nm_list[_m]])
                        )
                        sig_qm_list[_m] = 0.5 * (
                            _sigma_prop[Nm_list[_m]:, :Nm_list[_m]]
                            + np.transpose(_sigma_prop[:Nm_list[_m], Nm_list[_m]:])
                        )
                        sig_qq_list[_m] = _sigma_prop[Nm_list[_m]:, Nm_list[_m]:]
                    else:
                        sig_qq_list[_m] = np.atleast_2d(_sigma_prop)
                    sigma_list[_m] = _sigma_prop

                    # Recompute KF transition/measurement matrices from _Phi_prop
                    _phi_qq_p = np.zeros((Nq_list[_m]*_p_m, Nq_list[_m]))
                    _phi_qm_p = np.zeros((Nm_list[_m]*_p_m, Nq_list[_m]))
                    _phi_mm_p = np.zeros((Nm_list[_m]*_p_m, Nm_list[_m]))
                    _phi_mq_p = np.zeros((Nq_list[_m]*_p_m, Nm_list[_m]))
                    for _i in range(_p_m):
                        _phi_qq_p[Nq_list[_m]*_i:Nq_list[_m]*(_i+1), :] = _Phi_prop[
                            _i*(Nm_list[_m]+Nq_list[_m])+Nm_list[_m]:(_i+1)*(Nm_list[_m]+Nq_list[_m]),
                            Nm_list[_m]:
                        ]
                        _phi_qm_p[Nm_list[_m]*_i:Nm_list[_m]*(_i+1), :] = _Phi_prop[
                            _i*(Nm_list[_m]+Nq_list[_m]):_i*(Nm_list[_m]+Nq_list[_m])+Nm_list[_m],
                            Nm_list[_m]:
                        ]
                        _phi_mm_p[Nm_list[_m]*_i:Nm_list[_m]*(_i+1), :] = _Phi_prop[
                            _i*(Nm_list[_m]+Nq_list[_m]):_i*(Nm_list[_m]+Nq_list[_m])+Nm_list[_m],
                            :Nm_list[_m]
                        ]
                        _phi_mq_p[Nq_list[_m]*_i:Nq_list[_m]*(_i+1), :] = _Phi_prop[
                            _i*(Nm_list[_m]+Nq_list[_m])+Nm_list[_m]:(_i+1)*(Nm_list[_m]+Nq_list[_m]),
                            :Nm_list[_m]
                        ]
                    _phi_qc_p = _Phi_prop[-1, Nm_list[_m]:, np.newaxis]
                    _phi_mc_p = _Phi_prop[-1, :Nm_list[_m], np.newaxis]

                    GAMMAs_list[_m] = np.vstack((
                        np.hstack((np.transpose(_phi_qq_p), np.zeros((Nq_list[_m], Nq_list[_m])))),
                        np.hstack((np.eye(_p_m*Nq_list[_m]), np.zeros((_p_m*Nq_list[_m], Nq_list[_m]))))
                    ))
                    GAMMAz_list[_m] = np.vstack((
                        np.transpose(_phi_qm_p),
                        np.zeros((_p_m*Nq_list[_m], _p_m*Nm_list[_m]))
                    ))
                    GAMMAc_list[_m] = np.vstack((_phi_qc_p, np.zeros((_p_m*Nq_list[_m], 1))))
                    GAMMAu_list[_m] = np.vstack((
                        np.eye(Nq_list[_m]),
                        np.zeros((_p_m*Nq_list[_m], Nq_list[_m]))
                    ))

                    if self.temp_agg == "mean":
                        LAMBDAs_list[_m] = np.vstack((
                            np.hstack((np.zeros((Nm_list[_m], Nq_list[_m])), np.transpose(_phi_mq_p))),
                            1.0/freq_ratio_list[_m] * np.hstack((
                                np.tile(np.eye(Nq_list[_m]), freq_ratio_list[_m]),
                                np.zeros((Nq_list[_m], Nq_list[_m]*(_p_m-(freq_ratio_list[_m]-1))))
                            ))
                        ))
                    else:
                        LAMBDAs_list[_m] = np.vstack((
                            np.hstack((np.zeros((Nm_list[_m], Nq_list[_m])), np.transpose(_phi_mq_p))),
                            np.hstack((
                                np.tile(np.eye(Nq_list[_m]), freq_ratio_list[_m]),
                                np.zeros((Nq_list[_m], Nq_list[_m]*(_p_m-(freq_ratio_list[_m]-1))))
                            ))
                        ))
                    LAMBDAz_list[_m] = np.vstack((
                        np.transpose(_phi_mm_p),
                        np.zeros((Nq_list[_m], _p_m*Nm_list[_m]))
                    ))
                    LAMBDAc_list[_m] = np.vstack((_phi_mc_p, np.zeros((Nq_list[_m], 1))))
                    LAMBDAu_list[_m] = np.vstack((
                        np.eye(Nm_list[_m]),
                        np.zeros((Nq_list[_m], Nm_list[_m]))
                    ))
                    Wmatrix_list[_m] = np.hstack((
                        np.eye(Nm_list[_m]),
                        np.zeros((Nm_list[_m], Nq_list[_m]))
                    ))
                    LAMBDAs_t_list[_m] = Wmatrix_list[_m] @ LAMBDAs_list[_m]
                    LAMBDAz_t_list[_m] = Wmatrix_list[_m] @ LAMBDAz_list[_m]
                    LAMBDAc_t_list[_m] = Wmatrix_list[_m] @ LAMBDAc_list[_m]
                    LAMBDAu_t_list[_m] = Wmatrix_list[_m] @ LAMBDAu_list[_m]

                    # Update stored thinned draws for block _m if applicable
                    if j % self.thining == 0:
                        _jt = int(j / self.thining)
                        Sigmap_list[_m][_jt, :, :] = _sigma_prop
                        Phip_list[_m][_jt, :, :] = _Phi_prop
                        Cons_list[_m][_jt, :] = _Phi_prop[-1, :]

                    # Update downstream input for block _m+1 (takes effect next sweep)
                    YQ_list[_m+1] = _YQ_prop
                    Yq_list[_m+1] = _Yq_prop
                    YDATA_list[_m+1][:T_list[_m+1], Nm_list[_m+1]:] = _YQ_prop
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue  # auto-reject on any numerical failure

        #self.YYactsim = YYactsim_list[-1]
        #self.XXactsim = XXactsim_list[-1]
        self.Phip = Phip_list[-1]
        self.Sigmap = Sigmap_list[-1]
        self.YDATA = YDATA_list[-1]
        self.select_q = select_q
        self.select_m = select_m_list[-1]
        self.select = select_list[-1]
        self.varlist = varlist_list[-1]
        self.Nm = Nm_list[-1]
        self.nv = nv_list[-1]
        self.freq_ratio = freq_ratio_list[-1]
        self.T = T_list[-1]
        self.Tstar = Tstar_list[-1]
        self.YQX = YQX_list[-1]
        self.YMX = YMX_list[-1]
        self.YQ = YQ_list[-1]
        self.YM = YM_list[-1]
        self.YMh = YMh_list[-1]
        self.Ym = Ym_list[-1]
        self.Yq = Yq_list[-1]
        self.T0 = T0_list[-1]
        self.Tnew = Tnew_list[-1]
    
    if not(var_of_interest is None):
        #update varlist and select_list
        for m in range(len(frequencies)-1):
            idx = list(filter(lambda x: varlist_list[m][x] in (YMX_list[m].columns.tolist() + var_of_interest), range(len(varlist_list[m]))))
            varlist_list[m] = varlist_list[m][idx]
            select_list[m] = select_list[m][idx]
            
        for m in range(1, len(frequencies)-1):
            idx_q = list(filter(lambda x: (YMX_list[m-1].columns.tolist() + YQX_list[0].columns.tolist())[x] in  var_of_interest, range(len(YMX_list[m-1].columns.tolist() + YQX_list[0].columns.tolist()))))
            select_q[m] = select_q[m][idx_q]
    
    
    #save lists to self
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
    self.select_list = select_list
    self.Tnew_list = Tnew_list
    self.Ym_list = Ym_list
    self.Yq_list = Yq_list
    self.select_m_list = select_m_list
    self.select_q = select_q
    self.lstate_list = lstate_list
    self.nlags_list = nlags_list
    self.varlist_list = varlist_list
    self.YMX_list =YMX_list
    self.index_list = index_list
    self.select_list = select_list        
    self.var_of_interest = var_of_interest
    self.explosive_counter = explosive_counter
    # Stability-truncation shares; NaN when check_explosive was off.
    self.stability_proposals = stability_proposals
    self.stability_rejected = stability_rejected
    self.stability_rejection_share = (
        stability_rejected / stability_proposals if stability_proposals > 0
        else float("nan")
    )
    self.stability_proposals_backward = stability_proposals_backward
    self.stability_rejected_backward = stability_rejected_backward
    self.stability_rejection_share_backward = (
        stability_rejected_backward / stability_proposals_backward
        if stability_proposals_backward > 0 else float("nan")
    )
    self.valid_draws = [draw for draw in valid_draws if draw >= self.nburn/self.thining]
    # MH acceptance rates per block (Version 2)
    self.mh_accept_counts = mh_accept_counts
    self.mh_total_counts = mh_total_counts
    self.mh_accept_rates = [
        (c / t if t > 0 else float("nan"))
        for c, t in zip(mh_accept_counts, mh_total_counts)
    ]

    # Return MDD if requested (for hyperparameter optimization)
    if return_mdd:
        return mdd_list[-1]
        
def forecast(self, H, conditionals = None, seed = None):

    '''
    Method to generate the forecasts in the highest frequency.\n

    Parameters
    ----------
    H : int
        Forecast horizon in highest frequnecy
    conditionals : pandas DataFrame or None
        Conditional forecasts\n
        column names must be the variable names\n
        no index needed\n
        either values or np.nan

    Notes
    -----
    Dispatches on the estimation method recorded in :meth:`fit` (``self.method``).
    Both the Schorfheide--Song and Chan--Poon--Zhu estimators populate the same
    posterior-draw and latent-state attributes, so the forward simulation itself
    is shared; only the entry point differs.

    Returns
    -------
    None
        Results are stored on the model instance (e.g. ``self.forecast_draws_list``)
        and consumed by :meth:`aggregate`, the plotting and the saving methods.
    '''
    # Dispatch on the estimation method used in fit(). Both estimators populate
    # the same draw/state attributes, so the forward simulation is shared.
    if seed is None:
        seed = getattr(self, 'seed', 0)
    np.random.seed(seed)
    method = getattr(self, "method", "schorfheide_song")
    if method == "chan_poon_zhu":
        from ._estimation_cpz import forecast_cpz
        return forecast_cpz(self, H, conditionals=conditionals)
    return _forecast_impl(self, H, conditionals=conditionals)


def _forecast_impl(self, H, conditionals = None):

    '''
    Shared forward-simulation implementation used by both estimation methods.
    See :func:`forecast` for the public interface.
    '''
    self.H = H
    # First we need to extend the index
    # depending on the highest frequencies the approach differs
    
    index = copy.deepcopy(self.index_list[-1])
    
    if self.frequencies[-1] == "D":
        
        index = index.append(pd.date_range(start=index[-1] + Day(), periods = H, freq='D'))

        # Function to check if a month has more than 20 days
        def has_more_than_20_days(month, dti):
            return sum(dti.to_period('M') == month) > 20

        # Function to remove the last day of a month
        def remove_last_day_of_month(month, dti):
            return dti[~((dti.to_period('M') == month) and (dti.day > 20))]

        # Check each month in extended_dti
        for month in index.to_period('M').unique():
            # If a month has more than 20 days
            while has_more_than_20_days(month, index):
                # Remove the last day of that month
                index = remove_last_day_of_month(month, index)
                # Add an additional day at the end
                index = index.append(pd.DatetimeIndex([index[-1] + Day()]))
    
    if self.frequencies[-1] == 'W':
        index = index.append(pd.date_range(start=index[-1] + Week(), periods=H, freq='W'))

        # Function to check if a month has more than 4 weeks
        def has_more_than_4_weeks(month, dti):
            return sum(dti.to_period('M') == month) > 4

        # Function to remove the last week of a month
        def remove_last_week_of_month(month, dti):
            return dti[~((dti.to_period('M') == month) & (dti.day > 28))]

        # Check each month in extended_dti
        for month in index.to_period('M').unique():
            # If a month has more than 4 weeks
            while has_more_than_4_weeks(month, index):
                # Remove the last week of that month
                index = remove_last_week_of_month(month, index)
                # Add an additional week at the end
                index = index.append(pd.DatetimeIndex([index[-1] + Week()]))
                
    if self.frequencies[-1] == 'M':
        index = index.append(pd.date_range(start=index[-1] + MonthBegin(), periods=H, freq='MS'))
    
    if self.frequencies[-1] == 'Q':
        index = index.append(pd.date_range(start=index[-1] + QuarterBegin(), periods=H, freq='QS'))
        
    
    # Now we need to look at the conditional forecasts:
    
    YYcond = pd.DataFrame(np.nan, index =  index[-H:], columns= self.varlist_list[-1])
    
    if conditionals is not None:
        conditionals.index = YYcond.index[:len(conditionals.index)]
        
        YYcond.update(conditionals)
        YYcond = np.array(YYcond) 
        YYcond[:,(self.select_list[-1] == 1)] = YYcond[:,(self.select_list[-1] == 1)]/100
        YYcond[:,(self.select_list[-1] == 0)] = np.log(YYcond[:,(self.select_list[-1] == 0)])
    exc = np.array(~np.isnan(YYcond))
    YYcond = np.array(YYcond)
        
    YY_m_list = deque()
    YY_med_list = deque()
    YY_095_list = deque()
    YY_005_list = deque()
    YY_084_list = deque()
    YY_016_list = deque()
    forecast_draws_list = deque()
    
    
    H_= int(self.H)
            
    #Prepare index for output
    index = index[index.shape[0]-(self.lstate_list[-1].shape[2]+H_):]

    ###############  
    # Forecasting #
    ###############
    
    # store forecasts in monthly frequency
    YYvector_ml  = np.zeros((len(self.valid_draws),H_,self.Nm_list[-1]+self.Nq_list[-1]))     # collects now/forecast      
    
    # store forecasts in quarterly frequency
    YYvector_ql  = np.zeros((len(self.valid_draws),self.Nm_list[-1]+self.Nq_list[-1]))   
    YYvector_qg  = np.zeros((len(self.valid_draws),int(self.H/self.freq_ratio_list[-1]),self.Nm_list[-1]+self.Nq_list[-1]))
    
    print(" ", end = '\n')
    print("Multiple Frequency BVAR: Forecasting", end = "\n")
    print("Forecast Horizon: ", H_, end = "\n")
    print("Total Draws: ", len(self.valid_draws))
    
    
    for idx, jj in enumerate(tqdm(self.valid_draws)):
        
        YYact = np.squeeze(self.YYactsim_list[-1][jj, -1, :])
        XXact = np.squeeze(self.XXactsim_list[-1][jj, -1, :])
        post_phi = np.squeeze(self.Phip_list[-1][jj,:,:])
        post_sig = np.squeeze(self.Sigmap_list[-1][jj,:,:])
        

        # Bayesian Estimation Forecasting 
        ###################################

        
        YYpred = np.zeros((H_+1, self.nv_list[-1])) # forecasts from VAR
        YYpred[0,:] = YYact
        XXpred = np.zeros((H_+1, self.nv_list[-1]*self.nlags_list[-1]+1))
        XXpred[:,-1] = np.full((H_+1), fill_value = 1)
        XXpred[0,:] = XXact
        
        # given posterior draw, draw number (H+1) random sequence

        error_pred = np.zeros((H_+1, self.nv_list[-1]))
        
        for h in range(H_+1):
            if post_sig.size > 1:
                error_pred[h,:] = np.linalg.cholesky(post_sig) @ np.random.standard_normal(self.nv_list[-1])
            else:
                error_pred[h,:] = np.random.normal(loc = 0, scale = post_sig)
        # given posterior draw, iterate forward to construct forecasts
        
        for h in range(1,H_+1):
            
            XXpred[h,self.nv_list[-1]:-1] = XXpred[h-1, :-self.nv_list[-1]-1]
            XXpred[h, :self.nv_list[-1]] = YYpred[h-1, :]
            #YYpred[h,:] = (XXpred[h,:] @ post_phi + error_pred[h,:])
            YYpred[h,:] = (1-exc[h-1,:]) * (XXpred[h,:] @ post_phi + error_pred[h,:]) + exc[h-1,:] * np.nan_to_num(YYcond[h-1,:])
        
        YYpred1 = copy.deepcopy(YYpred)
        YYpred = YYpred[1:,:]
        

        
        
        # Now-/Forecasts
        # Store in hf
        YYvector_ml[idx,:,:] = YYpred
        
    forecast_draws_list.append(YYvector_ml)
    
    #mean — back-transform each draw to level scale before averaging (Jensen's inequality correction)
    # For forecasts (YYvector_ml): transform to levels draw-by-draw, then take mean
    YYvector_ml_levels = YYvector_ml.copy()
    YYvector_ml_levels[:, :, (self.select_list[-1] == 0)] = np.exp(YYvector_ml_levels[:, :, (self.select_list[-1] == 0)])
    YYvector_ml_levels[:, :, (self.select_list[-1] == 1)] = 100 * YYvector_ml_levels[:, :, (self.select_list[-1] == 1)]
    YYftr_m = np.nanmean(YYvector_ml_levels, axis = 0)

    # For nowcasts (YYactsim): transform to levels draw-by-draw, then take mean
    YYactsim_draws = self.YYactsim_list[-1][self.valid_draws,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]].copy()
    if YYactsim_draws.size:
        YYactsim_draws[:, :, (self.select_m_list[-1] == 0)] = np.exp(YYactsim_draws[:, :, (self.select_m_list[-1] == 0)])
        YYactsim_draws[:, :, (self.select_m_list[-1] == 1)] = 100 * YYactsim_draws[:, :, (self.select_m_list[-1] == 1)]
    YYnow_m = np.mean(YYactsim_draws, axis = 0)

    # For latent states (lstate): transform to levels draw-by-draw, then take mean
    lstate_draws = self.lstate_list[-1][self.valid_draws,:,:].copy()
    lstate_draws[:, (self.select_q[-1] == 0), :] = np.exp(lstate_draws[:, (self.select_q[-1] == 0), :])
    lstate_draws[:, (self.select_q[-1] == 1), :] = 100 * lstate_draws[:, (self.select_q[-1] == 1), :]
    lstate_m = np.mean(lstate_draws, axis = 0).T
    
    YMh_list = copy.deepcopy(self.YMh_list)
    
    YMh_len_correction = int(YMh_list[-1].shape[0] - lstate_m[:-(self.freq_ratio_list[-1]),:].shape[0])
    
    #if YMh_list[-1].size:
    #    YMh_list[-1][:, (self.select_m_list[-1] == 1)] = 100 * YMh_list[-1][:, (self.select_m_list[-1] == 1)]
    #    YMh_list[-1][:, (self.select_m_list[-1] == 0)] =  np.exp(YMh_list[-1][:, (self.select_m_list[-1] == 0)])
    
    if YMh_list[-1].size:
        YY_m_list.append(np.vstack((np.vstack((np.hstack((YMh_list[-1][YMh_len_correction:,:], lstate_m[:-(self.freq_ratio_list[-1]),:])), np.hstack((YYnow_m, lstate_m[-self.freq_ratio_list[-1]:,:])))), YYftr_m)))
        #YY_m_list.append(np.vstack((np.vstack((np.hstack((YMh_list[-1], lstate_m[:-(self.freq_ratio_list[-1]),:])), np.hstack((YYnow_m, lstate_m[-self.freq_ratio_list[-1]:,:])))), YYftr_m)))
    else:
        YY_m_list.append(np.vstack((lstate_m,YYftr_m)))
    
    
    
    
    #self.mean_phi = np.mean(self.Phip, axis = 0)
    
    #median
    YYftr_med = np.nanmedian(YYvector_ml, axis = 0)
    YYftr_med[:, (self.select_list[-1] == 1)] = 100 * YYftr_med[:, (self.select_list[-1] == 1)]
    YYftr_med[:, (self.select_list[-1] == 0)] = np.exp(YYftr_med[:, (self.select_list[-1] == 0)])
    
    YYnow_med = np.median(self.YYactsim_list[-1][self.valid_draws,1:(self.freq_ratio+1),:self.Nm_list[-1]], axis = 0) # actual/nowcast monthlies
    
    if YYnow_med.size:
        YYnow_med[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_med[:, (self.select_m_list[-1] == 1)]
        YYnow_med[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_med[:, (self.select_m_list[-1] == 0)])
    
    lstate_med = np.median(self.lstate_list[-1][self.valid_draws,:,:], axis = 0).T # hf obs for lf vars
    lstate_med[:, (self.select_q[-1] == 1)] = 100 * lstate_med[:, (self.select_q[-1] == 1)]
    lstate_med[:, (self.select_q[-1] == 0)] = np.exp(lstate_med[:, (self.select_q[-1]== 0)])
    
    if YMh_list[-1].size:
        YY_med_list.append(np.vstack((np.vstack((np.hstack((YMh_list[-1][YMh_len_correction:,:], lstate_med[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_med, lstate_m[-self.freq_ratio_list[-1]:,:])))), YYftr_med)))
    else:
        YY_med_list.append(np.vstack((lstate_med,YYftr_med)))
        
    # safe uncertainty
    # 95%
    YYftr_095 = np.nanquantile(YYvector_ml, q = 0.95 ,axis = 0)
    YYftr_095[:, (self.select_list[-1] == 1)] = 100 * YYftr_095[:, (self.select_list[-1] == 1)]
    YYftr_095[:, (self.select_list[-1] == 0)] = np.exp(YYftr_095[:, (self.select_list[-1] == 0)])
    
    YYnow_095 = np.quantile(self.YYactsim_list[-1][self.valid_draws,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.95, axis = 0) # actual/nowcast monthlies
    if YYnow_095.size:
        YYnow_095[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_095[:, (self.select_m_list[-1] == 1)]
        YYnow_095[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_095[:, (self.select_m_list[-1] == 0)])
    
    lstate_095 = np.quantile(self.lstate_list[-1][self.valid_draws,:,:], q = 0.95, axis = 0).T # hf obs for lf vars
    lstate_095[:, (self.select_q[-1] == 1)] = 100 * lstate_095[:, (self.select_q[-1] == 1)]
    lstate_095[:, (self.select_q[-1] == 0)] = np.exp(lstate_095[:, (self.select_q[-1] == 0)])
    
    YMna = np.full(YMh_list[-1].shape, np.nan)
    
    if YMna.size:
        YY_095_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_095[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_095, lstate_095[-self.freq_ratio_list[-1]:,:])))), YYftr_095)))
    else:
        YY_095_list.append(np.vstack((lstate_095,YYftr_095)))
    
    # 84%
    YYftr_084 = np.nanquantile(YYvector_ml, q = 0.84 ,axis = 0)
    YYftr_084[:, (self.select_list[-1] == 1)] = 100 * YYftr_084[:, (self.select_list[-1] == 1)]
    YYftr_084[:, (self.select_list[-1] == 0)] = np.exp(YYftr_084[:, (self.select_list[-1] == 0)])
    
    YYnow_084 = np.quantile(self.YYactsim_list[-1][self.valid_draws,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.84, axis = 0) # actual/nowcast monthlies
    if YYnow_084.size:
        YYnow_084[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_084[:, (self.select_m_list[-1] == 1)]
        YYnow_084[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_084[:, (self.select_m_list[-1] == 0)])
    
    lstate_084 = np.quantile(self.lstate_list[-1][self.valid_draws,:,:], q = 0.84, axis = 0).T # hf obs for lf vars
    lstate_084[:, (self.select_q[-1] == 1)] = 100 * lstate_084[:, (self.select_q[-1] == 1)]
    lstate_084[:, (self.select_q[-1] == 0)] = np.exp(lstate_084[:, (self.select_q[-1] == 0)])
    
    YMna = np.full(YMh_list[-1].shape, np.nan)
    
    if YMna.size:
        YY_084_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_084[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_084, lstate_084[-self.freq_ratio_list[-1]:,:])))), YYftr_084)))
    else:
        YY_084_list.append(np.vstack((lstate_084,YYftr_084)))
        
    # 16%
    
    YYftr_016 = np.nanquantile(YYvector_ml, q = 0.16 ,axis = 0)
    YYftr_016[:, (self.select_list[-1] == 1)] = 100 * YYftr_016[:, (self.select_list[-1] == 1)]
    YYftr_016[:, (self.select_list[-1] == 0)] = np.exp(YYftr_016[:, (self.select_list[-1] == 0)])
    
    YYnow_016 = np.quantile(self.YYactsim_list[-1][self.valid_draws,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.16, axis = 0) # actual/nowcast monthlies
    if YYnow_016.size:
        YYnow_016[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_016[:, (self.select_m_list[-1] == 1)]
        YYnow_016[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_016[:, (self.select_m_list[-1] == 0)])
    
    lstate_016 = np.quantile(self.lstate_list[-1][self.valid_draws,:,:], q = 0.16, axis = 0).T # hf obs for lf vars
    lstate_016[:, (self.select_q[-1] == 1)] = 100 * lstate_016[:, (self.select_q[-1] == 1)]
    lstate_016[:, (self.select_q[-1] == 0)] = np.exp(lstate_016[:, (self.select_q[-1] == 0)])
    
    YMna = np.full(YMh_list[-1].shape, np.nan)
    
    if YMna.size:
        YY_016_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:],lstate_016[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_016, lstate_016[-self.freq_ratio_list[-1]:,:])))), YYftr_016)))
    else:
        YY_016_list.append(np.vstack((lstate_016,YYftr_016)))
    
    
    # 5%    
    
    YYftr_005 = np.nanquantile(YYvector_ml, q = 0.05 ,axis = 0)
    YYftr_005[:, (self.select_list[-1] == 1)] = 100 * YYftr_005[:, (self.select_list[-1] == 1)]
    YYftr_005[:, (self.select_list[-1] == 0)] = np.exp(YYftr_005[:, (self.select_list[-1] == 0)])
    
    YYnow_005 = np.quantile(self.YYactsim_list[-1][self.valid_draws,1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]], q = 0.05, axis = 0) # actual/nowcast monthlies
    
    if YYnow_005.size:   
        YYnow_005[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_005[:, (self.select_m_list[-1] == 1)]
        YYnow_005[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_005[:, (self.select_m_list[-1] == 0)])
    
    lstate_005 = np.quantile(self.lstate_list[-1][self.valid_draws,:,:], q = 0.05, axis = 0).T # hf obs for lf vars
    lstate_005[:, (self.select_q[-1] == 1)] = 100 * lstate_005[:, (self.select_q[-1] == 1)]
    lstate_005[:, (self.select_q[-1] == 0)] = np.exp(lstate_005[:, (self.select_q[-1] == 0)])
    
    if YMna.size:
        YY_005_list.append(np.vstack((np.vstack((np.hstack((YMna[YMh_len_correction:,:], lstate_005[:-self.freq_ratio_list[-1],:])), np.hstack((YYnow_005, lstate_005[-self.freq_ratio_list[-1]:,:])))), YYftr_005)))
    else:
        YY_005_list.append(np.vstack((lstate_005,YYftr_005)))
            
    YY_mean_pd = pd.DataFrame(YY_m_list[-1], columns = self.varlist_list[-1])
    YY_mean_pd.index = index
    
    YY_median_pd = pd.DataFrame(YY_med_list[-1], columns = self.varlist_list[-1])
    YY_median_pd.index = index
    
    YY_095_pd = pd.DataFrame(YY_095_list[-1], columns = self.varlist_list[-1])
    YY_095_pd.index = index
    
    YY_005_pd = pd.DataFrame(YY_005_list[-1], columns = self.varlist_list[-1])
    YY_005_pd.index = index
    
    YY_084_pd = pd.DataFrame(YY_084_list[-1], columns = self.varlist_list[-1])
    YY_084_pd.index = index
    
    YY_016_pd = pd.DataFrame(YY_016_list[-1], columns = self.varlist_list[-1])
    YY_016_pd.index = index

    self.YY_095 = YY_095_list
    self.YY_084 = YY_084_list
    self.YY_016 = YY_016_list
    self.YY_005 = YY_005_list
    self.YY_mean = YY_m_list
    self.YY_median = YY_med_list
    self.forecast_draws_list = forecast_draws_list
    
    self.YY_mean_pd = YY_mean_pd
    self.YY_median_pd = YY_median_pd
    self.YY_095_pd = YY_095_pd
    self.YY_005_pd = YY_005_pd
    self.YY_084_pd = YY_084_pd
    self.YY_016_pd = YY_016_pd
    
    self.index_list[-1] = index
        
        
def aggregate(self, frequency, reset_index = True):
    
    '''
    Aggregates the Mean, Median and quantililes in the highest frequency to the desired frequency. \n
    The Function ensures, that we start at the beginning of a Year or Quarter depending on the chosen frequency \n
    
    Parameters
    ----------
    frequency : str
        The frequency to which the data should be aggregated to
    reset_index : boolean
        Schould index be changed to period Index

    '''
    if self.forecast_draws_list is None :
            sys.exit("Error: To gaggregate generate forecasts first")
            
    if frequency not in ["Y","Q"] :
            sys.exit("Error: Aggregation currently only implemented for aggregation to yearly and quarterly frequency")
    freq_lf = frequency
    freq_hf = self.frequencies[-1]
    
    YY_full_list = deque()
    YY_full_list_agg = deque()
    
    YYnow = copy.copy(self.YYactsim_list[-1])# actual/nowcast
    lstate = copy.copy(self.lstate_list[-1]) # hf obs for lf vars
    
    
    YMh_len_correction = int(self.YMh_list[-1].shape[0] - lstate[0][:,:-(self.freq_ratio_list[-1])].shape[1])
    
    if self.YMh_list[-1].size:
        for i in range(len(self.valid_draws)):
            lstate_temp = lstate[i].T
            lstate_temp[:, (self.select_q[-1] == 1)] = 100 * lstate_temp[:, (self.select_q[-1] == 1)]
            lstate_temp[:, (self.select_q[-1] == 0)] = np.exp(lstate_temp[:, (self.select_q[-1]== 0)])
            YYnow_temp = YYnow[i][1:(self.freq_ratio_list[-1]+1),:self.Nm_list[-1]]
            YYnow_temp[:, (self.select_m_list[-1] == 1)] = 100 * YYnow_temp[:, (self.select_m_list[-1] == 1)]
            YYnow_temp[:, (self.select_m_list[-1] == 0)] = np.exp(YYnow_temp[:, (self.select_m_list[-1] == 0)])
            forecast_draws_temp = copy.copy(self.forecast_draws_list[-1][i,:,:])
            forecast_draws_temp[:, (self.select_list[-1] == 1)] = 100 * forecast_draws_temp[:, (self.select_list[-1] == 1)]
            forecast_draws_temp[:, (self.select_list[-1] == 0)] = np.exp(forecast_draws_temp[:, (self.select_list[-1] == 0)])

            temp = np.vstack((np.vstack((np.hstack((self.YMh_list[-1][YMh_len_correction:,:], lstate_temp[:-(self.freq_ratio_list[-1]), :])), np.hstack((YYnow_temp, lstate_temp[-self.freq_ratio_list[-1]:,:])))), forecast_draws_temp))
            temp = pd.DataFrame(temp, columns = self.varlist_list[-1])
            temp.index = self.index_list[-1]
            YY_full_list.append(temp)
    else:
        for i in range(len(self.valid_draws)):
            temp = np.vstack((lstate[i,:,:],self.forecast_draws_list[i,:,:]))
            temp = pd.DataFrame(temp, columns = self.varlist_list[-1])
            temp.index = self.index_list[-1]
            YY_full_list.append(temp)
    
    
    def find_first_position(arr, numbers, count):
        for i in range(len(arr) - count + 1):
            if arr[i] in numbers and all(arr[i] == arr[j] for j in range(i+1, i+count)):
                return i
    
    
            
    def agg_helper(freq_lf, freq_hf, df):
        # Set the frequency ratio        
        if freq_hf == "Q" and freq_lf == "Y":
            freq_ratio = 4
        elif freq_hf == "M" and freq_lf == "Y":
            freq_ratio = 12
        elif freq_hf == "W" and freq_lf == "Y":
            freq_ratio = 48
        elif freq_hf == "D" and freq_lf == "Y":
            freq_ratio = 260
        elif freq_hf == "M" and freq_lf == "Q":
            freq_ratio = 3
        elif freq_hf == "W" and freq_lf == "Q":
            freq_ratio = 12
        elif freq_hf == "W" and freq_lf == "M":
            freq_ratio = 4
        elif freq_hf == "D" and freq_lf == "W":
            freq_ratio = 5
        elif freq_hf == "D" and freq_lf == "M":
            freq_ratio = 20
                
        if frequency == 'Q':
            if freq_hf == 'W':
                start = find_first_position(df.index.month, [1, 4, 7, 10], 4 )
            elif freq_hf == 'M':
                start = find_first_position(df.index.month, [1, 4, 7, 10], 1)
            elif freq_hf == 'D':
                start = find_first_position(df.index.month, [1, 4, 7, 10], 20)
        elif frequency == 'Y':
            if freq_hf == 'Q':
                start = find_first_position(df.index.month, [1,3], 1 )
            elif freq_hf == 'M':
                start = find_first_position(df.index.month, [1], 1)
            elif freq_hf == 'W':
                start = find_first_position(df.index.month, [1], 4)
            elif freq_hf == 'D':
                start = find_first_position(df.index.month, [1], 20)
        return freq_ratio, start
    
    
    freq_ratio, start = agg_helper(freq_lf, freq_hf, YY_full_list[0])
    print("Aggregating for each draw")
    for i in tqdm(range(len(self.valid_draws))):
        temp = YY_full_list[i].iloc[start:,].groupby(YY_full_list[i].iloc[start:,].reset_index().index // freq_ratio).filter(lambda x: len(x) == freq_ratio)
        if self.temp_agg == "mean":
            temp = temp.groupby(temp.reset_index().index // freq_ratio).mean()
        if self.temp_agg == "sum":
            temp = temp.groupby(temp.reset_index().index // freq_ratio).sum()
        temp.index = YY_full_list[i].iloc[start:,].index[::freq_ratio][:temp.shape[0]]
        temp.index = temp.index.map(lambda x: x.replace(day=1))
        YY_full_list_agg.append(temp)
        
    self.YY_mean_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).mean()
    self.YY_median_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).median()
    self.YY_095_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.95)
    self.YY_005_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.05)
    self.YY_084_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.84)
    self.YY_016_agg = pd.concat(YY_full_list_agg).groupby(pd.concat(YY_full_list_agg).index).quantile(0.16)
    
    hist = copy.copy(self.input_data)
    hist.appendleft(self.input_data_Q)
    i = 0
    for m in self.frequencies:
        if self.frequencies.index(m) == self.frequencies.index(frequency):
            # history ersetzen durch originaldaten 
            # interval durch NA
            idx = self.YY_mean_agg.index.intersection(hist[i].index, sort=False)
            self.YY_mean_agg.loc[idx, hist[i].columns] = hist[i].loc[idx, hist[i].columns]
            self.YY_median_agg.loc[idx, hist[i].columns] = hist[i].loc[idx, hist[i].columns]
            self.YY_095_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_005_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_084_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_016_agg.loc[idx, hist[i].columns] = np.nan
            
        if self.frequencies.index(m) > self.frequencies.index(frequency) & self.frequencies.index(m) < len(self.frequencies)-1:
            freq_ratio_temp, start_temp = agg_helper(frequency, m, hist[i])
            hist_agg = hist[i].iloc[start_temp:,].groupby(hist[i].iloc[start_temp:,].reset_index().index // freq_ratio_temp).filter(lambda x: len(x) == freq_ratio_temp)
            if self.temp_agg == 'mean':
                hist_agg = hist_agg.groupby(hist_agg.reset_index().index // freq_ratio_temp).mean()
            if self.temp_agg == 'sum':
                hist_agg = hist_agg.groupby(hist_agg.reset_index().index // freq_ratio_temp).sum()
            hist_agg.index = hist[i].iloc[start_temp:,].index[::freq_ratio_temp][:hist_agg.shape[0]]
            hist_agg.index = hist_agg.index.map(lambda x: x.replace(day=1))
            
            idx = hist_agg.index.intersection(self.YY_mean_agg.index, sort=False)
            self.YY_mean_agg.loc[idx, hist[i].columns] = hist_agg.loc[idx, hist[i].columns]
            self.YY_median_agg.loc[idx, hist[i].columns] = hist_agg.loc[idx, hist[i].columns]
            self.YY_095_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_005_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_084_agg.loc[idx, hist[i].columns] = np.nan
            self.YY_016_agg.loc[idx, hist[i].columns] = np.nan
        
        i = i+1
    
    if reset_index == True:
        index_new = pd.PeriodIndex(self.YY_mean_agg.index, freq= frequency)
        self.YY_mean_agg.index = index_new
        self.YY_median_agg.index = index_new
        self.YY_095_agg.index = index_new
        self.YY_005_agg.index = index_new
        self.YY_084_agg.index = index_new
        self.YY_016_agg.index = index_new
        
    
        
    if not(self.var_of_interest is None):
        idx_var_of_interest = list(filter(lambda x: self.YY_mean_agg.columns.tolist()[x] in self.YMX_list[-1].columns.tolist() + self.var_of_interest, range(len(self.YY_mean_agg.columns.tolist()))))
        self.YY_mean_agg = self.YY_mean_agg.iloc[:, idx_var_of_interest]
        self.YY_median_agg = self.YY_median_agg.iloc[:, idx_var_of_interest]
        self.YY_095_agg = self.YY_095_agg.iloc[:, idx_var_of_interest]
        self.YY_005_agg = self.YY_005_agg.iloc[:, idx_var_of_interest]
        self.YY_084_agg = self.YY_084_agg.iloc[:, idx_var_of_interest]
        self.YY_016_agg = self.YY_016_agg.iloc[:, idx_var_of_interest]
            
            
    self.agg_freq = frequency

def scenario_forecast(self, H, conditionals, names, agg = True):
    
    '''
    Generates the mean forecasts for multiple scenarios
    
    Parameters
    ----------
    H : int
        Forecast Horizon in highest frequency
    conditionals : list
        list of panda DataFrames 
    names : list
        list of the names for the scenarios
    agg : boolean
        If true aggregates output to lowest frequency\n
        Default is True
        
    Returns
    ----------
    out_dict : dict
        Dictionary containing the panda DataFrames for each szenario

    '''
    
    out = []
    
    for i in range(len(names)):
        
        temp = copy.deepcopy(self)
        if conditionals[i] is not None:
            temp.forecast(H, conditionals[i])
        else:
            temp.forecast(H)
        if agg == True:
            temp.aggregate(temp.frequencies[0], reset_index = True)
        
        out.append(temp.YY_mean_agg)
    out_dict = dict(zip(names, out))
    
    return out_dict