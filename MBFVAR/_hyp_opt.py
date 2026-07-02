import os
import sys

import numpy as np
import math

from collections import deque

from scipy.stats import invwishart
import pandas as pd
from scipy.stats import multivariate_normal
from datetime import datetime
from pandas.tseries.offsets import Week , MonthBegin, QuarterBegin, Day

import itertools

import matplotlib.pyplot as plt

#for progressbar
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position = 0, leave=True) # this line does the magic

import copy

# for hyperparameter tuning (lazy imports – only needed when calling update_hyperparameters*)
# from bayes_opt import BayesianOptimization
# from mango import scheduler, Tuner

from .mbfvar_data import mbfvar_data

#from MBFVAR.pseudo_inverse.pseudo_inverse import calculate_pseudo_inverse
from .mfbvar_funcs import mdd_, is_explosive
from .cholcov.cholcov_module import cholcovOrEigendecomp
from .inverse.matrix_inversion import invert_matrix



def update_hyperparameters(self, mbfvar_data, pbounds, init_points, n_iter, nsim, var_of_interest = None, temp_agg = 'mean', save = False, name = "hyp.txt"):
    
    '''
    This method uses bayesian optimization to find the hyperparameters with the highest mdd\n
    lambda 1: overall tightness\n
    lambda 2:  scaling down the variance for the coefficients of a distant lag\n
    lambda 3:  number of observations used for obtaining the prior for the covariance matrix of error terms, fixed to 1\n
    lambda 4: . tuning parameter for coefficients for constant\n
    lambda 5:  tuning parameter for the covariance between coefficients\n

    NOTE: This function now uses the SAME model as the main fit() function in _estimation.py,
    including the Metropolis-within-Gibbs backward correction for cross-block feedback.
    This ensures hyperparameters are optimized for the full model.

    Parameters
    ----------
    mbfvar_data : mbfvar_data class object
        data in the form of a mbfvar_data class object
    pbound : dict
        boundaries for each hyperparameter:\n
        - two frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1\n
        - three frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2\n
        - four frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2, lambda1_3, lambda2_3, lambda4_3, lambda5_3
    init_points : int
        How many steps of random exploration you want to perform
    n_iter : int
        How many steps of bayesian optimization you want to perform
    nsim : int
        number of draws in each MUFBVAR estimation
    var_of_interest: list of names of variables that we are interested in or None
        Only the variables that are in this list get used in every bi frequency var.
        If None all variables get taken into each higher frequency bi frequency var.
    temp_agg : str
        `mean` or `sum` defines the measurement equation
    save : boolean
        True if you want to save the hyperparameters as a txt
    name : str
        path where you want to save the hyperparameters

    Returns
    ----------

    hyp : list
        list containing the optimized hyperparameters


    '''
    from bayes_opt import BayesianOptimization

    def estim(mbfvar_data, hyp_list, nsim, var_of_interest, temp_agg):
        """
        Calculate MDD by calling the main fit() function with return_mdd=True.
        This ensures hyperparameter optimization uses the SAME model as main estimation,
        including the Metropolis-within-Gibbs backward correction.
        """
        # Temporarily store current nsim/nburn and set to optimization values
        original_nsim = self.nsim
        original_nburn = self.nburn_perc

        self.nsim = nsim
        self.nburn_perc = self.nburn_perc  # Keep the same burn-in proportion

        
        try:
            # Call the main fit() function with return_mdd=True
            mdd = self.fit(mbfvar_data, hyp_list, var_of_interest=var_of_interest,
                          temp_agg=temp_agg, return_mdd=True)
        except Exception:
            # Any numerical failure (IndexError, LinAlgError, etc.) for a bad
            # hyperparameter combination must return a penalty so joblib does
            # not propagate the exception and kill the optimisation.
            return -1e16
        
        finally:
            # Always restore the original values
            self.nsim = original_nsim
            self.nburn_perc = original_nburn

        if not np.isfinite(mdd):
            return -1e16
        # Restore original values
        self.nsim = original_nsim
        self.nburn_perc = original_nburn

        return mdd

    def calc_mdd_1(lambda1_1, lambda2_1, lambda4_1, lambda5_1):

        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1]]
        mdd = estim(mbfvar_data, hyp_list, nsim, var_of_interest, temp_agg)

        
        return mdd
    
    def calc_mdd_2(lambda1_1, lambda2_1, lambda4_1,
                lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2]]
        mdd = estim(mbfvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        
        return mdd
    
    def calc_mdd_3(lambda1_1, lambda2_1, lambda4_1,
                lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2,
                lambda1_3, lambda2_3, lambda4_3, lambda5_3):
        
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2],
                    [lambda1_3, lambda2_3, 1, lambda4_3, lambda5_3]]
        mdd = estim(mbfvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        
        return mdd
        
    if len(mbfvar_data.frequencies)-1 == 1:
        optimizer = BayesianOptimization(
        f= calc_mdd_1,
        pbounds=pbounds,
        verbose= 2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        )
        
        
    if len(mbfvar_data.frequencies)-1 == 2:
        optimizer = BayesianOptimization(
        f= calc_mdd_2,
        pbounds=pbounds,
        verbose= 2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        )
        
    if len(mbfvar_data.frequencies)-1 == 3:
        optimizer = BayesianOptimization(
        f = calc_mdd_3,
        pbounds=pbounds,
        verbose= 2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        )
        
    optimizer.maximize(
    init_points = init_points,
    n_iter = n_iter,
    )
    
    hyp_opt = optimizer.max
    
    sublists = [list(hyp_opt["params"].values())[i:i+4] for i in range(0, len(list(hyp_opt["params"].values())), 4)] 
    hyp = []
    for i in sublists:
        i.insert(2,1)
        hyp.append(i)
    
    
    if save == True:
        with open(name, 'w') as f:
            print(hyp, file=f)
            
    return hyp


def update_hyperparameters_mango(self, mbfvar_data, param_space, init_points, n_iter, nsim, njobs, var_of_interest = None, temp_agg = 'mean', save = False, name = "hyp.txt"):
    
    '''
    This method uses bayesian optimization to find the hyperparameters with the highest mdd\n
    lambda 1: overall tightness\n
    lambda 2:  scaling down the variance for the coefficients of a distant lag\n
    lambda 3:  number of observations used for obtaining the prior for the covariance matrix of error terms, fixed to 1\n
    lambda 4: . tuning parameter for coefficients for constant\n
    lambda 5:  tuning parameter for the covariance between coefficients\n

    NOTE: This function now uses the SAME model as the main fit() function in _estimation.py,
    including the Metropolis-within-Gibbs backward correction for cross-block feedback.
    This ensures hyperparameters are optimized for the full model.

    Parameters
    ----------
    mbfvar_data : mbfvar_data class object
        data in the form of a mbfvar_data class object
    param_space : dict
        boundaries for each hyperparameter:\n
        - two frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1\n
        - three frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2\n
        - four frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2, lambda1_3, lambda2_3, lambda4_3, lambda5_3
    init_points : int
        How many steps of random exploration you want to perform
    n_iter : int
        How many steps of bayesian optimization you want to perform
    nsim : int
        number of draws in each MUFBVAR estimation
    njobs : int
        number of parallel jobs
    var_of_interest: list of names of variables that we are interested in or None
        Only the variables that are in this list get used in every bi frequency var.
        If None all variables get taken into each higher frequency bi frequency var.
    temp_agg : str
        `mean` or `sum` defines the measurement equation
    save : boolean
        True if you want to save the hyperparameters as a txt
    name : str
        path where you want to save the hyperparameters

    Returns
    ----------
    
    hyp : list
        list containing the optimized hyperparameters
        

    '''
    from mango import scheduler, Tuner


    def estim(mbfvar_data, hyp_list, nsim, var_of_interest, temp_agg):
        """
        Calculate MDD by calling the main fit() function with return_mdd=True.
        This ensures hyperparameter optimization uses the SAME model as main estimation,
        including the Metropolis-within-Gibbs backward correction.
        """
        # Temporarily store current nsim/nburn and set to optimization values
        original_nsim = self.nsim
        original_nburn = self.nburn_perc

        self.nsim = nsim
        self.nburn_perc = self.nburn_perc  # Keep the same burn-in proportion

        # Call the main fit() function with return_mdd=True
        try:
            # Call the main fit() function with return_mdd=True
            mdd = self.fit(mbfvar_data, hyp_list, var_of_interest=var_of_interest,
                          temp_agg=temp_agg, return_mdd=True)
        except Exception:
            # Any numerical failure (IndexError, LinAlgError, etc.) for a bad
            # hyperparameter combination must return a penalty so joblib does
            # not propagate the exception and kill the optimisation.
            return -1e16
        finally:
            # Always restore the original values
            self.nsim = original_nsim
            self.nburn_perc = original_nburn

        if not np.isfinite(mdd):
            return -1e16
        
        # Restore original values
        self.nsim = original_nsim
        self.nburn_perc = original_nburn

        return mdd

    @scheduler.parallel(n_jobs = njobs)
    def calc_mdd_1(lambda1_1, lambda2_1, lambda4_1, lambda5_1):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1]]
        mdd = estim(mbfvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        return mdd

    @scheduler.parallel(n_jobs = njobs)
    def calc_mdd_2(lambda1_1, lambda2_1, lambda4_1,
                lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2]]
        mdd = estim(mbfvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        return mdd

    @scheduler.parallel(n_jobs = njobs)
    def calc_mdd_3(lambda1_1, lambda2_1, lambda4_1,
                lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2,
                lambda1_3, lambda2_3, lambda4_3, lambda5_3):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2],
                    [lambda1_3, lambda2_3, 1, lambda4_3, lambda5_3]]
        mdd = estim(mbfvar_data, hyp_list, nsim, var_of_interest, temp_agg)
        return mdd

    
    conf_dict = dict(num_iteration = n_iter, initial_random = init_points)
    
    
    if len(mbfvar_data.frequencies)-1 == 1:
        tuner = Tuner(param_space, calc_mdd_1, conf_dict)
        
    if len(mbfvar_data.frequencies)-1 == 2:
        tuner = Tuner(param_space, calc_mdd_2, conf_dict)
        
        
    if len(mbfvar_data.frequencies)-1 == 3:
        tuner = Tuner(param_space, calc_mdd_3, conf_dict)
        
        
    results = tuner.maximize()
    best_params = results["best_params"]
    if save == True:
        with open(name, 'w') as f:
            print(best_params, file=f)
            
    sublists = [list(best_params.values())[i:i+4] for i in range(0, len(list(best_params.values())), 4)] 
    hyp = []
    for i in sublists:
        i.insert(2,1)
        hyp.append(i)
    
    if save == True:
        with open(name, 'w') as f:
            print(hyp, file=f)
            
    return hyp


    
def update_hyperparameters_mango_rmse(self, mufbvar_data_in, param_space, H, init_points, n_iter, nsim, njobs, var_of_interest = None, temp_agg = 'mean', h_eval = None, n_eval = 1, save = False, name = "hyp.txt", method = 'schorfheide_song'):
    """
    Use Bayesian optimization to select hyperparameters minimizing out-of-sample RMSE for MUFBVAR models.

    This method tunes hyperparameters (lambdas) for the multifrequency VAR (MUFBVAR) model using Bayesian optimization (via Mango).
    It runs the model over a rolling forecast to evaluate the out-of-sample RMSE for each hyperparameter set, and returns the set with the lowest RMSE.

    NOTE: This function uses the main fit() function from _estimation.py, which includes
    the Metropolis-within-Gibbs backward correction for cross-block feedback.
    This ensures hyperparameters are optimized for the full model.

    Hyperparameters:
        - lambda1: overall tightness
        - lambda2: scaling factor for the variance of distant lags
        - lambda3: number of observations for the prior on the error covariance (fixed to 1)
        - lambda4: tuning for coefficients of the constant
        - lambda5: tuning for covariance between coefficients

    Parameters
    ----------
    mbfvar_data : MUFBVAR.mbfvar_data object
        Holds the input data for multifrequency VAR estimation.
    param_space : dict
        Dictionary with bounds for each hyperparameter, structured according to the number of frequencies:
            - Two frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1
            - Three frequencies: add lambda1_2, lambda2_2, lambda4_2, lambda5_2
            - Four frequencies: add lambda1_3, lambda2_3, lambda4_3, lambda5_3
    H : int
        Forecast horizon in the lowest frequency.
    init_points : int
        Number of initial random exploration steps for Bayesian optimization.
    n_iter : int
        Number of optimization iterations.
    nsim : int
        Number of simulation draws in MUFBVAR estimation.
    njobs : int
        Number of parallel jobs.
    var_of_interest : list of str or None, default None
        List of variable names to consider. If None, all variables are used.
    temp_agg : str, default 'mean'
        Temporal aggregation method ('mean' or 'sum'), defining the measurement equation.
    h_eval : int or None, default None
        Specific forecast horizon (1-indexed) at which to evaluate RMSE.
        If None, RMSE is calculated across all H forecast periods (original behavior).
        Must be between 1 and H (inclusive).
    n_eval : int, default 1
        Number of rolling forecast origins to use for evaluation. For each origin k
        (from 0 to n_eval-1), the train/test split is shifted back by k low-frequency
        periods, and the RMSE is computed across all origins. Using n_eval > 1 produces
        a more stable and reliable objective function for hyperparameter optimization.
        When n_eval=1, the behavior is equivalent to a single-origin evaluation.
    save : bool, default False
        If True, saves the best hyperparameters to a file.
    name : str, default "hyp.txt"
        Path to file for saving hyperparameters if `save` is True.

    Returns
    -------
    hyp : list
        List of optimized hyperparameters (best set found).
    """
    from mango import scheduler, Tuner

    if h_eval is not None and (not isinstance(h_eval, int) or h_eval < 1 or h_eval > H):
        raise ValueError(f"h_eval must be an integer between 1 and H ({H}) inclusive, got {h_eval}.")

    if not isinstance(n_eval, int) or n_eval < 1:
        raise ValueError(f"n_eval must be a positive integer, got {n_eval}.")

    nburn_perc =  self.nburn_perc
    nlags = self.nlags
    thining = self.thining

    def calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining, h_eval, n_eval):
        try:
            mufbvar_data_temp = copy.deepcopy(mufbvar_data_in)
            horizon_mapping = {f'{mufbvar_data_temp.frequencies[0]}' : H}
            for i, freq  in enumerate(mufbvar_data_temp.frequencies[1:]):
                horizon_mapping.update({f'{freq}' : math.prod(itertools.islice(mufbvar_data_temp.freq_ratio_list, 0 ,i+1)) * H})

            # Ratio of each frequency relative to the lowest frequency (for offset scaling)
            ratio_mapping = {f'{mufbvar_data_temp.frequencies[0]}' : 1}
            for i, freq in enumerate(mufbvar_data_temp.frequencies[1:]):
                ratio_mapping.update({f'{freq}' : math.prod(itertools.islice(mufbvar_data_temp.freq_ratio_list, 0 ,i+1))})

            mufbvar_data_temp.input_data.appendleft(mufbvar_data_temp.input_data_Q)
            data_list = list(mufbvar_data_temp.input_data)

            all_squared_errors = []  # collect squared errors across all evaluation origins

            for k in range(n_eval):
                # For each origin k, shift the cutoff back by k low-frequency periods
                result_in_sample = []
                result_out_sample = []

                skip_origin = False
                for df_freq, freq in zip(data_list, mufbvar_data_temp.frequencies):
                    horizon = horizon_mapping.get(freq)
                    ratio = ratio_mapping.get(freq)
                    # Total rows to hold out: the forecast horizon plus k extra periods
                    # (k low-frequency periods scaled to the current frequency)
                    total_holdout = horizon + k * ratio
                    if len(df_freq) <= total_holdout:
                        skip_origin = True
                        break
                    in_sample = df_freq.iloc[:-total_holdout].copy()
                    if k > 0:
                        out_sample = df_freq.iloc[-total_holdout:-k*ratio].copy()
                    else:
                        out_sample = df_freq.iloc[-total_holdout:].copy()
                    result_in_sample.append(in_sample)
                    result_out_sample.append(out_sample)

                if skip_origin:
                    continue  # not enough data for this origin

                data_in = mbfvar_data(result_in_sample, mufbvar_data_temp.trans, mufbvar_data_temp.frequencies)

                model_temp = self.__class__(nsim, nburn_perc, nlags, thining)
                # Note: check_explosive=False skips explosive VAR checks for faster hyperparameter optimization
                model_temp.fit(data_in, hyp = hyp_list, var_of_interest = var_of_interest, temp_agg = temp_agg, check_explosive = False, method = method)
                model_temp.forecast(H * math.prod(data_in.freq_ratio_list))
                model_temp.aggregate(frequency = data_in.frequencies[0])

                out_sample = result_out_sample[0]
                out_sample = out_sample[var_of_interest]
                if (data_in.frequencies[0] == "Q"):
                    out_sample = out_sample.assign(Index = pd.DatetimeIndex(out_sample.index).to_period('Q')).set_index('Index')
                    out_sample = out_sample.add_suffix('_out_sample')

                joined_df = model_temp.YY_mean_agg[var_of_interest].join(out_sample, how="inner", lsuffix="_train", rsuffix="_out")

                suffix = '_out_sample'
                for col in joined_df.columns:
                    if col.endswith(suffix):
                        pred_col = col.replace(suffix, '')
                        if pred_col in joined_df.columns:
                            if h_eval is not None:
                                # Evaluate squared error at the specific forecast horizon (1-indexed)
                                idx = h_eval - 1  # convert to 0-indexed
                                if idx < len(joined_df):
                                    error = joined_df[pred_col].iloc[idx] - joined_df[col].iloc[idx]
                                    all_squared_errors.append(error ** 2)
                                else:
                                    print(f"Warning: h_eval={h_eval} is out of bounds for origin k={k} (joined_df has {len(joined_df)} rows). Skipping this variable/origin.")
                            else:
                                # RMSE across all H forecast periods
                                errors = joined_df[pred_col][:H] - joined_df[col][:H]
                                all_squared_errors.extend((errors ** 2).tolist())

            if not all_squared_errors:
                print(f"No valid evaluation origins found. This may be caused by insufficient data for n_eval={n_eval} origins with forecast horizon H={H}. Consider reducing n_eval or using a longer dataset.")
                return 1e10
            mean_rmse = float(np.sqrt(np.mean(all_squared_errors)))
            # Return high error if mean_rmse is nan or inf
            if np.isnan(mean_rmse) or np.isinf(mean_rmse):
                print("RMSE is nan or inf, returning high error.")
                return 1e10
            return mean_rmse
        except Exception as e:
            print(f"Error in calc_rmse: {e}")
            return 1e10  # Return a very high RMSE if any error occurs

    @scheduler.parallel(n_jobs = njobs)   
    def calc_rmse_1(lambda1_1, lambda2_1, lambda4_1, lambda5_1):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1]]   
        return calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining, h_eval, n_eval)

    @scheduler.parallel(n_jobs = njobs)
    def calc_rmse_2(lambda1_1, lambda2_1, lambda4_1,
                lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2]]
        return calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining, h_eval, n_eval)

    @scheduler.parallel(n_jobs = njobs)
    def calc_rmse_3(lambda1_1, lambda2_1, lambda4_1,
                lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2,
                lambda1_3, lambda2_3, lambda4_3, lambda5_3):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2],
                    [lambda1_3, lambda2_3, 1, lambda4_3, lambda5_3]]
        return calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining, h_eval, n_eval)

    conf_dict = dict(num_iteration = n_iter, initial_random = init_points)

    if len(mufbvar_data_in.frequencies)-1 == 1:
        tuner = Tuner(param_space, calc_rmse_1, conf_dict)
    if len(mufbvar_data_in.frequencies)-1 == 2:
        tuner = Tuner(param_space, calc_rmse_2, conf_dict)
    if len(mufbvar_data_in.frequencies)-1 == 3:
        tuner = Tuner(param_space, calc_rmse_3, conf_dict)

    results = tuner.minimize()
    best_params = results["best_params"]

    sublists = [list(best_params.values())[i:i+4] for i in range(0, len(list(best_params.values())), 4)] 
    hyp = []
    for i in sublists:
        i.insert(2,1)
        hyp.append(i)

    if save == True:
        with open(name, 'w') as f:
            print(hyp, file=f)
    return hyp


def update_hyperparameters_mango_rmse_random(self, mufbvar_data_in, param_space, H, init_points, n_iter, nsim, njobs, var_of_interest=None, temp_agg='mean', h_eval=None, n_eval=1, min_T=None, random_seed=None, save=False, name="hyp.txt", method='schorfheide_song'):
    """
    Use Bayesian optimization to select hyperparameters minimizing out-of-sample RMSE,
    evaluating on a randomly sampled set of forecast origins.

    This method is identical to ``update_hyperparameters_mango_rmse`` except that the
    ``n_eval`` evaluation origins are drawn **at random** (without replacement) from all
    valid origins, rather than using the last ``n_eval`` consecutive periods.  The random
    draw is performed once before optimization begins, so every hyperparameter configuration
    is evaluated on the same set of origins (ensuring a fair comparison).

    Valid origins are those for which:

    * every frequency has enough observations to produce the required holdout window, AND
    * the resulting lowest-frequency in-sample period contains at least ``min_T``
      observations (when ``min_T`` is not ``None``).

    NOTE: This function uses the main fit() function from _estimation.py, which includes
    the Metropolis-within-Gibbs backward correction for cross-block feedback.
    This ensures hyperparameters are optimized for the full model.

    Hyperparameters:
        - lambda1: overall tightness
        - lambda2: scaling factor for the variance of distant lags
        - lambda3: number of observations for the prior on the error covariance (fixed to 1)
        - lambda4: tuning for coefficients of the constant
        - lambda5: tuning for covariance between coefficients

    Parameters
    ----------
    mufbvar_data_in : MUFBVAR.mbfvar_data object
        Holds the input data for multifrequency VAR estimation.
    param_space : dict
        Dictionary with bounds for each hyperparameter, structured according to the number
        of frequencies:
            - Two frequencies: lambda1_1, lambda2_1, lambda4_1, lambda5_1
            - Three frequencies: add lambda1_2, lambda2_2, lambda4_2, lambda5_2
            - Four frequencies: add lambda1_3, lambda2_3, lambda4_3, lambda5_3
    H : int
        Forecast horizon in the lowest frequency.
    init_points : int
        Number of initial random exploration steps for Bayesian optimization.
    n_iter : int
        Number of optimization iterations.
    nsim : int
        Number of simulation draws in MUFBVAR estimation.
    njobs : int
        Number of parallel jobs.
    var_of_interest : list of str or None, default None
        List of variable names to consider. If None, all variables are used.
    temp_agg : str, default 'mean'
        Temporal aggregation method ('mean' or 'sum'), defining the measurement equation.
    h_eval : int or None, default None
        Specific forecast horizon (1-indexed) at which to evaluate RMSE.
        If None, RMSE is calculated across all H forecast periods.
        Must be between 1 and H (inclusive).
    n_eval : int, default 1
        Number of forecast origins to draw at random for evaluation. Must not exceed the
        total number of valid origins available in the data (raises ValueError otherwise).
    min_T : int or None, default None
        Minimum number of in-sample observations required in the lowest frequency after
        holding out the forecast window. Origins that would result in fewer than ``min_T``
        lowest-frequency observations are excluded from the pool of valid origins. When
        ``None``, no minimum sample-size constraint is applied.
    random_seed : int or None, default None
        Seed for the random number generator used to draw the evaluation origins. Set to
        a fixed integer for reproducible results across runs.
    save : bool, default False
        If True, saves the best hyperparameters to a file.
    name : str, default "hyp.txt"
        Path to file for saving hyperparameters if ``save`` is True.

    Returns
    -------
    hyp : list
        List of optimized hyperparameters (best set found).

    Raises
    ------
    ValueError
        If ``h_eval`` is outside [1, H].
    ValueError
        If ``n_eval`` is not a positive integer.
    ValueError
        If ``min_T`` is not a positive integer (when provided).
    ValueError
        If no valid origins exist (data too short or ``min_T`` too large).
    ValueError
        If ``n_eval`` exceeds the number of valid origins available.
    """
    from mango import scheduler, Tuner

    if h_eval is not None and (not isinstance(h_eval, int) or h_eval < 1 or h_eval > H):
        raise ValueError(f"h_eval must be an integer between 1 and H ({H}) inclusive, got {h_eval}.")

    if not isinstance(n_eval, int) or n_eval < 1:
        raise ValueError(f"n_eval must be a positive integer, got {n_eval}.")

    if min_T is not None and (not isinstance(min_T, int) or min_T < 1):
        raise ValueError(f"min_T must be a positive integer or None, got {min_T}.")

    nburn_perc = self.nburn_perc
    nlags = self.nlags
    thining = self.thining

    # ------------------------------------------------------------------
    # Compute the pool of valid origins and draw sampled_ks once so that
    # every hyperparameter configuration is evaluated on the same origins.
    # ------------------------------------------------------------------
    _data_tmp = copy.deepcopy(mufbvar_data_in)
    _horizon_mapping = {f'{_data_tmp.frequencies[0]}': H}
    for i, freq in enumerate(_data_tmp.frequencies[1:]):
        _horizon_mapping[f'{freq}'] = math.prod(itertools.islice(_data_tmp.freq_ratio_list, 0, i + 1)) * H

    _ratio_mapping = {f'{_data_tmp.frequencies[0]}': 1}
    for i, freq in enumerate(_data_tmp.frequencies[1:]):
        _ratio_mapping[f'{freq}'] = math.prod(itertools.islice(_data_tmp.freq_ratio_list, 0, i + 1))

    _data_tmp.input_data.appendleft(_data_tmp.input_data_Q)
    _data_list = list(_data_tmp.input_data)

    # Largest k allowed by data length at every frequency
    max_k_data = min(
        (len(df) - _horizon_mapping[str(freq)] - 1) // _ratio_mapping[str(freq)]
        for df, freq in zip(_data_list, _data_tmp.frequencies)
    )

    # Largest k allowed by the minimum in-sample size constraint (lowest frequency)
    if min_T is not None:
        len_lowest = len(_data_list[0])
        horizon_lowest = _horizon_mapping[str(_data_tmp.frequencies[0])]
        # in-sample length at origin k = len_lowest - horizon_lowest - k >= min_T
        # => k <= len_lowest - horizon_lowest - min_T
        max_k_min_T = len_lowest - horizon_lowest - min_T
        max_k = min(max_k_data, max_k_min_T)
    else:
        max_k = max_k_data

    if max_k < 0:
        if min_T is not None:
            raise ValueError(
                f"No valid forecast origins exist. The data may be too short or min_T={min_T} "
                f"is too large. Consider reducing min_T or using a longer dataset."
            )
        else:
            raise ValueError(
                f"No valid forecast origins exist. The data is too short for a forecast "
                f"horizon of H={H}."
            )

    n_valid = max_k + 1  # origins k = 0, 1, ..., max_k
    if n_eval > n_valid:
        raise ValueError(
            f"n_eval={n_eval} exceeds the number of valid forecast origins ({n_valid}). "
            f"Reduce n_eval or relax min_T."
        )

    rng = np.random.default_rng(random_seed)
    sampled_ks = sorted(rng.choice(n_valid, size=n_eval, replace=False).tolist())

    # ------------------------------------------------------------------
    # Inner RMSE function — identical to update_hyperparameters_mango_rmse
    # except the loop iterates over sampled_ks instead of range(n_eval).
    # ------------------------------------------------------------------
    def calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining, h_eval, sampled_ks):
        try:
            mufbvar_data_temp = copy.deepcopy(mufbvar_data_in)
            horizon_mapping = {f'{mufbvar_data_temp.frequencies[0]}': H}
            for i, freq in enumerate(mufbvar_data_temp.frequencies[1:]):
                horizon_mapping.update({f'{freq}': math.prod(itertools.islice(mufbvar_data_temp.freq_ratio_list, 0, i + 1)) * H})

            ratio_mapping = {f'{mufbvar_data_temp.frequencies[0]}': 1}
            for i, freq in enumerate(mufbvar_data_temp.frequencies[1:]):
                ratio_mapping.update({f'{freq}': math.prod(itertools.islice(mufbvar_data_temp.freq_ratio_list, 0, i + 1))})

            mufbvar_data_temp.input_data.appendleft(mufbvar_data_temp.input_data_Q)
            data_list = list(mufbvar_data_temp.input_data)

            all_squared_errors = []

            for k in sampled_ks:
                result_in_sample = []
                result_out_sample = []

                skip_origin = False
                for df_freq, freq in zip(data_list, mufbvar_data_temp.frequencies):
                    horizon = horizon_mapping.get(freq)
                    ratio = ratio_mapping.get(freq)
                    total_holdout = horizon + k * ratio
                    if len(df_freq) <= total_holdout:
                        skip_origin = True
                        break
                    in_sample = df_freq.iloc[:-total_holdout].copy()
                    if k > 0:
                        out_sample = df_freq.iloc[-total_holdout:-k * ratio].copy()
                    else:
                        out_sample = df_freq.iloc[-total_holdout:].copy()
                    result_in_sample.append(in_sample)
                    result_out_sample.append(out_sample)

                if skip_origin:
                    continue

                data_in = mbfvar_data(result_in_sample, mufbvar_data_temp.trans, mufbvar_data_temp.frequencies)

                model_temp = self.__class__(nsim, nburn_perc, nlags, thining)
                model_temp.fit(data_in, hyp=hyp_list, var_of_interest=var_of_interest, temp_agg=temp_agg, check_explosive=False, method=method)
                model_temp.forecast(H * math.prod(data_in.freq_ratio_list))
                model_temp.aggregate(frequency=data_in.frequencies[0])

                out_sample = result_out_sample[0]
                out_sample = out_sample[var_of_interest]
                if data_in.frequencies[0] == "Q":
                    out_sample = out_sample.assign(Index=pd.DatetimeIndex(out_sample.index).to_period('Q')).set_index('Index')
                    out_sample = out_sample.add_suffix('_out_sample')

                joined_df = model_temp.YY_mean_agg[var_of_interest].join(out_sample, how="inner", lsuffix="_train", rsuffix="_out")

                suffix = '_out_sample'
                for col in joined_df.columns:
                    if col.endswith(suffix):
                        pred_col = col.replace(suffix, '')
                        if pred_col in joined_df.columns:
                            if h_eval is not None:
                                idx = h_eval - 1
                                if idx < len(joined_df):
                                    error = joined_df[pred_col].iloc[idx] - joined_df[col].iloc[idx]
                                    all_squared_errors.append(error ** 2)
                                else:
                                    print(f"Warning: h_eval={h_eval} is out of bounds for origin k={k} (joined_df has {len(joined_df)} rows). Skipping this variable/origin.")
                            else:
                                errors = joined_df[pred_col][:H] - joined_df[col][:H]
                                all_squared_errors.extend((errors ** 2).tolist())

            if not all_squared_errors:
                print(f"No valid evaluation origins found for sampled_ks={sampled_ks}.")
                return 1e10
            mean_rmse = float(np.sqrt(np.mean(all_squared_errors)))
            if np.isnan(mean_rmse) or np.isinf(mean_rmse):
                print("RMSE is nan or inf, returning high error.")
                return 1e10
            return mean_rmse
        except Exception as e:
            print(f"Error in calc_rmse: {e}")
            return 1e10

    @scheduler.parallel(n_jobs=njobs)
    def calc_rmse_1(lambda1_1, lambda2_1, lambda4_1, lambda5_1):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1]]
        return calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining, h_eval, sampled_ks)

    @scheduler.parallel(n_jobs=njobs)
    def calc_rmse_2(lambda1_1, lambda2_1, lambda4_1,
                    lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2]]
        return calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining, h_eval, sampled_ks)

    @scheduler.parallel(n_jobs=njobs)
    def calc_rmse_3(lambda1_1, lambda2_1, lambda4_1,
                    lambda5_1, lambda1_2, lambda2_2, lambda4_2, lambda5_2,
                    lambda1_3, lambda2_3, lambda4_3, lambda5_3):
        hyp_list = [[lambda1_1, lambda2_1, 1, lambda4_1, lambda5_1],
                    [lambda1_2, lambda2_2, 1, lambda4_2, lambda5_2],
                    [lambda1_3, lambda2_3, 1, lambda4_3, lambda5_3]]
        return calc_rmse(hyp_list, mufbvar_data_in, H, nsim, var_of_interest, temp_agg, nlags, nburn_perc, thining, h_eval, sampled_ks)

    conf_dict = dict(num_iteration=n_iter, initial_random=init_points)

    if len(mufbvar_data_in.frequencies) - 1 == 1:
        tuner = Tuner(param_space, calc_rmse_1, conf_dict)
    if len(mufbvar_data_in.frequencies) - 1 == 2:
        tuner = Tuner(param_space, calc_rmse_2, conf_dict)
    if len(mufbvar_data_in.frequencies) - 1 == 3:
        tuner = Tuner(param_space, calc_rmse_3, conf_dict)

    results = tuner.minimize()
    best_params = results["best_params"]

    sublists = [list(best_params.values())[i:i + 4] for i in range(0, len(list(best_params.values())), 4)]
    hyp = []
    for i in sublists:
        i.insert(2, 1)
        hyp.append(i)

    if save == True:
        with open(name, 'w') as f:
            print(hyp, file=f)
    return hyp