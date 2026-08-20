import os
import sys

import pandas as pd
import numpy as np

from collections import deque
import copy
import itertools

class MixedFrequencyBVAR:
    """
    Mixed-Frequency Bayesian Vector Autoregression (MBFVAR) model.

    This class implements a Bayesian VAR model for multi-frequency time series data,
    allowing for joint modeling and forecasting of variables observed at different
    frequencies (e.g., quarterly, monthly, weekly). The model uses a Minnesota-type
    prior and Gibbs sampling for posterior inference.

    Parameters
    ----------
    nsim : int
        Total number of posterior draws to generate from the MCMC sampler.
        Typical values: 1000-10000 depending on convergence requirements.

    nburn_perc : float
        Proportion of initial draws to discard as burn-in, between 0 and 1.
        These draws are discarded to allow the Markov chain to reach its
        stationary distribution.
        Typical value: 0.5 (discard first 50% of draws)

    nlags : list of int
        Number of lags for each frequency transition. Must contain one element
        for each pair of consecutive frequencies. Each value must be at least
        as large as the corresponding frequency ratio.

        Example: For frequencies ["Q", "M", "W"] with ratios [3, 4]:
            nlags = [6, 4] means 6 lags for Q→M transition, 4 for M→W

    thining : int
        Thinning parameter to reduce autocorrelation in posterior draws.
        Keep every nth draw where n = thining. Use 1 to keep all draws.
        Higher values reduce storage but may require more total simulations.

    Attributes
    ----------
    nsim : int
        Number of simulations
    nburn_perc : float
        Burn-in percentage
    nlags : list of int
        Lag structure
    thining : int
        Thinning parameter

    Methods
    -------
    fit(data_in, hyp, var_of_interest=None)
        Estimate the model parameters using Gibbs sampling

    forecast(H, conditionals=None)
        Generate forecasts for H periods ahead in the highest frequency

    aggregate(frequency)
        Aggregate forecasts to a lower frequency

    fanchart(variables, save=False, show=True, agg=False, nhist=10, name=None)
        Create fan chart visualization of forecasts with uncertainty bands

    mean_plot(variables, save=False, show=True, agg=False, name=None)
        Plot mean forecast paths

    scenario_forecast(H, conditionals, names, agg=False)
        Generate and compare multiple forecast scenarios

    scenario_plot(scenario_dict, variables, save=False, show=True, nhist=10, name=None)
        Visualize comparison of multiple forecast scenarios

    compare_models(models, model_names, variables, agg=False, save=False, show=True, nhist=5, name=None)
        Compare forecasts from multiple model vintages

    update_hyperparameters(data_in, pbounds, init_points, n_iter, nsim, var_of_interest=None, save=False, name=None)
        Optimize hyperparameters using Bayesian optimization

    update_hyperparameters_mango_rmse(data_in, param_space, H, init_points, n_iter, nsim, njobs, var_of_interest=None, temp_agg='mean', save=False, name=None)
        Optimize hyperparameters using Mango optimizer with RMSE criterion

    update_hyperparameters_mango_rmse_random(data_in, param_space, H, init_points, n_iter, nsim, njobs, var_of_interest=None, temp_agg='mean', h_eval=None, n_eval=1, min_T=None, random_seed=None, save=False, name=None)
        Optimize hyperparameters using Mango with RMSE criterion over randomly sampled forecast origins

    to_excel(filename, agg=False)
        Export forecasts to Excel file

    save(filename)
        Save fitted model object to disk

    Notes
    -----
    The model uses a Minnesota-type prior with the following structure:

    - Own lags have tighter priors (more influence)
    - Other variables' lags have looser priors (less influence)
    - More distant lags are down-weighted
    - The prior is controlled by 5 hyperparameters per frequency level

    The Gibbs sampler alternates between:
    1. Drawing VAR coefficients conditional on error covariance
    2. Drawing error covariance conditional on VAR coefficients

    Examples
    --------
    Basic usage:

    >>> import MBFVAR
    >>> import numpy as np
    >>>
    >>> # Initialize model
    >>> model = MBFVAR.MixedFrequencyBVAR(
    ...     nsim=1000,      # 1000 posterior draws
    ...     nburn=0.5,      # Discard first 50%
    ...     nlags=[6, 4],   # Lag structure
    ...     thining=1       # Keep all draws
    ... )
    >>>
    >>> # Fit model with hyperparameters
    >>> hyp = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]]
    >>> model.fit(data_in, hyp=hyp)
    >>>
    >>> # Generate 52-period ahead forecast
    >>> model.forecast(H=52)
    >>>
    >>> # Visualize results
    >>> model.fanchart(variables="all", show=True)

    Conditional forecasting:

    >>> import pandas as pd
    >>>
    >>> # Impose constraints on specific variables
    >>> conditionals = pd.DataFrame({
    ...     'w_1': [0.02, 0.025, np.nan, np.nan],  # Constrained periods
    ...     'm_2': [np.nan, 0.01, 0.01, np.nan]    # np.nan = unconstrained
    ... })
    >>> model.forecast(H=52, conditionals=conditionals)

    Scenario analysis:

    >>> # Define multiple scenarios
    >>> scenarios = [
    ...     pd.DataFrame({'w_1': [0.02, 0.025, np.nan, np.nan]}),  # Optimistic
    ...     pd.DataFrame({'w_1': [-0.02, -0.025, np.nan, np.nan]}), # Pessimistic
    ...     None  # Baseline (unconditional)
    ... ]
    >>> names = ["Optimistic", "Pessimistic", "Baseline"]
    >>>
    >>> # Generate and plot scenarios
    >>> out = model.scenario_forecast(H=52, conditionals=scenarios, names=names, agg=True)
    >>> model.scenario_plot(scenario_dict=out, variables="all", show=True)

    References
    ----------
    Schorfheide, F., & Song, D. (2015). Real-time forecasting with a mixed-frequency
    VAR. Journal of Business & Economic Statistics, 33(3), 366-380.

    Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector
    autoregressions. Review of Economics and Statistics, 97(2), 436-451.
    """
    
    def __init__(self, nsim, nburn_perc, nlags, thining, seed=0):

        self.nsim = nsim
        self.nburn_perc = nburn_perc
        self.nlags = nlags
        self.thining = thining
        self.seed = seed
        
    # Imported methods
    from ._estimation import fit, forecast, aggregate, scenario_forecast
    from ._plots import fanchart, mean_plot, scenario_plot, compare_models
    from ._save import to_excel, save
    from ._hyp_opt import update_hyperparameters, update_hyperparameters_mango, update_hyperparameters_mango_rmse, update_hyperparameters_mango_rmse_random
