"""
Chan--Poon--Zhu (CPZ) Mixed-Frequency Example
==============================================

This example mirrors ``basic_example.py`` but selects the alternative
mixed-frequency estimator of Chan, Poon & Zhu (2024) via ``method``.

The CPZ path uses a conditionally-Gaussian latent-state sampler together with a
common stochastic-volatility (SV) process, adapted to each bi-frequency block of
MBFVAR's sequential chaining.  It is fully interchangeable with the default
Schorfheide--Song estimator: the same ``forecast`` / ``aggregate`` / plotting /
saving methods work unchanged, because ``fit`` populates the same posterior
draw and latent-state attributes.

The sample dataset hist.xlsx contains:
- Quarterly data (Q): 2 variables
- Monthly data (M): 3 variables
- Weekly data (W): 3 variables
"""

import MBFVAR
import pandas as pd
import numpy as np
import os

# Change to examples directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Loading data...")

io_data = "hist.xlsx"
frequencies = ["Q", "M", "W"]  # Lowest to highest frequency

data = []
for freq in frequencies:
    data_temp = pd.read_excel(io_data, sheet_name=freq, index_col=0)
    data.append(data_temp)
    print(f"  {freq}: {data_temp.shape[0]} observations, {data_temp.shape[1]} variables")

# ============================================================================
# 2. SPECIFY TRANSFORMATIONS
# ============================================================================
# 0 = take natural log, 1 = divide by 100 (for percentage/rate data)
trans = [
    np.array([1, 1]),        # Quarterly: 2 variables
    np.array([1, 1, 1]),     # Monthly: 3 variables
    np.array([1, 1, 1]),     # Weekly: 3 variables
]

# ============================================================================
# 3. PREPARE DATA
# ============================================================================
print("\nPreparing data for MBFVAR...")
data_in = MBFVAR.mbfvar_data(data, trans, frequencies)

# ============================================================================
# 4. SPECIFY MODEL PARAMETERS
# ============================================================================
nsim = 1000         # Number of posterior draws
nburn = 0.5         # Proportion of draws to discard as burn-in (50%)
nlags = [6, 4]      # Lags: 6 for first frequency pair, 4 for second
thining = 1         # Keep every nth draw

# Minnesota prior hyperparameters per frequency step
hyp = [
    [0.09, 4.3, 1, 2.7, 4.3],  # first frequency pair
    [0.09, 4.3, 1, 2.7, 4.3],  # second frequency pair
]

# ============================================================================
# 5. INITIALIZE AND FIT MODEL (Chan-Poon-Zhu method)
# ============================================================================
print("\nFitting model with the Chan-Poon-Zhu estimator...")
model = MBFVAR.MixedFrequencyBVAR(nsim, nburn, nlags, thining)

# The only difference from basic_example.py is ``method="chan_poon_zhu"``.
model.fit(data_in, hyp=hyp, method="chan_poon_zhu")
print("  Model fitted successfully!")

# The per-block common stochastic-volatility paths are available in model.h_list
print(f"  Stored SV paths per block: {[h.shape for h in model.h_list]}")

# ============================================================================
# 6. FORECAST, AGGREGATE, SAVE, VISUALIZE (identical to the default method)
# ============================================================================
H = 52  # Forecast horizon in highest frequency (52 weeks)
print(f"\nGenerating {H}-week forecast...")
model.forecast(H)

print("Aggregating forecasts to quarterly frequency...")
model.aggregate(frequency="Q")

print("Saving results...")
model.to_excel("forecasts_weekly_cpz.xlsx", agg=False)
model.to_excel("forecasts_quarterly_cpz.xlsx", agg=True)

print("Generating visualizations...")
model.fanchart(variables="all", save=True, show=False, agg=True,
               nhist=10, name="fanchart_quarterly_cpz")
model.mean_plot(variables="all", save=True, show=False, name="mean_forecast_cpz")

print("\n" + "=" * 70)
print("CPZ EXAMPLE COMPLETED SUCCESSFULLY!")
print("=" * 70)
