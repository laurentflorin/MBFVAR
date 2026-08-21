import os

import MBFVAR
import pandas as pd
import numpy as np 
import pickle


# Preparations
#---------------------

io_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hist.xlsx")

#Model Specification
H = 96          # forecast horizon
nsim = 40      # number of draws from Posterior Density
nburn = 0.5     # number of draws to discard
nlags = [3,4]   # Number of lags
thining = 1     # Thining 

hyp = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]] # Hyperparameters see documentation for details

frequencies = ["Q","M","W"] # Frequencies


# Load the data
# --------------
data = []
for freq in range(len(frequencies)):
        freq = frequencies[freq]
        data_temp = pd.read_excel(io_data, sheet_name = freq, index_col = 0)
        data.append(data_temp)

#Transformations
trans = [np.array((1,1)), np.array((1,1,1)), np.array((1,1,1))]    


# Initialize data class            
mbfvar_data_in = MBFVAR.mbfvar_data(data, trans, frequencies)


# Fit and Forecast
#--------------------

# Initialize model class    
model =  MBFVAR.MixedFrequencyBVAR(nsim, nburn, nlags, thining)

from scipy.stats import uniform

param_space = dict(lambda1_1 = uniform(0.001, 20), lambda2_1= uniform(0.01, 10), lambda4_1= uniform(0.01, 10), lambda5_1= uniform(0.01, 10), lambda1_2= uniform(0.001, 20), lambda2_2= uniform(0.01, 10), lambda4_2= uniform(0.01, 10), lambda5_2= uniform(0.01, 10))
init_points = 3 # number of random points
n_iter = 8 # number of baysian optimization steps
nsim = 1000 # number of simulations 
njobs = 1

H = 2
hyp = model.update_hyperparameters_mango_rmse(mbfvar_data_in, param_space, H, init_points, n_iter, nsim, njobs, var_of_interest = ["q_1"], temp_agg = 'mean', save = False, name = "hyp.txt")




# Estimate the model
model.fit(mbfvar_data_in, hyp = hyp, var_of_interest = None)

# Conditional forecasts

conditionals = pd.DataFrame({'w_1' : [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05]})   

# Create forecasts in highest frequency
model.forecast(H)

model.to_excel("test.xlsx")
# Aggregate
model.aggregate(frequency = "Q")
model.to_excel("test_q.xlsx", agg=True)
# Save results
#------------
#model.to_excel('out_test.xlsx', agg = True)
#model.save("model_2002_Q4")

# Plots
#---------

model.mean_plot(variables = "all", save = False, show = True)

model.fanchart(variables = "all", save = False, show = True, agg = True, nhist = 10)

model.fanchart(variables = "all", save = False, show = True, agg = False, nhist = 50)

# Optimizing Hyperparameters
#------------------------------

# Define boundaries for each hyperparameter, see documentation for details
pbounds = {'lambda1_1': (0.001, 20), 'lambda2_1': (0.01, 10), 'lambda4_1': (0.01, 10), 'lambda5_1': (0.01, 10), 'lambda1_2': (0.001, 20), 'lambda2_2': (0.01, 10), 'lambda4_2': (0.01, 10), 'lambda5_2': (0.01, 10)}
init_points = 3 # number of random points
n_iter = 8 # number of baysian optimization steps
nsim = 100 # number of simulations 

hyp = model.update_hyperparameters(mbfvar_data, pbounds, init_points, n_iter, nsim, var_of_interest = ["q_1"], save = False, name = "hyp.txt")

# Using mango
from scipy.stats import uniform

param_space = dict(lambda1_1 = uniform(0.001, 20), lambda2_1= uniform(0.01, 10), lambda4_1= uniform(0.01, 10), lambda5_1= uniform(0.01, 10), lambda1_2= uniform(0.001, 20), lambda2_2= uniform(0.01, 10), lambda4_2= uniform(0.01, 10), lambda5_2= uniform(0.01, 10))
init_points = 3 # number of random points
n_iter = 8 # number of baysian optimization steps
nsim = 100 # number of simulations 
njobs = 1

H = 2
model.update_hyperparameters_mango_rmse(mbfvar_data, param_space, H, init_points, n_iter, nsim, njobs, var_of_interest = ["q_1"], temp_agg = 'mean', save = False, name = "hyp.txt")

model.update_hyperparameters_mango(mbfvar_data, param_space, init_points, n_iter, nsim, njobs, var_of_interest = ["q_1"], temp_agg = 'mean', save = False, name = "hyp.txt")
#-------------------------------

# We can compare different scenarios

conditionals = [pd.DataFrame({'w_1' : [0.018, 0.025, np.nan, np.nan, 0.0228, 0.05],
                        'm_2' : [ 0.3, 0.002, 0.01 , 0.01, np.nan, np.nan]}),
                pd.DataFrame({'w_1' : [-0.02, -0.25, np.nan, np.nan, -0.228, 0.1],
                        'm_2' : [ -0.2, -0.012, 0 , 0.1, np.nan, np.nan]}), 
                None
                ]

names = ["good", "bad", "base"]

out_scenarios = model.scenario_forecast(H, conditionals, names, agg = True)

# Scenario Plot
model.scenario_plot(scenario_dict = out_scenarios, variables = "all", save = False, name = "Scenario", show = True, nhist = 10)

# Compare with model from last quarter
#--------------------------------------

#load previous quarter model:
file = open("model_2001_Q3.pkl",'rb')
model_2001_Q3 = pickle.load(file)

model_names = ["2001-Q4", "2001-Q3"]
multifrquency_var_models = [model_2001_Q3]

model.compare_models(multifrquency_var_models, model_names, agg = True, variables = "all", save = False, name = "Comparison", show = True, nhist = 5)

model.compare_models(multifrquency_var_models, model_names, agg = False, variables = ["q_1"], save = False, name = "Comparison", show = True, nhist = 20)