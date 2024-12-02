# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:05:29 2024

@author: xuyun
"""

#%% libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import TruncatedNormal
from sklearn.preprocessing import MinMaxScaler
import time
import yfinance as yf
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle
from keras.optimizers import Adam
from statsmodels.tsa.api import VAR
from tensorflow.keras.models import load_model
import os
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error
from scipy.optimize import shgo
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.losses import categorical_crossentropy
from scipy.optimize import differential_evolution
from pyswarm import pso
from bayes_opt import BayesianOptimization
import itertools
from sklearn.linear_model import LinearRegression
from itertools import product



# '''
# general NN-approximated solution 的 convergence to the real solution 的数学研究，看在我们这里是不是也可以 apply
# 我之前写出来的Found a system of equations that the optimal control law satisfies that is yet too difficult to find the closed-form solution，
# 看我现在做一些numerical experiment能否satisfy that system of equations
# '''

'''
Markov decision if VAR(2): E(V_t(Y_t)| Y_{t-1}, Y_{t-2})，只要有了两个sets of information也是markov
'''


#%% global parameters and quantizer set up

T = 5
num_processes = 4

directory = '/Users/xuyunpeng/Documents/Time-consistent planning/Meeting19/'

dataframes = []

# Load the dataframes for t from 1 to T (1 to 5)
for t in range(1, T+1):
    file_path = os.path.join(directory, f'quantizer_vecBrownian/quantization_grid_t_{t}.csv')
    df = pd.read_csv(file_path)
    dataframes.append(df)

number_of_rows = dataframes[0].shape[0]  # Get number of rows from the first dataframe

# Create the quantize_grid array without an empty entry yet
quantize_grid = np.zeros((T, number_of_rows, num_processes))

# Fill quantize_grid for t = 1 to 5 (index 0 to 4)
for t, df in enumerate(dataframes):
    quantize_grid[t] = df.values

# Create an empty grid for t = 0 with the same shape as other grids
empty_grid = np.zeros((1, number_of_rows, num_processes))

# Prepend the empty grid to the quantize_grid, effectively shifting everything by 1
quantize_grid = np.concatenate((empty_grid, quantize_grid), axis=0)

# Now quantize_grid[0] is empty, and the original grids are shifted from t = 1 to t = T+1
weights = np.full((number_of_rows,), 1/number_of_rows)



#%% DO NOT RUN: choosing the best 3 stocks


# Define your asset categories
assets = {
    'Government Bonds': [
        'TLT', 'IEF', 'SHY', 'VGIT', 'TLH', 'TIP', 'BIL', 'SPTL', 'GOVT', 'SCHR', 'VTIP', 'STIP'
    ],
    'Corporate Bonds': [
        'LQD', 'HYG', 'VCIT', 'VCLT', 'IGIB', 'SPLB', 'ANGL', 'LQDH', 'HYLB', 'BNDX', 'REM'
    ],
    'Stock ETFs': [
        'VIG', 'SPY', 'QQQ', 'VTI', 'VOO', 'DIA', 'SCHD', 'IWM', 'ARKK', 'XLF', 'XLK', 'XLV', 
        'MTUM', 'VYM', 'FDVV', 'RSP'
    ]
}

# Download data for all assets
tickers = sum(assets.values(), [])
S = yf.download(tickers, start='2020-01-01', end='2023-04-01', interval='1mo', progress=False)['Adj Close']
S = S.dropna()

# Log differences to calculate returns
ln_S = np.log(S)
d_ln_S = ln_S.diff().dropna()

# Function to fit VAR model and calculate means/variances
def fit_var_and_compute_means_variances(data):
    try:
        # Fit the VAR model
        model = VAR(data)
        results = model.fit(maxlags=2)

        # Extract coefficients
        phi_matrices = results.params
        mu = phi_matrices.loc['const'].to_numpy()

        # Variance-covariance matrix of residuals
        variances = np.diag(results.sigma_u)

        return mu, variances, results
    except Exception as e:
        print(f"Error fitting VAR model: {e}")
        return None, None, None

# Generate all combinations of one stock ETF, one corporate bond, and one government bond

stock_etfs = assets['Stock ETFs']
corporate_bonds = assets['Corporate Bonds']
government_bonds = assets['Government Bonds']

combinations = list(product(stock_etfs, corporate_bonds, government_bonds))

best_combination = None
best_likelihood = -np.inf

# Store results for analysis
results_list = []

# Iterate through combinations
for combo in combinations:
    # Subset data for the combination
    subset_data = d_ln_S[list(combo)]
    
    # Fit the VAR model and calculate metrics
    mu, variances, results = fit_var_and_compute_means_variances(subset_data)
    if mu is None or variances is None or results is None:
        print('===var failed===')
        continue

    # Ensure high/medium/low mean-variance assets are present
    sorted_indices = np.argsort(mu)
    if not (variances[sorted_indices[0]] < variances[sorted_indices[1]] < variances[sorted_indices[2]]):
        continue

    # Simulate v[k]
    v = np.ones((6, len(combo)))
    phi_matrices = results.params
    L1_columns = [f'L1.{ticker}' for ticker in combo] 
    L2_columns = [f'L2.{ticker}' for ticker in combo] 
    
    Phi1 = phi_matrices.loc[L1_columns].to_numpy()
    Phi2 = phi_matrices.loc[L2_columns].to_numpy()
    
    for k in range(2, 6):
        v[k] = Phi1 @ v[k - 1] + Phi2 @ v[k - 2]
    
    # Check constraint for v[6] >= 0.5 * v[1]
    if not np.any(v[5] >= 0.5 * v[0]):
        continue

    # Calculate individual log-likelihood contributions
    residuals = results.resid
    individual_log_likelihoods = -0.5 * (
        np.log(2 * np.pi)
        + np.log(np.diag(results.sigma_u))
        + residuals ** 2 / np.diag(results.sigma_u)
    )
    total_likelihood = np.sum(individual_log_likelihoods.values.flatten())
    
    # Calculate the variance of individual contributions
    contribution_variances = np.var([np.sum(individual_log_likelihoods[col]) for col in subset_data.columns])

    # Store results if total likelihood is high and contributions are equal
    results_list.append({
        'combo': combo,
        'total_likelihood': total_likelihood,
        'contribution_variance': contribution_variances
    })

# Sort results by total likelihood and filter for low contribution variance
results_df = pd.DataFrame(results_list)
filtered_results = results_df.sort_values(
    ['contribution_variance', 'total_likelihood'], 
    ascending=[True, False])

filtered_results.head()

#%% claims data

# heart disease https://data.cdc.gov/NCHS/Monthly-Provisional-Counts-of-Deaths-by-Select-Cau/9dzk-mvmi/about_data
# click Actions, API, download file, 
# assume the company's policy portfolio includes 1% of total heart disease death in US, death benefit is 100,000 per death, inflation adjusted. 
print('unit is per thousand for monetary amounts')
# policies inforce since 2020-03-01
database = pd.read_csv('/Users/xuyunpeng/Documents/Time-consistent planning/Meeting20/Monthly_Provisional_Counts_of_Deaths_by_Select_Causes__2020-2023_20240826.csv')
selected_data = database.iloc[1:40, 6] #from 1/1/20 to 4/1/23

num_death = selected_data*0.01 
'lambda * E(X) 换成 0.001*selected_data, premium就会变少'
inflation_rates = {
    2020: [0.001, 0.001, 0.002, -0.004, -0.008, -0.001, 0.006, 0.006, 0.004, 0.000, 0.002, 0.004],
    2021: [0.003, 0.004, 0.006, 0.008, 0.006, 0.009, 0.005, 0.003, 0.004, 0.009, 0.008, 0.005],
    2022: [0.006, 0.008, 0.012, 0.003, 0.010, 0.013, 0.000, 0.001, 0.004, 0.004, 0.001, -0.001],
    2023: [0.005, 0.004, 0.001]
} # https://www.investing.com/economic-calendar/cpi-69

initial_benefit = 1 # means 10k death benefit
inflated_benefit = initial_benefit
all_rates = []
all_rates.extend(inflation_rates[2020])
all_rates.extend(inflation_rates[2021])
all_rates.extend(inflation_rates[2022])
all_rates.extend(inflation_rates[2023])
inflated_values = np.zeros(len(all_rates))
for month_index, rate in enumerate(all_rates):
    inflated_benefit *= (1 + rate / 100)  # Apply the inflation rate
    inflated_values[month_index] = inflated_benefit  # Store the result

ln_L = np.log(inflated_values*num_death)
d_ln_L = np.diff(ln_L)

initial_capital = initial_benefit*1000

'''
US insurer life average equity, average life insurance benefit
check inflation_rates annualised?
assume base date is 31/3/2023
'''

#%% claim and stock price stack into 1 df and find VAR model

# top3 = ['NFLX', 'REM', 'KO']
# top3 = ['PG', 'REM', 'AGG']
top3 = ['VYM', 'HYG', 'TIP']

S = yf.download(top3, start='2020-01-01', end='2023-04-01', interval='1mo', progress=False)['Adj Close']

S = S.dropna()
ln_S = np.log(S)
d_ln_S = ln_S.diff().dropna()

print(d_ln_S)

toCSV = pd.DataFrame(d_ln_S)
toCSV['claim'] = d_ln_L

toCSV = toCSV.dropna()

toCSV.to_csv('stock_differences.csv', index=True)
data = pd.read_csv('stock_differences.csv', index_col=0)

data = data.apply(pd.to_numeric)

model = VAR(data)
results = model.fit(maxlags=2)
print(results.summary())

phi_matrices = results.params

Sigma = results.sigma_u

mu = phi_matrices.loc['const']
mu = mu.to_numpy()

L1_columns = [f'L1.{ticker}' for ticker in top3] + ['L1.claim']
L2_columns = [f'L2.{ticker}' for ticker in top3] + ['L2.claim']

Phi1 = phi_matrices.loc[L1_columns].to_numpy()
Phi2 = phi_matrices.loc[L2_columns].to_numpy()

actual_list_of_top3 = list(results.params.columns[:-1])
top3 = actual_list_of_top3


#%% prices simulation for testing and validation

numTrain = 2**14
numSim = numTrain

def VARMA_sim1(current_d_ln_x, last_d_ln_x, Sigma, numSim, T, validation=False):
    
    noise_test, d_ln_S1_test, d_ln_S2_test, d_ln_S3_test, d_ln_L_test = {}, {}, {}, {}, {}
    
    for t in range(T+5):  # Including 0 for noise
        d_ln_S1_test[t], d_ln_S2_test[t], d_ln_S3_test[t], d_ln_L_test[t] = [], [], [], []
        noise_test[t] = []
    
    for path in range(numSim):
        noise = {}
        noise[0] = np.random.normal(0, 1, 4)

        current_d_ln_x_temp = current_d_ln_x.copy()  
        
        last_d_ln_x_temp = last_d_ln_x.copy()
        
        d_ln_S1_test[0].append(current_d_ln_x_temp[0])  # d_ln_S1_0
        d_ln_S2_test[0].append(current_d_ln_x_temp[1])  # d_ln_S2_0
        d_ln_S3_test[0].append(current_d_ln_x_temp[2])  # d_ln_S3_0
        d_ln_L_test[0].append(current_d_ln_x_temp[3])
        
        for t in range(1, T+5):
            
            noise[t] = np.random.multivariate_normal(mean=np.zeros(4), cov=Sigma)
                                   
            current_d_ln_x_temp_list = current_d_ln_x_temp - mu
            last_d_ln_x_temp_list = last_d_ln_x_temp - mu
            
            d_ln_S1 = mu[0] + np.dot(Phi1, current_d_ln_x_temp_list)[0] + np.dot(Phi2, last_d_ln_x_temp_list)[0] + noise[t][0]
            d_ln_S2 = mu[1] + np.dot(Phi1, current_d_ln_x_temp_list)[1] + np.dot(Phi2, last_d_ln_x_temp_list)[1] + noise[t][1]
            d_ln_S3 = mu[2] + np.dot(Phi1, current_d_ln_x_temp_list)[2] + np.dot(Phi2, last_d_ln_x_temp_list)[2] + noise[t][2]
            d_ln_L  = mu[3] + np.dot(Phi1, current_d_ln_x_temp_list)[3] + np.dot(Phi2, last_d_ln_x_temp_list)[3] + noise[t][3]
            
            last_d_ln_x_temp_list = current_d_ln_x_temp_list
            
            d_ln_S1_test[t].append((d_ln_S1))
            d_ln_S2_test[t].append((d_ln_S2))
            d_ln_S3_test[t].append((d_ln_S3))
            d_ln_L_test[t].append((d_ln_L))
                        
            current_d_ln_x_temp_list = [d_ln_S1, d_ln_S2, d_ln_S3, d_ln_L]            
            
            noise_test[t].append(noise[t])

    # Storing initial noise separately as it does not change with t
    noise_test[0] = [noise[0] for _ in range(numSim)]
    
    return noise_test, d_ln_S1_test, d_ln_S2_test, d_ln_S3_test, d_ln_L_test 

d_ln_S1_0, d_ln_S1_minus1 = d_ln_S[top3[0]][-1], d_ln_S[top3[0]][-2]
d_ln_S2_0, d_ln_S2_minus1 = d_ln_S[top3[1]][-1], d_ln_S[top3[1]][-2] 
d_ln_S3_0, d_ln_S3_minus1 = d_ln_S[top3[2]][-1], d_ln_S[top3[2]][-2] 
d_ln_L_0, d_ln_L_minus1 = d_ln_L[-1], d_ln_L[-2]

current_R_test = np.array([d_ln_S1_0, d_ln_S2_0, d_ln_S3_0, d_ln_L_0])
last_R_test = np.array([d_ln_S1_minus1, d_ln_S2_minus1, d_ln_S3_minus1, d_ln_L_minus1])

noise_vali, d_ln_S1_vali, d_ln_S2_vali, d_ln_S3_vali, d_ln_L_vali = VARMA_sim1( current_R_test, last_R_test, Sigma, numSim, T)
noise_test, d_ln_S1_test, d_ln_S2_test, d_ln_S3_test, d_ln_L_test = VARMA_sim1( current_R_test, last_R_test,  Sigma, numSim, T)

print('S1 is BAC, S2 is BND, S3 is PG')

#%% ln x_t simulation plot

ln_S1_0 = ln_S[top3[0]].iloc[-1]
ln_S2_0 = ln_S[top3[1]].iloc[-1]
ln_S3_0 = ln_S[top3[2]].iloc[-1]
ln_L_0 = np.log(initial_benefit)

ln_S1_vali = np.zeros((numSim, T+1))
ln_S2_vali = np.zeros((numSim, T+1))
ln_S3_vali = np.zeros((numSim, T+1))
ln_L_vali = np.zeros((numSim, T+1))

time_steps = range(1, T+2)
fig, axs = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

for i in range(numSim):
    ln_S1_vals = [(ln_S1_0 + sum(d_ln_S1_vali[j][i] for j in range(1, t))) for t in time_steps]
    ln_S2_vals = [(ln_S2_0 + sum(d_ln_S2_vali[j][i] for j in range(1, t))) for t in time_steps]
    ln_S3_vals = [(ln_S3_0 + sum(d_ln_S3_vali[j][i] for j in range(1, t))) for t in time_steps]
    ln_L_vals = [(ln_L_0 + sum(d_ln_L_vali[j][i] for j in range(1, t))) for t in time_steps]

    ln_S1_vali[i, :] = ln_S1_vals
    ln_S2_vali[i, :] = ln_S2_vals
    ln_S3_vali[i, :] = ln_S3_vals
    ln_L_vali[i, :] = ln_L_vals

    axs[0].plot(time_steps, ln_S1_vals, 'b-', alpha=0.2)
    axs[1].plot(time_steps, ln_S2_vals, 'r-', alpha=0.2)
    axs[2].plot(time_steps, ln_S3_vals, 'g-', alpha=0.2)
    axs[3].plot(time_steps, ln_L_vals, 'purple', alpha=0.2)


axs[0].set_title('Simulated Paths for lnS1 ' + actual_list_of_top3[0])
axs[1].set_title('Simulated Paths for lnS2 ' + actual_list_of_top3[1])
axs[2].set_title('Simulated Paths for lnS3 ' + actual_list_of_top3[2])
axs[3].set_title('Simulated Paths for lnL')

for ax in axs:
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True)

plt.tight_layout()
plt.show()

#%% prices simulation plot


ln_S1_0 = ln_S[top3[0]].iloc[-1]
ln_S2_0 = ln_S[top3[1]].iloc[-1]
ln_S3_0 = ln_S[top3[2]].iloc[-1]
ln_L_0 = np.log(initial_benefit)

S1_vali = np.zeros((numSim, T+1))
S2_vali = np.zeros((numSim, T+1))
S3_vali = np.zeros((numSim, T+1))
L_vali = np.zeros((numSim, T+1))

time_steps = range(1, T+2)
fig, axs = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

for i in range(numSim):
    S1_vals = [np.exp(ln_S1_0 + sum(d_ln_S1_vali[j][i] for j in range(1, t))) for t in time_steps]
    S2_vals = [np.exp(ln_S2_0 + sum(d_ln_S2_vali[j][i] for j in range(1, t))) for t in time_steps]
    S3_vals = [np.exp(ln_S3_0 + sum(d_ln_S3_vali[j][i] for j in range(1, t))) for t in time_steps]
    L_vals = [np.exp(ln_L_0 + sum(d_ln_L_vali[j][i] for j in range(1, t))) for t in time_steps]

    S1_vali[i, :] = S1_vals
    S2_vali[i, :] = S2_vals
    S3_vali[i, :] = S3_vals
    L_vali[i, :] = L_vals

    axs[0].plot(time_steps, S1_vals, 'b-', alpha=0.2)
    axs[1].plot(time_steps, S2_vals, 'r-', alpha=0.2)
    axs[2].plot(time_steps, S3_vals, 'g-', alpha=0.2)
    axs[3].plot(time_steps, L_vals, 'purple', alpha=0.2)


axs[0].set_title('Simulated Paths for S1 ' + actual_list_of_top3[0])
axs[1].set_title('Simulated Paths for S2 ' + actual_list_of_top3[1])
axs[2].set_title('Simulated Paths for S3 ' + actual_list_of_top3[2])
axs[3].set_title('Simulated Paths for L')

for ax in axs:
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True)

plt.tight_layout()
plt.show()


#%% return rate plot

R1_vali = np.zeros((numSim, T))  
R2_vali = np.zeros((numSim, T))
R3_vali = np.zeros((numSim, T))

for i in range(numSim):
    R1_vali[i, :] = S1_vali[i, 1:] / S1_vali[i, :-1]
    R2_vali[i, :] = S2_vali[i, 1:] / S2_vali[i, :-1]
    R3_vali[i, :] = S3_vali[i, 1:] / S3_vali[i, :-1]

time_steps = range(1, T+1)
fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

for i in range(numSim):
    axs[0].plot(time_steps, R1_vali[i, :], 'b-', alpha=0.2)
    axs[1].plot(time_steps, R2_vali[i, :], 'r-', alpha=0.2)
    axs[2].plot(time_steps, R3_vali[i, :], 'g-', alpha=0.2)
    
axs[0].set_title('Simulated Return Rates for R1 ' + actual_list_of_top3[0])
axs[1].set_title('Simulated Return Rates for R2 ' + actual_list_of_top3[1])
axs[2].set_title('Simulated Return Rates for R3 ' + actual_list_of_top3[2])

for ax in axs:
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Return Rate')
    ax.grid(True)

plt.tight_layout()
plt.show()

#%% original return rate for the past periods plot

R = yf.download(top3, start='2020-01-01', end='2023-04-01', interval='1mo', progress=False)['Adj Close']
R = R.pct_change().dropna()
R = (1 + R)

plt.figure(figsize=(10, 6))
for stock in top3:
    plt.plot(R.index, R[stock], label=stock)

plt.title('Actual Return Rates for Selected Stocks')
plt.xlabel('Date')
plt.ylabel('Return Rate')

plt.legend()
plt.show()

#%% plot VARIMA

time_steps = range(1, T+1)

fig, axs = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

for i in range(numSim):
    axs[0].plot(time_steps, 
                [d_ln_S1_vali[t][i] for t in time_steps], 'b-', alpha=0.2)
    axs[1].plot(time_steps, 
                [d_ln_S2_vali[t][i] for t in time_steps], 'r-', alpha=0.2)
    axs[2].plot(time_steps, 
                [d_ln_S3_vali[t][i] for t in time_steps], 'g-', alpha=0.2)
    axs[3].plot(time_steps, 
                [d_ln_L_vali[t][i] for t in time_steps], 'purple', alpha=0.2)

axs[0].set_title('Simulated Paths for d_ln_S1')
axs[1].set_title('Simulated Paths for d_ln_S2')
axs[2].set_title('Simulated Paths for d_ln_S3')
axs[3].set_title('Simulated Paths for d_ln_L')

for ax in axs:
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True)

plt.tight_layout()
plt.show()


#%% training data 

p = initial_benefit*1.2
gamma = 0.1
v = 0.8
gbond_up = 0.8
gbond_lo = 0.6
cbond_up = 0.2
cbond_lo = 0
div_up = 0.1
div_lo = 0

#loss is 10k average, capital is 10mil average
c_max = initial_capital*((1.3)**(T-1)) + p+ v*p + (v**2)*p + (v**3)*p + (v**4)*p
c_min = initial_capital*((0.8)**(T-1)) + p+ v*p + (v**2)*p + (v**3)*p + (v**4)*p

sobol = qmc.Sobol(d=13)
sobol_samples = sobol.random(n=numTrain)

d_ln_S1_range = [-0.1, 0.1]
d_ln_S2_range = [-0.09, 0.09]
d_ln_S3_range = [-0.15, 0.20]
d_ln_L_range = [-0.4, 0.4]
ln_S1_range = [3.9, 4.4]
ln_S2_range = [4.45, 4.85]
ln_S3_range = [4.4, 5.2]
ln_L_range = [-1, 0.75]
#C_range = [38456.339199999995, 1336.3392]

d_ln_S1t_train = sobol_samples[:, 0] * (d_ln_S1_range[1] - d_ln_S1_range[0]) + d_ln_S1_range[0]
d_ln_S1t_minus1_train = sobol_samples[:, 1] * (d_ln_S1_range[1] - d_ln_S1_range[0]) + d_ln_S1_range[0]

d_ln_S2t_train = sobol_samples[:, 2] * (d_ln_S2_range[1] - d_ln_S2_range[0]) + d_ln_S2_range[0]
d_ln_S2t_minus1_train = sobol_samples[:, 3] * (d_ln_S2_range[1] - d_ln_S2_range[0]) + d_ln_S2_range[0]

d_ln_S3t_train = sobol_samples[:, 4] * (d_ln_S3_range[1] - d_ln_S3_range[0]) + d_ln_S3_range[0]
d_ln_S3t_minus1_train = sobol_samples[:, 5] * (d_ln_S3_range[1] - d_ln_S3_range[0]) + d_ln_S3_range[0]

d_ln_Lt_train = sobol_samples[:, 6] * (d_ln_L_range[1] - d_ln_L_range[0]) + d_ln_L_range[0]
d_ln_Lt_minus1_train = sobol_samples[:, 7] * (d_ln_L_range[1] - d_ln_L_range[0]) + d_ln_L_range[0]

ln_S1t_train = sobol_samples[:, 8] * (ln_S1_range[1] - ln_S1_range[0]) + ln_S1_range[0]
ln_S2t_train = sobol_samples[:, 9] * (ln_S2_range[1] - ln_S2_range[0]) + ln_S2_range[0]
ln_S3t_train = sobol_samples[:, 10] * (ln_S3_range[1] - ln_S3_range[0]) + ln_S3_range[0]
ln_Lt_train = sobol_samples[:, 11] * (ln_L_range[1] - ln_L_range[0]) + ln_L_range[0]

c_train = sobol_samples[:, 12] * (c_max - c_min) + c_min


#%% terminal value function set up

def U(x):
    return 1/gamma * np.sign(x) * (np.abs(x)) ** gamma


def V_T(C_T):
    return U(C_T)


def B_minus_1(c, u,
              d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
              d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
              ln_S1t, ln_S2t, ln_S3t, ln_Lt, 
              quantizer, t):
        
    d_ln_Xt_vec = np.array([d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt]) 
    d_ln_Xt_minus1_vec = np.array([d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1]) 
    
    d_ln_S1t_plus1 = mu[0] + np.dot(Phi1, d_ln_Xt_vec-mu)[0] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[0] + quantizer[t+1][:, 0]
    d_ln_S2t_plus1 = mu[1] + np.dot(Phi1, d_ln_Xt_vec-mu)[1] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[1] + quantizer[t+1][:, 1]
    d_ln_S3t_plus1 = mu[2] + np.dot(Phi1, d_ln_Xt_vec-mu)[2] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[2] + quantizer[t+1][:, 2]
    d_ln_Lt_plus1 = mu[3] + np.dot(Phi1, d_ln_Xt_vec-mu)[3] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[3] + quantizer[t+1][:, 3]
    
    d_ln_S1t = d_ln_S1t
    d_ln_S2t = d_ln_S2t
    d_ln_S3t = d_ln_S3t
    d_ln_Lt = d_ln_Lt
    
    ln_S1t_plus1 = ln_S1t + d_ln_S1t_plus1
    ln_S2t_plus1 = ln_S2t + d_ln_S2t_plus1
    ln_S3t_plus1 = ln_S3t + d_ln_S3t_plus1
    ln_Lt_plus1 = ln_Lt + d_ln_Lt_plus1
    
    R1_t_plus1 = np.exp(ln_S1t_plus1)/np.exp(ln_S1t)
    R2_t_plus1 = np.exp(ln_S2t_plus1)/np.exp(ln_S2t)
    R3_t_plus1 = np.exp(ln_S3t_plus1)/np.exp(ln_S3t)
    u3 = 1-u[1]-u[2]
    
    C_t_plus1 = (u[1]*R1_t_plus1 + u[2]*R2_t_plus1 + u3*R3_t_plus1) * (c + p - u[0]*c) - np.exp(ln_Lt_plus1)
    
    return C_t_plus1,\
                d_ln_S1t_plus1,d_ln_S2t_plus1,d_ln_S3t_plus1,d_ln_Lt_plus1,\
                    d_ln_S1t,d_ln_S2t,d_ln_S3t,d_ln_Lt,\
                        ln_S1t_plus1,ln_S2t_plus1,ln_S3t_plus1,ln_Lt_plus1
                        
                        

def V_theta_t_plus1(c1,
                    d_ln_S1t_plus1,d_ln_S2t_plus1,d_ln_S3t_plus1,d_ln_Lt_plus1,
                    d_ln_S1t,d_ln_S2t,d_ln_S3t,d_ln_Lt,
                    ln_S1t_plus1,ln_S2t_plus1,ln_S3t_plus1,ln_Lt_plus1,
                    nnweights, inputscaler, outputscaler, scaleOutput = 1):
    
    inputdata = np.concatenate((
                                c1.reshape(-1,1),
                                
                                d_ln_S1t_plus1.reshape(-1,1), 
                                d_ln_S2t_plus1.reshape(-1,1), 
                                d_ln_S3t_plus1.reshape(-1,1), 
                                d_ln_Lt_plus1.reshape(-1,1), 
                                
                                d_ln_S1t.reshape(-1,1), 
                                d_ln_S2t.reshape(-1,1), 
                                d_ln_S3t.reshape(-1,1), 
                                d_ln_Lt.reshape(-1,1), 
                                                                                                
                                ln_S1t_plus1.reshape(-1,1), 
                                ln_S2t_plus1.reshape(-1,1), 
                                ln_S3t_plus1.reshape(-1,1), 
                                ln_Lt_plus1.reshape(-1,1),
                                
                                # noise_R1.reshape(-1,1), 
                                # noise_R2.reshape(-1,1), 
                                # noise_R3.reshape(-1,1), 
                                # noise_L.reshape(-1,1), 
                                
                                ), axis = 1)
    
    inputdata = inputscaler.transform(inputdata)
    
    layer1out = np.dot(inputdata, nnweights[0]) + nnweights[1]
    
    layer1out = tf.keras.activations.elu(layer1out).numpy()
    
    layer2out = np.dot(layer1out, nnweights[2]) + nnweights[3]
    
    layer2out = tf.keras.activations.elu(layer2out).numpy()
    
    layer3out = np.dot(layer2out, nnweights[4]) + nnweights[5]
    
    layer3out = tf.keras.activations.elu(layer3out).numpy()
    
    layer4out = np.dot(layer3out, nnweights[6]) + nnweights[7]

    if scaleOutput == 0:   
        output = tf.keras.activations.sigmoid(layer4out).numpy() 
    if scaleOutput == 1:  
        output = outputscaler.inverse_transform(layer4out)
    
    return output

#%% V_T_minus1

def V_T_minus1(c, u, 
                d_ln_S1T_minus1,d_ln_S2T_minus1,d_ln_S3T_minus1,d_ln_LT_minus1,
                d_ln_S1t_minus2, d_ln_S2t_minus2, d_ln_S3t_minus2, d_ln_Lt_minus2,
                ln_S1T_minus1, ln_S2T_minus1, ln_S3T_minus1, ln_LT_minus1, 
                quantizer, t=T-1
                ):
    
    C_T,\
        d_ln_S1T,d_ln_S2T,d_ln_S3T,d_ln_LT, \
            d_ln_S1T_minus1,d_ln_S2T_minus1,d_ln_S3T_minus1,d_ln_LT_minus1,\
                ln_S1T,ln_S2T,ln_S3T,ln_LT \
                    = B_minus_1(
                                c, 
                                u,
                                d_ln_S1T_minus1,d_ln_S2T_minus1,d_ln_S3T_minus1,d_ln_LT_minus1,
                                d_ln_S1t_minus2, d_ln_S2t_minus2, d_ln_S3t_minus2, d_ln_Lt_minus2,
                                ln_S1T_minus1, ln_S2T_minus1, ln_S3T_minus1, ln_LT_minus1, 
                                quantizer, t=T-1
                                )

    E_V_T = np.sum( weights * V_T(C_T) )
    
    V_T_minus1 = U(u[0]*c) + v * E_V_T
    
    return V_T_minus1



#%% value function surrogate set up

def V_t(c, u,
        d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
        d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
        ln_S1t, ln_S2t, ln_S3t, ln_Lt, 
        # noise_R1, noise_R2, noise_R3, noise_L,
        nnweights, inputscaler, outputscaler, quantizer, t
        ):
    
    numWeights = len(quantizer[0])
    
    C_t_plus1,\
        d_ln_S1t_plus1,d_ln_S2t_plus1,d_ln_S3t_plus1,d_ln_Lt_plus1,\
            d_ln_S1t,d_ln_S2t,d_ln_S3t,d_ln_Lt,\
                ln_S1t_plus1,ln_S2t_plus1,ln_S3t_plus1,ln_Lt_plus1 \
                    = B_minus_1(
                                c,
                                u,
                                d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
                                d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
                                ln_S1t, ln_S2t, ln_S3t, ln_Lt, 
                                quantizer, t)

    approx_V_t_plus1 = V_theta_t_plus1(
                               C_t_plus1,
                       
                               d_ln_S1t_plus1, 
                               d_ln_S2t_plus1, 
                               d_ln_S3t_plus1, 
                               d_ln_Lt_plus1, 
                               
                               np.ones(numWeights) * d_ln_S1t, 
                               np.ones(numWeights) * d_ln_S2t, 
                               np.ones(numWeights) * d_ln_S3t, 
                               np.ones(numWeights) * d_ln_Lt, # no more np.ones(numWeights) * 
                               
                               ln_S1t_plus1, 
                               ln_S2t_plus1, 
                               ln_S3t_plus1, 
                               ln_Lt_plus1, 
                        
                               nnweights, inputscaler, outputscaler)
    
    E_V_t_plus1 = np.sum(approx_V_t_plus1.flatten() * weights)
            
    V_t = U(u[0]*c) + v * E_V_t_plus1
                
    return V_t

#%% train


def BuildAndTrainModel(c1_train, 
                       d_ln_S1t_train, d_ln_S2t_train, d_ln_S3t_train, d_ln_Lt_train,
                       d_ln_S1t_minus1_train, d_ln_S2t_minus1_train, d_ln_S3t_minus1_train, d_ln_Lt_minus1_train,
                       ln_S1t_train, ln_S2t_train, ln_S3t_train, ln_Lt_train, 

                       quantizer, 
                       
                       nn_dim = 13, 
                       node_num = 30, 
                       batch_num = 64, 
                       epoch_num = 500, 
                       initializer = TruncatedNormal(mean = 0.0, stddev = 0.01, seed = 0) 
                       
                       ):
        
        
    input_train = np.concatenate((
                                    c1_train.reshape(-1,1),
                                    # gamma_train.reshape(-1,1), 
                                    
                                    d_ln_S1t_train.reshape(-1,1), 
                                    d_ln_S2t_train.reshape(-1,1), 
                                    d_ln_S3t_train.reshape(-1,1), 
                                    d_ln_Lt_train.reshape(-1,1), 
                                    
                                    d_ln_S1t_minus1_train.reshape(-1,1), 
                                    d_ln_S2t_minus1_train.reshape(-1,1), 
                                    d_ln_S3t_minus1_train.reshape(-1,1), 
                                    d_ln_Lt_minus1_train.reshape(-1,1), 
                                    
                                    ln_S1t_train.reshape(-1,1), 
                                    ln_S2t_train.reshape(-1,1), 
                                    ln_S3t_train.reshape(-1,1), 
                                    ln_Lt_train.reshape(-1,1),

                                    ), axis = 1) 
    
    
    input_scaler = MinMaxScaler(feature_range = (0,1))
    input_scaler.fit(input_train)
    input_train_scaled = input_scaler.transform(input_train)
    
    V_train = np.zeros((T+1, numTrain))
    u_cbond_train = np.zeros((T+1, numTrain))
    u_gbond_train = np.zeros((T+1, numTrain))
    u_stketf_train = np.zeros((T+1, numTrain))
    u_divid_train = np.zeros((T+1, numTrain))
    
    output_scaler = np.empty(T+1, dtype = object)
    V_hat_theta = np.empty(T+1, dtype = object)
    u_hat_theta_cbond = np.empty(T+1, dtype = object)
    u_hat_theta_gbond = np.empty(T+1, dtype = object)
    u_hat_theta_stketf = np.empty(T+1, dtype = object)
    u_hat_theta_divid = np.empty(T+1, dtype = object)
        
    start = time.perf_counter() 
    check = {}
    
    for j in range(T-1, 0, -1): # j is equivalent to t
            
        input_scaler = MinMaxScaler(feature_range = (0,1))
        input_scaler.fit(input_train)
        input_train_scaled = input_scaler.transform(input_train)
        print(np.isnan(input_train_scaled).any())
                    
        start_i = time.perf_counter()
        print("Time step " + str(j))
        check[j] = {}

        for i in range(numTrain):
            # check[j][i] = {}
            
            if j < (T-1):
                            
                def f_i(u):
                    V = V_t(
                                c1_train[i], 
                                u,
                                d_ln_S1t_train[i], d_ln_S2t_train[i], d_ln_S3t_train[i], d_ln_Lt_train[i],
                                d_ln_S1t_minus1_train[i], d_ln_S2t_minus1_train[i], d_ln_S3t_minus1_train[i], d_ln_Lt_minus1_train[i], 
                                ln_S1t_train[i], ln_S2t_train[i], ln_S3t_train[i], ln_Lt_train[i],
                                V_hat_theta[j+1].get_weights(),
                                input_scaler, output_scaler[j+1], quantizer, j
                                )
                    return V *-1
            else:
                def f_i(u):
                    V = V_T_minus1(
                                       c1_train[i], #gamma_train[i],
                                       u,
                                       d_ln_S1t_train[i], d_ln_S2t_train[i], d_ln_S3t_train[i], d_ln_Lt_train[i],
                                       d_ln_S1t_minus1_train[i], d_ln_S2t_minus1_train[i], d_ln_S3t_minus1_train[i], d_ln_Lt_minus1_train[i], 
                                       ln_S1t_train[i], ln_S2t_train[i], ln_S3t_train[i], ln_Lt_train[i],
                                       quantizer, j
                                       )                        
                    return V *-1
            
            # dividend, corporate bond \in[0,0.3], government bond\in[0.7,1] \righarrow  stock ETF\in[0,0.3]
            bounds = [(div_lo, div_up), (cbond_lo, cbond_up), (gbond_lo, gbond_up)]
            
            initial_guess = [(div_lo+div_up)/2, (cbond_lo+cbond_up)/2, (gbond_lo+gbond_up)/2]
            result = minimize(f_i, initial_guess, method='L-BFGS-B', bounds=bounds)
            
            v_hat = result.fun * -1
            u_hat = result.x
                                    
            V_train[j][i] = v_hat #*-1
            u_cbond_train[j][i] = u_hat[1]
            u_gbond_train[j][i] = u_hat[2]
            u_stketf_train[j][i] = 1-u_hat[1]-u_hat[2]
            u_divid_train[j][i] = u_hat[0]
            # if i == numTrain/2:
            #     print('     optimization half done')
            
            check[j][i] = [u_hat[1], u_hat[2], 1-u_hat[1]-u_hat[2], v_hat]   
                
        assert V_train[j].size > 0, f"valuefun_train at time step {j} is empty"
        assert np.std(V_train[j]) != 0, f"valuefun_train at time step {j} has constant values"
        
        end_i = time.perf_counter()
        print("     all optimizations done: " + str(round((end_i-start_i)/60,2)) + " min.")
        
        
        
        start_i = time.perf_counter()
        output_scaler[j] = MinMaxScaler(feature_range = (0,1))
        output_scaler[j].fit(V_train[j].reshape(-1, 1))
        V_train_scaled = output_scaler[j].transform(V_train[j].reshape(-1,1))    
        V_hat_theta[j] = Sequential([
                                    Input(shape=(nn_dim,)),  # Explicit input layer specification
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(1, activation=None, kernel_initializer=initializer, bias_initializer=initializer)
                                    ])
        optimizer = Adam(learning_rate=0.0001)
        V_hat_theta[j].compile(optimizer = optimizer, loss = 'mean_squared_error')
        V_hat_theta[j].fit(input_train_scaled, V_train_scaled, epochs = epoch_num, batch_size = batch_num, verbose = 0)
        end_i = time.perf_counter()
        print("     train value function done: " + str(round((end_i-start_i)/60,2)) + " min.")     
        in_sample_pred = V_hat_theta[j].predict(input_train_scaled)
        mse_valuefun = mean_squared_error(V_train_scaled, in_sample_pred)
        print(f"     MSE: {mse_valuefun}" )
        
        
        
        start_i = time.perf_counter()        
        u_hat_theta_cbond[j] = Sequential([
                                    Input(shape=(nn_dim,)),  # Explicit input layer specification
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(1, activation='sigmoid', kernel_initializer=initializer, bias_initializer=initializer)
                                    ])
        optimizer = Adam(learning_rate=0.001)
        u_hat_theta_cbond[j].compile(optimizer=optimizer, loss='mean_squared_error')
        u_hat_theta_cbond[j].fit(input_train_scaled, u_cbond_train[j], epochs=epoch_num, batch_size=batch_num, verbose=0)
        end_i = time.perf_counter()
        print("         train u_cbond done: " + str(round((end_i-start_i)/60,2)) + " min.")     
        # in_sample_pred = u_hat_theta_cbond[j].predict(input_train_scaled)
        # mse_valuefun = mean_squared_error(u_cbond_train[j], in_sample_pred)
        # print(f"     MSE: {mse_valuefun}" )
        
        
        
        start_i = time.perf_counter()        
        u_hat_theta_gbond[j] = Sequential([
                                    Input(shape=(nn_dim,)),  # Explicit input layer specification
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(1, activation='sigmoid', kernel_initializer=initializer, bias_initializer=initializer)
                                    ])
        optimizer = Adam(learning_rate=0.001)
        u_hat_theta_gbond[j].compile(optimizer=optimizer, loss='mean_squared_error')
        u_hat_theta_gbond[j].fit(input_train_scaled, u_gbond_train[j], epochs=epoch_num, batch_size=batch_num, verbose=0)
        end_i = time.perf_counter()
        print("         train u_gbond done: " + str(round((end_i-start_i)/60,2)) + " min.")     
        # in_sample_pred = u_hat_theta_gbond[j].predict(input_train_scaled)
        # mse_valuefun = mean_squared_error(u_gbond_train[j], in_sample_pred)
        # print(f"     MSE: {mse_valuefun}" )
        
        
        
        start_i = time.perf_counter()        
        u_hat_theta_stketf[j] = Sequential([
                                    Input(shape=(nn_dim,)),  # Explicit input layer specification
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(1, activation='sigmoid', kernel_initializer=initializer, bias_initializer=initializer)
                                    ])
        optimizer = Adam(learning_rate=0.001)
        u_hat_theta_stketf[j].compile(optimizer=optimizer, loss='mean_squared_error')
        u_hat_theta_stketf[j].fit(input_train_scaled, u_stketf_train[j], epochs=epoch_num, batch_size=batch_num, verbose=0)
        end_i = time.perf_counter()
        print("         train u_stketf done: " + str(round((end_i-start_i)/60,2)) + " min.")     
        # in_sample_pred = u_hat_theta_stketf[j].predict(input_train_scaled)
        # mse_valuefun = mean_squared_error(u_stketf_train[j], in_sample_pred)
        # print(f"     MSE: {mse_valuefun}" )
        
        
        
        start_i = time.perf_counter()        
        u_hat_theta_divid[j] = Sequential([
                                    Input(shape=(nn_dim,)),  # Explicit input layer specification
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(1, activation='sigmoid', kernel_initializer=initializer, bias_initializer=initializer)
                                    ])
        optimizer = Adam(learning_rate=0.001)
        u_hat_theta_divid[j].compile(optimizer=optimizer, loss='mean_squared_error')
        u_hat_theta_divid[j].fit(input_train_scaled, u_divid_train[j], epochs=epoch_num, batch_size=batch_num, verbose=0)
        end_i = time.perf_counter()
        print("         train u_divid done: " + str(round((end_i-start_i)/60,2)) + " min.")     
        # in_sample_pred = u_hat_theta_divid[j].predict(input_train_scaled)
        # mse_valuefun = mean_squared_error(u_divid_train[j], in_sample_pred)
        # print(f"     MSE: {mse_valuefun}" )
        
        
    end = time.perf_counter()
    duration = (end-start)/60

    print("Duration: " + str(duration) + " min.")
    
    return V_hat_theta, u_hat_theta_cbond, u_hat_theta_gbond, u_hat_theta_stketf, u_hat_theta_divid, input_scaler, output_scaler #, check

'print出来mean square error，把这些expected utility都存下来'
'做validation：做一个validation data set（可以是simulation），把由不同的hyper parameter的train出来的model放进去（相当于现在的test的步骤），比较expected utility'
'再用由最大expected utility的validation的hyper parameter的model去test' 
'我们假设simulation的data是真正的不知道的未来的data。我们假设我们有一个对未来的预估，即，validation dataset，用最好的validation的hyper parameter去test。'

#%% Train

V_hat_theta, u_hat_theta_cbond, u_hat_theta_gbond, u_hat_theta_stketf, u_hat_theta_divid, in_scaler, out_scaler\
= BuildAndTrainModel(c_train,  
                        d_ln_S1t_train, d_ln_S2t_train, d_ln_S3t_train, d_ln_Lt_train,
                        d_ln_S1t_minus1_train, d_ln_S2t_minus1_train, d_ln_S3t_minus1_train, d_ln_Lt_minus1_train, 
                        ln_S1t_train, ln_S2t_train, ln_S3t_train, ln_Lt_train, 
                     quantize_grid)


#%% check reasonability

def plot_relationship_c1_vs_nn(V_hat_theta, input_scaler, out_scaler_valuefun, fixed_inputs, c1_range):
    plt.figure(figsize=(10, 6))
    
    # Color map for each time step j to differentiate them visually
    colors = plt.cm.viridis(np.linspace(0, 1, len(V_hat_theta)))
    
    for j in range(1, len(V_hat_theta)-1):
        nn_model = V_hat_theta[j]
        output_scaler = out_scaler_valuefun[j]

        # Prepare input data for predictions by varying c1
        predictions = []
        for c1 in c1_range:
            # Create input by combining c1 with the fixed inputs
            input_data = np.concatenate(([c1], fixed_inputs)).reshape(1, -1)
            
            # Scale the input
            input_data_scaled = input_scaler.transform(input_data)
            
            # Predict the value function using the trained model for time step j
            pred = nn_model.predict(input_data_scaled)
            
            # Inverse scale the output using the appropriate scaler for this time step
            pred_inverse_scaled = output_scaler.inverse_transform(pred) if output_scaler else pred
            predictions.append(pred_inverse_scaled[0][0])  # Extract the predicted value
        
        # Plot the predictions as a function of c1, with color corresponding to j
        plt.plot(c1_range, predictions, label=f"Time step t={j}", color=colors[j-1])
    
    # Label the axes with LaTeX formatting
    plt.xlabel('C_t', fontsize=14)
    plt.ylabel('V^{\hat{theta}_t}', fontsize=14)
    plt.title('Relationship between C_t and V^{\hat{theta}_t,'+ f'gamma = {gamma}', fontsize=16)
    
    # Show the legend with color-coded labels for each j
    plt.legend(title="Time step (t)", loc='best', fontsize=10)
    plt.grid(True)
    plt.show()


input_train = np.concatenate((
                                c_train.reshape(-1,1),
                                # gamma_train.reshape(-1,1), 
                                
                                d_ln_S1t_train.reshape(-1,1), 
                                d_ln_S2t_train.reshape(-1,1), 
                                d_ln_S3t_train.reshape(-1,1), 
                                d_ln_Lt_train.reshape(-1,1), 
                                
                                d_ln_S1t_minus1_train.reshape(-1,1), 
                                d_ln_S2t_minus1_train.reshape(-1,1), 
                                d_ln_S3t_minus1_train.reshape(-1,1), 
                                d_ln_Lt_minus1_train.reshape(-1,1), 
                                
                                ln_S1t_train.reshape(-1,1), 
                                ln_S2t_train.reshape(-1,1), 
                                ln_S3t_train.reshape(-1,1), 
                                ln_Lt_train.reshape(-1,1),

                                ), axis = 1) 
    
# Assume fixed values for all other inputs (mean values or specific fixed values from the training data)
fixed_inputs = np.mean(input_train[:, 1:], axis=0)  # Use the mean values of all other variables except c1

# Define the range of c1 values to explore (e.g., from min to max value in the training data)
c1_min = np.min(c_train)
c1_max = np.max(c_train)
c1_range = np.linspace(c1_min, c1_max, 10)  # 100 points between the min and max of c1

# Call the function to plot the relationship
plot_relationship_c1_vs_nn(V_hat_theta, in_scaler, out_scaler, fixed_inputs, c1_range)




#%% save locally

output_dir = '/Users/xuyunpeng/Documents/Time-consistent planning/Meeting22_quantizing/models22'
os.makedirs(output_dir, exist_ok=True)

# Save each model to disk using the recommended .keras format
for j in range(1, T):
    model_path = os.path.join(output_dir, f'V_hat_theta_o1_{j}.keras')  # Use .keras instead of .h5
    V_hat_theta[j].save(model_path)  # Save model using native Keras format
    print(f"Saved model for time step {j} at {model_path}")

with open(os.path.join(output_dir, 'input_scaler_o1.pkl'), 'wb') as f:
    pickle.dump(in_scaler, f)

# Save each output scaler for different time steps
for j in range(1, T):
    with open(os.path.join(output_dir, f'output_scaler_o1{j}.pkl'), 'wb') as f:
        pickle.dump(out_scaler[j], f)
        print(f"Saved output scaler for time step {j}")


#%% retrieve models

output_dir = '/Users/xuyunpeng/Documents/Time-consistent planning/Meeting22_quantizing/models22'
os.makedirs(output_dir, exist_ok=True)

V_hat_theta_load = {}
for j in range(1, T):
    model_path = os.path.join(output_dir, f'V_hat_theta_{j}.keras')  # Use .keras extension
    V_hat_theta_load[j] = load_model(model_path)  # Load the model
    print(f"Loaded model for time step {j} from {model_path}")
    
with open(os.path.join(output_dir, 'input_scaler.pkl'), 'rb') as f:
    input_scaler_load = pickle.load(f)

output_scaler_load = {}
for j in range(1, T):
    with open(os.path.join(output_dir, f'output_scaler_{j}.pkl'), 'rb') as f:
        output_scaler_load[j] = pickle.load(f)
    
#%% testing prep

def portfolio_return(returns, weights, capital, liability):
    return -1* (
        U( weights[0]*capital) + U(
        (weights[1] * returns[0] + weights[2] * returns[1] + (1-weights[1]-weights[2]) * returns[2]) * (capital +p- weights[0]*capital) - liability
                                   )
        )
        

#%% test one path

def IndividualTest(c0, #gamma, 
                   path, T,
                   d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
                   d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1, 
                   quantizer, t,
                   input_scaler_valuefun, out_scaler_valuefun, V_hat_theta
                   ):
    
    samples = np.ones((6, 5, T+2))  
    # 5 strategies; 4 controls + 1 capital; T+2 horizon
                                                   
    samples[:,4,0:T+2] = c0
    
    for t in range(0, T):
        
        ln_S1_t = ((ln_S1_0) + sum((d_ln_S1t[j][path]) for j in range(0, t)))
        ln_S2_t = ((ln_S2_0) + sum((d_ln_S2t[j][path]) for j in range(0, t)))
        ln_S3_t = ((ln_S3_0) + sum((d_ln_S3t[j][path]) for j in range(0, t)))
        ln_L_t = ((ln_L_0) + sum((d_ln_Lt[j][path]) for j in range(0, t)))
        
        R1_t_plus1 = np.exp((ln_S1_0) + sum((d_ln_S1t[j][path]) for j in range(0, t+1)))/np.exp(ln_S1_t)
        R2_t_plus1 = np.exp((ln_S2_0) + sum((d_ln_S2t[j][path]) for j in range(0, t+1)))/np.exp(ln_S2_t)
        R3_t_plus1 = np.exp((ln_S3_0) + sum((d_ln_S3t[j][path]) for j in range(0, t+1)))/np.exp(ln_S3_t)
        
        bounds = [(div_lo, div_up), (cbond_lo, cbond_up), (gbond_lo, gbond_up)]
        initial_guess = [(div_lo+div_up)/2, (cbond_lo+cbond_up)/2, (gbond_lo+gbond_up)/2]
        
        
        
        
        # t = 0,1,2,..., T-1
        if t < T-1:
            def g_i(u): 
                V = V_t(samples[0][4][t], u, 
                            d_ln_S1t[t][path], d_ln_S2t[t][path], d_ln_S3t[t][path], d_ln_Lt[t][path],         
                            d_ln_S1t_minus1[t][path], d_ln_S2t_minus1[t][path], d_ln_S3t_minus1[t][path], d_ln_Lt_minus1[t][path], 
                            ln_S1_t, ln_S2_t, ln_S3_t, ln_L_t, 
                            V_hat_theta[t+1].get_weights(), input_scaler_valuefun, out_scaler_valuefun[t+1],  quantizer, t)
                return V
        else:
            def g_i(u): 
                V = V_T_minus1(samples[0][4][t], u, 
                            d_ln_S1t[t][path], d_ln_S2t[t][path], d_ln_S3t[t][path], d_ln_Lt[t][path],         
                            d_ln_S1t_minus1[t][path], d_ln_S2t_minus1[t][path], d_ln_S3t_minus1[t][path], d_ln_Lt_minus1[t][path], 
                            ln_S1_t, ln_S2_t, ln_S3_t, ln_L_t, 
                            quantizer, t)         
                return V
        result = minimize(g_i, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000})
        u_hat = result.x
        samples[0][0][t] = u_hat[0]
        samples[0][1][t] = u_hat[1]
        samples[0][2][t] = u_hat[2]
        samples[0][3][t] = 1-u_hat[1]-u_hat[2]
        samples[0][4][t+1] = \
            (samples[0][1][t] * R1_t_plus1 +
              samples[0][2][t] * R2_t_plus1 +
              samples[0][3][t] * R3_t_plus1)*\
            (samples[0][4][t] +p - samples[0][4][t]*samples[0][0][t]) - \
                    np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )
        samples[0][0][t] = samples[0][4][t]*samples[0][0][t]
                    
        
        
        
        # all in one stock strategies (3 of them) start here
        
        return_rates = [np.sum(weights*R1_t_plus1), np.sum(weights*R2_t_plus1), np.sum(weights*R3_t_plus1)]
        liab = np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)))
        def func(weights): 
            value = portfolio_return(return_rates, weights, samples[5][4][t], liab)    
            return value
        test_result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds)
        samples[1][0][t] = test_result.x[0]
        samples[1][1][t] = test_result.x[1]
        samples[1][2][t] = test_result.x[2]
        samples[1][3][t] = 1-test_result.x[1]-test_result.x[2]
        samples[1][4][t+1] = \
            (samples[1][1][t] * R1_t_plus1 +
             samples[1][2][t] * R2_t_plus1 +
             samples[1][3][t] * R3_t_plus1) *\
            (samples[1][4][t] +p - samples[1][4][t]*samples[1][0][t]) - \
                    np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )
                    
                    
                    
                    
        samples[2][1][t] = 0.25
        samples[2][2][t] = 0.5
        samples[2][3][t] = 0.25
        samples[2][4][t+1] = \
            (samples[2][1][t] * R1_t_plus1 +
             samples[2][2][t] * R2_t_plus1 +
             samples[2][3][t] * R3_t_plus1) *\
            (samples[2][4][t] +p - samples[2][4][t]*samples[0][0][t]) - \
                    np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )
                    
                    
                    
                    
        samples[3][1][t] = 0.15
        samples[3][2][t] = 0.7
        samples[3][3][t] = 0.15
        samples[3][4][t+1] = \
            (samples[3][1][t] * R1_t_plus1 +
             samples[3][2][t] * R2_t_plus1 +
             samples[3][3][t] * R3_t_plus1) *\
            (samples[3][4][t] +p - samples[3][4][t]*samples[0][0][t]) - \
                    np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )



            
        # same proportion strategy starts here
        samples[4][1][t] = 0
        samples[4][2][t] = 1
        samples[4][3][t] = 0
        samples[4][4][t+1] = \
            (samples[4][1][t] * R1_t_plus1 +
             samples[4][2][t] * R2_t_plus1 +
             samples[4][3][t] * R3_t_plus1) *\
            (samples[4][4][t] +p - samples[4][4][t]*samples[0][0][t]) - \
                    np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )




         # V_T optimization strategy here
        j = t
        def obj(u): 
            V = V_T_minus1(samples[5][4][t], u, 
                        d_ln_S1t[t][path], d_ln_S2t[t][path], d_ln_S3t[t][path], d_ln_Lt[t][path],         
                        d_ln_S1t_minus1[t][path], d_ln_S2t_minus1[t][path], d_ln_S3t_minus1[t][path], d_ln_Lt_minus1[t][path], 
                        ln_S1_t, ln_S2_t, ln_S3_t, ln_L_t, 
                        quantizer, t=j)         
            return V
        test_result = minimize(obj, initial_guess, method='L-BFGS-B', bounds=bounds)
        # if path == 100 or path == 500 or path == 1000: 
        #     print(ep_u1, ep_u2, ep_u3)
        samples[5][0][t] = test_result.x[0]
        samples[5][1][t] = test_result.x[1]
        samples[5][2][t] = test_result.x[2]
        samples[5][3][t] = 1-test_result.x[1]-test_result.x[2]
        samples[5][4][t+1] = \
            (samples[5][1][t] * R1_t_plus1 +
             samples[5][2][t] * R2_t_plus1 +
             samples[5][3][t] * R3_t_plus1) *\
            (samples[5][4][t] +p - samples[5][4][t]*samples[5][0][t]) - \
                    np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )
        
                
        
        
                
        loss_count = 0
        minimum_capital = c0 - sum(
                                np.exp(ln_L_0 + sum(
                                    d_ln_Lt[j][path] for j in range(1, t))
                                    ) for t in range(1, T)
                                ) + (T-1)*p

        if samples[0][4][T-1] < minimum_capital or samples[0][4][T-1] < samples[1][4][T-1] or samples[0][4][T-1] < samples[2][4][T-1] or samples[0][4][T-1] < samples[3][4][T-1] or samples[0][4][T-1] < samples[4][4][T-1] or samples[0][4][T-1] < samples[5][4][T-1] :
            loss_count = 1
        
    return samples, loss_count


def RunTests(c0, #gamma, 
                   T,
                   d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
                   d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1, 
                   quantizer, t,
                   input_scaler, out_scaler_valuefun, nnsolver_valuefun  
                    ):    
    
    start = time.perf_counter() 
    results = {}
    total_loss_coun = 0
    start_i = time.perf_counter() 
    for path in range(1,numTest-1):
        
        
        
        samples, loss_coun = IndividualTest(c0, #gamma, 
                                               path, T,
                                               d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
                                               d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1, 
                                               quantizer, t,
                                               input_scaler, out_scaler_valuefun, nnsolver_valuefun
                                            )
     
        results[path] = samples
        total_loss_coun += loss_coun 
        
        if path == numTest/2: 
            end_i = time.perf_counter() 
            duration_i = (end_i-start_i)/60
            print('50% done:' + str(duration_i) + " min.")
            
        elif path == numTest/4:
            print('25% done')
        elif path == numTest/4*3:
            print('75% done')
    
    end = time.perf_counter() 
    duration = (end-start)/60
    print("Duration: " + str(duration) + " min.")
        
    return results, total_loss_coun


#%% Testing data prep
#
numTest = numSim

d_ln_S1t_minus1_vali = {0: [d_ln_S1_minus1 for _ in range(numTest)]}
d_ln_S2t_minus1_vali = {0: [d_ln_S2_minus1 for _ in range(numTest)]}
d_ln_S3t_minus1_vali = {0: [d_ln_S3_minus1 for _ in range(numTest)]}
d_ln_Lt_minus1_vali = {0: [d_ln_L_minus1 for _ in range(numTest)]}
for t in range(1, T + 1):
    d_ln_S1t_minus1_vali[t] = d_ln_S1_vali[t - 1]
    d_ln_S2t_minus1_vali[t] = d_ln_S2_vali[t - 1]
    d_ln_S3t_minus1_vali[t] = d_ln_S3_vali[t - 1]
    d_ln_Lt_minus1_vali[t] = d_ln_L_vali[t - 1]




#%% Validation results

results_vali, total_loss_count_vali = RunTests(initial_capital,
                                                T,
                                                d_ln_S1_vali, d_ln_S2_vali, d_ln_S3_vali, d_ln_L_vali,
                                                d_ln_S1t_minus1_vali, d_ln_S2t_minus1_vali, d_ln_S3t_minus1_vali, d_ln_Lt_minus1_vali,
                                                quantize_grid, t,
                                                in_scaler, out_scaler, V_hat_theta 
                                                )

'question: how do we compare the utility coming from dividend? we simply sum them up, or do TVM at each time point?'


#%%
capit_NN = []
capit_1 = []
capit_2 = []
capit_3 = []
capit_sp = []
capit_ep = []

utili_NN = []
for path in results_vali:
    NN_c = results_vali[path][0][4][T-1]
    capit_NN.append(U(NN_c))
    
    allS1_c = results_vali[path][1][4][T-1]
    capit_1.append(U(allS1_c))
    
    allS2_c = results_vali[path][2][4][T-1]
    capit_2.append(U(allS2_c))
    
    allS3_c = results_vali[path][3][4][T-1]
    capit_3.append(U(allS3_c))
    
    sp_c = results_vali[path][4][4][T-1]
    capit_sp.append(U(sp_c))
    
    ep_c = results_vali[path][5][4][T-1]
    capit_ep.append(U(ep_c))
    
to_box_plot = [capit_NN, capit_1, 
               capit_2, capit_3, capit_sp, capit_ep]

plt.figure(figsize=(10, 7))

# Create boxplot
box = plt.boxplot(to_box_plot, labels=["NN", "1",
                                        "2", 
                                       "3", "4", "5"], 
                  patch_artist=True, showfliers=False)
# Add mean points
means = [np.mean(capit_NN), 
          np.mean(capit_1),
         np.mean(capit_2), np.mean(capit_3), np.mean(capit_sp), np.mean(capit_ep)]
plt.scatter(range(1, 7), means, color='red', label='Mean', zorder=3)

# Add labels and grid
plt.ylabel("Utility")
plt.grid(True)
plt.legend()

# Show plot
plt.show()

print('Pr(NN loss money)=',total_loss_count_vali/numSim)


#%% print some results in detail

some_paths = [random.randint(1, numTest-1) for _ in range(5)]

for some in range(1,len(some_paths)):
    print('')
    print(f'for tested path no.{some}')
    for t in range(1,T):
        print(
             f'u_t={t}:', round(results_vali[some_paths[some]][0][0][t],4),round(results_vali[some_paths[some]][0][1][t],4), round(results_vali[some_paths[some]][0][2][t],4), round(results_vali[some_paths[some]][0][3][t],7), 
             f'L_t={t} :', round(np.exp((ln_L_0) + sum((d_ln_L_vali[j][some_paths[some]]) for j in range(1, t))),2), 
             f'C_t={t} :', round(results_vali[some_paths[some]][0][4][t],2), 
             round(U(results_vali[some_paths[some]][0][4][t]),2)
              )


#%% distribution plot

bottomLine = U(initial_capital)

# Plotting the histograms
plt.figure(figsize=(10, 6))
plt.hist((capit_NN), bins=80, color='blue', edgecolor='black')
plt.axvline(x=bottomLine, color='red', linestyle='--', linewidth=2, label=initial_capital)

plt.title('Histogram of portfolio value')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

'universal approx. theorem, so dont need to assume some theta hat?'

#%% sensitivity

capitalizations = [capit_NN, capit_1, capit_2, capit_3, capit_sp, capit_ep]
labels = ["NN", "S1", "S2", "S3", "same proportion", "expected value"]

plt.figure(figsize=(10, 7))

# Plot empirical PMF for each capitalization
for i, cap in enumerate(capitalizations):
    # Create histogram to approximate PMF
    counts, bin_edges = np.histogram(cap, bins=30, density=True)
    
    # Bin centers for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Plot the empirical PMF
    plt.plot(bin_centers, counts, label=labels[i])

# Add labels, legend, and grid
plt.title("Empirical Probability Mass Function for Asset Allocations")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.grid(True)
plt.xlim(500, 5000)
plt.legend()

# Show plot
plt.show()




