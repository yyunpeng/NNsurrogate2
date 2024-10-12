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

#%% global parameters and quantizer set up

T=5
num_processes = 4

directory = '/Users/xuyunpeng/Documents/yourPath'

dataframes = []

for i in range(1, 6):
    file_path = os.path.join(directory, f'quantizer_vecBrownian/quantization_grid_t_{i}.csv')
    df = pd.read_csv(file_path)
    dataframes.append(df)

number_of_rows = dataframes[0].shape[0] 
quantize_grid = np.zeros((T, number_of_rows,  num_processes))

for i, df in enumerate(dataframes):
    quantize_grid[i] = df.values

weights = np.full((number_of_rows,), 1/number_of_rows)


#%% claims data

print('unit is per thousand for monetary amounts')
# policies inforce since 2020-03-01
database = pd.read_csv('/Users/xuyunpeng/Documents/yourPath/Monthly_Provisional_Counts_of_Deaths_by_Select_Causes__2020-2023_20240826.csv')
selected_data = database.iloc[1:40, 6] #from 1/1/20 to 4/1/23

num_death = selected_data*0.01 
inflation_rates = {
    2020: [0.001, 0.001, 0.002, -0.004, -0.008, -0.001, 0.006, 0.006, 0.004, 0.000, 0.002, 0.004],
    2021: [0.003, 0.004, 0.006, 0.008, 0.006, 0.009, 0.005, 0.003, 0.004, 0.009, 0.008, 0.005],
    2022: [0.006, 0.008, 0.012, 0.003, 0.010, 0.013, 0.000, 0.001, 0.004, 0.004, 0.001, -0.001],
    2023: [0.005, 0.004, 0.001]
} # https://www.investing.com/economic-calendar/cpi-69

initial_benefit = 10 # means 10k death benefit
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


#%% claim and stock price stack into 1 df and find VAR model

top3 = ['PFE', 'REM', 'BND']

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


#%% prices simulation for testing and validation

numTrain = 1024
numSim = numTrain

def VARMA_sim1(current_d_ln_x, last_d_ln_x, Sigma, numSim, T, validation=False):
    
    noise_test, d_ln_S1_test, d_ln_S2_test, d_ln_S3_test, d_ln_L_test = {}, {}, {}, {}, {}
    
    for t in range(T+5):  # Including 0 for noise
        if t != 0: # For d_R1, d_R2, d_R3, d_L, we start from t=1
            d_ln_S1_test[t], d_ln_S2_test[t], d_ln_S3_test[t], d_ln_L_test[t] = [], [], [], []
        noise_test[t] = []
    
    for path in range(numSim):
        noise = {}
        noise[0] = np.random.normal(0, 1, 4)

        current_d_ln_x_temp = current_d_ln_x.copy()  
        
        last_d_ln_x_temp = last_d_ln_x.copy()
        
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

actual_list_of_top3 = ['BND', 'PFE', 'REM']

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

top3 = ['PFE', 'REM', 'BND']
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

y = initial_benefit*1.2
gamma = 0.9
v = 0.8

#loss is 10k average, capital is 10mil average
c_min = initial_capital-y*(T-1)
c_max = initial_capital+y*(T-1)

# sobol sequences used to cover the 13-dimensional space more uniformly than random sampling
sobol = qmc.Sobol(d=13)
sobol_samples = sobol.random(n=numTrain)

d_ln_S1_range = [-0.06, 0.08]
d_ln_S2_range = [-0.20, 0.30]
d_ln_S3_range = [-0.40, 0.30]
d_ln_L_range = [-0.40, 0.40]
S1_range = [34, 47]
S2_range = [10, 40]
S3_range = [20, 120]
L_range = [4, 18]

d_ln_S1t_train = sobol_samples[:, 0] * (d_ln_S1_range[1] - d_ln_S1_range[0]) + d_ln_S1_range[0]
d_ln_S1t_minus1_train = sobol_samples[:, 1] * (d_ln_S1_range[1] - d_ln_S1_range[0]) + d_ln_S1_range[0]

d_ln_S2t_train = sobol_samples[:, 2] * (d_ln_S2_range[1] - d_ln_S2_range[0]) + d_ln_S2_range[0]
d_ln_S2t_minus1_train = sobol_samples[:, 3] * (d_ln_S2_range[1] - d_ln_S2_range[0]) + d_ln_S2_range[0]

d_ln_S3t_train = sobol_samples[:, 4] * (d_ln_S3_range[1] - d_ln_S3_range[0]) + d_ln_S3_range[0]
d_ln_S3t_minus1_train = sobol_samples[:, 5] * (d_ln_S3_range[1] - d_ln_S3_range[0]) + d_ln_S3_range[0]

d_ln_Lt_train = sobol_samples[:, 6] * (d_ln_L_range[1] - d_ln_L_range[0]) + d_ln_L_range[0]
d_ln_Lt_minus1_train = sobol_samples[:, 7] * (d_ln_L_range[1] - d_ln_L_range[0]) + d_ln_L_range[0]

S1t_train = sobol_samples[:, 8] * (S1_range[1] - S1_range[0]) + S1_range[0]
S2t_train = sobol_samples[:, 9] * (S2_range[1] - S2_range[0]) + S2_range[0]
S3t_train = sobol_samples[:, 10] * (S3_range[1] - S3_range[0]) + S3_range[0]
Lt_train = sobol_samples[:, 11] * (L_range[1] - L_range[0]) + L_range[0]

c_train = sobol_samples[:, 12] * (c_max - c_min) + c_min


#%% terminal value function set up

def V_T(C_T):
    return 1/gamma * np.sign(C_T) * (np.abs(C_T)) ** gamma

def V_T_minus1(c, u , 
                d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
                d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
                S1t, S2t, S3t, Lt, 
                quantizer, t=T-1
                ):
    
    d_ln_Xt_vec = np.array([d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt]) 
    d_ln_Xt_minus1_vec = np.array([d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1]) 
    # noise_t_vec = np.array([noise_R1, noise_R2, noise_R3, noise_L])    

    S1_T = np.exp(
                  np.log(S1t) + (mu[0] + np.dot(Phi1, d_ln_Xt_vec-mu)[0] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[0] #+ dotProduc(matrix_beta,noise_t_vec)[0]
                    + quantizer[t][:, 0])
                  )
    S2_T = np.exp(
                  np.log(S2t) + (mu[1] + np.dot(Phi1, d_ln_Xt_vec-mu)[1] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[1] #+ dotProduc(matrix_beta,noise_t_vec)[0]
                    + quantizer[t][:, 1])
                  )
    S3_T = np.exp(
                  np.log(S2t) + (mu[2] + np.dot(Phi1, d_ln_Xt_vec-mu)[2] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[2] #+ dotProduc(matrix_beta,noise_t_vec)[0]
                    + quantizer[t][:, 2])
                  )
    L_T = np.exp(
                np.log(Lt) + (mu[3] + np.dot(Phi1, d_ln_Xt_vec)[3] + np.dot(Phi2, d_ln_Xt_minus1_vec)[3] #+ dotProduc(matrix_beta,noise_t_vec)[3]
                    + quantizer[t][:, 3])
                )
    
    R1_T = S1_T/S1t
    R2_T = S2_T/S1t
    R3_T = S3_T/S1t

    u2 = 1-u[0]-u[1]
    
    C_T = ( u[0]*R1_T + u[1]*R2_T + u2*R3_T )*(c + y - u[2]*c) - L_T
    
    E_V_T = np.sum( weights * V_T(C_T) )
    
    V_T_minus1 = 1/gamma * np.sign(u[2]*c) * (np.abs(u[2]*c)) ** gamma + v * E_V_T
    
    return V_T_minus1


#%% NN Predictor set up

def NN_Surrogate(c1,
                d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
                d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
                S1t, S2t, S3t, Lt, 
                nnweights, inputscaler, outputscaler, scaleOutput = 1):
    
    inputdata = np.concatenate((
                                c1.reshape(-1,1),
                                
                                d_ln_S1t.reshape(-1,1), 
                                d_ln_S2t.reshape(-1,1), 
                                d_ln_S3t.reshape(-1,1), 
                                d_ln_Lt.reshape(-1,1), 
                                
                                d_ln_S1t_minus1.reshape(-1,1), 
                                d_ln_S2t_minus1.reshape(-1,1), 
                                d_ln_S3t_minus1.reshape(-1,1), 
                                d_ln_Lt_minus1.reshape(-1,1), 
                                                                                                
                                S1t.reshape(-1,1), 
                                S2t.reshape(-1,1), 
                                S3t.reshape(-1,1), 
                                Lt.reshape(-1,1),
                                
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
    
    layer4out = tf.keras.activations.elu(layer4out).numpy()
    
    layer5out = np.dot(layer4out, nnweights[8]) + nnweights[9]
    
    layer5out = tf.keras.activations.elu(layer5out).numpy()
    
    layer6out = np.dot(layer5out, nnweights[10]) + nnweights[11]
    
    layer6out = tf.keras.activations.elu(layer6out).numpy()
    
    layer7out = np.dot(layer6out, nnweights[12]) + nnweights[13]
    
    layer7out = tf.keras.activations.elu(layer7out).numpy()
    
    layer8out = np.dot(layer7out, nnweights[14]) + nnweights[15]
    
    if scaleOutput == 0:   
        output = tf.keras.activations.sigmoid(layer8out).numpy() 
    if scaleOutput == 1:  
        output = outputscaler.inverse_transform(layer8out)
    
    return output


#%% value function surrogate set up

def V_t(c, u, 
        d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
        d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
        S1t, S2t, S3t, Lt, 
        # noise_R1, noise_R2, noise_R3, noise_L,
        nnweights, inputscaler, outputscaler, quantizer, t
        ):
    
    numWeights = len(quantizer[0])
    
    d_ln_Xt_vec = np.array([d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt]) 
    d_ln_Xt_minus1_vec = np.array([d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1]) 
    # noise_t_vec = np.array([noise_R1, noise_R2, noise_R3, noise_L])    

    S1_t_plus1 = np.exp(
                        np.log(S1t) + (mu[0] + np.dot(Phi1, d_ln_Xt_vec-mu)[0] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[0] #+ dotProduc(matrix_beta,noise_t_vec)[0]
                         + quantizer[t][:, 0])
                        )
    S2_t_plus1 = np.exp(
                        np.log(S2t) + (mu[1] + np.dot(Phi1, d_ln_Xt_vec-mu)[1] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[1] #+ dotProduc(matrix_beta,noise_t_vec)[0]
                         + quantizer[t][:, 1])
                        )
    S3_t_plus1 = np.exp(
                        np.log(S3t) + (mu[2] + np.dot(Phi1, d_ln_Xt_vec-mu)[2] + np.dot(Phi2, d_ln_Xt_minus1_vec-mu)[2] #+ dotProduc(matrix_beta,noise_t_vec)[0]
                         + quantizer[t][:, 2])
                        )
    L_t_plus1 = np.exp(
            np.log(Lt) + (mu[3] + np.dot(Phi1, d_ln_Xt_vec)[3] + np.dot(Phi2, d_ln_Xt_minus1_vec)[3] #+ dotProduc(matrix_beta,noise_t_vec)[3]
             + quantizer[t][:, 3])
            )
    
    R1_t_plus1 = S1_t_plus1/S1t
    R2_t_plus1 = S2_t_plus1/S1t
    R3_t_plus1 = S3_t_plus1/S1t
    u2 = 1-u[0]-u[1]
    
    V_t_plus1 = NN_Surrogate(
                       np.ones(numWeights) * (u[0]*R1_t_plus1 + u[1]*R2_t_plus1 + u2*R3_t_plus1) * (c + y - u[2]*c) - L_t_plus1 ,
                       
                       np.ones(numWeights) * d_ln_S1t, 
                       np.ones(numWeights) * d_ln_S2t, 
                       np.ones(numWeights) * d_ln_S3t, 
                       np.ones(numWeights) * d_ln_Lt, 
                       
                       np.ones(numWeights) * d_ln_S1t_minus1, 
                       np.ones(numWeights) * d_ln_S2t_minus1, 
                       np.ones(numWeights) * d_ln_S3t_minus1, 
                       np.ones(numWeights) * d_ln_Lt_minus1, 
                       
                       np.ones(numWeights) * S1t, 
                       np.ones(numWeights) * S2t, 
                       np.ones(numWeights) * S3t, 
                       np.ones(numWeights) * Lt, 
                
                   nnweights, inputscaler, outputscaler)
        
    E_V_t_plus1 = np.sum(weights * V_t_plus1)
    
    V_t = 1/gamma * np.sign(u[2]*c) * (np.abs(u[2]*c)) ** gamma + v * E_V_t_plus1
                
    return V_t

#%% train

div_upper = 0.05

def custom_activation(x):
    return tf.nn.sigmoid(x)*div_upper

def BuildAndTrainModel(c1_train, #gamma_train, 
                       d_ln_S1t_train, d_ln_S2t_train, d_ln_S3t_train, d_ln_Lt_train,
                       d_ln_S1t_minus1_train, d_ln_S2t_minus1_train, d_ln_S3t_minus1_train, d_ln_Lt_minus1_train,
                       S1_t_train, S2_t_train, S3_t_train, L_t_train, 

                       quantizer, 
                       
                       nn_dim = 13, 
                       node_num = 300, 
                       batch_num = 128, 
                       epoch_num = 500, 
                       initializer = TruncatedNormal(mean = 0.0, stddev = 0.05, seed = 0) 
                       
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
                                    
                                    S1_t_train.reshape(-1,1), 
                                    S2_t_train.reshape(-1,1), 
                                    S3_t_train.reshape(-1,1), 
                                    L_t_train.reshape(-1,1),

                                    ), axis = 1) 
    
    
    input_scaler = MinMaxScaler(feature_range = (0,1))
    input_scaler.fit(input_train)
    input_train_scaled = input_scaler.transform(input_train)
    
    valuefun_train = np.zeros((T+1, numTrain))
    proportion_train = np.zeros((3, T+1, numTrain))
    dividend_train = np.zeros((T+1, numTrain))
    
    output_scaler_valuefun = np.empty(T+1, dtype = object)
    nnsolver_valuefun = np.empty(T+1, dtype = object)
    nnsolver_proportion = np.empty(T+1, dtype=object)
    nnsolver_dividend = np.empty(T+1, dtype=object)
        
    start = time.perf_counter() 
    check = {}
    
    for j in range(T-1, 0, -1): # j is equivalent to t
    
        input_scaler = MinMaxScaler(feature_range = (0,1))
        input_scaler.fit(input_train)
        input_train_scaled = input_scaler.transform(input_train)
                    
        start_i = time.perf_counter()
        print("Time step " + str(j))

        for i in range(numTrain):
            
            
            if j < (T-1):
                output_scaler = output_scaler_valuefun[j+1]  
                
                def f_i(u):
                        V = V_t(
                                c1_train[i], 
                                u,
                                d_ln_S1t_train[i], d_ln_S2t_train[i], d_ln_S3t_train[i], d_ln_Lt_train[i],
                                d_ln_S1t_minus1_train[i], d_ln_S2t_minus1_train[i], d_ln_S3t_minus1_train[i], d_ln_Lt_minus1_train[i], 
                                S1_t_train[i], S2_t_train[i], S3_t_train[i], L_t_train[i],
                                nnsolver_valuefun[j+1].get_weights(),
                                input_scaler, output_scaler, quantizer, j
                                )
                        return -1*V 

            else:

                def f_i(u):
                        V = V_T_minus1(
                                       c1_train[i], 
                                       u,
                                       d_ln_S1t_train[i], d_ln_S2t_train[i], d_ln_S3t_train[i], d_ln_Lt_train[i],
                                       d_ln_S1t_minus1_train[i], d_ln_S2t_minus1_train[i], d_ln_S3t_minus1_train[i], d_ln_Lt_minus1_train[i], 
                                       S1_t_train[i], S2_t_train[i], S3_t_train[i], L_t_train[i],
                                       quantizer, j
                                       )                        
                        return -1*V 
            
            bounds = [(0, 1), (0, 1), (0, 0.005)]
            guess_u = np.array([1/3,1/3,0.003])
            optio = {
                        'maxiter': 10000
                    }
            result = minimize(f_i, guess_u, method='L-BFGS-B', options=optio, bounds=bounds)
            if not result.success:
                print("Optimization failed:", result.message)
            
            if i == int(numTrain/2) or i == int(numTrain/4*3) or i == int(numTrain/4):
                print(f'          {i}th optimization done')
                
            v_hat = result.fun*-1
            u_hat = result.x
            
            proportion_train[0][j][i], proportion_train[1][j][i], proportion_train[2][j][i] = u_hat[0], u_hat[1], 1-u_hat[0]-u_hat[1]
            dividend_train[j][i] = u_hat[-1] 
            valuefun_train[j][i] = v_hat
        
        end_i = time.perf_counter()
        print("     all optimizations done: " + str(round((end_i-start_i)/60,2)) + " min.")
        
        start_i = time.perf_counter()
        output_scaler_valuefun[j] = MinMaxScaler(feature_range = (0,1))
        output_scaler_valuefun[j].fit(valuefun_train[j].reshape(-1, 1))
        valuefun_train_scaled = output_scaler_valuefun[j].transform(valuefun_train[j].reshape(-1,1))    
        nnsolver_valuefun[j] = Sequential([
                                        Input(shape=(nn_dim,)),  # Explicit input layer specification
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                        Dense(1, activation=None, kernel_initializer=initializer, bias_initializer=initializer)
                                        ])
        optimizer = Adam(learning_rate=0.0001)
        nnsolver_valuefun[j].compile(optimizer = optimizer, loss = 'mean_squared_error')
        nnsolver_valuefun[j].fit(input_train_scaled, valuefun_train_scaled,
                              epochs = epoch_num, batch_size = batch_num, verbose = 0)
        end_i = time.perf_counter()
        print("     train value function done: " + str(round((end_i-start_i)/60,2)) + " min.")     
        
        valuefun_train_scaled_pred = nnsolver_valuefun[j].predict(input_train_scaled)
        mse_valuefun = mean_squared_error(valuefun_train_scaled, valuefun_train_scaled_pred)
        print(f"     MSE for value function: {mse_valuefun}")
        
        start_i = time.perf_counter()        
        nnsolver_proportion[j] = Sequential([
                                    Input(shape=(nn_dim,)),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),                                    
                                    Dense(3, activation='softmax', kernel_initializer=initializer, bias_initializer=initializer)
                                    ])
        optimizer = Adam(learning_rate=0.00001)
        nnsolver_proportion[j].compile(optimizer=optimizer, loss='mean_squared_error')
        nnsolver_proportion[j].fit(input_train_scaled, proportion_train[:, j, :].T,
                               epochs=epoch_num, batch_size=batch_num, verbose=0) 
        end_i = time.perf_counter()
        print("     train proportion done: " + str(round((end_i-start_i)/60,2)) + " min.")
        
        proportion_train_pred = nnsolver_proportion[j].predict(input_train_scaled)
        mse_proportion = mean_squared_error(proportion_train[:, j, :].T, proportion_train_pred)
        print(f"     MSE for proportion: {mse_proportion}")
        
        start_i = time.perf_counter()   

        nnsolver_dividend[j] = Sequential([
                                    Input(shape=(nn_dim,)),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),
                                    Dense(node_num, activation='elu', kernel_initializer=initializer, bias_initializer=initializer),      
                                    Dense(1, activation=custom_activation, kernel_initializer=initializer, bias_initializer=initializer)
                                    ])
        optimizer = Adam(learning_rate=0.0001)
        nnsolver_dividend[j].compile(optimizer=optimizer, loss='mean_squared_error')
        nnsolver_dividend[j].fit(input_train_scaled, dividend_train[j].reshape(-1, 1),
                               epochs=epoch_num, batch_size=batch_num, verbose=0) 
        end_i = time.perf_counter()
        print("     train dividend done: " + str(round((end_i-start_i)/60,2)) + " min.")
        
        dividend_train_pred = nnsolver_dividend[j].predict(input_train_scaled)
        mse_dividend = mean_squared_error(dividend_train[j].reshape(-1, 1), dividend_train_pred)
        print(f"     MSE for dividend: {mse_dividend}")
        
    end = time.perf_counter()
    duration = (end-start)/60

    print("Duration: " + str(duration) + " min.")
    
    return nnsolver_proportion, nnsolver_dividend, nnsolver_valuefun, input_scaler, output_scaler_valuefun

#%% Train

nnsolver_proportion, nnsolver_dividend, nnsolver_valuefun, in_scaler, out_scaler_valuefun, check_minimise \
= BuildAndTrainModel(c_train, #gamma_train, 
                        d_ln_S1t_train, d_ln_S2t_train, d_ln_S3t_train, d_ln_Lt_train,
                        d_ln_S1t_minus1_train, d_ln_S2t_minus1_train, d_ln_S3t_minus1_train, d_ln_Lt_minus1_train, 
                        S1t_train, S2t_train, S3t_train, Lt_train, 
                     quantize_grid)



#%% save locally

np.save('/Users/xuyunpeng/Documents/yourPath/nnsolver_proportion_21-3', nnsolver_proportion)
np.save('/Users/xuyunpeng/Documents/yourPath/nnsolver_valuefun_21-3', nnsolver_valuefun)
np.save('/Users/xuyunpeng/Documents/yourPath/out_scaler_valuefun_21-3', out_scaler_valuefun)

for j in range(1,T):
    nnsolver_dividend[j].save(f'nnsolver_dividend_21-3_t{j}.keras')

file_path = '/Users/xuyunpeng/Documents/yourPath/in_scaler_21-3.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(in_scaler, file)


#%% retrieve models

print('remember to change working directory, otherwise File not found')

loaded_proportion = np.load('/Users/xuyunpeng/Documents/yourPath/nnsolver_proportion_21-1.npy', allow_pickle=True)

loaded_dividend = []
for j in range(1,T):
    loaded_dividend.append(load_model(f'nnsolver_dividend_21-1_t{j}.keras', custom_objects={'custom_activation': custom_activation}))
loaded_dividend = [None] + loaded_dividend + [None]
    
file_path = '/Users/xuyunpeng/Documents/yourPath/in_scaler_21-1.pkl'
with open(file_path, 'rb') as file:
    loaded_in_scaler = pickle.load(file)
    

#%% test one path

def IndividualTest(c0,
                   nnsolver_proportion, nnsolver_dividend, path, input_scaler, T,
                   d_ln_S1t, d_ln_S2t, d_ln_S3t, d_ln_Lt,
                   d_ln_S1t_minus1, d_ln_S2t_minus1, d_ln_S3t_minus1, d_ln_Lt_minus1,
                   ):
    
    samples = np.ones((5, 5, T+2))  
    # 5 strategies; 4 controls + 1 capital; T+2 horizon
                                                   
    samples[:,4,0:T+2] = c0
    
    for t in range(1, T):
                
        if t < (T):
            
            # NN strategy starts here"
            
            u_t = np.empty(3)
            
            for i in range(3):
                
                S1_t = np.exp((ln_S1_0) + sum((d_ln_S1t[j][path]) for j in range(1, t)))
                S2_t = np.exp((ln_S2_0) + sum((d_ln_S2t[j][path]) for j in range(1, t)))
                S3_t = np.exp((ln_S3_0) + sum((d_ln_S3t[j][path]) for j in range(1, t)))
                L_t = np.exp((ln_L_0) + sum((d_ln_Lt[j][path]) for j in range(1, t)))
            
                u_t[i] = NN_Surrogate(samples[0][4][t],  
                                             
                                             d_ln_S1t[t][path], 
                                             d_ln_S2t[t][path], 
                                             d_ln_S3t[t][path], 
                                             d_ln_Lt[t][path],         
                                             
                                             d_ln_S1t_minus1[t][path], 
                                             d_ln_S2t_minus1[t][path], 
                                             d_ln_S3t_minus1[t][path], 
                                             d_ln_Lt_minus1[t][path], 
                                             
                                             np.array(S1_t), 
                                             np.array(S2_t), 
                                             np.array(S3_t), 
                                             np.array(L_t), 
                                             
                                             nnsolver_proportion[t].get_weights(), input_scaler, 
                                             outputscaler=None, scaleOutput=0)[0,i]
            
            
            
            # make sure they are proportions and sum up to 1
            u_t[0] = u_t[0]/sum(u_t[i] for i in range(3))
            u_t[1] = u_t[1]/sum(u_t[i] for i in range(3))
            u_t[2] = u_t[2]/sum(u_t[i] for i in range(3))
            
            samples[0][3][t] = NN_Surrogate(samples[0][4][t], 
                                             
                                              d_ln_S1t[t][path], 
                                              d_ln_S2t[t][path], 
                                              d_ln_S3t[t][path], 
                                              d_ln_Lt[t][path],         
                                             
                                              d_ln_S1t_minus1[t][path], 
                                              d_ln_S2t_minus1[t][path], 
                                              d_ln_S3t_minus1[t][path], 
                                              d_ln_Lt_minus1[t][path], 
                                             
                                              np.array(S1_t), 
                                              np.array(S2_t), 
                                              np.array(S3_t), 
                                              np.array(L_t), 
                                             
                                               nnsolver_dividend[t].get_weights(), input_scaler, 
                                               outputscaler=None, scaleOutput=0)[0][0]*div_upper
            

            
            h1_t = u_t[0]*(samples[0][4][t]+y- samples[0][3][t]*samples[0][4][t])/(S1_t)
            h2_t = u_t[1]*(samples[0][4][t]+y- samples[0][3][t]*samples[0][4][t])/(S2_t)
            h3_t = u_t[2]*(samples[0][4][t]+y- samples[0][3][t]*samples[0][4][t])/(S3_t)
            
            samples[0][0][t] = h1_t
            samples[0][1][t] = h2_t
            samples[0][2][t] = h3_t

            samples[0][4][t+1] = \
            (samples[0][0][t] * np.exp((ln_S1_0) + sum((d_ln_S1t[j][path]) for j in range(1, t+1))) +
             samples[0][1][t] * np.exp((ln_S2_0) + sum((d_ln_S2t[j][path]) for j in range(1, t+1))) +
             samples[0][2][t] * np.exp((ln_S3_0) + sum((d_ln_S3t[j][path]) for j in range(1, t+1))) ) -\
                (samples[0][4][t]*samples[0][3][t]) - \
                    np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )
            
            
            # all in one stock strategies (3 of them) start here
            all_S1_h_t = 1*(samples[1][4][t]+y- samples[0][3][t]*samples[1][4][t])/(S1_t)
            
            samples[1][4][t+1] = \
                (all_S1_h_t * np.exp((ln_S1_0) + sum((d_ln_S1t[j][path]) for j in range(1, t+1))) ) -\
                (samples[1][4][t]*samples[0][3][t]) - \
                np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )
                        
            all_S2_h_t = 1*(samples[2][4][t]+y- samples[0][3][t]*samples[2][4][t])/(S2_t)
            
            samples[2][4][t+1] = \
                (all_S2_h_t * np.exp((ln_S2_0) + sum((d_ln_S2t[j][path]) for j in range(1, t+1))) ) -\
                (samples[2][4][t]*samples[0][3][t]) - \
                np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )
                        
            all_S3_h_t = 1*(samples[3][4][t]+y- samples[0][3][t]*samples[3][4][t])/(S3_t)
            
            samples[3][4][t+1] = \
                (all_S3_h_t * np.exp((ln_S3_0) + sum((d_ln_S3t[j][path]) for j in range(1, t+1))) ) -\
                (samples[3][4][t]*samples[0][3][t]) - \
                np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )
             
                
            # same proportion strategy starts here
            sp_h1_t = 1/3*(samples[0][4][t]+y- samples[0][3][t]*samples[0][4][t])/(S1_t)
            sp_h2_t = 1/3*(samples[0][4][t]+y- samples[0][3][t]*samples[0][4][t])/(S2_t)
            sp_h3_t = 1/3*(samples[0][4][t]+y- samples[0][3][t]*samples[0][4][t])/(S3_t)
            
            samples[4][4][t+1] = \
            (sp_h1_t * np.exp((ln_S1_0) + sum((d_ln_S1t[j][path]) for j in range(1, t+1))) +
             sp_h2_t * np.exp((ln_S2_0) + sum((d_ln_S2t[j][path]) for j in range(1, t+1))) +
             sp_h3_t * np.exp((ln_S3_0) + sum((d_ln_S3t[j][path]) for j in range(1, t+1))) ) -\
                (samples[4][4][t]*samples[0][3][t]) - \
                    np.exp(ln_L_0 + sum((d_ln_Lt[j][path]) for j in range(1, t+1)) )
                    
    loss_count = 0
    minimum_capital = c0 - sum(
                            np.exp(ln_L_0 + sum(
                                d_ln_Lt[j][path] for j in range(1, t))
                                ) for t in range(1, T)
                            ) + (T-1)*y 
    if samples[0][4][T-1] < minimum_capital or samples[0][4][T-1] < samples[1][4][T-1] or samples[0][4][T-1] < samples[2][4][T-1] or samples[0][4][T-1] < samples[3][4][T-1] or samples[0][4][T-1] < samples[4][4][T-1] :
        loss_count = 1
        
    return samples, loss_count


def RunTests(c0, 
             nnsolver_proportion, nnsolver_dividend, input_scaler, T, numSim,
                    d_ln_R1t, d_ln_R2t, d_ln_R3t, d_ln_Lt,
                    d_ln_R1t_minus1, d_ln_R2t_minus1, d_ln_R3t_minus1, d_ln_Lt_minus1,
                    ):    
    
    results = {}
    total_loss_coun = 0

    for path in range(1,numSim):
        
        samples, loss_coun = IndividualTest(c0, 
                                            nnsolver_proportion, nnsolver_dividend, path, input_scaler, T,
                                            d_ln_R1t, d_ln_R2t, d_ln_R3t, d_ln_Lt,
                                            d_ln_R1t_minus1, d_ln_R2t_minus1, d_ln_R3t_minus1, d_ln_Lt_minus1,
                                            )
        
        results[path] = samples
        total_loss_coun += loss_coun 

    return results, total_loss_coun

#%% Testing data prep

d_ln_S1t_vali = {t: d_ln_S1_vali[t] for t in range(1, T + 1)}
d_ln_S2t_vali = {t: d_ln_S2_vali[t] for t in range(1, T + 1)}
d_ln_S3t_vali = {t: d_ln_S3_vali[t] for t in range(1, T + 1)}
d_ln_Lt_vali = {t: d_ln_L_vali[t] for t in range(1, T + 1)}

d_ln_S1t_minus1_vali = {1: [d_ln_S1_0 for _ in range(numSim)]}
d_ln_S2t_minus1_vali = {1: [d_ln_S2_0 for _ in range(numSim)]}
d_ln_S3t_minus1_vali = {1: [d_ln_S3_0 for _ in range(numSim)]}
d_ln_Lt_minus1_vali = {1: [d_ln_L_0 for _ in range(numSim)]}
for t in range(2, T + 1):
    d_ln_S1t_minus1_vali[t] = d_ln_S1t_vali[t - 1]
    d_ln_S2t_minus1_vali[t] = d_ln_S2t_vali[t - 1]
    d_ln_S3t_minus1_vali[t] = d_ln_S3t_vali[t - 1]
    d_ln_Lt_minus1_vali[t] = d_ln_Lt_vali[t - 1]

def U(x):
    return 1/gamma * np.sign(x) * (np.abs(x)) ** gamma

#%% Validation results


results_vali, total_loss_count_vali = RunTests(initial_capital, #np.array(gamma_test),
                                     
                    # loaded_proportion, loaded_dividend, loaded_in_scaler, 
                    nnsolver_proportion, nnsolver_dividend, in_scaler,
                    
                    T, numSim,
                    d_ln_S1t_vali, d_ln_S2t_vali, d_ln_S3t_vali, d_ln_Lt_vali,
                    d_ln_S1t_minus1_vali, d_ln_S2t_minus1_vali, d_ln_S3t_minus1_vali, d_ln_Lt_minus1_vali
                   )

capit_NN = []
capit_1 = []
capit_2 = []
capit_3 = []
capit_sp = []

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
    
    utili_NN.append(1/gamma * np.sign(NN_c) * (np.abs(NN_c)) ** gamma)

to_box_plot = [capit_NN,capit_1, capit_2, capit_3, capit_sp]

plt.figure(figsize=(10, 6))

# Create boxplot
box = plt.boxplot(to_box_plot, labels=["NN", "S1", "S2", "S3", "same proportion"], patch_artist=True)

# Add mean points
means = [np.mean(capit_NN), np.mean(capit_1), np.mean(capit_2), np.mean(capit_3), np.mean(capit_sp)]
plt.scatter(range(1, 6), means, color='red', label='Mean', zorder=3)

# Add labels and grid
plt.title("Boxplot of Different Asset Allocation Strategies (with Mean)")
plt.ylabel("Utility")
plt.grid(True)
plt.legend()

# Show plot
plt.show()

print('Pr(NN loss money)=',total_loss_count_vali/numSim)


#%% print some results in detail

some_paths = [random.randint(1, numSim-1) for _ in range(5)]

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

#%%

import matplotlib.pyplot as plt

import random

bottomLine = U(initial_capital)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(utili_NN, bins=80, color='blue', edgecolor='black')
plt.axvline(x=bottomLine, color='red', linestyle='--', linewidth=2, label=initial_capital)

plt.title('Histogram of utility')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


bottomLine = (initial_capital)

# Plotting the histograms
plt.figure(figsize=(10, 6))
plt.hist(capit_NN, bins=80, color='blue', edgecolor='black')
plt.axvline(x=bottomLine, color='red', linestyle='--', linewidth=2, label=initial_capital)

plt.title('Histogram of portfolio value')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



