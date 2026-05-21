import pandas as pd
import numpy as np
import sys
import os

# Set paths
sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.data import get_lookback_data
from hrp_engine.denoiser import denoise_covariance
from hrp_engine.hrp import optimize_hrp
from hrp_engine.backtest import get_rebalance_dates

# Load cached prices
prices_df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

# Clean prices
prices_clean = prices_df.ffill().bfill()
rebalance_dates = get_rebalance_dates(prices_clean.index, 'quarterly')

# Keep only rebalance dates that have 4 years of history
start_date = prices_clean.index[0]
valid_rebalance_dates = [
    d for d in rebalance_dates 
    if d >= start_date + pd.DateOffset(years=4) and d <= prices_clean.index[-1]
]

# We will run a simple walk-forward simulation for HRP, IVP, and EWP without tax/costs to compare raw performance
dates = prices_clean.loc[valid_rebalance_dates[0]:].index
hrp_vals = []
ivp_vals = []
ewp_vals = []

# Initial setup
T_0 = valid_rebalance_dates[0]
lookback_returns, active_assets = get_lookback_data(prices_df, T_0, 4)
cov_emp = lookback_returns.cov()
cov_denoised = denoise_covariance(cov_emp.values, len(lookback_returns))
cov_denoised_df = pd.DataFrame(cov_denoised, index=active_assets, columns=active_assets)

w_hrp = optimize_hrp(cov_denoised_df, 'single')

# IVP weights
diag_cov = np.diag(cov_denoised_df)
w_ivp = 1.0 / diag_cov
w_ivp /= w_ivp.sum()
w_ivp = pd.Series(w_ivp, index=active_assets)

# EWP weights
w_ewp = pd.Series(1.0 / len(active_assets), index=active_assets)

# To track daily values
hrp_val = 1.0
ivp_val = 1.0
ewp_val = 1.0

current_hrp_shares = w_hrp * hrp_val / prices_clean.loc[T_0, active_assets]
current_ivp_shares = w_ivp * ivp_val / prices_clean.loc[T_0, active_assets]
current_ewp_shares = w_ewp * ewp_val / prices_clean.loc[T_0, active_assets]

hrp_vals.append(hrp_val)
ivp_vals.append(ivp_val)
ewp_vals.append(ewp_val)

# Run simulation
rebalance_idx = 1
for i in range(1, len(dates)):
    curr_date = dates[i]
    
    # Calculate daily values
    hrp_val = (prices_clean.loc[curr_date, active_assets] * current_hrp_shares).sum()
    ivp_val = (prices_clean.loc[curr_date, active_assets] * current_ivp_shares).sum()
    ewp_val = (prices_clean.loc[curr_date, active_assets] * current_ewp_shares).sum()
    
    hrp_vals.append(hrp_val)
    ivp_vals.append(ivp_val)
    ewp_vals.append(ewp_val)
    
    # Check if rebalance date
    if rebalance_idx < len(valid_rebalance_dates) and curr_date >= valid_rebalance_dates[rebalance_idx]:
        T_k = valid_rebalance_dates[rebalance_idx]
        lookback_returns, active_assets = get_lookback_data(prices_df, T_k, 4)
        cov_emp = lookback_returns.cov()
        cov_denoised = denoise_covariance(cov_emp.values, len(lookback_returns))
        cov_denoised_df = pd.DataFrame(cov_denoised, index=active_assets, columns=active_assets)
        
        w_hrp = optimize_hrp(cov_denoised_df, 'single')
        
        diag_cov = np.diag(cov_denoised_df)
        w_ivp = 1.0 / diag_cov
        w_ivp /= w_ivp.sum()
        w_ivp = pd.Series(w_ivp, index=active_assets)
        
        w_ewp = pd.Series(1.0 / len(active_assets), index=active_assets)
        
        # Reset shares based on current portfolio value
        current_hrp_shares = w_hrp * hrp_val / prices_clean.loc[curr_date, active_assets]
        current_ivp_shares = w_ivp * ivp_val / prices_clean.loc[curr_date, active_assets]
        current_ewp_shares = w_ewp * ewp_val / prices_clean.loc[curr_date, active_assets]
        
        rebalance_idx += 1

# Calculate annualized returns
years = (dates[-1] - dates[0]).days / 365.25
ann_ret_hrp = (hrp_vals[-1]) ** (1.0 / years) - 1.0
ann_ret_ivp = (ivp_vals[-1]) ** (1.0 / years) - 1.0
ann_ret_ewp = (ewp_vals[-1]) ** (1.0 / years) - 1.0

print(f"Simulation Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({years:.2f} years)")
print(f"HRP Annualized Return (No TC/Tax): {ann_ret_hrp*100:.2f}% (Final value: {hrp_vals[-1]:.4f})")
print(f"IVP Annualized Return (No TC/Tax): {ann_ret_ivp*100:.2f}% (Final value: {ivp_vals[-1]:.4f})")
print(f"EWP Annualized Return (No TC/Tax): {ann_ret_ewp*100:.2f}% (Final value: {ewp_vals[-1]:.4f})")
