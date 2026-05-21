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

# Run simulation
dates = prices_clean.loc[valid_rebalance_dates[0]:].index

# Portfolios to simulate:
# 1. HRP Denoised
# 2. HRP Empirical (No denoising)
# 3. IVP Denoised
# 4. IVP Empirical (No denoising)
p_vals = {
    "HRP Denoised": [1.0],
    "HRP Empirical": [1.0],
    "IVP Denoised": [1.0],
    "IVP Empirical": [1.0]
}

# Initial setup
T_0 = valid_rebalance_dates[0]
lookback_returns, active_assets = get_lookback_data(prices_df, T_0, 4)
cov_emp = lookback_returns.cov()
cov_denoised = pd.DataFrame(denoise_covariance(cov_emp.values, len(lookback_returns)), index=active_assets, columns=active_assets)

w_hrp_denoised = optimize_hrp(cov_denoised, 'single')
w_hrp_emp = optimize_hrp(cov_emp, 'single')

w_ivp_denoised = pd.Series(1.0 / np.diag(cov_denoised), index=active_assets)
w_ivp_denoised /= w_ivp_denoised.sum()

w_ivp_emp = pd.Series(1.0 / np.diag(cov_emp), index=active_assets)
w_ivp_emp /= w_ivp_emp.sum()

shares = {
    "HRP Denoised": w_hrp_denoised * 1.0 / prices_clean.loc[T_0, active_assets],
    "HRP Empirical": w_hrp_emp * 1.0 / prices_clean.loc[T_0, active_assets],
    "IVP Denoised": w_ivp_denoised * 1.0 / prices_clean.loc[T_0, active_assets],
    "IVP Empirical": w_ivp_emp * 1.0 / prices_clean.loc[T_0, active_assets]
}

rebalance_idx = 1
for i in range(1, len(dates)):
    curr_date = dates[i]
    
    # Calculate daily values
    for name in p_vals.keys():
        val = (prices_clean.loc[curr_date, active_assets] * shares[name]).sum()
        p_vals[name].append(val)
        
    # Check if rebalance date
    if rebalance_idx < len(valid_rebalance_dates) and curr_date >= valid_rebalance_dates[rebalance_idx]:
        T_k = valid_rebalance_dates[rebalance_idx]
        lookback_returns, active_assets = get_lookback_data(prices_df, T_k, 4)
        cov_emp = lookback_returns.cov()
        cov_denoised = pd.DataFrame(denoise_covariance(cov_emp.values, len(lookback_returns)), index=active_assets, columns=active_assets)
        
        w_hrp_denoised = optimize_hrp(cov_denoised, 'single')
        w_hrp_emp = optimize_hrp(cov_emp, 'single')
        
        w_ivp_denoised = pd.Series(1.0 / np.diag(cov_denoised), index=active_assets)
        w_ivp_denoised /= w_ivp_denoised.sum()
        
        w_ivp_emp = pd.Series(1.0 / np.diag(cov_emp), index=active_assets)
        w_ivp_emp /= w_ivp_emp.sum()
        
        # Reset shares
        shares["HRP Denoised"] = w_hrp_denoised * p_vals["HRP Denoised"][-1] / prices_clean.loc[curr_date, active_assets]
        shares["HRP Empirical"] = w_hrp_emp * p_vals["HRP Empirical"][-1] / prices_clean.loc[curr_date, active_assets]
        shares["IVP Denoised"] = w_ivp_denoised * p_vals["IVP Denoised"][-1] / prices_clean.loc[curr_date, active_assets]
        shares["IVP Empirical"] = w_ivp_emp * p_vals["IVP Empirical"][-1] / prices_clean.loc[curr_date, active_assets]
        
        rebalance_idx += 1

years = (dates[-1] - dates[0]).days / 365.25
print(f"Simulation Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({years:.2f} years)")
for name in p_vals.keys():
    ann_ret = (p_vals[name][-1]) ** (1.0 / years) - 1.0
    print(f"  {name:<15} | Ann. Return: {ann_ret*100:5.2f}% | Final Value: {p_vals[name][-1]:.4f}")
