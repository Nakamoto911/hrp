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

# Load cached prices
prices_df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
rebalance_date = pd.Timestamp('2026-05-19')

# Calculate raw target weights
lookback_returns, active_assets = get_lookback_data(prices_df, rebalance_date, 4)
cov_emp = lookback_returns.cov().values
n_obs = len(lookback_returns)
cov_denoised = denoise_covariance(cov_emp, n_obs)
cov_denoised_df = pd.DataFrame(cov_denoised, index=active_assets, columns=active_assets)

w_star = optimize_hrp(cov_denoised_df, 'single')
print("Raw HRP target weights (w_star):")
print(w_star.sort_values(ascending=False).to_string())
print(f"Sum of w_star: {w_star.sum():.6f}")
