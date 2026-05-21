import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.data import get_lookback_data
from hrp_engine.denoiser import denoise_covariance

# Load cached prices
prices_df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

# Slices prices to a sample lookback window
rebalance_date = pd.Timestamp('2022-03-30')
lookback_returns, active_assets = get_lookback_data(prices_df, rebalance_date, 4)

cov_emp = lookback_returns.cov().values
N = cov_emp.shape[0]
std = np.sqrt(np.diag(cov_emp))
corr_emp = cov_emp / np.outer(std, std)
corr_emp = np.clip(corr_emp, -1.0, 1.0)

eigenvalues, _ = np.linalg.eigh(corr_emp)
eigenvalues = eigenvalues[::-1] # descending

# Standard MP estimation:
# Trace of correlation matrix is N.
# Noise variance sigma2 is estimated as the mean of the eigenvalues <= 1.0 or by fitting.
# In the code:
eigenvalues_lte_1 = eigenvalues[eigenvalues <= 1.0]
sigma2_code = np.var(eigenvalues_lte_1) if len(eigenvalues_lte_1) > 0 else 1.0
sigma2_mean = np.mean(eigenvalues_lte_1) if len(eigenvalues_lte_1) > 0 else 1.0

# Calculate lambda_max for both
q = N / len(lookback_returns)
lambda_max_code = sigma2_code * (1.0 + np.sqrt(q))**2
lambda_max_mean = sigma2_mean * (1.0 + np.sqrt(q))**2

n_noise_code = np.sum(eigenvalues <= lambda_max_code)
n_noise_mean = np.sum(eigenvalues <= lambda_max_mean)

print(f"Number of assets (N): {N}")
print(f"Number of observations (T): {len(lookback_returns)}")
print(f"q (N/T): {q:.6f}")
print("\nEigenvalues:")
print(eigenvalues)
print("\nCode Implementation (np.var):")
print(f"  sigma2: {sigma2_code:.6f}")
print(f"  lambda_max: {lambda_max_code:.6f}")
print(f"  Number of noise eigenvalues: {n_noise_code} out of {N}")

print("\nAlternative Implementation (np.mean):")
print(f"  sigma2: {sigma2_mean:.6f}")
print(f"  lambda_max: {lambda_max_mean:.6f}")
print(f"  Number of noise eigenvalues: {n_noise_mean} out of {N}")
