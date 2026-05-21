import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.backtest import run_strategy_backtest

# Load cache
df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

# 3-day median cleaning
df_cleaned = df.copy()
for col in df.columns:
    series = df[col]
    if series.notna().sum() > 5:
        valid_idx = series.dropna().index
        cleaned_valid = series.loc[valid_idx].rolling(window=3, min_periods=1, center=True).median()
        df_cleaned.loc[valid_idx, col] = cleaned_valid

# Define corrected denoiser
def corrected_denoise_covariance(cov_emp: np.ndarray, n_obs: int) -> np.ndarray:
    N = cov_emp.shape[0]
    if N <= 1:
        return cov_emp.copy()
    std = np.sqrt(np.diag(cov_emp))
    std_safe = np.where(std == 0, 1e-8, std)
    corr_emp = cov_emp / np.outer(std_safe, std_safe)
    corr_emp = np.clip(corr_emp, -1.0, 1.0)
    
    eigenvalues, eigenvectors = np.linalg.eigh(corr_emp)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    eigenvalues_lte_1 = eigenvalues[eigenvalues <= 1.0]
    sigma2 = np.mean(eigenvalues_lte_1) if len(eigenvalues_lte_1) > 0 else 1.0
    if sigma2 <= 0:
        sigma2 = 1e-8
        
    q = N / n_obs
    lambda_max = sigma2 * (1.0 + np.sqrt(q))**2
    
    is_noise = eigenvalues <= lambda_max
    n_noise = np.sum(is_noise)
    
    if n_noise > 0:
        average_noise_eigenvalue = np.mean(eigenvalues[is_noise])
        eigenvalues_denoised = eigenvalues.copy()
        eigenvalues_denoised[is_noise] = average_noise_eigenvalue
    else:
        eigenvalues_denoised = eigenvalues.copy()
        
    corr_denoised_raw = np.dot(eigenvectors * eigenvalues_denoised, eigenvectors.T)
    diag_val = np.diag(corr_denoised_raw)
    diag_val_safe = np.clip(diag_val, 1e-8, None)
    diag_inv_sqrt = 1.0 / np.sqrt(diag_val_safe)
    corr_denoised = corr_denoised_raw * np.outer(diag_inv_sqrt, diag_inv_sqrt)
    corr_denoised = np.clip(corr_denoised, -1.0, 1.0)
    
    cov_denoised = corr_denoised * np.outer(std, std)
    return cov_denoised

import hrp_engine.backtest
hrp_engine.backtest.denoise_covariance = corrected_denoise_covariance

params = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

# Run corrected denoised strategy
hrp_cum, hrp_diag, _, _, _ = run_strategy_backtest(df_cleaned, params, 'european')
weights_denoised = pd.DataFrame.from_dict(hrp_diag["weight_history"], orient='index')

# Run empirical strategy (by using a mock denoiser that just returns the input)
def identity_denoise(cov_emp: np.ndarray, n_obs: int) -> np.ndarray:
    return cov_emp.copy()

hrp_engine.backtest.denoise_covariance = identity_denoise
_, hrp_diag_emp, _, _, _ = run_strategy_backtest(df_cleaned, params, 'european')
weights_empirical = pd.DataFrame.from_dict(hrp_diag_emp["weight_history"], orient='index')

# Compare average weights
summary = []
for col in df.columns:
    avg_w_den = weights_denoised[col].mean() if col in weights_denoised.columns else 0.0
    avg_w_emp = weights_empirical[col].mean() if col in weights_empirical.columns else 0.0
    summary.append({
        "Ticker": col,
        "Denoised Weight": avg_w_den,
        "Empirical Weight": avg_w_emp,
        "Difference": avg_w_den - avg_w_emp
    })

summary_df = pd.DataFrame(summary).sort_values(by="Denoised Weight", ascending=False)
print("AVERAGE WEIGHTS COMPARISON (DENOISED VS EMPIRICAL):")
print(summary_df.to_string(index=False, formatters={
    'Denoised Weight': '{:.2%}'.format,
    'Empirical Weight': '{:.2%}'.format,
    'Difference': '{:+.2%}'.format
}))
