import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

# Load cache
df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

# Trim SXR8.DE before 2010-11-01
df_trimmed = df.copy()
df_trimmed.loc[:"2010-10-29", "SXR8.DE"] = np.nan

# Also apply corrected denoiser
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

from hrp_engine.config import StrategyParams
from hrp_engine.backtest import run_strategy_backtest
from hrp_engine.reporting import compute_metrics

params = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

# Clean single-day bad ticks using rolling 3-day median
df_cleaned = df_trimmed.copy()
for col in df_trimmed.columns:
    series = df_trimmed[col]
    if series.notna().sum() > 5:
        valid_idx = series.dropna().index
        cleaned_valid = series.loc[valid_idx].rolling(window=3, min_periods=1, center=True).median()
        df_cleaned.loc[valid_idx, col] = cleaned_valid

# Run backtests
hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(df_cleaned, params, 'european')

metrics_hrp = compute_metrics(hrp_cum, hrp_diag["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)
metrics_sf = compute_metrics(sixty_forty_cum, sf_diag["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)
metrics_sp = compute_metrics(sp500_cum, 0.0, params.transaction_cost_bps, 100000.0)

print("TRIMMED SXR8.DE & DENOISED EUROPEAN POOL PERFORMANCE:")
print(f"HRP:   Return: {metrics_hrp['annualized_return']*100:.2f}% | Vol: {metrics_hrp['annualized_volatility']*100:.2f}% | Sharpe: {metrics_hrp['sharpe_ratio']:.2f} | MDD: {metrics_hrp['max_drawdown']*100:.2f}%")
print(f"60/40: Return: {metrics_sf['annualized_return']*100:.2f}% | Vol: {metrics_sf['annualized_volatility']*100:.2f}% | Sharpe: {metrics_sf['sharpe_ratio']:.2f} | MDD: {metrics_sf['max_drawdown']*100:.2f}%")
print(f"S&P:   Return: {metrics_sp['annualized_return']*100:.2f}% | Vol: {metrics_sp['annualized_volatility']*100:.2f}% | Sharpe: {metrics_sp['sharpe_ratio']:.2f} | MDD: {metrics_sp['max_drawdown']*100:.2f}%")
