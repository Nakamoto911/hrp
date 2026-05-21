import pandas as pd
import numpy as np
import sys
import os

# Set paths
sys.path.append(os.path.abspath('.'))

# Define a temporary corrected denoiser function
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
    if len(eigenvalues_lte_1) > 0:
        # Corrected: use np.mean instead of np.var
        sigma2 = np.mean(eigenvalues_lte_1)
    else:
        sigma2 = 1.0
        
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

# Let's monkeypatch the denoiser in the hrp_engine modules and run the backtest
import hrp_engine.backtest
hrp_engine.backtest.denoise_covariance = corrected_denoise_covariance

from hrp_engine.config import StrategyParams
from hrp_engine.backtest import run_strategy_backtest
from hrp_engine.reporting import compute_metrics

# Load cache
df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
df_cleaned = df.rolling(window=3, min_periods=1, center=True).median()

params = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

# Run backtests
hrp_cum_raw, hrp_diag_raw, _, _, _ = run_strategy_backtest(df, params, 'european')
hrp_cum_cl, hrp_diag_cl, _, _, _ = run_strategy_backtest(df_cleaned, params, 'european')

metrics_raw = compute_metrics(hrp_cum_raw, hrp_diag_raw["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)
metrics_cl = compute_metrics(hrp_cum_cl, hrp_diag_cl["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)

print("WITH CORRECTED DENOISER (RAW DATA):")
for k, v in metrics_raw.items():
    if isinstance(v, float):
        print(f"  {k:<35}: {v:.4f}")

print("\nWITH CORRECTED DENOISER (CLEANED DATA):")
for k, v in metrics_cl.items():
    if isinstance(v, float):
        print(f"  {k:<35}: {v:.4f}")
