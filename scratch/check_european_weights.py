import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

# Monkeypatch denoiser
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

hrp_cum, hrp_diag, _, _, _ = run_strategy_backtest(df_cleaned, params, 'european')

# Collect weight history
weight_history = hrp_diag["weight_history"]
weights_df = pd.DataFrame.from_dict(weight_history, orient='index')

# Calculate average weights and asset returns over the backtest period
# The backtest start date is the first rebalance date
backtest_start = weights_df.index[0]
prices_bt = df_cleaned.loc[backtest_start:]

# Compute annualized return of each asset over the backtest period
asset_ann_returns = {}
for col in df_cleaned.columns:
    series = prices_bt[col].dropna()
    if len(series) > 1:
        ann_ret = (series.iloc[-1] / series.iloc[0]) ** (252.0 / len(series)) - 1.0
        # standard deviation of daily returns
        ann_vol = series.pct_change().std() * np.sqrt(252.0)
        asset_ann_returns[col] = (ann_ret, ann_vol)
    else:
        asset_ann_returns[col] = (0.0, 0.0)

summary = []
for col in weights_df.columns:
    avg_weight = weights_df[col].mean()
    ann_ret, ann_vol = asset_ann_returns.get(col, (0.0, 0.0))
    summary.append({
        "Ticker": col,
        "Avg Weight": avg_weight,
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol
    })

summary_df = pd.DataFrame(summary).sort_values(by="Avg Weight", ascending=False)
print("AVERAGE HRP WEIGHTS AND ASSET PERFORMANCE IN EUROPEAN POOL:")
print(summary_df.to_string(index=False, formatters={
    'Avg Weight': '{:.2%}'.format,
    'Ann. Return': '{:.2%}'.format,
    'Ann. Volatility': '{:.2%}'.format
}))
