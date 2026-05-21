import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.backtest import run_strategy_backtest
from hrp_engine.reporting import compute_metrics

corrected_tickers = [
    'SXR8.DE',  # S&P 500
    'SXRZ.DE',  # Nikkei 225
    'SXRW.DE',  # FTSE 100
    'SXRT.DE',  # EuroSTOXX 50
    'IS3N.DE',  # MSCI EM
    'IUSM.DE',  # US 10Y Treasuries
    'IS0L.DE',  # Bund Allemand
    'XJSE.DE',  # JGB Japonais
    'IGLT.MI',  # U.K. Gilts
    'IBCQ.DE',  # Credit IG
    'IHYG.MI',  # Credit HY
    '4GLD.DE',  # Gold
    'CRUD.MI',  # Oil
    'COPA.MI',  # Copper
    'AIGP.MI',  # Agriculture
    'IS3S.DE',  # Equity Value
    'IS3R.DE',  # Equity Momentum
    'IS3Q.DE',  # Equity Quality
    'IQQ0.DE',  # Equity Defensive
    'BTCE.DE'   # Bitcoin
]

print("Downloading new tickers from yfinance...")
prices = {}
for t in corrected_tickers:
    df_t = yf.download(t, start="2008-01-02", end="2026-05-20", progress=False)
    if not df_t.empty:
        prices[t] = df_t["Adj Close"] if "Adj Close" in df_t else df_t["Close"]
        prices[t] = prices[t].squeeze()

prices_df = pd.DataFrame(prices)

# Do NOT ffill/bfill the entire DataFrame to preserve NaNs before inception.
# Only ffill internal gaps for each series, but keep the initial NaNs before the first valid index!
df_cleaned = pd.DataFrame(index=prices_df.index)
for col in prices_df.columns:
    series = prices_df[col]
    first_valid = series.first_valid_index()
    if first_valid is not None:
        # Slice from first valid to end
        valid_part = series.loc[first_valid:]
        # ffill internal gaps
        valid_part = valid_part.ffill()
        # Apply 3-day median cleaning to the valid part
        cleaned_valid = valid_part.rolling(window=3, min_periods=1, center=True).median()
        # Assign back to df_cleaned, keeping NaNs before first_valid
        df_cleaned.loc[first_valid:, col] = cleaned_valid

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

hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(df_cleaned, params, 'european')
metrics = compute_metrics(hrp_cum, hrp_diag["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)
sp500_m = compute_metrics(sp500_cum, 0.0, params.transaction_cost_bps, 100000.0)
sf_m = compute_metrics(sixty_forty_cum, sf_diag["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)

print("\n--- RESULTS WITH CORRECTED TICKERS & PRESERVED NANS ---")
print(f"HRP:   Return: {metrics['annualized_return']*100:.2f}% | Vol: {metrics['annualized_volatility']*100:.2f}% | Sharpe: {metrics['sharpe_ratio']:.2f} | MDD: {metrics['max_drawdown']*100:.2f}%")
print(f"60/40: Return: {sf_m['annualized_return']*100:.2f}% | Vol: {sf_m['annualized_volatility']*100:.2f}% | Sharpe: {sf_m['sharpe_ratio']:.2f} | MDD: {sf_m['max_drawdown']*100:.2f}%")
print(f"S&P:   Return: {sp500_m['annualized_return']*100:.2f}% | Vol: {sp500_m['annualized_volatility']*100:.2f}% | Sharpe: {sp500_m['sharpe_ratio']:.2f} | MDD: {sp500_m['max_drawdown']*100:.2f}%")
