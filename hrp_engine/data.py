import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple

# Define the asset pools
ETF_POOL = ['IVV', 'IJH', 'IWM', 'EFA', 'EEM', 'AGG', 'SPTL', 'HYG', 'SPBO', 'IYR', 'DBC', 'GLD']
MUTUAL_FUND_POOL = ['^SP500TR', 'VIMSX', 'NAESX', 'FDIVX', 'VEIEX', 'VBMFX', 'VUSTX', 'VWEHX', 'VWESX', 'FRESX', '^SPGSCI', 'GC=F']
EUROPEAN_POOL = [
    'SXR8.DE', 'EXX5.DE', 'EXS1.DE', 'SXRT.DE', 'IS3N.DE',
    'IUSM.DE', 'IS0L.DE', 'XJSE.DE', 'IGLT.L',
    'IBCQ.DE', 'IHYG.MI',
    '4GLD.DE', 'CRUD.MI', 'COPA.MI', 'AIGP.MI',
    '5MVW.DE', '36BZ.DE', 'QDVB.DE', 'IUS3.DE',
    'BTCE.DE'
]

def generate_synthetic_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generates realistic daily price series for testing.
    Includes drift, volatilies, and correlations.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    n_assets = len(tickers)
    
    # Generate a random correlation matrix
    # Make a random matrix A and form A * A^T, then normalize to correlation
    rng = np.random.default_rng(42)
    A = rng.normal(size=(n_assets, n_assets))
    cov = np.dot(A, A.T)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    
    # Ensure positive definiteness
    corr = corr + np.eye(n_assets) * 0.05
    std = np.sqrt(np.diag(corr))
    corr = corr / np.outer(std, std)
    
    # Decompose correlation to simulate correlated returns
    L = np.linalg.cholesky(corr)
    
    # Asset specific annual returns (drifts) and volatilities
    # Equity-like: higher drift and vol; bond-like: lower drift and vol
    drifts = rng.uniform(0.02, 0.09, size=n_assets) / 252.0
    vols = rng.uniform(0.04, 0.18, size=n_assets) / np.sqrt(252.0)
    
    raw_returns = rng.normal(size=(n_days, n_assets))
    correlated_returns = np.dot(raw_returns, L.T)
    
    # Apply drifts and vols
    daily_returns = drifts + vols * correlated_returns
    
    # Convert to prices starting at 100
    prices = 100.0 * np.exp(np.cumsum(daily_returns, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    
    # Introduce different inception dates for some assets to test variable histories
    # For example, let some assets start later
    for i, ticker in enumerate(tickers):
        if i % 3 == 1:
            # Starts 5 years late
            start_idx = min(int(n_days * 0.25), n_days - 100)
            df.iloc[:start_idx, i] = np.nan
        elif i % 3 == 2:
            # Starts 2 years late
            start_idx = min(int(n_days * 0.10), n_days - 100)
            df.iloc[:start_idx, i] = np.nan
            
    return df

def fetch_data(pool_name: str, cache_dir: str = 'cache', force_refresh: bool = False) -> pd.DataFrame:
    """
    Downloads daily price series for a pool, caches them to a CSV file.
    Falls back to synthetic data if the download fails.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{pool_name.lower()}_prices.csv")
    
    if pool_name.lower() == 'etf':
        tickers = ETF_POOL
    elif pool_name.lower() == 'european':
        tickers = EUROPEAN_POOL
    else:
        tickers = MUTUAL_FUND_POOL
    start_date = '2000-01-01'
    end_date = '2026-05-20'
    
    if not force_refresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            # Ensure all tickers are present
            if all(t in df.columns for t in tickers):
                return df
        except Exception:
            pass
            
    print(f"Downloading historical data for pool: {pool_name}...")
    try:
        # Download adjusted closing prices
        df_raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if isinstance(df_raw.columns, pd.MultiIndex):
            # Try to get Adj Close, otherwise Close
            if 'Adj Close' in df_raw.columns.levels[0]:
                df = df_raw['Adj Close']
            else:
                df = df_raw['Close']
        else:
            df = df_raw
            
        # Drop columns not in tickers or entirely empty
        df = df[tickers]
        
        # Save to cache
        df.to_csv(cache_path)
        print(f"Data saved to {cache_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error downloading data: {e}. Generating realistic synthetic data instead.")
        df = generate_synthetic_prices(tickers, start_date, end_date)
        df.to_csv(cache_path)
        return df

def get_lookback_data(prices_df: pd.DataFrame, rebalance_date: pd.Timestamp, lookback_years: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Slices prices to the lookback window ending at rebalance_date.
    Identifies assets that are active in this window, and returns their returns.
    """
    start_date = rebalance_date - pd.DateOffset(years=lookback_years)
    # Slice the dataframe
    window_df = prices_df.loc[start_date:rebalance_date]
    
    # We require an asset to have data for at least 80% of the lookback period
    # and specifically must have a valid price at the rebalance date to be tradeable.
    active_assets = []
    for col in window_df.columns:
        series = window_df[col]
        # Count non-NaN values
        valid_count = series.notna().sum()
        total_count = len(series)
        if total_count > 0 and (valid_count / total_count) >= 0.8:
            # Check if current price is valid
            if not pd.isna(series.iloc[-1]):
                active_assets.append(col)
                
    if not active_assets:
        raise ValueError(f"No active assets found in the lookback window ending at {rebalance_date}")
        
    # Slice prices for active assets
    sliced_prices = window_df[active_assets].copy()
    
    # Fill remaining NaNs (e.g. if an asset started slightly after start_date)
    # Forward fill then backward fill to ensure no NaNs remain
    sliced_prices = sliced_prices.ffill().bfill()
    
    # Calculate daily returns
    returns_df = sliced_prices.pct_change().dropna(how='all')
    
    return returns_df, active_assets

def get_least_history_info(prices_df: pd.DataFrame) -> Tuple[pd.Timestamp, str]:
    """
    Finds the start date and ticker of the asset with the least history in the pool.
    """
    first_valid_dates = {col: prices_df[col].first_valid_index() for col in prices_df.columns}
    least_history_asset = max(first_valid_dates, key=first_valid_dates.get)
    start_date = first_valid_dates[least_history_asset]
    return start_date, least_history_asset

