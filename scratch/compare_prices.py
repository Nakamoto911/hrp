import pandas as pd
import yfinance as yf
import numpy as np

# Load cache
cache_df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

# Download again like in scratch script
european_pool_mi = [
    'SXR8.DE', 'EXX5.DE', 'EXS1.DE', 'SXRT.DE', 'IS3N.DE',
    'IUSM.DE', 'IS0L.DE', 'XJSE.DE', 'IGLT.MI',
    'IBCQ.DE', 'IHYG.MI',
    '4GLD.DE', 'CRUD.MI', 'COPA.MI', 'AIGP.MI',
    '5MVW.DE', '36BZ.DE', 'QDVB.DE', 'IUS3.DE',
    'BTCE.DE'
]
df_raw = yf.download(european_pool_mi, start='2008-01-02', end='2026-05-20', progress=False)
scratch_df = df_raw['Adj Close'] if 'Adj Close' in df_raw.columns.levels[0] else df_raw['Close']
scratch_df = scratch_df[european_pool_mi]

# Compare shapes and dates
print("Cache shape:", cache_df.shape)
print("Scratch shape:", scratch_df.shape)

common_idx = cache_df.index.intersection(scratch_df.index)
print("Common index size:", len(common_idx))

# Check differences
diffs = {}
for col in cache_df.columns:
    diff = np.abs(cache_df.loc[common_idx, col] - scratch_df.loc[common_idx, col])
    max_diff = diff.max()
    diffs[col] = max_diff
    
print("Max absolute differences:")
for col, val in diffs.items():
    print(f"  {col}: {val}")
