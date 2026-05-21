import yfinance as yf
import pandas as pd
import numpy as np

# Let's download a few tickers with known jumps
tickers = ['EXX5.DE', 'IUS3.DE', 'IGLT.MI']
df_raw = yf.download(tickers, start='2008-01-01', end='2026-05-20', progress=False)

print("Columns returned:", df_raw.columns)

# Safely extract Close and Adj Close
for ticker in tickers:
    for price_col in ['Close', 'Adj Close']:
        try:
            if isinstance(df_raw.columns, pd.MultiIndex):
                # Columns are MultiIndex, like (price_col, ticker)
                if (price_col, ticker) in df_raw.columns:
                    series = df_raw[(price_col, ticker)].dropna()
                elif (price_col.lower(), ticker) in df_raw.columns:
                    series = df_raw[(price_col.lower(), ticker)].dropna()
                else:
                    print(f"  {ticker} ({price_col}) not found in MultiIndex columns.")
                    continue
            else:
                series = df_raw[ticker].dropna()
                
            rets = series.pct_change().dropna()
            extreme_mask = (rets > 0.10) | (rets < -0.10)
            n_extreme = extreme_mask.sum()
            print(f"  {ticker} ({price_col}): {n_extreme} extreme daily returns (Ann. Vol: {rets.std()*np.sqrt(252.0)*100:.2f}%)")
            if n_extreme > 0:
                extreme_idx = rets[extreme_mask].index
                for idx in extreme_idx[:3]:
                    p_prev = series.loc[:idx].iloc[-2]
                    p_curr = series.loc[idx]
                    print(f"    {idx.strftime('%Y-%m-%d')}: {rets.loc[idx]*100:.2f}% ({p_prev:.4f} -> {p_curr:.4f})")
        except Exception as e:
            print(f"  Failed for {ticker} ({price_col}): {e}")
