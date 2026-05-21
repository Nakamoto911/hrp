import yfinance as yf
import pandas as pd

try:
    df = yf.download('IGLT.MI', start='2000-01-01', end='2026-05-20', progress=False)
    if not df.empty:
        prices = df['Adj Close'] if 'Adj Close' in df.columns.levels[0] else df['Close']
        print(f"IGLT.MI: Downloaded successfully. Shape: {prices.shape}")
        print(f"  First: {prices.index[0].strftime('%Y-%m-%d')} -> {prices.iloc[0]}")
        print(f"  Last: {prices.index[-1].strftime('%Y-%m-%d')} -> {prices.iloc[-1]}")
    else:
        print("IGLT.MI: Returned empty DataFrame")
except Exception as e:
    print(f"IGLT.MI: Error {e}")
