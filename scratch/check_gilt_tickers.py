import yfinance as yf
import pandas as pd

tickers = ['VGVA.DE', 'SYBG.DE', 'CBUS.DE', 'CEMM.DE', 'VGOV.DE', 'VGUE.DE']

for t in tickers:
    try:
        df = yf.download(t, start='2010-01-01', end='2026-05-20', progress=False)
        if not df.empty:
            prices = df['Adj Close'] if 'Adj Close' in df.columns.levels[0] else df['Close']
            print(f"{t}: Downloaded successfully. Shape: {prices.shape}")
            print(f"  First: {prices.index[0].strftime('%Y-%m-%d')} -> {prices.iloc[0]}")
            print(f"  Last: {prices.index[-1].strftime('%Y-%m-%d')} -> {prices.iloc[-1]}")
        else:
            print(f"{t}: Returned empty DataFrame")
    except Exception as e:
        print(f"{t}: Error {e}")
