import yfinance as yf
import pandas as pd

start_date = '2000-01-01'
end_date = '2026-05-20'

print("Downloading 2B7M.DE (iShares Core UK Gilts UCITS ETF on Xetra)...")
df = yf.download('2B7M.DE', start=start_date, end=end_date, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    prices = df['Adj Close'] if 'Adj Close' in df.columns.levels[0] else df['Close']
else:
    prices = df

print("\nData shape:", prices.shape)
if not prices.empty:
    print("First date:", prices.index[0].strftime('%Y-%m-%d'), "->", prices.iloc[0])
    print("Last date:", prices.index[-1].strftime('%Y-%m-%d'), "->", prices.iloc[-1])
else:
    print("No data returned.")
