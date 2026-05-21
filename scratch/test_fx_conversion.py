import yfinance as yf
import pandas as pd

start_date = '2000-01-01'
end_date = '2026-05-20'

print("Downloading...")
df = yf.download(['IGLT.L', 'GBPEUR=X'], start=start_date, end=end_date, progress=False)

print("\nDataFrame columns:")
print(df.columns)
print("\nDataFrame head:")
print(df.head())
