import pandas as pd

df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
print("Cached data range:", df.index[0].strftime('%Y-%m-%d'), "to", df.index[-1].strftime('%Y-%m-%d'))
print("DataFrame columns:", list(df.columns))
print("First valid indices:")
for col in df.columns:
    print(f"  {col}: {df[col].first_valid_index().strftime('%Y-%m-%d') if df[col].first_valid_index() is not None else 'None'}")
