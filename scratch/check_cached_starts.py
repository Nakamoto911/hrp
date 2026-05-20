import pandas as pd

df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
print("Ticker start dates in cache:")
first_valid_dates = {col: df[col].first_valid_index() for col in df.columns}
for col, date in sorted(first_valid_dates.items(), key=lambda x: str(x[1])):
    print(f"  {col}: {date.strftime('%Y-%m-%d') if pd.notna(date) else 'NaT'}")
