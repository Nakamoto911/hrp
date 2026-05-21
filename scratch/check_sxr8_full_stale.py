import pandas as pd

df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
sxr8 = df["SXR8.DE"].dropna().loc[:"2010-11-05"]

print("SXR8.DE history up to 2010-11-05:")
for d, p in sxr8.items():
    print(f"  {d.strftime('%Y-%m-%d')}: {p:.4f}")
