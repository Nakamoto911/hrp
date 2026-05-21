import pandas as pd

df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
sxr8 = df["SXR8.DE"].dropna()

print(f"First valid index of SXR8.DE: {sxr8.index[0].strftime('%Y-%m-%d')} (Price: {sxr8.iloc[0]:.4f})")
print("\nSXR8.DE prices in 2010-05 to 2010-11:")
print(sxr8.loc["2010-05-01":"2010-11-10"])
