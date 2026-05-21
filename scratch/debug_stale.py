import pandas as pd

df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
series = df["SXR8.DE"].copy()

first_valid = series.first_valid_index()
start_slice = series.loc[first_valid:].dropna().head(10)
print("First 10 valid values of SXR8.DE:")
print(start_slice)
print(f"Number of unique values: {start_slice.nunique()}")
print(f"Unique values: {start_slice.unique()}")
