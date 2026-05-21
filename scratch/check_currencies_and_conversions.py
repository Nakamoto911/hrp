import pandas as pd

# Load cached prices
df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

# Select a few columns including IGLT.L and some European ones
cols = ['SXR8.DE', 'EXX5.DE', 'EXS1.DE', 'IGLT.L', 'IS0L.DE']
print(df[cols].dropna().head(10))
print("\nDescriptive statistics:")
print(df[cols].describe())
