import pandas as pd

df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
iglt = df['IGLT.MI'].dropna()
print("IGLT.MI First 30 prices:")
print(iglt.head(30))
print("\nIGLT.MI Last 10 prices:")
print(iglt.tail(10))
