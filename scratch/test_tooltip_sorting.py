import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Create dummy weights data
dates = pd.date_range(start="2025-01-01", periods=3, freq="ME")
data = {
    "Oil (CRUD.MI)": [0.1, 0.6, 0.3],
    "Copper (COPA.MI)": [0.4, 0.1, 0.2],
    "Gold (GLD)": [0.5, 0.3, 0.5]
}
weights_raw_df = pd.DataFrame(data, index=dates)

# Prepare hover texts sorted by weight for each date
hover_texts = []
for idx, row in weights_raw_df.iterrows():
    sorted_items = sorted(row.items(), key=lambda x: x[1], reverse=True)
    lines = [f"{name}: {val:.2%}" for name, val in sorted_items]
    hover_texts.append("<br>".join(lines))

print("Hover text for date 1:")
print(hover_texts[0])
