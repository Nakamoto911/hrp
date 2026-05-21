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

# Plot stacked area chart
weights_melted = weights_raw_df.reset_index().rename(columns={'index': 'Date'}).melt(
    id_vars='Date', 
    var_name='Asset Name (Symbol)', 
    value_name='Weight'
)

fig = px.area(
    weights_melted, 
    x='Date', 
    y='Weight', 
    color='Asset Name (Symbol)',
    title="HRP Dynamic Asset Weight Allocation Over Time"
)

# Sort columns by their value in descending order for each date
hover_texts = []
for idx, row in weights_raw_df.iterrows():
    sorted_items = sorted(row.items(), key=lambda x: x[1], reverse=True)
    lines = [f"{name}: {val:.2%}" for name, val in sorted_items]
    hover_texts.append("<br>".join(lines))

# Disable hover on all area traces
for trace in fig.data:
    trace.hoverinfo = 'skip'
    trace.hovertemplate = None

# Add transparent scatter trace at the top for unified hover
fig.add_trace(go.Scatter(
    x=weights_raw_df.index,
    y=[1.0] * len(weights_raw_df),
    line=dict(color='rgba(0,0,0,0)'),
    showlegend=False,
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_texts,
    name=""
))

fig.update_layout(
    template="plotly_dark",
    hovermode="x unified"
)

# Save to html
fig.write_html("/Volumes/PRO-G40/Code/hrp/scratch/test_hover.html")
print("HTML written successfully.")
