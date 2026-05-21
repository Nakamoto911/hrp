import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Dummy data
dates = pd.date_range(start="2025-01-01", periods=3, freq="ME")
weights_raw_df = pd.DataFrame({
    "Oil (CRUD.MI)": [0.1, 0.6, 0.3],
    "Copper (COPA.MI)": [0.4, 0.1, 0.2],
    "Gold (GLD)": [0.5, 0.3, 0.5]
}, index=dates)

# Plot stacked area chart
weights_melted = weights_raw_df.reset_index().rename(columns={'index': 'Date'}).melt(
    id_vars='Date', 
    var_name='Asset Name (Symbol)', 
    value_name='Weight'
)

fig_weights = px.bar(
    weights_melted, 
    x='Date', 
    y='Weight', 
    color='Asset Name (Symbol)',
    title="HRP Dynamic Asset Weight Allocation Over Time",
    color_discrete_sequence=px.colors.qualitative.Plotly
)

# Map asset names to trace colors BEFORE editing traces
asset_colors = {}
for trace in fig_weights.data:
    if trace.name and hasattr(trace, 'marker') and getattr(trace.marker, 'color', None):
        asset_colors[trace.name] = trace.marker.color

# Sorting logic
hover_texts = []
for idx, row in weights_raw_df.iterrows():
    row_clean = row.fillna(0.0)
    sorted_items = sorted(row_clean.items(), key=lambda x: x[1], reverse=True)
    lines = []
    for name, val in sorted_items:
        color = asset_colors.get(name, '#FFFFFF')
        bullet = f'<span style="color:{color}; font-size:14px;">●</span>'
        lines.append(f"{bullet} {name}: {val:.2%}")
    hover_texts.append("<br>".join(lines))

# Disable hover on all area traces
for trace in fig_weights.data:
    trace.hoverinfo = 'skip'
    trace.hovertemplate = None

# Add transparent scatter trace
fig_weights.add_trace(go.Scatter(
    x=weights_raw_df.index,
    y=[1.0] * len(weights_raw_df),
    line=dict(color='rgba(0,0,0,0)', width=0),
    marker=dict(color='rgba(0,0,0,0)', size=0),
    showlegend=False,
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_texts,
    name=""
))

fig_weights.update_layout(
    template="plotly_dark",
    xaxis_title="Rebalance Date",
    yaxis_title="Weight (1.0 = 100%)",
    legend_title="Asset Universe",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=50, b=40)
)

print("Figure created successfully!")
print("Sample hover text line:")
print(hover_texts[0])
