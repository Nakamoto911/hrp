import pandas as pd
import plotly.express as px

dates = pd.date_range(start="2005-01-01", end="2010-01-01", freq="QE")
df = pd.DataFrame({
    'Date': dates,
    'Weight': [1.0] * len(dates),
    'Asset': ['A'] * len(dates)
})

fig = px.bar(df, x='Date', y='Weight', color='Asset')

diffs = pd.Series(dates).diff().dropna()
median_diff_ms = diffs.median().total_seconds() * 1000
bar_width = median_diff_ms * 0.95

print("bar_width in ms:", bar_width)
fig.update_traces(width=bar_width, selector=dict(type='bar'))
fig.write_html("scratch/test_bar_width.html")
print("Done writing HTML!")
