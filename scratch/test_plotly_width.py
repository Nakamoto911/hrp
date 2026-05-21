import pandas as pd
import plotly.express as px

# Create sample datetime data
dates = pd.to_datetime(['2005-03-31', '2005-06-30', '2005-09-30'])
df = pd.DataFrame({
    'Date': dates,
    'Weight': [0.4, 0.5, 0.6],
    'Asset': ['A', 'A', 'A']
})

fig = px.bar(df, x='Date', y='Weight', color='Asset')

# Calculate width
diffs = pd.Series(dates).diff().dropna()
median_diff_ms = diffs.median().total_seconds() * 1000
bar_width = median_diff_ms * 0.95

print(f"Median diff ms: {median_diff_ms}")
print(f"Applying width: {bar_width}")

fig.update_traces(width=bar_width, selector=dict(type='bar'))

# Export to JSON/HTML or check trace fields
for trace in fig.data:
    print(f"Trace name: {trace.name}, type: {trace.type}, width: {getattr(trace, 'width', None)}")
