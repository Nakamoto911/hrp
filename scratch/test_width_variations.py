import pandas as pd
import plotly.express as px

# Create sample datetime data spanning 3 quarters
dates = pd.to_datetime(['2005-03-31', '2005-06-30', '2005-09-30'])
df = pd.DataFrame({
    'Date': dates,
    'Weight': [0.4, 0.5, 0.6],
    'Asset': ['A', 'A', 'A']
})

# Test 1: Width in milliseconds (approx 86 days = 7.4e9 ms)
fig1 = px.bar(df, x='Date', y='Weight', color='Asset')
fig1.update_traces(width=7469280000.0, selector=dict(type='bar'))
fig1.update_layout(template="plotly_dark", title="Width in Milliseconds (7.4e9)")
fig1.write_html("scratch/test_width_ms.html")

# Test 2: Width in days (approx 86 days = 86)
fig2 = px.bar(df, x='Date', y='Weight', color='Asset')
fig2.update_traces(width=86.0, selector=dict(type='bar'))
fig2.update_layout(template="plotly_dark", title="Width in Days (86)")
fig2.write_html("scratch/test_width_days.html")

# Test 3: Width in microseconds/nanoseconds or something else
# Let's see what happens if we set width to 86 * 24 * 60 * 60 * 1000 * 1000 (microseconds)
fig3 = px.bar(df, x='Date', y='Weight', color='Asset')
fig3.update_traces(width=86 * 24 * 60 * 60 * 1000 * 1000, selector=dict(type='bar'))
fig3.update_layout(template="plotly_dark", title="Width in Microseconds")
fig3.write_html("scratch/test_width_us.html")

print("Generated test HTML files for different widths.")
