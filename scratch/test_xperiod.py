import pandas as pd
import plotly.express as px

# Create sample datetime data spanning 4 quarters
dates = pd.to_datetime(['2005-03-31', '2005-06-30', '2005-09-30', '2005-12-31'])
df = pd.DataFrame({
    'Date': dates,
    'Weight': [0.4, 0.5, 0.6, 0.3],
    'Asset': ['A', 'A', 'A', 'A']
})

fig = px.bar(df, x='Date', y='Weight', color='Asset')

# Set xperiod to 'M3' (3 months) and alignment
fig.update_traces(xperiod="M3", xperiodalignment="middle", selector=dict(type='bar'))
fig.update_layout(template="plotly_dark", title="Stacked Bar with xperiod='M3'")
fig.write_html("scratch/test_xperiod.html")

print("Generated test_xperiod.html")
