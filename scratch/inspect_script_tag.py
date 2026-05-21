with open('scratch/test_width_ms.html') as f:
    html = f.read()

import re
matches = re.findall(r'<script.*?>([\s\S]*?)</script>', html)
for i, m in enumerate(matches):
    if "Plotly.newPlot" in m:
        print(f"Match {i} (length {len(m)}):")
        idx = m.index("Plotly.newPlot")
        print(m[idx:idx+1500])
        break
