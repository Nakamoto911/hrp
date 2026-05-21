with open('scratch/test_width_ms.html') as f:
    html = f.read()

import re
import json

# Locate the window.PLOTLYENV or Plotly.newPlot configuration where the data/layout are actually defined.
# Plotly HTML contains window.PLOTLYENV = window.PLOTLYENV || {};
# and then something like Plotly.newPlot(gd, [trace1, trace2], layout) or similar.
# Let's search for "window.PLOTLYENV" or "document.getElementById"
matches = re.findall(r'document\.getElementById\([^)]+\)\s*,\s*(\[.*?\])\s*,\s*({.*?})\s*,\s*\{', html, re.DOTALL)
if matches:
    print("Found direct document.getElementById match!")
    data_str, layout_str = matches[0]
    data = json.loads(data_str)
    print("Traces:", [t.get('name') for t in data])
    for t in data:
        print("Trace name:", t.get('name'), "width:", t.get('width'))
else:
    # Let's search for the script tag containing Plotly.newPlot and print a larger context around it
    script_matches = re.findall(r'<script.*?>([\s\S]*?)</script>', html)
    for m in script_matches:
        if "Plotly.newPlot" in m:
            # Let's search for the definition of data/layout arrays
            # Often it is like:
            # Plotly.newPlot(gd, [{"type": "bar", ...}], {"template": ...})
            # Let's search for the arguments before Plotly.newPlot
            print("Found script containing Plotly.newPlot.")
            # Let's search for the data array using regex
            data_match = re.search(r'gd\s*,\s*(\[.*?\])\s*,\s*({.*?})\s*,', m, re.DOTALL)
            if data_match:
                print("Found gd data/layout match!")
                data = json.loads(data_match.group(1))
                for t in data:
                    print("Trace name:", t.get('name'), "width:", t.get('width'))
            else:
                print("Could not match gd data/layout using regex, printing context around Plotly.newPlot:")
                idx = m.index("Plotly.newPlot")
                start = max(0, idx - 1000)
                print(m[start:idx+200])
