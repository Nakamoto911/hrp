import json
import re

files = [
    ("ms", "scratch/test_width_ms.html"),
    ("days", "scratch/test_width_days.html"),
    ("us", "scratch/test_width_us.html")
]

for name, path in files:
    try:
        with open(path) as f:
            html = f.read()
        # Find all JSON-like trace data
        # In Plotly HTML output, data is typically inside a JSON block or a JS array inside a script tag
        # Let's search for "data" array in the Plotly.newPlot call
        match = re.search(r'Plotly\.newPlot\s*\(\s*[^,]+,\s*(\[.*?\])\s*,', html, re.DOTALL)
        if match:
            # Let's find width in the matched string using regex
            width_matches = re.findall(r'"width"\s*:\s*([0-9\.\+e]+)', match.group(1))
            print(f"File: {name} | Widths found in data: {width_matches}")
        else:
            # Maybe it is in a different format, let's search for "width" in the whole HTML
            width_matches = re.findall(r'"width"\s*:\s*([0-9\.\+e]+)', html)
            print(f"File: {name} | Widths found in HTML: {width_matches}")
    except Exception as e:
        print(f"Error reading {name}: {e}")
