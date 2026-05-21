import re
import json

def parse_html(path):
    print(f"\nParsing {path}...")
    with open(path) as f:
        html = f.read()
    
    matches = re.findall(r'<script.*?>([\s\S]*?)</script>', html)
    for m in matches:
        if "Plotly.newPlot" in m and "PLOTLYENV" in m:
            print("Found target script block!")
            # Find the actual Plotly.newPlot call in this script block
            match = re.search(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[.*?\])\s*,\s*({.*?})\s*,\s*\{', m, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                for t in data:
                    print("Trace:", t.get('name'), "type:", t.get('type'), "width:", t.get('width'))
                return
            else:
                # Let's do a simpler scan for "width" inside the script tag
                print("Could not parse with exact regex. Printing width occurrences in script tag:")
                width_matches = re.findall(r'"width"\s*:\s*([^\s,}\]]+)', m)
                print(width_matches)
                return

parse_html("scratch/test_width_ms.html")
parse_html("scratch/test_width_days.html")
parse_html("scratch/test_width_us.html")
