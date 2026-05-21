import subprocess
import time
import os

def capture_screenshot(html_path, output_png):
    print(f"Opening {html_path} in default browser...")
    # Open the HTML file in the default browser on macOS
    subprocess.run(["open", html_path])
    
    # Wait for the page and Plotly to render
    time.run_sec = 3
    time.sleep(time.run_sec)
    
    # Take a screenshot of the main screen
    print(f"Capturing screenshot to {output_png}...")
    subprocess.run(["screencapture", "-x", output_png])
    print("Screenshot captured.")

# Let's clean up any previous screenshots
for f in ["scratch/screenshot_ms.png", "scratch/screenshot_days.png"]:
    if os.path.exists(f):
        os.remove(f)

# Capture for milliseconds
capture_screenshot("scratch/test_width_ms.html", "scratch/screenshot_ms.png")

# Give it a moment, then capture for days
time.sleep(2)
capture_screenshot("scratch/test_width_days.html", "scratch/screenshot_days.png")
