#!/usr/bin/env python3
"""
Debug RLE format from Label Studio
"""

import os, json, requests
from io import BytesIO
from zipfile import ZipFile

# Load config
try:
    from config import LS_TOKEN
except ImportError:
    LS_TOKEN = os.environ.get("LS_TOKEN")

# Settings
LS_HOST = "http://127.0.0.1:8080"
PROJECT_ID = 7

# Download data
export_url = f"{LS_HOST}/api/projects/{PROJECT_ID}/export?exportType=JSON&download_all_tasks=true"
resp = requests.get(export_url, headers={"Authorization": f"Token {LS_TOKEN}"}, timeout=120)
resp.raise_for_status()

ctype = resp.headers.get("content-type", "")
if ctype.startswith("application/zip"):
    with ZipFile(BytesIO(resp.content)) as zf:
        jname = next(n for n in zf.namelist() if n.endswith(".json"))
        tasks = json.loads(zf.read(jname))
else:
    tasks = resp.json()

# Examine RLE data structure
task = tasks[0]
annotations = task.get("annotations", [])
results = annotations[0].get("result", []) if annotations else []

print("=== RLE DATA ANALYSIS ===")
for i, res in enumerate(results):
    if res["type"] == "brushlabels":
        name = res["value"]["brushlabels"][0]
        rle = res["value"]["rle"]
        
        print(f"\n--- {name} ---")
        print(f"RLE type: {type(rle)}")
        print(f"RLE length: {len(rle)}")
        print(f"First 10 values: {rle[:10]}")
        print(f"Last 10 values: {rle[-10:]}")
        print(f"Min value: {min(rle)}")
        print(f"Max value: {max(rle)}")
        print(f"Sum: {sum(rle)}")
        
        # Check if it might be coordinates vs run-length
        if len(rle) > 20:
            print(f"Values 10-20: {rle[10:20]}")
        
        # Look for patterns
        even_vals = rle[0::2][:5]  # start positions
        odd_vals = rle[1::2][:5]   # lengths
        print(f"Even indices (starts): {even_vals}")
        print(f"Odd indices (lengths): {odd_vals}") 