#!/usr/bin/env python3
"""
Test the correct decode_rle function from Label Studio
"""

import os, json, cv2, numpy as np, requests
from io import BytesIO
from zipfile import ZipFile

# Import the correct function
from label_studio_converter.brush import decode_rle

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

# Test decode_rle function
task = tasks[0]
annotations = task.get("annotations", [])
results = annotations[0].get("result", []) if annotations else []

print("=== TESTING CORRECT decode_rle ===")

# Get image dimensions
rel = task["data"]["image"].split("/data/upload")[-1]
IMG_ROOT = "/home/lab/.local/share/label-studio/media/upload"
src_img = f"{IMG_ROOT}/{rel.lstrip('/')}"
img = cv2.imread(src_img)
h, w = img.shape[:2]
print(f"Image dimensions: {w}x{h}")

for i, res in enumerate(results):
    if res["type"] == "brushlabels":
        name = res["value"]["brushlabels"][0]
        rle = res["value"]["rle"]
        
        print(f"\n--- Testing {name} ---")
        try:
            # Try decode_rle function
            mask = decode_rle(rle, (h, w))
            print(f"✓ decode_rle succeeded!")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask pixels: {mask.sum()}")
            
            if mask.sum() > 0:
                ys, xs = np.nonzero(mask)
                print(f"  Bounding box: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")
                print(f"  Center: ({xs.mean():.1f}, {ys.mean():.1f})")
                
                # Save mask for visual inspection
                mask_img = (mask * 255).astype(np.uint8)
                cv2.imwrite(f"correct_{name}_mask.png", mask_img)
                print(f"  Saved correct_{name}_mask.png")
            else:
                print(f"  ⚠️  Empty mask!")
                
        except Exception as e:
            print(f"✗ decode_rle failed: {e}")
            
            # Try different parameter orders
            try:
                mask = decode_rle(rle)
                print(f"✓ decode_rle with single param succeeded!")
                print(f"  Mask shape: {mask.shape}")
            except Exception as e2:
                print(f"✗ decode_rle single param also failed: {e2}")

print(f"\nTest complete!") 