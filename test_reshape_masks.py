#!/usr/bin/env python3
"""
Test reshaping the decoded RLE masks to proper image dimensions
"""

import os, json, cv2, numpy as np, requests
from io import BytesIO
from zipfile import ZipFile
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

# Test reshaping masks
task = tasks[0]
annotations = task.get("annotations", [])
results = annotations[0].get("result", []) if annotations else []

print("=== TESTING MASK RESHAPING ===")

# Get image
rel = task["data"]["image"].split("/data/upload")[-1]
IMG_ROOT = "/home/lab/.local/share/label-studio/media/upload"
src_img = f"{IMG_ROOT}/{rel.lstrip('/')}"
img = cv2.imread(src_img)
h, w = img.shape[:2]
print(f"Image dimensions: {w}x{h}")

masks = {}
for i, res in enumerate(results):
    if res["type"] == "brushlabels":
        name = res["value"]["brushlabels"][0]
        rle = res["value"]["rle"]
        
        print(f"\n--- Processing {name} ---")
        
        # Decode the RLE
        flat_mask = decode_rle(rle)
        print(f"Flat mask shape: {flat_mask.shape}")
        print(f"Flat mask pixels: {flat_mask.sum()}")
        
        # Try different reshaping approaches
        total_pixels = w * h
        flat_size = len(flat_mask)
        
        print(f"Image pixels: {total_pixels}")
        print(f"Flat mask size: {flat_size}")
        print(f"Ratio: {flat_size / total_pixels}")
        
        # Try reshaping to image dimensions
        if flat_size == total_pixels:
            # Perfect match - reshape directly
            mask = flat_mask.reshape(h, w)
        elif flat_size == total_pixels * 4:
            # Might be RGBA - take first channel
            mask = flat_mask.reshape(h, w, 4)[:, :, 0]
        else:
            # Unknown format - try multiple approaches
            print(f"  Trying to find best reshape...")
            
            # Try as RGBA
            if flat_size % (h * w) == 0:
                channels = flat_size // (h * w)
                print(f"  Possible channels: {channels}")
                mask = flat_mask.reshape(h, w, channels)[:, :, 0]
            else:
                print(f"  Cannot reshape cleanly - using modulo approach")
                mask = flat_mask[:total_pixels].reshape(h, w)
        
        print(f"Reshaped mask: {mask.shape}")
        print(f"Reshaped pixels: {mask.sum()}")
        
        if mask.sum() > 0:
            ys, xs = np.nonzero(mask)
            print(f"Bounding box: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")
            print(f"Center: ({xs.mean():.1f}, {ys.mean():.1f})")
            
            # Save the properly reshaped mask
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(f"reshaped_{name}_mask.png", mask_img)
            print(f"Saved reshaped_{name}_mask.png")
            
            masks[name] = mask
        else:
            print(f"Empty mask after reshaping!")

print(f"\nFound {len(masks)} valid reshaped masks: {list(masks.keys())}") 