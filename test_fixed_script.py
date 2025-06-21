#!/usr/bin/env python3
"""
Test the fixed ls2yolo logic on one image
"""

import os, json, cv2, numpy as np, pathlib, requests, shutil
from io import BytesIO
from zipfile import ZipFile

# Import correct RLE decoder from Label Studio
try:
    from label_studio_converter.brush import decode_rle
except ImportError:
    raise ImportError("label_studio_converter is required. Install with: pip install label-studio-converter")

def decode_ls_rle(rle, h, w):
    """Decode Label Studio RLE format to 2D mask."""
    flat_mask = decode_rle(rle)
    
    # Handle different RLE formats
    total_pixels = h * w
    flat_size = len(flat_mask)
    
    if flat_size == total_pixels * 4:
        # RGBA format - take first channel
        mask = flat_mask.reshape(h, w, 4)[:, :, 0]
    elif flat_size == total_pixels:
        # Direct format
        mask = flat_mask.reshape(h, w)
    else:
        # Unknown format - try to extract what we can
        mask = flat_mask[:total_pixels].reshape(h, w)
    
    return mask.astype(np.uint8)

# Load config
try:
    from config import LS_TOKEN
except ImportError:
    LS_TOKEN = os.environ.get("LS_TOKEN")

# Settings
LS_HOST = "http://127.0.0.1:8080"
PROJECT_ID = 7
IMG_ROOT = pathlib.Path("/home/lab/.local/share/label-studio/media/upload")
LABEL2KP = {"pupil_mask": 0, "purkinje1_mask": 1, "purkinje4_mask": 2}

def poly_to_mask(poly, h, w):
    pts = np.array(poly, np.float32).reshape(-1,2)
    mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask

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

print(f"✓ {len(tasks)} tasks downloaded")

# Test first task
task = tasks[0]
rel = task["data"]["image"].split("/data/upload")[-1]
src_img = IMG_ROOT/rel.lstrip("/")

print(f"\n=== TESTING FIRST IMAGE ===")
print(f"Processing: {src_img.name}")

img = cv2.imread(str(src_img))
h, w = img.shape[:2]
results = task["annotations"][0].get("result", []) if task["annotations"] else []
masks = {}

for res in results:
    if res["type"]!="brushlabels": continue
    name = res["value"]["brushlabels"][0]
    if "points" in res["value"]:
        pts = [(p[0]*w/100, p[1]*h/100) for p in res["value"]["points"]]
        mask = poly_to_mask(pts, h, w)
    else:
        mask = decode_ls_rle(res["value"]["rle"], h, w)
    if mask.sum()==0: continue
    masks[name] = mask

print(f"Found masks: {list(masks.keys())}")

# Generate YOLO labels
labels = []
norm = lambda v,d: v/d

for mask_name, mask in masks.items():
    class_id = LABEL2KP[mask_name]
    
    # Individual bounding box for this mask
    ys, xs = np.nonzero(mask)
    x_min, x_max = xs.min()-5, xs.max()+5
    y_min, y_max = ys.min()-5, ys.max()+5
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w-1, x_max), min(h-1, y_max)
    
    if x_max <= x_min or y_max <= y_min:
        print(f"⚠️  Degenerate box for {mask_name}")
        continue
        
    # YOLO format: class_id center_x center_y width height
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    bw, bh = x_max - x_min, y_max - y_min
    
    label_line = f"{class_id} {norm(cx,w):.6f} {norm(cy,h):.6f} {norm(bw,w):.6f} {norm(bh,h):.6f}"
    labels.append(label_line)
    
    print(f"{mask_name} (class {class_id}):")
    print(f"  Pixels: {mask.sum()}")
    print(f"  Bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
    print(f"  YOLO: {label_line}")

print(f"\nGenerated {len(labels)} YOLO labels:")
for label in labels:
    print(f"  {label}") 