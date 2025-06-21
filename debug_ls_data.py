#!/usr/bin/env python3
"""
Debug script to examine Label Studio data structure and masks
"""

import os, json, cv2, numpy as np, pathlib, requests
from io import BytesIO
from zipfile import ZipFile

# Load config
try:
    from config import LS_TOKEN
except ImportError:
    LS_TOKEN = os.environ.get("LS_TOKEN")
    if not LS_TOKEN:
        raise RuntimeError("Set LS_TOKEN in config.py or export LS_TOKEN")

# rle2mask function
try:
    from label_studio_converter.utils import rle2mask
except ImportError:
    try:
        from label_studio_converter.brush import rle2mask
    except ImportError:
        def rle2mask(rle, shape):
            h, w = shape
            mask = np.zeros(h * w, dtype=np.uint8)
            for s, l in zip(rle[0::2], rle[1::2]):
                mask[int(s): int(s)+int(l)] = 1
            return mask.reshape(h, w)

# Settings
LS_HOST = "http://127.0.0.1:8080"
PROJECT_ID = 7
IMG_ROOT = pathlib.Path("/home/lab/.local/share/label-studio/media/upload")

def poly_to_mask(poly, h, w):
    pts = np.array(poly, np.float32).reshape(-1,2)
    mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask

# Download data
export_url = f"{LS_HOST}/api/projects/{PROJECT_ID}/export?exportType=JSON&download_all_tasks=true"
print("Downloading data...")

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

# Debug first task
task = tasks[0]
print(f"\n=== DEBUGGING FIRST TASK ===")
print(f"Image path: {task['data']['image']}")

# Check annotations structure
annotations = task.get("annotations", [])
print(f"Number of annotations: {len(annotations)}")

if annotations:
    results = annotations[0].get("result", [])
    print(f"Number of results: {len(results)}")
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i} ---")
        print(f"Type: {res.get('type')}")
        if res.get("type") == "brushlabels":
            print(f"Label: {res['value'].get('brushlabels', [])}")
            print(f"Has points: {'points' in res['value']}")
            print(f"Has rle: {'rle' in res['value']}")
            
            if 'points' in res['value']:
                print(f"Points (first 3): {res['value']['points'][:3]}")
            if 'rle' in res['value']:
                print(f"RLE length: {len(res['value']['rle'])}")

# Process one image to see what masks look like
rel = task["data"]["image"].split("/data/upload")[-1]
src_img = IMG_ROOT/rel.lstrip("/")
print(f"\nLooking for image: {src_img}")

if src_img.exists():
    img = cv2.imread(str(src_img))
    h, w = img.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    masks = {}
    for res in results:
        if res["type"] != "brushlabels": 
            continue
        name = res["value"]["brushlabels"][0]
        print(f"\nProcessing mask: {name}")
        
        if "points" in res["value"]:
            pts = [(p[0]*w/100, p[1]*h/100) for p in res["value"]["points"]]
            print(f"  Converting {len(pts)} points to mask")
            mask = poly_to_mask(pts, h, w)
        else:
            print(f"  Converting RLE to mask")
            mask = rle2mask(res["value"]["rle"], (h, w))
        
        if mask.sum() == 0:
            print(f"  ⚠️  Empty mask!")
            continue
            
        masks[name] = mask
        
        # Analyze mask
        ys, xs = np.nonzero(mask)
        print(f"  Mask pixels: {mask.sum()}")
        print(f"  Bounding box: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")
        print(f"  Center: ({xs.mean():.1f}, {ys.mean():.1f})")
        
        # Save individual mask for visual inspection
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(f"debug_{name}_mask.png", mask_img)
        print(f"  Saved debug_{name}_mask.png")
        
    print(f"\nFound {len(masks)} valid masks: {list(masks.keys())}")
else:
    print(f"⚠️  Image not found: {src_img}") 