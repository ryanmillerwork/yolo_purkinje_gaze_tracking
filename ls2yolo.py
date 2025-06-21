#!/usr/bin/env python3
"""
ls2yolo.py
──────────
• Downloads the latest JSON export from Label Studio.
• Converts pupil / Purkinje-1 / Purkinje-4 masks → YOLO-Pose labels.
• Saves color-overlay preview images for quick QC.

Run inside the same Python venv you use for MobileSAM.

Dependencies:
    pip install label-studio-converter opencv-python numpy requests
"""

import os, json, cv2, numpy as np, pathlib, requests, shutil
from io import BytesIO
from zipfile import ZipFile

# ─────────────── Load the LS access token ────────────────
try:                                  # preferred: config.py
    from config import LS_TOKEN
except ImportError:                   # fallback: environment variable
    LS_TOKEN = os.environ.get("LS_TOKEN")
    if not LS_TOKEN:
        raise RuntimeError(
            "Set LS_TOKEN in config.py or export LS_TOKEN in the shell."
        )

# rle2mask import that works with any converter version
try:
    from label_studio_converter.utils import rle2mask
except ImportError:
    try:
        from label_studio_converter.brush import rle2mask
    except ImportError:
        def rle2mask(rle, shape):
            """Minimal RLE decoder (start, length, …)."""
            h, w = shape
            mask = np.zeros(h * w, dtype=np.uint8)
            for s, l in zip(rle[0::2], rle[1::2]):
                mask[int(s): int(s)+int(l)] = 1
            return mask.reshape(h, w)

# ─────────────── user paths & constants ───────────────
LS_HOST    = "http://127.0.0.1:8080"
PROJECT_ID = 7
OUT_ROOT   = pathlib.Path("yolo_dualp")
IMG_ROOT   = pathlib.Path("/home/lab/.local/share/label-studio/media/upload")

LABEL2KP = {"pupil_mask": 0, "purkinje1_mask": 1, "purkinje4_mask": 2}
REQUIRED  = set(LABEL2KP)            # all three masks must be present

MASK_COLOR = {                       # BGR colors for previews
    "pupil_mask":     (255,128,  0),
    "purkinje1_mask": (  0,128,255),
    "purkinje4_mask": ( 60,  0,255)
}

# ───── 1. Download the export straight from Label Studio ─────
export_url = f"{LS_HOST}/api/projects/{PROJECT_ID}/export?exportType=JSON&download_all_tasks=true"
print("Requesting:", export_url)

resp = requests.get(export_url,
                    headers={"Authorization": f"Token {LS_TOKEN}"},
                    timeout=120, stream=True)
resp.raise_for_status()

ctype = resp.headers.get("content-type", "")
if ctype.startswith("application/zip"):
    with ZipFile(BytesIO(resp.content)) as zf:
        jname = next(n for n in zf.namelist() if n.endswith(".json"))
        tasks = json.loads(zf.read(jname))
else:
    tasks = resp.json()

print(f"✓ {len(tasks)} tasks downloaded from project {PROJECT_ID}")

# ───── 2. helpers ─────
def poly_to_mask(poly, h, w):
    pts = np.array(poly, np.float32).reshape(-1,2)
    mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask

def draw_overlay(img, masks, boxes):
    view = img.copy()
    
    # Draw mask overlays with colors
    for name, m in masks.items():
        color = np.array(MASK_COLOR[name])
        view[m.astype(bool)] = 0.6*view[m>0] + 0.4*color
    
    # Draw individual bounding boxes
    for x1, y1, x2, y2, mask_name in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label text
        class_id = LABEL2KP[mask_name]
        label_text = f"{mask_name} (cls:{class_id})"
        cv2.putText(view, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return view

# create output dirs
for sub in ("images/train", "labels/train", "preview"):
    (OUT_ROOT/sub).mkdir(parents=True, exist_ok=True)

# ───── 3. Convert each task ─────
for i, task in enumerate(tasks, 1):
    rel = task["data"]["image"].split("/data/upload")[-1]
    src_img = IMG_ROOT/rel.lstrip("/")
    if not src_img.exists():
        print(f"⚠️  Missing image {src_img}, skipping"); continue

    img = cv2.imread(str(src_img));  h, w = img.shape[:2]
    results = task["annotations"][0].get("result", []) if task["annotations"] else []
    masks = {}

    for res in results:
        if res["type"]!="brushlabels": continue
        name = res["value"]["brushlabels"][0]
        if "points" in res["value"]:
            pts = [(p[0]*w/100, p[1]*h/100) for p in res["value"]["points"]]
            mask = poly_to_mask(pts, h, w)
        else:
            mask = rle2mask(res["value"]["rle"], (h, w))
        if mask.sum()==0: continue
        masks[name] = mask

    # Create separate labels for each detected mask
    if not masks:
        print(f"⚠️  No masks found in {src_img.name}, skipping")
        continue

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
            print(f"⚠️  Degenerate box for {mask_name} in {src_img.name}, skipping this mask")
            continue
            
        # YOLO format: class_id center_x center_y width height
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        bw, bh = x_max - x_min, y_max - y_min
        
        label_line = f"{class_id} {norm(cx,w):.6f} {norm(cy,h):.6f} {norm(bw,w):.6f} {norm(bh,h):.6f}"
        labels.append(label_line)
    
    if not labels:
        print(f"⚠️  No valid labels generated for {src_img.name}, skipping")
        continue

    stem = src_img.stem
    (OUT_ROOT/"labels/train"/f"{stem}.txt").write_text("\n".join(labels))
    shutil.copy(src_img, OUT_ROOT/"images/train"/f"{stem}.jpg")
    # Create preview with individual bboxes for each mask
    individual_boxes = []
    for mask_name, mask in masks.items():
        ys, xs = np.nonzero(mask)
        x_min, x_max = xs.min()-5, xs.max()+5
        y_min, y_max = ys.min()-5, ys.max()+5
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w-1, x_max), min(h-1, y_max)
        individual_boxes.append((x_min, y_min, x_max, y_max, mask_name))
    
    cv2.imwrite(str(OUT_ROOT/"preview"/f"{stem}_preview.jpg"),
                draw_overlay(img, masks, individual_boxes))

    if i % 50 == 0:
        print(f"  processed {i}/{len(tasks)} images")

print("✓ YOLO dataset (images, labels, previews) written to", OUT_ROOT)
