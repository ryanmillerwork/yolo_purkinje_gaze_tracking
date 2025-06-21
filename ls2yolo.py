"""
labelstudio_to_yolo_pose.py
───────────────────────────
• Downloads the latest JSON export from Label Studio via REST.
• Converts pupil / Purkinje-1 / Purkinje-4 masks → YOLO-Pose labels.
• Saves color-annotated preview JPEGs for quick QC.

Run inside the same Python environment as your MobileSAM backend.

Dependencies:
    pip install label-studio-sdk opencv-python numpy requests
"""

import os, json, cv2, numpy as np, pathlib, requests, shutil
from io import BytesIO
from zipfile import ZipFile
from config import LS_TOKEN

# ─────────────────────── user settings ────────────────────────
LS_HOST    = "http://127.0.0.1:8080"               # Label Studio host
PROJECT_ID = 7                                    # project ID
OUT_ROOT   = pathlib.Path("yolo_dualp")            # output dataset folder
IMG_ROOT   = "/home/lab/.local/share/label-studio/media/upload"  # LS upload dir on server
# ───────────────────────────────────────────────────────────────

LABEL2KP   = {"pupil_mask":0, "purkinje1_mask":1, "purkinje4_mask":2}
MASK_COLOR = {                                  # BGR for OpenCV
    "pupil_mask":     (255,128,  0),
    "purkinje1_mask": (  0,128,255),
    "purkinje4_mask": ( 60,  0,255)
}

# ----------------------------------------------------------------
# 1.  Download fresh JSON export
# ----------------------------------------------------------------
export_url = f"{LS_HOST}/api/projects/{PROJECT_ID}/export?exportType=JSON&download_all_tasks=true"
headers    = {"Authorization": f"Token {LS_TOKEN}"}
print(f"Requesting export from: {export_url}")
response   = requests.get(export_url, headers=headers, timeout=120, stream=True)

print(f"Response status: {response.status_code}")
print(f"Response headers: {dict(response.headers)}")
print(f"Response content type: {response.headers.get('content-type', 'unknown')}")
print(f"Response content length: {len(response.content)} bytes")

# Debug: show first 200 chars of response if it's not a zip
if not response.headers.get('content-type', '').startswith('application/zip'):
    print(f"Response content preview: {response.content[:200]}")

response.raise_for_status()

# Handle both ZIP and direct JSON responses
content_type = response.headers.get('content-type', '')
if content_type.startswith('application/zip'):
    # ZIP file containing JSON
    with ZipFile(BytesIO(response.content)) as zf:
        json_file = next((n for n in zf.namelist() if n.endswith(".json")), None)
        if not json_file:
            raise RuntimeError("Export zip had no JSON file inside.")
        tasks = json.loads(zf.read(json_file))
elif 'json' in content_type:
    # Direct JSON response
    tasks = response.json()
else:
    raise RuntimeError(f"Unexpected content type: {content_type}. Expected JSON or ZIP file.")

print(f"✓ Downloaded {len(tasks)} tasks from project {PROJECT_ID}")

# ----------------------------------------------------------------
# 2.  Utilities
# ----------------------------------------------------------------
def poly_to_mask(poly, h, w):
    pts = np.array(poly, np.float32).reshape(-1,2)
    mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask

def rle_to_mask(rle_data, h, w):
    """Convert RLE format to binary mask"""
    # RLE format: [start1, length1, start2, length2, ...]
    mask = np.zeros(h * w, dtype=np.uint8)
    for i in range(0, len(rle_data), 2):
        if i + 1 < len(rle_data):
            start = int(rle_data[i])
            length = int(rle_data[i + 1])
            mask[start:start + length] = 1
    return mask.reshape(h, w)

def draw_overlay(img, masks, bbox, kps):
    out = img.copy()
    for name, m in masks.items():
        out[m.astype(bool)] = (
            0.4 * np.array(MASK_COLOR[name]) + 0.6 * out[m.astype(bool)]
        ).astype(np.uint8)
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    for x,y in kps:
        cv2.drawMarker(out, (int(x),int(y)), (0,0,0),
                       cv2.MARKER_TILTED_CROSS, 5, 2)
    return out

# create output dirs
for sub in ("images/train", "labels/train", "preview"):
    (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------
# 3.  Convert each task
# ----------------------------------------------------------------
for i, task in enumerate(tasks):
    print(f"\nProcessing task {i+1}/{len(tasks)}")
    print(f"Raw image path: {task['data']['image']}")
    
    rel_path = task["data"]["image"].split("/data/upload")[-1]
    src_img  = pathlib.Path(IMG_ROOT + rel_path)
    print(f"Constructed path: {src_img}")
    print(f"File exists: {src_img.exists()}")
    
    if not src_img.exists():
        # Try to find the file by searching for it
        filename = src_img.name
        print(f"⚠️  File not found at: {src_img}")
        print(f"Searching for: {filename}")
        
        # Search in common Label Studio locations
        search_paths = [
            f"/data/upload/{filename}",
            f"/data/upload/7/{filename}",
            f"/label-studio/data/upload/{filename}",
            f"/label-studio/data/upload/7/{filename}",
            f"/home/lab/labeling/data/upload/{filename}",
            f"/home/lab/labeling/data/upload/7/{filename}",
        ]
        
        found_file = None
        for search_path in search_paths:
            if pathlib.Path(search_path).exists():
                found_file = pathlib.Path(search_path)
                print(f"✓ Found file at: {found_file}")
                break
        
        if not found_file:
            print(f"⚠️  Could not find {filename} in any common locations")
            continue
        
        src_img = found_file
        
    img = cv2.imread(str(src_img))
    if img is None:
        print(f"⚠️  Failed to load image: {src_img}")
        continue
        
    h, w = img.shape[:2]

    # Skip tasks without annotations
    if not task["annotations"] or len(task["annotations"]) == 0:
        print(f"⚠️  No annotations for this task, skipping...")
        continue
        
    if not task["annotations"][0].get("result"):
        print(f"⚠️  No annotation results for this task, skipping...")
        continue

    masks, kps = {}, np.zeros((3,2), np.float32)

    for ann in task["annotations"][0]["result"]:
        if ann["type"] != "brushlabels":
            continue
        name = ann["value"]["brushlabels"][0]
        
        # Handle different annotation formats
        if "points" in ann["value"]:
            # Polygon points format
            poly = [(p[0]*w/100, p[1]*h/100) for p in ann["value"]["points"]]
            mask = poly_to_mask(poly, h, w)
        elif "rle" in ann["value"]:
            # RLE format
            rle_data = ann["value"]["rle"]
            mask = rle_to_mask(rle_data, h, w)
        else:
            print(f"⚠️  Unknown annotation format for {name}")
            continue
            
        masks[name] = mask
        ys, xs = np.nonzero(mask)
        if len(xs) > 0 and len(ys) > 0:
            kps[LABEL2KP[name]] = (xs.mean(), ys.mean())
        else:
            print(f"⚠️  Empty mask for {name}")
            continue

    # bounding box + YOLO line
    x_min, y_min = kps.min(0) - 5
    x_max, y_max = kps.max(0) + 5
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    bw, bh = x_max - x_min,    y_max - y_min

    norm = lambda v, dim: max(min(v / dim, 1), 0)
    items = [0, norm(cx,w), norm(cy,h), norm(bw,w), norm(bh,h)]
    for x, y in kps:
        items += [norm(x,w), norm(y,h), 2]      # vis=2 (visible)
    yolo_line = " ".join(f"{v:.6f}" for v in items)

    stem = src_img.stem
    (OUT_ROOT/"labels/train"/f"{stem}.txt").write_text(yolo_line)
    shutil.copy(src_img, OUT_ROOT/"images/train"/f"{stem}.jpg")

    preview = draw_overlay(img, masks, (x_min,y_min,x_max,y_max), kps)
    cv2.imwrite(str(OUT_ROOT/"preview"/f"{stem}_preview.jpg"), preview)

print("✓ YOLO dataset with previews written to", OUT_ROOT)
