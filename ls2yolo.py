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

# ──────────────── Load the LS access token ────────────────
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
        # minimal fallback
        import numpy as np
        def rle2mask(rle, shape):
            h, w = shape
            mask = np.zeros(h * w, dtype=np.uint8)
            for start, length in zip(rle[0::2], rle[1::2]):
                mask[int(start): int(start) + int(length)] = 1
            return mask.reshape(h, w)

# ──────────────── user paths & constants ────────────────
LS_HOST    = "http://127.0.0.1:8080"
PROJECT_ID = 7                                         # change to your project
OUT_ROOT   = pathlib.Path("yolo_dualp")                # dataset output
IMG_ROOT   = pathlib.Path("/home/lab/.local/share/label-studio/media/upload")

LABEL2KP   = {"pupil_mask": 0,
              "purkinje1_mask": 1,
              "purkinje4_mask": 2}

MASK_COLOR = {  # BGR for OpenCV overlays
    "pupil_mask":     (255,128,  0),
    "purkinje1_mask": (  0,128,255),
    "purkinje4_mask": ( 60,  0,255)
}

# 1 ───── Download the export straight from Label Studio ─────
export_url = (
    f"{LS_HOST}/api/projects/{PROJECT_ID}/export"
    "?exportType=JSON&download_all_tasks=true"
)
print("Requesting:", export_url)

resp = requests.get(export_url,
                    headers={"Authorization": f"Token {LS_TOKEN}"},
                    timeout=120,
                    stream=True)
resp.raise_for_status()

ctype = resp.headers.get("content-type", "")
if ctype.startswith("application/zip"):
    with ZipFile(BytesIO(resp.content)) as zf:
        json_name = next(n for n in zf.namelist() if n.endswith(".json"))
        tasks = json.loads(zf.read(json_name))
else:                        # direct JSON
    tasks = resp.json()

print(f"✓ {len(tasks)} tasks downloaded from project {PROJECT_ID}")

# 2 ───── utility helpers ─────
def poly_to_mask(poly, h, w):
    pts = np.array(poly, np.float32).reshape(-1, 2)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask

def draw_overlay(img, masks, box, kps):
    view = img.copy()
    for name, m in masks.items():
        color = np.array(MASK_COLOR[name])
        view[m.astype(bool)] = 0.6 * view[m > 0] + 0.4 * color
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(view, (x1,y1), (x2,y2), (0,255,0), 2)
    for x,y in kps:
        cv2.drawMarker(view, (int(x),int(y)), (0,0,0),
                       cv2.MARKER_TILTED_CROSS, 5, 2)
    return view

# make output folders
for sub in ("images/train", "labels/train", "preview"):
    (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

# 3 ───── Convert each task ─────
for i, task in enumerate(tasks, 1):
    rel_path = task["data"]["image"].split("/data/upload")[-1]
    src_img  = IMG_ROOT / rel_path.lstrip("/")
    if not src_img.exists():
        print(f"⚠️  Missing file: {src_img} — skipping")
        continue

    img = cv2.imread(str(src_img))
    if img is None:
        print(f"⚠️  Could not read {src_img}")
        continue
    h, w = img.shape[:2]

    if not task["annotations"]:
        print(f"⚠️  No annotations in task {i}, skipping")
        continue
    results = task["annotations"][0].get("result", [])
    if not results:
        print(f"⚠️  Empty results in task {i}, skipping")
        continue

    masks, kps = {}, np.zeros((3, 2), np.float32)

    for res in results:
        if res["type"] != "brushlabels":
            continue
        name = res["value"]["brushlabels"][0]

        # polygon or RLE
        if "points" in res["value"]:
            pts = [(p[0] * w / 100, p[1] * h / 100) for p in res["value"]["points"]]
            mask = poly_to_mask(pts, h, w)
        else:
            mask = rle2mask(res["value"]["rle"], (h, w))

        if mask.sum() == 0:
            continue

        masks[name] = mask
        ys, xs = np.nonzero(mask)
        kps[LABEL2KP[name]] = (xs.mean(), ys.mean())

    if not masks:
        print(f"⚠️  No usable masks in task {i}")
        continue

    # bounding box
    x_min, y_min = np.clip(kps.min(0) - 5, 0, [w-1, h-1])
    x_max, y_max = np.clip(kps.max(0) + 5, 0, [w-1, h-1])
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    bw, bh = x_max - x_min, y_max - y_min

    norm = lambda v, d: v / d
    yolo_items = [0, norm(cx,w), norm(cy,h), norm(bw,w), norm(bh,h)]
    for x, y in kps:
        yolo_items += [norm(x,w), norm(y,h), 1]

    stem = src_img.stem
    (OUT_ROOT/"labels/train"/f"{stem}.txt").write_text(
        " ".join(f"{v:.6f}" for v in yolo_items)
    )
    shutil.copy(src_img, OUT_ROOT/"images/train"/f"{stem}.jpg")

    prev = draw_overlay(img, masks, (x_min,y_min,x_max,y_max), kps)
    cv2.imwrite(str(OUT_ROOT/"preview"/f"{stem}_preview.jpg"), prev)

    if i % 50 == 0:
        print(f"  processed {i}/{len(tasks)} images")

print("✓ YOLO dataset (images, labels, previews) written to", OUT_ROOT)
