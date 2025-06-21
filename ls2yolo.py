#!/usr/bin/env python3
import os, json, cv2, numpy as np, pathlib, requests, shutil
from io import BytesIO
from zipfile import ZipFile
from label_studio_sdk import Client
# try the new location first
try:
    from label_studio_converter.utils import rle2mask
except ImportError:
    # fall back to old (<0.0.58) or roll-your-own
    try:
        from label_studio_converter.brush import rle2mask
    except ImportError:
        import numpy as np
        def rle2mask(rle, shape):
            """Minimal decoder for LS RLE (start, length, …)."""
            h, w = shape
            mask = np.zeros(h * w, dtype=np.uint8)
            for s, l in zip(rle[0::2], rle[1::2]):
                mask[int(s): int(s)+int(l)] = 1
            return mask.reshape(h, w)

try:
    from config import LS_TOKEN              # 1️⃣ preferred method
except ImportError:
    LS_TOKEN = os.environ.get("LS_TOKEN")    # 2️⃣ fallback
    if not LS_TOKEN:
        raise RuntimeError(
            "Label-Studio token not found. "
            "Either create config.py with LS_TOKEN='…' "
            "or export LS_TOKEN env variable."
        )

LS_HOST  = "http://127.0.0.1:8080"
PROJECT  = 7
OUT      = pathlib.Path("yolo_dualp")
MEDIA    = pathlib.Path("/home/lab/.local/share/label-studio/media/upload")

LABEL2KP = {"pupil_mask":0, "purkinje1_mask":1, "purkinje4_mask":2}
COLORS   = {"pupil_mask":(255,128,0), "purkinje1_mask":(0,128,255),"purkinje4_mask":(60,0,255)}

def overlay(img,masks,box,kps):
    ov = img.copy()
    for n,m in masks.items():
        color = np.array(COLORS[n]); ov[m.astype(bool)] = 0.6*ov[m>0]+0.4*color
    x1,y1,x2,y2 = map(int,box); cv2.rectangle(ov,(x1,y1),(x2,y2),(0,255,0),2)
    for x,y in kps: cv2.drawMarker(ov,(int(x),int(y)),(0,0,0),cv2.MARKER_TILTED_CROSS,5,2)
    return ov

def export_yolo(debug=False):
    cli   = Client(url=LS_HOST, api_key=LS_TOKEN)
    tasks = cli.get_project(PROJECT).export("JSON")

    for split in ("train","val"):
        (OUT/f"images/{split}").mkdir(parents=True,exist_ok=True)
        (OUT/f"labels/{split}").mkdir(parents=True,exist_ok=True)
    (OUT/"preview").mkdir(exist_ok=True)

    for i,t in enumerate(tasks):
        img_rel = pathlib.Path(t["data"]["image"].split("/data/upload")[-1])
        src_img = MEDIA/img_rel
        if not src_img.exists():
            if debug: print("Missing:", src_img); continue
        img = cv2.imread(str(src_img)); h,w = img.shape[:2]

        masks,kps = {}, np.zeros((3,2), np.float32)
        for res in t["annotations"][0]["result"]:
            if res["type"]!="brushlabels": continue
            name = res["value"]["brushlabels"][0]
            if "rle" in res["value"]:
                mask = rle2mask(res["value"]["rle"], (h,w))
            else:
                pts  = [(p[0]*w/100,p[1]*h/100) for p in res["value"]["points"]]
                mask = cv2.fillPoly(np.zeros((h,w),np.uint8),
                                    [np.array(pts,np.int32)],1)
            if mask.sum()==0: continue
            masks[name]=mask
            ys,xs=np.nonzero(mask); kps[LABEL2KP[name]]=(xs.mean(),ys.mean())

        if not masks: continue
        x_min,y_min = np.clip(kps.min(0)-5, 0, [w-1,h-1])
        x_max,y_max = np.clip(kps.max(0)+5, 0, [w-1,h-1])
        cx,cy,bw,bh = (x_min+x_max)/2,(y_min+y_max)/2,x_max-x_min,y_max-y_min

        norm = lambda v,d: v/d
        vals = [0,norm(cx,w),norm(cy,h),norm(bw,w),norm(bh,h)]
        for x,y in kps: vals += [norm(x,w),norm(y,h),2]
        split = "val" if i%5==0 else "train"
        stem  = src_img.stem
        (OUT/f"labels/{split}/{stem}.txt").write_text(" ".join(f"{v:.6f}" for v in vals))
        shutil.copy(src_img, OUT/f"images/{split}/{stem}.jpg")
        cv2.imwrite(str(OUT/f"preview/{stem}_preview.jpg"),
                    overlay(img,masks,(x_min,y_min,x_max,y_max),kps))

    print("✓ YOLO dataset saved in", OUT)

if __name__ == "__main__":
    export_yolo(debug=True)
