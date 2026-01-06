import os, json, argparse, math
from pathlib import Path
from collections import defaultdict

import cv2
import pandas as pd
from tqdm import tqdm


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def laplacian_var(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def mean_luma(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean())


def xywh_to_yolo(x, y, w, h, W, H):
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    bw = w / W
    bh = h / H
    return clamp(cx), clamp(cy), clamp(bw), clamp(bh)


def normalize_bbox_to_xywh(bbox):
    """
    NightOwls COCO-like bbox should be [x, y, w, h] (pixels).
    For robustness, also accept [x1, y1, x2, y2] and convert.
    """
    if bbox is None or len(bbox) != 4:
        return None
    a, b, c, d = bbox
    a = float(a); b = float(b); c = float(c); d = float(d)

    # Heuristic: if c>d? no. Better: if c and d look like x2,y2 (usually > a,b)
    # If c > a and d > b and (c - a) > 1 and (d - b) > 1, treat as xyxy.
    # Otherwise treat as xywh.
    if (c > a) and (d > b) and ((c - a) > 1.0) and ((d - b) > 1.0):
        # could be xyxy
        x1, y1, x2, y2 = a, b, c, d
        w = x2 - x1
        h = y2 - y1
        return x1, y1, w, h
    else:
        # xywh
        x, y, w, h = a, b, c, d
        return x, y, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", required=True, help="NightOwls training json path")
    ap.add_argument("--img_dir", required=True, help="dir containing png frames")
    ap.add_argument("--out_root", required=True, help="output root, e.g. nightowls_person_yolo")

    ap.add_argument("--target_keep", type=int, default=8000, help="how many images to keep (after filtering)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min_area_ratio", type=float, default=0.0005, help="min bbox area ratio to keep")
    ap.add_argument("--min_short_side", type=int, default=480, help="drop images with min(w,h) < this")

    ap.add_argument("--blur_thr", type=float, default=0.0, help="if >0, drop images with lap_var < thr")
    ap.add_argument("--night_luma_thr", type=float, default=60.0, help="night luma threshold for is_night flag (NightOwls mostly night)")
    args = ap.parse_args()

    json_path = Path(args.json_path)
    img_dir = Path(args.img_dir)

    out_root = Path(args.out_root)
    out_img_dir = out_root / "images" / "train"
    out_lbl_dir = out_root / "labels" / "train"
    ensure_dir(str(out_img_dir))
    ensure_dir(str(out_lbl_dir))

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- categories: find pedestrian/person ---
    cats = data.get("categories", [])
    ped_ids = []
    for c in cats:
        name = str(c.get("name", "")).lower()
        if name in ("pedestrian", "person") or ("ped" in name) or ("person" in name):
            ped_ids.append(int(c["id"]))
    if not ped_ids:
        raise RuntimeError("No pedestrian/person category found in categories.")
    # You already verified: {'name':'pedestrian','id':1}
    ped_id = ped_ids[0]

    # --- images dict ---
    imgs = {int(im["id"]): im for im in data.get("images", [])}

    # --- annotations by image_id (only pedestrian) ---
    anns_by_img = defaultdict(list)
    for a in data.get("annotations", []):
        if int(a.get("category_id", -1)) != ped_id:
            continue
        if int(a.get("iscrowd", 0)) == 1:
            continue
        bbox = a.get("bbox", None)
        if bbox is None:
            continue
        anns_by_img[int(a["image_id"])].append(bbox)

    # --- candidates: images that have at least 1 ped bbox ---
    candidates = []
    for img_id, im in imgs.items():
        if img_id not in anns_by_img:
            continue
        # ensure file exists
        fn = im.get("file_name")
        if not fn:
            continue
        if not (img_dir / fn).exists():
            continue
        candidates.append(img_id)

    if not candidates:
        raise RuntimeError("No candidate images found with pedestrian annotations.")

    # --- sort by timestamp for de-correlation ---
    def get_ts(img_id):
        im = imgs[img_id]
        ts = im.get("timestamp", 0)
        try:
            return int(ts)
        except:
            return 0

    candidates.sort(key=get_ts)

    # --- subsample to target_keep uniformly (avoid over-dense frames) ---
    target = max(1, int(args.target_keep))
    if len(candidates) > target:
        step = int(math.ceil(len(candidates) / target))
        selected = candidates[::step][:target]
    else:
        selected = candidates

    rows = []
    kept = 0
    dropped = 0
    drop_boxes = 0

    for img_id in tqdm(selected, desc="NightOwls -> YOLO"):
        im = imgs[img_id]
        W = int(im["width"])
        H = int(im["height"])
        if min(W, H) < args.min_short_side:
            dropped += 1
            continue

        fn = im["file_name"]
        src = img_dir / fn
        if not src.exists():
            dropped += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            dropped += 1
            continue

        lv = laplacian_var(img)
        if args.blur_thr > 0 and lv < args.blur_thr:
            dropped += 1
            continue

        ml = mean_luma(img)

        # parse and filter bboxes
        img_area = float(W * H)
        yolo_lines = []
        person_count = 0

        for bbox in anns_by_img[img_id]:
            xywh = normalize_bbox_to_xywh(bbox)
            if xywh is None:
                drop_boxes += 1
                continue
            x, y, w, h = xywh

            # clip + sanity
            x = max(0.0, min(x, W - 1.0))
            y = max(0.0, min(y, H - 1.0))
            w = max(0.0, min(w, W - x))
            h = max(0.0, min(h, H - y))
            if w <= 1 or h <= 1:
                drop_boxes += 1
                continue

            area_ratio = (w * h) / img_area
            if area_ratio < args.min_area_ratio:
                drop_boxes += 1
                continue

            cx, cy, bw, bh = xywh_to_yolo(x, y, w, h, W, H)
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            person_count += 1

        if person_count == 0:
            dropped += 1
            continue

        out_base = f"no_{img_id}_{Path(fn).stem}"
        out_img_path = out_img_dir / (out_base + ".jpg")
        out_lbl_path = out_lbl_dir / (out_base + ".txt")

        cv2.imwrite(str(out_img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        out_lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        rows.append({
            "image": str(out_img_path),
            "label": str(out_lbl_path),
            "domain": "nightowls",
            "is_night": 1 if ml < args.night_luma_thr else 1,  # NightOwls training is night; keep as 1
            "w": W, "h": H,
            "person_count": person_count,
            "mean_luma": ml,
            "lap_var": lv,
            "timestamp": im.get("timestamp", None),
        })
        kept += 1

    ensure_dir(str(out_root))
    meta_path = out_root / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("\n=== DONE ===")
    print(f"Candidates with ped: {len(candidates)}")
    print(f"Selected for processing: {len(selected)}")
    print(f"Kept images: {kept}")
    print(f"Dropped images: {dropped}")
    print(f"Dropped boxes: {drop_boxes}")
    print(f"Meta saved: {meta_path}")
    print(f"Images out: {out_img_dir}")
    print(f"Labels out: {out_lbl_dir}")


if __name__ == "__main__":
    main()
