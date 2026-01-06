import os, json, argparse, shutil
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import cv2
import pandas as pd

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def mean_luma(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean())

def laplacian_var(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def coco_bbox_to_yolo(bbox_xywh, w, h):
    x, y, bw, bh = bbox_xywh
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    bw = bw / w
    bh = bh / h
    return clamp(cx), clamp(cy), clamp(bw), clamp(bh)

def find_ann_file(root: Path, name: str):
    # 兼容 train / train.json / train.jsonl 之类
    candidates = [root / name, root / f"{name}.json", root / f"{name}.jsonl"]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Cannot find annotation file for '{name}' in {root}. Tried: {candidates}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch_root", required=True, help="CrowdHuman root containing Images/ and train/val annotation files")
    ap.add_argument("--out_root", required=True, help="output root e.g. /home/didu/datasets/crowdhuman_person_yolo")
    ap.add_argument("--min_area_ratio", type=float, default=0.001)
    ap.add_argument("--min_short_side", type=int, default=480)
    ap.add_argument("--crowd_count_thr", type=int, default=60)
    ap.add_argument("--crowd_median_area_ratio", type=float, default=0.0015)
    ap.add_argument("--blur_thr", type=float, default=0.0)
    ap.add_argument("--dark_thr", type=float, default=0.0)
    args = ap.parse_args()

    root = Path(args.ch_root)
    img_dir = root / "Images"
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing Images dir: {img_dir}")

    ann_train = find_ann_file(root, "train")
    ann_val   = find_ann_file(root, "val")

    out_img_dir = Path(args.out_root) / "images" / "train"
    out_lbl_dir = Path(args.out_root) / "labels" / "train"
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    rows = []
    kept = 0
    dropped = 0

    for split_name, ann_path in [("train", ann_train), ("val", ann_val)]:
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # person category id（一般是 1）
        person_cat_ids = [c["id"] for c in coco.get("categories", []) if c.get("name") == "person"]
        if not person_cat_ids:
            raise RuntimeError(f"No 'person' category found in {ann_path}")
        person_cat_id = person_cat_ids[0]

        # images: id -> info
        imgs = {im["id"]: im for im in coco.get("images", [])}

        # anns by image_id
        anns_by_img = defaultdict(list)
        for a in coco.get("annotations", []):
            if a.get("iscrowd", 0) == 1:
                continue
            if a.get("category_id") != person_cat_id:
                continue
            if "bbox" not in a:
                continue
            anns_by_img[a.get("image_id")].append(a)

        for img_id, im in tqdm(imgs.items(), desc=f"Processing {split_name}"):
            if img_id not in anns_by_img:
                continue

            w = int(im.get("width", 0))
            h = int(im.get("height", 0))
            if w <= 0 or h <= 0:
                # 尝试从图读尺寸
                # file_name 可能没有，优先用 image_id
                pass

            # file_name 兼容：有些 CrowdHuman COCO 格式可能没有 file_name
            file_name = im.get("file_name", None)
            if file_name:
                # 确保有后缀
                if not (file_name.endswith(".jpg") or file_name.endswith(".png")):
                    # 很多 CrowdHuman 图片就是 .jpg
                    file_name = file_name + ".jpg"
                src_path = img_dir / file_name
            else:
                # 用 image_id 拼 jpg
                src_path = img_dir / (str(img_id) + ".jpg")

            if not src_path.exists():
                # 再试一次：如果 image_id 是 "xxx,yyy" 这种，图片可能就是这个名字
                alt = img_dir / (str(im.get("image_id", img_id)) + ".jpg")
                if alt.exists():
                    src_path = alt
                else:
                    dropped += 1
                    continue

            img = cv2.imread(str(src_path))
            if img is None:
                dropped += 1
                continue
            hh, ww = img.shape[:2]
            if w <= 0 or h <= 0:
                w, h = ww, hh

            if min(w, h) < args.min_short_side:
                dropped += 1
                continue

            lv = laplacian_var(img)
            ml = mean_luma(img)
            if args.blur_thr > 0 and lv < args.blur_thr:
                dropped += 1
                continue
            if args.dark_thr > 0 and ml < args.dark_thr:
                dropped += 1
                continue

            # bbox 过滤
            img_area = float(w * h)
            areas = []
            yolo_lines = []
            for a in anns_by_img[img_id]:
                x, y, bw, bh = a["bbox"]
                if bw <= 1 or bh <= 1:
                    continue
                area_ratio = (bw * bh) / img_area
                if area_ratio < args.min_area_ratio:
                    continue
                cx, cy, bw_n, bh_n = coco_bbox_to_yolo(a["bbox"], w, h)
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")
                areas.append(area_ratio)

            if not yolo_lines:
                dropped += 1
                continue

            person_count = len(yolo_lines)
            med_area = float(pd.Series(areas).median()) if areas else 0.0

            # 极端点阵过滤（CrowdHuman 会更密，所以阈值比 COCO 放宽）
            if person_count > args.crowd_count_thr and med_area < args.crowd_median_area_ratio:
                dropped += 1
                continue

            out_base = f"ch_{split_name}_{str(img_id).replace('/','_')}"
            out_img_path = out_img_dir / (out_base + ".jpg")
            out_lbl_path = out_lbl_dir / (out_base + ".txt")

            shutil.copy2(str(src_path), str(out_img_path))
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")

            rows.append({
                "image": str(out_img_path),
                "label": str(out_lbl_path),
                "domain": "crowdhuman",
                "is_night": 0,
                "w": w, "h": h,
                "person_count": person_count,
                "median_area_ratio": med_area,
                "mean_luma": ml,
                "lap_var": lv,
            })
            kept += 1

    ensure_dir(args.out_root)
    meta_path = Path(args.out_root) / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("\n=== DONE ===")
    print(f"Kept images: {kept}")
    print(f"Dropped images: {dropped}")
    print(f"Meta saved: {meta_path}")
    print(f"Images out: {out_img_dir}")
    print(f"Labels out: {out_lbl_dir}")

if __name__ == "__main__":
    main()
