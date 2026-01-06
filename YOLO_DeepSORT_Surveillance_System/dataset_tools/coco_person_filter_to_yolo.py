import os, json, shutil, argparse, math
from collections import defaultdict
from tqdm import tqdm
import cv2
import pandas as pd

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def laplacian_var(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def mean_luma(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean())

def coco_bbox_to_yolo(bbox_xywh, w, h):
    x, y, bw, bh = bbox_xywh
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    bw = bw / w
    bh = bh / h
    return clamp(cx), clamp(cy), clamp(bw), clamp(bh)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True, help="COCO root containing train2017/ val2017/ annotations_trainval2017/")
    ap.add_argument("--out_root", required=True, help="output root, e.g. coco_person_yolo")
    ap.add_argument("--min_area_ratio", type=float, default=0.001, help="min bbox area ratio to keep")
    ap.add_argument("--crowd_count_thr", type=int, default=30, help="if person_count > thr and median area too small -> drop image")
    ap.add_argument("--crowd_median_area_ratio", type=float, default=0.002, help="median area ratio threshold for crowd drop")
    ap.add_argument("--min_short_side", type=int, default=480, help="drop images with min(w,h) < this")
    ap.add_argument("--blur_thr", type=float, default=0.0, help="if >0, drop images with lap_var < thr")
    ap.add_argument("--dark_thr", type=float, default=0.0, help="if >0, drop images with mean_luma < thr")
    args = ap.parse_args()

    coco_root = args.coco_root
    ann_dir = os.path.join(coco_root, "annotations")
    train_img_dir = os.path.join(coco_root, "train2017")
    val_img_dir = os.path.join(coco_root, "val2017")

    ann_files = [
        ("train2017", os.path.join(ann_dir, "instances_train2017.json"), train_img_dir),
        ("val2017",   os.path.join(ann_dir, "instances_val2017.json"),   val_img_dir),
    ]

    out_img_dir = os.path.join(args.out_root, "images", "train")
    out_lbl_dir = os.path.join(args.out_root, "labels", "train")
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    rows = []
    kept = 0
    dropped = 0

    for split_name, ann_path, img_dir in ann_files:
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # 找 person 的 category_id（COCO 一般是 1，但我们不硬编码）
        person_cat_ids = [c["id"] for c in coco["categories"] if c["name"] == "person"]
        if not person_cat_ids:
            raise RuntimeError(f"No 'person' category found in {ann_path}")
        person_cat_id = person_cat_ids[0]

        # image_id -> image info
        imgs = {im["id"]: im for im in coco["images"]}

        # image_id -> list of person anns
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            if a.get("iscrowd", 0) == 1:
                # iscrowd 的 bbox 往往不稳定，这里直接跳过（更干净）
                continue
            if a["category_id"] != person_cat_id:
                continue
            if "bbox" not in a:
                continue
            anns_by_img[a["image_id"]].append(a)

        for img_id, im in tqdm(imgs.items(), desc=f"Processing {split_name}"):
            if img_id not in anns_by_img:
                continue  # 没有人，直接不要

            w, h = int(im["width"]), int(im["height"])
            if min(w, h) < args.min_short_side:
                dropped += 1
                continue

            file_name = im["file_name"]
            src_path = os.path.join(img_dir, file_name)
            if not os.path.exists(src_path):
                # COCO 有时路径不一致，直接跳过
                dropped += 1
                continue

            # 读图做质量过滤（可选）
            img = cv2.imread(src_path)
            if img is None:
                dropped += 1
                continue

            if args.blur_thr > 0:
                lv = laplacian_var(img)
                if lv < args.blur_thr:
                    dropped += 1
                    continue
            else:
                lv = laplacian_var(img)

            if args.dark_thr > 0:
                ml = mean_luma(img)
                if ml < args.dark_thr:
                    dropped += 1
                    continue
            else:
                ml = mean_luma(img)

            # bbox 过滤：太小的删
            valid = []
            areas = []
            img_area = float(w * h)
            for a in anns_by_img[img_id]:
                x, y, bw, bh = a["bbox"]
                if bw <= 1 or bh <= 1:
                    continue
                area_ratio = (bw * bh) / img_area
                if area_ratio < args.min_area_ratio:
                    continue
                valid.append(a["bbox"])
                areas.append(area_ratio)

            if not valid:
                dropped += 1
                continue

            # 极端 crowd 过滤
            person_count = len(valid)
            med_area = float(pd.Series(areas).median())
            if person_count > args.crowd_count_thr and med_area < args.crowd_median_area_ratio:
                dropped += 1
                continue

            # 输出：复制图片 + 写 yolo label
            # 为避免 train/val 重名，前面加 split 前缀
            out_base = f"{split_name}_{os.path.splitext(file_name)[0]}"
            out_img_path = os.path.join(out_img_dir, out_base + ".jpg")
            out_lbl_path = os.path.join(out_lbl_dir, out_base + ".txt")

            # 保存 jpg（统一格式，后面更省事）
            cv2.imwrite(out_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            with open(out_lbl_path, "w", encoding="utf-8") as f:
                for bbox in valid:
                    cx, cy, bw, bh = coco_bbox_to_yolo(bbox, w, h)
                    # class_id 0 代表 person
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            rows.append({
                "image": out_img_path,
                "label": out_lbl_path,
                "domain": "coco",
                "is_night": 0,
                "w": w, "h": h,
                "person_count": person_count,
                "median_area_ratio": med_area,
                "mean_luma": ml,
                "lap_var": lv
            })
            kept += 1

    ensure_dir(args.out_root)
    meta_path = os.path.join(args.out_root, "meta.csv")
    pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("\n=== DONE ===")
    print(f"Kept images: {kept}")
    print(f"Dropped images: {dropped}")
    print(f"Meta saved: {meta_path}")
    print(f"Images out: {out_img_dir}")
    print(f"Labels out: {out_lbl_dir}")

if __name__ == "__main__":
    main()
