import os, json, argparse, random, shutil
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

def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = bw / w
    bh = bh / h
    return clamp(cx), clamp(cy), clamp(bw), clamp(bh)

def parse_one_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # 你的样例里是 dict: {name, frames, attributes}
    frames = d.get("frames", [])
    if not frames:
        return None
    # BDD 这类单图 json 一般只有一个 frame
    objs = frames[0].get("objects", [])
    return objs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd_root", required=True, help="BDD root containing 100k/train 100k/val")
    ap.add_argument("--out_root", required=True, help="output root e.g. /home/didu/datasets/bdd_person_yolo")
    ap.add_argument("--keep_categories", default="person,pedestrian", help="comma-separated categories to keep")
    ap.add_argument("--min_area_ratio", type=float, default=0.001)
    ap.add_argument("--min_short_side", type=int, default=480)
    ap.add_argument("--blur_thr", type=float, default=0.0)
    ap.add_argument("--dark_thr", type=float, default=0.0)
    ap.add_argument("--night_luma_thr", type=float, default=60.0, help="mean_luma < this => is_night=1 in meta")
    ap.add_argument("--target_keep", type=int, default=20000, help="random downsample to this many images (0=keep all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keep_cats = set([c.strip() for c in args.keep_categories.split(",") if c.strip()])
    bdd_root = Path(args.bdd_root)

    # 我们把 train+val 都当素材池（不碰 test）
    src_dirs = [
        ("train", bdd_root / "100k" / "train"),
        ("val",   bdd_root / "100k" / "val"),
    ]

    candidates = []
    for tag, d in src_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing directory: {d}")
        # 找所有 json
        for jp in d.glob("*.json"):
            imgp = d / (jp.stem + ".jpg")
            if imgp.exists():
                candidates.append((tag, jp, imgp))

    print(f"Found (json+jpg) pairs: {len(candidates)}")

    out_img_dir = Path(args.out_root) / "images" / "train"
    out_lbl_dir = Path(args.out_root) / "labels" / "train"
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    rows = []
    kept = 0
    dropped = 0

    # 先筛出“含 person bbox 的样本”，再随机下采样到 target_keep
    valid_samples = []
    for tag, jp, imgp in tqdm(candidates, desc="Scanning BDD for person"):
        objs = parse_one_json(jp)
        if objs is None:
            continue

        # 先快速判断是否有目标类别 + box2d
        has = False
        for o in objs:
            cat = o.get("category", "")
            if cat in keep_cats and "box2d" in o:
                has = True
                break
        if has:
            valid_samples.append((tag, jp, imgp))

    print(f"Has-person candidates: {len(valid_samples)}")

    random.seed(args.seed)
    if args.target_keep and args.target_keep > 0 and len(valid_samples) > args.target_keep:
        random.shuffle(valid_samples)
        valid_samples = valid_samples[:args.target_keep]
        print(f"Downsampled to target_keep={args.target_keep}")

    for tag, jp, imgp in tqdm(valid_samples, desc="Converting to YOLO"):
        img = cv2.imread(str(imgp))
        if img is None:
            dropped += 1
            continue
        h, w = img.shape[:2]
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

        objs = parse_one_json(jp)
        if not objs:
            dropped += 1
            continue

        img_area = float(w * h)
        yolo_lines = []
        person_count = 0
        occ_count = 0
        trunc_count = 0

        for o in objs:
            cat = o.get("category", "")
            if cat not in keep_cats:
                continue
            b = o.get("box2d", None)
            if b is None:
                continue
            x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 1 or bh <= 1:
                continue
            area_ratio = (bw * bh) / img_area
            if area_ratio < args.min_area_ratio:
                continue

            cx, cy, bw_n, bh_n = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")
            person_count += 1

            attrs = o.get("attributes", {}) or {}
            if attrs.get("occluded", False):
                occ_count += 1
            if attrs.get("truncated", False):
                trunc_count += 1

        if person_count == 0:
            dropped += 1
            continue

        out_base = f"bdd_{tag}_{jp.stem}"
        out_img_path = out_img_dir / (out_base + ".jpg")
        out_lbl_path = out_lbl_dir / (out_base + ".txt")

        # 直接 copy 原图（更快、更省）
        shutil.copy2(str(imgp), str(out_img_path))
        with open(out_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        rows.append({
            "image": str(out_img_path),
            "label": str(out_lbl_path),
            "domain": "bdd",
            "is_night": 1 if ml < args.night_luma_thr else 0,
            "w": w, "h": h,
            "person_count": person_count,
            "occluded_count": occ_count,
            "truncated_count": trunc_count,
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
