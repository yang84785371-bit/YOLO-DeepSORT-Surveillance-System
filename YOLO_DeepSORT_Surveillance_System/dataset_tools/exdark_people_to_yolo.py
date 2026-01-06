import os, argparse, shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import pandas as pd

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def xywh_to_yolo(x, y, w, h, W, H):
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    bw = w / W
    bh = h / H
    return clamp(cx), clamp(cy), clamp(bw), clamp(bh)

def laplacian_var(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def mean_luma(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean())

def parse_bbgt_txt(label_path: Path, keep_name="People"):
    """
    bbGt v3 format:
      % bbGt version=3
      People x y w h <ignored...>
    """
    lines = label_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    boxes = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("%"):
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        name = parts[0]
        if name != keep_name:
            continue
        x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
        boxes.append((x, y, w, h))
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exdark_root", required=True, help="ExDark root containing images/ and Annnotations/")
    ap.add_argument("--out_root", required=True, help="Output root e.g. /home/didu/datasets/exdark_person_yolo")
    ap.add_argument("--min_area_ratio", type=float, default=0.001, help="min bbox area ratio in image to keep")
    ap.add_argument("--min_short_side", type=int, default=480, help="drop images with min(W,H) < this")
    ap.add_argument("--night_luma_thr", type=float, default=80.0, help="mean_luma < thr => is_night=1")
    ap.add_argument("--blur_thr", type=float, default=0.0, help="if >0, drop images with lap_var < thr")
    ap.add_argument("--max_keep", type=int, default=0, help="if >0, cap kept images (random not applied, just first N)")
    args = ap.parse_args()

    root = Path(args.exdark_root)
    img_people = root / "images" / "People"
    lbl_people = root / "Annnotations" / "People"  # 注意 ExDark 的拼写就是 Annnotations

    if not img_people.exists():
        raise FileNotFoundError(f"Missing: {img_people}")
    if not lbl_people.exists():
        raise FileNotFoundError(f"Missing: {lbl_people}")

    out_img_dir = Path(args.out_root) / "images" / "train"
    out_lbl_dir = Path(args.out_root) / "labels" / "train"
    ensure_dir(out_img_dir); ensure_dir(out_lbl_dir)

    rows = []
    kept = 0
    dropped = 0

    # People 图片可能是 jpg/png/jpeg/JPEG 混合
    exts = ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]
    img_list = []
    for e in exts:
        img_list += list(img_people.glob(e))
    img_list = sorted(img_list)

    for img_path in tqdm(img_list, desc="ExDark People"):
        # label file name: <image_name>.<ext>.txt 例如 2015_06246.jpg.txt
        label_path = lbl_people / f"{img_path.name}.txt"
        if not label_path.exists() or label_path.stat().st_size == 0:
            dropped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            dropped += 1
            continue
        H, W = img.shape[:2]
        if min(W, H) < args.min_short_side:
            dropped += 1
            continue

        if args.blur_thr > 0:
            lv = laplacian_var(img)
            if lv < args.blur_thr:
                dropped += 1
                continue
        else:
            lv = laplacian_var(img)

        ml = mean_luma(img)
        is_night = 1 if ml < args.night_luma_thr else 0

        boxes = parse_bbgt_txt(label_path, keep_name="People")
        if not boxes:
            dropped += 1
            continue

        img_area = float(W * H)
        yolo_lines = []
        for (x, y, w, h) in boxes:
            # clip to image bounds
            x = max(0.0, min(x, W - 1))
            y = max(0.0, min(y, H - 1))
            w = max(0.0, min(w, W - x))
            h = max(0.0, min(h, H - y))
            if w <= 1 or h <= 1:
                continue
            area_ratio = (w * h) / img_area
            if area_ratio < args.min_area_ratio:
                continue
            cx, cy, bw, bh = xywh_to_yolo(x, y, w, h, W, H)
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            dropped += 1
            continue

        out_base = f"exdark_{img_path.stem}"  # stem 不含扩展名
        out_img_path = out_img_dir / f"{out_base}{img_path.suffix}"
        out_lbl_path = out_lbl_dir / f"{out_base}.txt"

        shutil.copy2(str(img_path), str(out_img_path))
        out_lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        rows.append({
            "image": str(out_img_path),
            "label": str(out_lbl_path),
            "domain": "exdark",
            "is_night": is_night,
            "w": W, "h": H,
            "person_count": len(yolo_lines),
            "mean_luma": ml,
            "lap_var": lv,
        })
        kept += 1
        if args.max_keep > 0 and kept >= args.max_keep:
            break

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

