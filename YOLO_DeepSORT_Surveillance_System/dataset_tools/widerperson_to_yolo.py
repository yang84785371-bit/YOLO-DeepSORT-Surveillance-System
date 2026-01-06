import os, argparse, shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import pandas as pd

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def clamp(x,a=0.0,b=1.0): return max(a, min(b, x))

def mean_luma(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean())

def laplacian_var(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def xywh_to_yolo(x, y, w, h, W, H):
    cx = (x + w/2.0) / W
    cy = (y + h/2.0) / H
    bw = w / W
    bh = h / H
    return clamp(cx), clamp(cy), clamp(bw), clamp(bh)


def parse_wider_txt(p: Path):
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []

    try:
        n = int(lines[0].strip())
        rows = lines[1:1+n]
    except:
        rows = lines[1:]

    anns = []
    for ln in rows:
        parts = ln.strip().split()
        if len(parts) < 5:
            continue

        cls = int(parts[0])
        x1 = float(parts[1])
        y1 = float(parts[2])
        x2 = float(parts[3])
        y2 = float(parts[4])

        anns.append((cls, x1, y1, x2, y2))

    return anns



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wp_root", required=True, help="WiderPerson root containing Images/ and Annotations/")
    ap.add_argument("--out_root", required=True, help="output root e.g. /home/didu/datasets/widerperson_person_yolo")
    ap.add_argument("--keep_cls", default="1,2", help="keep these WiderPerson classes as person (comma-separated)")
    ap.add_argument("--drop_cls", default="3,4", help="drop these classes (comma-separated)")
    ap.add_argument("--min_area_ratio", type=float, default=0.001)
    ap.add_argument("--min_short_side", type=int, default=480)
    ap.add_argument("--blur_thr", type=float, default=0.0)
    ap.add_argument("--dark_thr", type=float, default=0.0)
    args = ap.parse_args()

    keep_cls = set(int(x) for x in args.keep_cls.split(",") if x.strip())
    drop_cls = set(int(x) for x in args.drop_cls.split(",") if x.strip())

    root = Path(args.wp_root)
    img_dir = root / "Images"
    ann_dir = root / "Annotations"
    if not img_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError("Expect Images/ and Annotations/ under wp_root")

    out_img_dir = Path(args.out_root) / "images" / "train"
    out_lbl_dir = Path(args.out_root) / "labels" / "train"
    ensure_dir(out_img_dir); ensure_dir(out_lbl_dir)

    rows = []
    kept = 0
    dropped = 0

    ann_files = sorted(ann_dir.glob("*.jpg.txt"))
    print("Annotation files:", len(ann_files))

    for apath in tqdm(ann_files, desc="Processing WiderPerson"):
        stem = apath.name.replace(".jpg.txt", "")
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
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

        lv = laplacian_var(img)
        ml = mean_luma(img)
        if args.blur_thr > 0 and lv < args.blur_thr:
            dropped += 1
            continue
        if args.dark_thr > 0 and ml < args.dark_thr:
            dropped += 1
            continue

        anns = parse_wider_txt(apath)
        if not anns:
            dropped += 1
            continue

        img_area = float(W * H)
        yolo_lines = []
        kept_boxes = 0
        drop_boxes = 0

        for cls, x1, y1, x2, y2 in anns:
            if cls in drop_cls:
                drop_boxes += 1
                continue

            # ===== ① clip 到图像范围（防止越界）=====
            x1 = max(0.0, min(x1, W - 1))
            y1 = max(0.0, min(y1, H - 1))
            x2 = max(0.0, min(x2, W - 1))
            y2 = max(0.0, min(y2, H - 1))

            # ===== ② xyxy → xywh =====
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                drop_boxes += 1
                continue

            # ===== ③ 面积过滤 =====
            area_ratio = (w * h) / img_area
            if area_ratio < args.min_area_ratio:
                drop_boxes += 1
                continue

            # ===== ④ xywh → YOLO =====
            cx, cy, bw, bh = xywh_to_yolo(x1, y1, w, h, W, H)
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            kept_boxes += 1

        if kept_boxes == 0:
            dropped += 1
            continue

        out_base = f"wp_{stem}"
        out_img_path = out_img_dir / f"{out_base}.jpg"
        out_lbl_path = out_lbl_dir / f"{out_base}.txt"

        shutil.copy2(str(img_path), str(out_img_path))
        with open(out_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        rows.append({
            "image": str(out_img_path),
            "label": str(out_lbl_path),
            "domain": "widerperson",
            "is_night": 0,
            "w": W, "h": H,
            "person_count": kept_boxes,
            "dropped_boxes": drop_boxes,
            "mean_luma": ml,
            "lap_var": lv
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
