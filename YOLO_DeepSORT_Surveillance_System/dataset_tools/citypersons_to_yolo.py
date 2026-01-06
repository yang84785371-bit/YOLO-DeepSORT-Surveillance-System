import os, argparse, shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_citypersons_label(label_path: Path, keep_class_ids=set([0]), drop_track_id=True):
    """
    CityPersons labels_with_ids format (per line):
      class_id  track_id  cx  cy  w  h
    We output YOLO format:
      class_id  cx  cy  w  h
    """
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    yolo_lines = []
    dropped = 0
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) < 6:
            continue
        cls = int(float(parts[0]))
        if cls not in keep_class_ids:
            dropped += 1
            continue
        # parts[1] is track_id, ignore
        cx = float(parts[2])
        cy = float(parts[3])
        bw = float(parts[4])
        bh = float(parts[5])

        # basic sanity clamp (CityPersons already normalized)
        if bw <= 0 or bh <= 0:
            dropped += 1
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
            dropped += 1
            continue

        yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return yolo_lines, dropped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp_root", required=True, help="Root containing Citypersons/Citypersons/images and labels_with_ids")
    ap.add_argument("--out_root", required=True, help="Output root e.g. /home/didu/datasets/citypersons_person_yolo")
    ap.add_argument("--keep_classes", default="0", help="Keep these class ids as person (comma-separated), default=0")
    ap.add_argument("--min_area_ratio", type=float, default=0.0, help="Optional: drop boxes with (w*h) < this in normalized space")
    args = ap.parse_args()

    keep_class_ids = set(int(x) for x in args.keep_classes.split(",") if x.strip())

    cp_root = Path(args.cp_root)
    base = cp_root / "Citypersons"
    img_base = base / "images"
    lbl_base = base / "labels_with_ids"

    if not img_base.exists() or not lbl_base.exists():
        raise FileNotFoundError("Expect Citypersons/images and Citypersons/labels_with_ids under cp_root/Citypersons")

    out_img_dir = Path(args.out_root) / "images" / "train"
    out_lbl_dir = Path(args.out_root) / "labels" / "train"
    ensure_dir(out_img_dir); ensure_dir(out_lbl_dir)

    rows = []
    kept_images = 0
    dropped_images = 0

    # We take train + val (test has no public labels typically)
    splits = ["train", "val"]

    for split in splits:
        split_img_dir = img_base / split
        split_lbl_dir = lbl_base / split
        if not split_img_dir.exists() or not split_lbl_dir.exists():
            print(f"[WARN] Missing split: {split}")
            continue

        # traverse city folders
        city_dirs = [p for p in split_img_dir.iterdir() if p.is_dir()]
        for city_dir in tqdm(city_dirs, desc=f"CityPersons {split}"):
            city = city_dir.name
            imgs = sorted(city_dir.glob("*.png")) + sorted(city_dir.glob("*.jpg"))
            for img_path in imgs:
                stem = img_path.stem
                label_path = split_lbl_dir / city / f"{stem}.txt"
                if not label_path.exists():
                    dropped_images += 1
                    continue
                if label_path.stat().st_size == 0:
                    dropped_images += 1
                    continue

                yolo_lines, dropped_boxes = parse_citypersons_label(label_path, keep_class_ids=keep_class_ids)

                # optional min box area filter in normalized space
                if args.min_area_ratio > 0 and yolo_lines:
                    filtered = []
                    for ln in yolo_lines:
                        _, cx, cy, bw, bh = ln.split()
                        bw = float(bw); bh = float(bh)
                        if (bw * bh) >= args.min_area_ratio:
                            filtered.append(ln)
                    yolo_lines = filtered

                if not yolo_lines:
                    dropped_images += 1
                    continue

                out_base = f"cp_{split}_{city}_{stem}"
                out_img_path = out_img_dir / f"{out_base}{img_path.suffix}"
                out_lbl_path = out_lbl_dir / f"{out_base}.txt"

                shutil.copy2(str(img_path), str(out_img_path))
                out_lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

                rows.append({
                    "image": str(out_img_path),
                    "label": str(out_lbl_path),
                    "domain": "citypersons",
                    "is_night": 0,
                    "split_src": split,
                    "city": city,
                    "person_count": len(yolo_lines),
                    "dropped_boxes": dropped_boxes
                })
                kept_images += 1

    ensure_dir(args.out_root)
    meta_path = Path(args.out_root) / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("\n=== DONE ===")
    print(f"Kept images: {kept_images}")
    print(f"Dropped images: {dropped_images}")
    print(f"Meta saved: {meta_path}")
    print(f"Images out: {out_img_dir}")
    print(f"Labels out: {out_lbl_dir}")

if __name__ == "__main__":
    main()
