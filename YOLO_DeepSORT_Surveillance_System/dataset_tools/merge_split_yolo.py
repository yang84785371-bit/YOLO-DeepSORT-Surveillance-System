import os, csv, random, argparse, shutil
from pathlib import Path
import pandas as pd

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, mode: str):
    ensure(dst.parent)
    if dst.exists():
        return
    if mode == "link":
        os.link(src, dst)  # hardlink, same filesystem only
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)

def make_group_key(row, group_nightowls=True):
    # 防止 nightowls 近邻帧泄漏：按文件名前缀分组（同一段进同一split）
    img = Path(row["image"])
    domain = str(row.get("domain", ""))
    stem = img.stem
    if group_nightowls and "nightowls" in domain:
        # 58c58133bc260137e096a56a -> 取前 10~12 位做组（可调整）
        return stem[:12]
    return stem  # 其他数据集用自身唯一名即可

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["copy", "link", "symlink"], default="link",
                    help="link=硬链接(省空间，要求同一磁盘分区); copy=复制; symlink=软链接")
    ap.add_argument("--val_main", type=int, default=2500)
    ap.add_argument("--val_night", type=int, default=1200)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="each root contains meta.csv + images/train + labels/train")
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_root)

    # 1) 读入所有 meta
    dfs = []
    for r in args.roots:
        r = Path(r)
        meta = r / "meta.csv"
        if not meta.exists():
            raise FileNotFoundError(f"missing meta.csv: {meta}")
        df = pd.read_csv(meta)
        # 保险：确保字段存在
        for c in ["image", "label", "domain", "is_night"]:
            if c not in df.columns:
                raise RuntimeError(f"{meta} missing col {c}")
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    # 路径转 Path，过滤掉不存在文件
    ok = []
    for _, row in all_df.iterrows():
        ip = Path(str(row["image"]))
        lp = Path(str(row["label"]))
        if ip.exists() and lp.exists():
            ok.append(row)
    all_df = pd.DataFrame(ok).reset_index(drop=True)

    # 2) 去重（同名/同路径重复会导致泄漏）
    all_df["img_abs"] = all_df["image"].astype(str)
    all_df = all_df.drop_duplicates(subset=["img_abs"]).reset_index(drop=True)

    # 3) 分 night / non-night
    night_df = all_df[all_df["is_night"].astype(int) == 1].copy()
    day_df   = all_df[all_df["is_night"].astype(int) == 0].copy()

    if len(night_df) < args.val_night:
        raise RuntimeError(f"not enough night samples: {len(night_df)} < val_night {args.val_night}")
    if len(day_df) < args.val_main:
        # 不够就从 all_df（非 night 优先）补
        print(f"[WARN] day samples {len(day_df)} < val_main {args.val_main}, will sample from all_df as fallback")

    # 4) 按“组”抽样（nightowls 防近邻泄漏）
    def sample_by_groups(df, target_n, must_nightowls_group=True):
        df = df.copy()
        df["g"] = df.apply(lambda r: make_group_key(r, group_nightowls=must_nightowls_group), axis=1)
        groups = list(df.groupby("g"))
        random.shuffle(groups)

        picked = []
        cnt = 0
        for gname, gdf in groups:
            if cnt >= target_n:
                break
            picked.append(gdf)
            cnt += len(gdf)
        picked_df = pd.concat(picked, ignore_index=True)
        # 如果超了，最后再随机裁到 target_n（仍然尽量保留组完整，裁剪只发生在末尾）
        if len(picked_df) > target_n:
            picked_df = picked_df.sample(n=target_n, random_state=args.seed).reset_index(drop=True)
        return picked_df

    val_night_df = sample_by_groups(night_df, args.val_night, must_nightowls_group=True)

    # val_main：优先 day_df，不够则用 all_df 里剩余补
    day_pool = day_df[~day_df["img_abs"].isin(val_night_df["img_abs"])].copy()
    if len(day_pool) >= args.val_main:
        val_main_df = sample_by_groups(day_pool, args.val_main, must_nightowls_group=True)
    else:
        # fallback：从 all_df 剩余里补
        remain = all_df[~all_df["img_abs"].isin(val_night_df["img_abs"])].copy()
        val_main_df = sample_by_groups(remain, args.val_main, must_nightowls_group=True)

    # train = 全部 - 两个 val
    used = set(val_night_df["img_abs"]).union(set(val_main_df["img_abs"]))
    train_df = all_df[~all_df["img_abs"].isin(used)].copy()

    # 5) 输出文件（images/ labels/）
    splits = {
        "train": train_df,
        "val_main": val_main_df,
        "val_night": val_night_df,
    }

    for split, df in splits.items():
        img_out = out_root / "images" / split
        lab_out = out_root / "labels" / split
        ensure(img_out); ensure(lab_out)

        for _, row in df.iterrows():
            src_img = Path(str(row["image"]))
            src_lab = Path(str(row["label"]))
            # 保留原扩展名（png/jpg/JPEG 都行）
            dst_img = img_out / src_img.name
            dst_lab = lab_out / src_lab.name
            link_or_copy(src_img, dst_img, args.mode)
            link_or_copy(src_lab, dst_lab, args.mode)

        df.drop(columns=["img_abs"], errors="ignore").to_csv(out_root / f"meta_{split}.csv",
                                                           index=False, encoding="utf-8-sig")

    # 6) 生成 ultralytics yaml
    yaml_path = out_root / "ped_mix.yaml"
    yaml_path.write_text(
        f"""path: {out_root}
train: images/train
val: images/val_main
# 你训练时可以手动额外跑一次：yolo val data=ped_mix.yaml split=val_night（或单独写一个yaml）
names:
  0: person
""", encoding="utf-8"
    )

    print("\n=== SPLIT DONE ===")
    for k, df in splits.items():
        print(k, "images:", len(df), "night_ratio:", round(float(df["is_night"].mean()), 3))
    print("out:", out_root)
    print("yaml:", yaml_path)

if __name__ == "__main__":
    main()
