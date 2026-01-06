from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path("/mnt/d/wsl_rescue/work_dirs")  # 修改成你的目录

EXPERIMENTS = [
    "ped_y11s_val_full",
    "ped_y11s_val_full_night",
    "ped_y11s_val_pretrain",
    "ped_y11s_val_pretrain_night",
]

# 正则匹配：容忍任意空白符
PAT_MAP05 = re.compile(r"mAP@0\.5:\s*([0-9.]+)", re.IGNORECASE)
PAT_MAP5095 = re.compile(r"mAP@0\.5:0\.95:\s*([0-9.]+)", re.IGNORECASE)
PAT_PREC = re.compile(r"Precision:\s*([0-9.]+)", re.IGNORECASE)
PAT_RECALL = re.compile(r"Recall:\s*([0-9.]+)", re.IGNORECASE)

def extract_metric(text, pattern):
    m = pattern.search(text)
    return float(m.group(1)) if m else None

rows = []

for name in EXPERIMENTS:
    metrics_path = BASE_DIR / name / "metrics.txt"

    if not metrics_path.exists():
        print(f"[WARN] 跳过 {name}: 未找到文件 {metrics_path}")
        continue

    text = metrics_path.read_text(encoding="utf-8", errors="ignore")
    
    row = {
        "experiment": name,
        "mAP@0.5": extract_metric(text, PAT_MAP05),
        "mAP@0.5:0.95": extract_metric(text, PAT_MAP5095),
        "Precision": extract_metric(text, PAT_PREC),
        "Recall": extract_metric(text, PAT_RECALL),
    }
    
    rows.append(row)

df = pd.DataFrame(rows)

print("\n===== 验证结果汇总 =====")
print(df.to_string(index=False))

out_path = BASE_DIR / "val_summary.csv"
df.to_csv(out_path, index=False)
print(f"\nCSV 已保存至: {out_path}")

