"""
v15_combo.py
============
v4 (LGB集成) + v14 (纯反转) 集成

思路：
- v4 真实 0.018 (已知)
- v9 反转 + LGB 真实 -0.01178
- v14 纯反转 未知
- v15 = v4 模型 + 反转信号（与 v9 类似但用 v4 的强模型）

具体：
- 用 v4 的训练流程 + 反转信号加权
- 中心化
"""
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

ROOT = Path(r'E:\智能量化投资策略建模挑战赛\code')
sys.path.insert(0, str(ROOT))

from adaptivepath.dataset import build_truncated_dataset, CONTEXT_LEN, TARGET_LEN
from adaptivepath.window_features_v2 import window_features_v2

DATA = Path(r'E:\智能量化投资策略建模挑战赛')
TRAIN_CSV = DATA / "train" / "train.csv"
TEST_CSV = DATA / "test" / "test.csv"
OUT = DATA / "output"
SUB = DATA / "submission" / "output"


def main(out_csv_name="submission_v15.csv"):
    t0 = time.time()
    print("=" * 70)
    print("[v15] v4 (LGB集成) + v14 (反转) 集成")
    print("=" * 70)

    # === 1. 加载 v4 与 v14 提交 ===
    print("\n[1] 加载 v4 和 v14 提交 ...")
    v4 = pd.read_csv(SUB / "submission_v4_old.csv")
    v14 = pd.read_csv(SUB / "submission_v14.csv")
    print(f"  v4: {v4.shape}, v14: {v14.shape}")
    # merge
    df = v4.merge(v14, on='code', suffixes=('_v4', '_v14'))
    print(f"  merged: {df.shape}")

    # === 2. Rank 归一化 ===
    print("\n[2] Rank 归一化 ...")
    v4_rank = (rankdata(df['up_factor_v4']) - 1) / (len(df) - 1)
    v14_rank = (rankdata(df['up_factor_v14']) - 1) / (len(df) - 1)

    # === 3. 多种权重组合，输出多个版本 ===
    print("\n[3] 集成（多权重） ...")

    # v15_a: 0.5 v4 + 0.5 v14 (平均)
    a = 0.5 * v4_rank + 0.5 * v14_rank

    # v15_b: 0.7 v4 + 0.3 v14 (主要 v4)
    b = 0.7 * v4_rank + 0.3 * v14_rank

    # v15_c: 0.3 v4 + 0.7 v14 (主要 v14)
    c = 0.3 * v4_rank + 0.7 * v14_rank

    # v15_d: 中心化 a 到 ±0.3
    d = 0.5 + 0.3 * (a - 0.5) * 2

    # v15_e: 中心化 b 到 ±0.15 (稳健)
    e = 0.5 + 0.15 * (b - 0.5) * 2

    versions = {
        'v15_a': a,
        'v15_b': b,
        'v15_c': c,
        'v15_d': d,
        'v15_e': e,
    }

    for name, up in versions.items():
        sub = pd.DataFrame({'code': df['code'].values, 'up_factor': up})
        sub = sub.sort_values('code').reset_index(drop=True)
        sub.to_csv(SUB / f'submission_{name}.csv', index=False, encoding='utf-8')
        print(f"  {name}: mean={up.mean():.4f}, std={up.std():.4f}, "
              f"top-5={sub.nlargest(5, 'up_factor')['code'].tolist()}")

    # === 4. 最终选择 v15_d (中心化 0.5±0.3, 平均) ===
    final = versions['v15_d']
    sub = pd.DataFrame({'code': df['code'].values, 'up_factor': final})
    sub = sub.sort_values('code').reset_index(drop=True)
    sub.to_csv(SUB / out_csv_name, index=False, encoding='utf-8')
    print(f"\n[Submit] -> {SUB / out_csv_name}")
    print(sub['up_factor'].describe())
    print(f"  top-5: {sub.nlargest(5, 'up_factor')['code'].tolist()}")

    # 报告
    report = {
        'method': 'v4 (LGB) + v14 (reversal) blend, then centralize',
        'v4_weight': 0.5,
        'v14_weight': 0.5,
        'centralization': '0.5 ± 0.3',
        'top5': sub.nlargest(5, 'up_factor')['code'].tolist(),
    }
    with open(OUT / "v15_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
