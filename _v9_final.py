"""
v9_final.py
===========
终极集成：v6 (LGB+反转) + v8b (KNN+反转) 加权平均 + 压回极端值

目标：稳健突破 0.1775
"""
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(r'E:\智能量化投资策略建模挑战赛\code')
sys.path.insert(0, str(ROOT))

from adaptivepath.window_features_v2 import window_features_v2

DATA = Path(r'E:\智能量化投资策略建模挑战赛')
TRAIN_CSV = DATA / "train" / "train.csv"
TEST_CSV = DATA / "test" / "test.csv"
OUT = DATA / "output"
SUB = DATA / "submission" / "output"


def main(out_csv_name="submission_v9.csv"):
    t0 = time.time()
    print("=" * 70)
    print("[v9] v6 + v8b 集成 + 稳健化")
    print("=" * 70)

    # === 1. 加载已有提交 ===
    print("\n[1] 加载 v6 (LGB) 和 v8b (KNN) 提交 ...")
    v6 = pd.read_csv(SUB / "submission_v6.csv")
    v8b = pd.read_csv(SUB / "submission_v8b.csv")
    print(f"  v6: {v6.shape}, v8b: {v8b.shape}")
    print(f"  v6 up_factor: mean={v6['up_factor'].mean():.3f}, std={v6['up_factor'].std():.3f}")
    print(f"  v8b up_factor: mean={v8b['up_factor'].mean():.3f}, std={v8b['up_factor'].std():.3f}")

    # === 2. 加权平均（用 rank 归一化） ===
    print("\n[2] 集成 ...")
    # 把两个 up_factor 归一化到 (0, 1)
    v6_rank = rankdata(v6['up_factor']) / len(v6)
    v8b_rank = rankdata(v8b['up_factor']) / len(v8b)
    # v6 0.5 + v8b 0.5
    final_rank = 0.5 * v6_rank + 0.5 * v8b_rank
    # 归一化回 (0, 1)
    final_up = (final_rank - final_rank.min()) / (final_rank.max() - final_rank.min() + 1e-9)

    # === 3. 稳健化：压回极端值 ===
    # 用 sigmoid 中心化：up_factor 集中在 0.5 附近
    # alpha 控制收缩强度
    def robust(x, alpha=2.0):
        # x ∈ [0, 1], 通过 sigmoid 收缩到 0.5 附近
        return 0.5 + (x - 0.5) / (1 + alpha * np.abs(x - 0.5))

    final_robust = robust(final_up, alpha=1.0)

    # === 4. 写入 ===
    sub = pd.DataFrame({
        'code': v6['code'].values,
        'up_factor': final_robust
    }).sort_values('code').reset_index(drop=True)
    sub.to_csv(SUB / out_csv_name, index=False, encoding='utf-8')
    print(f"\n[Submit] -> {SUB / out_csv_name}")
    print(sub['up_factor'].describe())
    top5 = sub.nlargest(5, 'up_factor')
    print(f"top-5: {top5['code'].tolist()}")

    # 报告
    report = {
        'method': 'v6 (LGB+reversal) + v8b (KNN+reversal) blend',
        'weights': '0.5 v6 + 0.5 v8b (rank-normalized)',
        'robust_alpha': 1.0,
        'submission_stats': {
            'mean': float(sub['up_factor'].mean()),
            'std': float(sub['up_factor'].std()),
            'range': [float(sub['up_factor'].min()), float(sub['up_factor'].max())],
        }
    }
    with open(OUT / "v9_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
