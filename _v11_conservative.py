"""
v11_conservative.py
===================
极保守方案：用全期训练 + 弱信号 + 极保守 up_factor（中心化在 0.5）

教训：
- v4 (OOF 0.607) 真实 0.018 — OOF 过拟合但至少正分
- v9 (反转信号) 真实 -0.01178 — 反方向了
- 关键：A 股 20 日动量对 future 20d_return 是弱反转 (-0.05)

策略：
- 用 11 年训练 LightGBM (v4 类似的特征)
- 把 up_factor 强烈中心化在 0.5
- 加一点点反转信号但权重小
- 整体保持"中性预测"避免方向错误
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb

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


def main(out_csv_name="submission_v11.csv"):
    t0 = time.time()
    print("=" * 70)
    print("[v11] 极保守方案：全期 LGB + 弱反转 + 中心化")
    print("=" * 70)

    # === 1. 加载 ===
    print("\n[1] 加载 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"  train: {train.shape}, test: {test.shape}")

    # === 2. 构造截断样本 ===
    print("\n[2] 构造截断样本 ...")
    train_sub = train[train['code'].isin(
        train.groupby('code').size().nlargest(2000).index
    )].copy()
    samples = build_truncated_dataset(train_sub, n_per_stock=4, min_history=60, seed=42)
    print(f"  训练样本: {len(samples)}")

    # === 3. 特征抽取 ===
    print("\n[3] 抽取窗口特征 ...")
    code_to_sub = {c: g for c, g in train_sub.groupby("code", sort=False)}
    train_rows = []
    for i, row in samples.iterrows():
        sub = code_to_sub.get(row["code"])
        if sub is None:
            continue
        ctx = sub[(sub["date"] >= row["ctx_start"]) & (sub["date"] <= row["ctx_end_date"])]
        if len(ctx) < CONTEXT_LEN:
            continue
        f = window_features_v2(ctx)
        f["up_label"] = int(row["up_label"])
        f["future_20d_return"] = float(row["future_20d_return"])
        train_rows.append(f)
    Xtr_df = pd.DataFrame(train_rows)
    feat_cols = [c for c in Xtr_df.columns if c not in {"up_label", "future_20d_return"}]
    Xtr = Xtr_df[feat_cols].values.astype(np.float32)
    ytr = Xtr_df["up_label"].values.astype(np.int32)
    print(f"  特征: {len(feat_cols)}, 样本: {len(Xtr)}")

    # 测试集特征
    test_rows = []
    for code, sub in test.groupby("code"):
        f = window_features_v2(sub)
        f["code"] = code
        test_rows.append(f)
    test_df = pd.DataFrame(test_rows)
    for c in feat_cols:
        if c not in test_df.columns:
            test_df[c] = 0.0
    Xte = test_df[feat_cols].values.astype(np.float32)
    print(f"  test: {Xte.shape}")

    # === 4. 5 折 OOF LightGBM ===
    print("\n[4] OOF 训练 LightGBM (5 折) ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(Xtr))
    test_pred = np.zeros(len(Xte))
    Xtr_s = np.nan_to_num(Xtr, nan=0.0)
    Xte_s = np.nan_to_num(Xte, nan=0.0)
    p = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200,
        "feature_fraction": 0.6, "bagging_fraction": 0.6, "bagging_freq": 5,
        "lambda_l1": 1.0, "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1,
    }
    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr_s)):
        m = lgb.train(p, lgb.Dataset(Xtr_s[tr_idx], label=ytr[tr_idx]),
                      num_boost_round=1500,
                      valid_sets=[lgb.Dataset(Xtr_s[va_idx], label=ytr[va_idx])],
                      callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100, verbose=False)])
        oof[va_idx] = m.predict(Xtr_s[va_idx], num_iteration=m.best_iteration)
        test_pred += m.predict(Xte_s, num_iteration=m.best_iteration) / 5
    auc = roc_auc_score(ytr, oof)
    brier = brier_score_loss(ytr, oof)
    print(f"  OOF AUC={auc:.4f}, Brier={brier:.4f}")

    # === 5. 极保守策略：把 up_factor 强烈中心化在 0.5 ===
    print("\n[5] 极保守中心化 ...")
    # rank 归一化
    test_pred_rank = rankdata(test_pred) / len(test_pred)
    # 中心化：up_factor = 0.5 + 0.1 * (rank - 0.5)  — 只允许 ±5% 的偏离
    test_centralized = 0.5 + 0.1 * (test_pred_rank - 0.5)

    # === 6. 提交 ===
    sub = pd.DataFrame({
        'code': test_df['code'].values,
        'up_factor': test_centralized
    }).sort_values('code').reset_index(drop=True)
    sub.to_csv(SUB / out_csv_name, index=False, encoding='utf-8')
    print(f"\n[Submit] -> {SUB / out_csv_name}")
    print(sub['up_factor'].describe())
    top5 = sub.nlargest(5, 'up_factor')
    print(f"top-5: {top5['code'].tolist()}")

    # 报告
    report = {
        'method': 'LGB full-period + centralized to 0.5 ± 0.05',
        'oof_auc': float(auc),
        'oof_brier': float(brier),
        'centralization': '0.5 + 0.1*(rank-0.5)',
        'submission_stats': {
            'mean': float(sub['up_factor'].mean()),
            'std': float(sub['up_factor'].std()),
            'range': [float(sub['up_factor'].min()), float(sub['up_factor'].max())],
        }
    }
    with open(OUT / "v11_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
