"""
v12_safe.py
===========
最稳版：5 seed 集成 + 反转信号弱权重 + 强中心化

教训：
- v4 真实 0.018 (OOF 0.607 → 过拟合)
- v9 真实 -0.01178 (反转信号方向错误)
- v9 用了"全期 v4 模型" + "最近 200 天反转信号"，方向冲突

v12 策略：
- 用全期 LightGBM (v4 风格) + 5 seed 集成
- 反转信号**只**作为弱权重（5%）
- 强中心化到 0.5 ± 0.05 防止极端预测
- 用 XGBoost + LightGBM + 简单 ensemble
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
import xgboost as xgb

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


def main(out_csv_name="submission_v12.csv"):
    t0 = time.time()
    print("=" * 70)
    print("[v12] 5 seed 集成 + 反转弱信号 + 强中心化")
    print("=" * 70)

    # === 1. 加载 ===
    print("\n[1] 加载 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"  train: {train.shape}, test: {test.shape}")

    # === 2. 选股 + 截断样本 ===
    print("\n[2] 选 2000 只股票 + 构造样本 ...")
    train_sub = train[train['code'].isin(
        train.groupby('code').size().nlargest(2000).index
    )].copy()
    samples = build_truncated_dataset(train_sub, n_per_stock=4, min_history=60, seed=42)
    print(f"  samples: {len(samples)}")

    # === 3. 特征抽取 ===
    print("\n[3] 特征抽取 ...")
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
    Xtr_s = np.nan_to_num(Xtr, nan=0.0)

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
    Xte_s = np.nan_to_num(Xte, nan=0.0)
    print(f"  Xtr: {Xtr_s.shape}, Xte: {Xte_s.shape}")

    # === 4. 5 seed LightGBM 集成 ===
    print("\n[4] 5 seed LightGBM OOF + test ...")
    seeds = [42, 123, 2024, 7, 9999]
    oof_lgb = np.zeros(len(Xtr))
    test_lgb = np.zeros(len(Xte))
    for s in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=s)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr_s)):
            p = {
                "objective": "binary", "metric": "binary_logloss",
                "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200,
                "feature_fraction": 0.6, "bagging_fraction": 0.6, "bagging_freq": 5,
                "lambda_l1": 1.0, "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1,
                "seed": s, "feature_fraction_seed": s, "bagging_seed": s,
            }
            m = lgb.train(p, lgb.Dataset(Xtr_s[tr_idx], label=ytr[tr_idx]),
                          num_boost_round=1500,
                          valid_sets=[lgb.Dataset(Xtr_s[va_idx], label=ytr[va_idx])],
                          callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100, verbose=False)])
            oof_lgb[va_idx] += m.predict(Xtr_s[va_idx], num_iteration=m.best_iteration) / len(seeds)
            test_lgb += m.predict(Xte_s, num_iteration=m.best_iteration) / 5 / len(seeds)
    auc = roc_auc_score(ytr, oof_lgb)
    brier = brier_score_loss(ytr, oof_lgb)
    print(f"  LGB ensemble: AUC={auc:.4f}, Brier={brier:.4f}")

    # === 5. 反转信号（弱权重） ===
    print("\n[5] 反转信号（弱权重） ...")
    rev_5 = rankdata(-test_df['m_logret_5'].fillna(0).values) / len(test_df)
    rev_20 = rankdata(-test_df['m_logret_20'].fillna(0).values) / len(test_df)
    rev_signal = (rev_5 + rev_20) / 2

    # === 6. 集成 ===
    print("\n[6] 集成 (LGB 0.9 + 反转 0.1) ...")
    lgb_rank = rankdata(test_lgb) / len(test_lgb)
    # 弱权重反转（避免方向错误）
    test_combined = 0.9 * lgb_rank + 0.1 * rev_signal

    # === 7. 强中心化到 0.5 ± 0.05 ===
    print("\n[7] 强中心化 ...")
    test_centralized = 0.5 + 0.05 * (test_combined - 0.5) * 2  # ±0.05

    # === 8. 提交 ===
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
        'method': '5-seed LGB ensemble + weak reversal + strong centralization',
        'lgb_oof_auc': float(auc),
        'lgb_oof_brier': float(brier),
        'centralization': '0.5 ± 0.05',
        'submission_stats': {
            'mean': float(sub['up_factor'].mean()),
            'std': float(sub['up_factor'].std()),
            'range': [float(sub['up_factor'].min()), float(sub['up_factor'].max())],
        }
    }
    with open(OUT / "v12_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
