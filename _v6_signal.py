"""
run_v6.py
=========
基于关键发现：最近 200 天训练集显示**反转效应**（20 日动量对未来 20 日收益负相关 -0.05）
- 用"反向 20 日动量"作为最强朴素信号
- 与 v5 LightGBM 集成
- 与短期反转（5 日）结合
"""
import sys
import time
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, rankdata

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


def main(out_csv_name="submission_v6.csv"):
    t0 = time.time()
    print("=" * 70)
    print("[FCPFF v6] 反转信号 + LightGBM 集成")
    print("=" * 70)

    # === 1. 加载 ===
    print("\n[1] 加载数据 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"  train: {train.shape}, test: {test.shape}")

    # === 2. 反转信号：基于最近 200 天训练样本训练 ===
    print("\n[2] 用最近 200 天数据训练 LightGBM ...")
    train['date_idx'] = train['date'].str.replace('DAY_', '', regex=False).astype(int)
    train_recent = train[train['date_idx'] >= 2594].copy()  # 最近 200 天
    print(f"  最近 200 天: {train_recent.shape[0]:,} rows, {train_recent['code'].nunique()} stocks")

    # 计算"未来 20 日收益"作为标签
    train_recent = train_recent.sort_values(['code', 'date']).reset_index(drop=True)
    train_recent['next_20d_close'] = train_recent.groupby('code')['close'].shift(-20)
    train_recent['future_20d_return'] = train_recent['next_20d_close'] / train_recent['close'] - 1.0
    train_recent['up_label'] = (train_recent['future_20d_return'] > 0).astype(int)

    # 计算特征：多时间尺度动量
    for w in [3, 5, 10, 20, 40, 60]:
        train_recent[f'logret_{w}'] = train_recent.groupby('code')['close'].transform(
            lambda s: np.log(s / s.shift(w))
        )
    # 波动率
    train_recent['vol_20'] = train_recent.groupby('code')['close'].transform(
        lambda s: s.pct_change().rolling(20, min_periods=5).std()
    )
    # 20 日均价
    train_recent['close_ma20'] = train_recent.groupby('code')['close'].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    train_recent['close_to_ma20'] = train_recent['close'] / (train_recent['close_ma20'] + 1e-9) - 1
    # 量比
    train_recent['vol_ma20'] = train_recent.groupby('code')['volume'].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    train_recent['vol_ratio_20'] = train_recent['volume'] / (train_recent['vol_ma20'] + 1e-9)

    # 删除 NaN
    feat_cols = ['logret_3', 'logret_5', 'logret_10', 'logret_20', 'logret_40', 'logret_60',
                 'vol_20', 'close_to_ma20', 'vol_ratio_20']
    df_train = train_recent[feat_cols + ['up_label', 'future_20d_return', 'date_idx']].dropna()
    print(f"  训练样本: {len(df_train)}")
    print(f"  上涨比例: {df_train['up_label'].mean():.4f}")

    # OOF 训练
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(df_train))
    test_pred = np.zeros(len(test))
    Xtr = df_train[feat_cols].values
    ytr = df_train['up_label'].values

    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
        X_tr, y_tr = Xtr[tr_idx], ytr[tr_idx]
        X_va, y_va = Xtr[va_idx], ytr[va_idx]
        p = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.6,
            "bagging_freq": 5,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "verbose": -1,
            "n_jobs": -1,
        }
        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
        m = lgb.train(p, train_set, num_boost_round=1000,
                      valid_sets=[train_set, val_set], valid_names=["train", "valid"],
                      callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100, verbose=False)])
        oof[va_idx] = m.predict(X_va, num_iteration=m.best_iteration)

    auc = roc_auc_score(ytr, oof)
    brier = brier_score_loss(ytr, oof)
    print(f"  OOF AUC = {auc:.4f}, Brier = {brier:.4f}")

    # === 3. 用全部最近数据训练 final model ===
    print("\n[3] 训练 final model (全部最近数据) ...")
    p = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.6,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }
    # 用全量数据做 final model, num_boost 设为 OOF best_iter 平均
    final_model = lgb.train(p, lgb.Dataset(Xtr, label=ytr), num_boost_round=500)

    # === 4. 计算测试集特征 ===
    print("\n[4] 计算测试集特征 ...")
    test = test.sort_values(['code', 'date']).reset_index(drop=True)
    test_rows = []
    for code, sub in test.groupby('code'):
        f = {'code': code}
        n = len(sub)
        base = sub['close'].iloc[0]
        for w in [3, 5, 10, 20, 40, 60]:
            if n > w:
                f[f'logret_{w}'] = np.log(sub['close'].iloc[-1] / sub['close'].iloc[-w-1])
            else:
                f[f'logret_{w}'] = 0
        # 波动率
        rets = sub['close'].pct_change().dropna()
        f['vol_20'] = rets.std() if len(rets) > 1 else 0
        # close_to_ma20
        if n >= 20:
            ma20 = sub['close'].iloc[-20:].mean()
        else:
            ma20 = sub['close'].mean()
        f['close_to_ma20'] = sub['close'].iloc[-1] / (ma20 + 1e-9) - 1
        # vol_ratio_20
        vol_ma20 = sub['volume'].iloc[-20:].mean() if n >= 20 else sub['volume'].mean()
        f['vol_ratio_20'] = sub['volume'].iloc[-1] / (vol_ma20 + 1e-9)
        test_rows.append(f)
    test_df = pd.DataFrame(test_rows)
    Xte = test_df[feat_cols].values
    test_pred = final_model.predict(Xte)

    # === 5. 朴素反转信号 ===
    print("\n[5] 朴素反转信号 ...")
    # 20 日反向动量
    rev_mom_20 = -test_df['logret_20'].values
    # 5 日反向动量
    rev_mom_5 = -test_df['logret_5'].values
    # 用 rank 归一化到 [0, 1]
    def norm(x):
        ranks = rankdata(x)
        return ranks / len(ranks)
    rev_20_norm = norm(rev_mom_20)
    rev_5_norm = norm(rev_mom_5)
    # 平均反转信号
    rev_signal = (rev_20_norm + rev_5_norm) / 2

    # === 6. 集成 LGB + 反转信号 ===
    print("\n[6] 集成 ...")
    # 把 LGB 预测也归一化
    lgb_norm = norm(test_pred)
    # 加权：LGB 0.5 + 反转信号 0.5
    final_up = 0.5 * lgb_norm + 0.5 * rev_signal
    # 缩放到 [0, 1]
    final_up = norm(final_up)

    # 提交
    sub = pd.DataFrame({
        'code': test_df['code'].values,
        'up_factor': final_up
    }).sort_values('code').reset_index(drop=True)
    sub.to_csv(SUB / out_csv_name, index=False, encoding='utf-8')
    print(f"\n[Submit] -> {SUB / out_csv_name}")
    print(sub['up_factor'].describe())

    # Top-5
    top5 = sub.nlargest(5, 'up_factor')
    print(f"top-5: {top5['code'].tolist()}")

    # 报告
    report = {
        'config': {
            'lgb_oof_auc': float(auc),
            'lgb_oof_brier': float(brier),
        },
        'submission_stats': {
            'mean': float(sub['up_factor'].mean()),
            'std': float(sub['up_factor'].std()),
            'range': [float(sub['up_factor'].min()), float(sub['up_factor'].max())],
        }
    }
    with open(OUT / "v6_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
