"""
run_v7.py
=========
终极版：walk-forward 验证 + 多重反转信号 + 更稳健的集成

核心洞察：
- 训练集最后 200 天（DAY_2594-2794）显示**反转效应**（20 日动量对未来 20 日收益 -0.05 相关）
- 但这个相关性在更早时期不稳定
- 用"训练集全部"会引入过时的"动量"信号

策略：
1. 训练样本只用"最近 500 天"（DAY_2294-2794）—— 平衡样本量与时效性
2. 多重反转信号：5/10/20 日动量反向、波动率倒数
3. 集成：朴素反转信号 + LightGBM（用 walk-forward 验证）
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


def norm_rank(x):
    """用 rank 归一化到 (0, 1)"""
    ranks = rankdata(x)
    return ranks / len(ranks)


def main(out_csv_name="submission_v7.csv"):
    t0 = time.time()
    print("=" * 70)
    print("[FCPFF v7] Walk-Forward + 多重反转 + 集成")
    print("=" * 70)

    # === 1. 加载 ===
    print("\n[1] 加载 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train['date_idx'] = train['date'].str.replace('DAY_', '', regex=False).astype(int)
    print(f"  train: {train.shape}, test: {test.shape}")

    # === 2. 用最近 500 天数据训练 ===
    print("\n[2] 选最近 500 天 (DAY_2294-2794) ...")
    train_recent = train[train['date_idx'] >= 2294].copy()
    train_recent = train_recent.sort_values(['code', 'date']).reset_index(drop=True)
    print(f"  rows: {len(train_recent):,}, codes: {train_recent['code'].nunique()}")

    # 未来 20 日收益
    train_recent['next_20d_close'] = train_recent.groupby('code')['close'].shift(-20)
    train_recent['future_20d_return'] = train_recent['next_20d_close'] / train_recent['close'] - 1.0
    train_recent['up_label'] = (train_recent['future_20d_return'] > 0).astype(int)

    # 特征
    for w in [3, 5, 10, 20, 40, 60]:
        train_recent[f'logret_{w}'] = train_recent.groupby('code')['close'].transform(
            lambda s: np.log(s / s.shift(w))
        )
    train_recent['vol_20'] = train_recent.groupby('code')['close'].transform(
        lambda s: s.pct_change().rolling(20, min_periods=5).std()
    )
    train_recent['vol_60'] = train_recent.groupby('code')['close'].transform(
        lambda s: s.pct_change().rolling(60, min_periods=10).std()
    )
    train_recent['close_ma20'] = train_recent.groupby('code')['close'].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    train_recent['close_to_ma20'] = train_recent['close'] / (train_recent['close_ma20'] + 1e-9) - 1
    train_recent['vol_ma20'] = train_recent.groupby('code')['volume'].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    train_recent['vol_ratio_20'] = train_recent['volume'] / (train_recent['vol_ma20'] + 1e-9)
    # 高低价差
    train_recent['hl_range'] = (train_recent['high'] - train_recent['low']) / (train_recent['close'] + 1e-9)
    # 跳空
    train_recent['gap'] = train_recent['open'] / train_recent.groupby('code')['close'].shift(1) - 1
    # 连涨连跌
    train_recent['sign'] = np.sign(train_recent.groupby('code')['close'].pct_change()).fillna(0)
    train_recent['streak'] = train_recent.groupby('code')['sign'].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    ) * train_recent['sign']

    feat_cols = [
        'logret_3', 'logret_5', 'logret_10', 'logret_20', 'logret_40', 'logret_60',
        'vol_20', 'vol_60', 'close_to_ma20', 'vol_ratio_20', 'hl_range', 'gap', 'streak',
    ]
    df_train = train_recent[feat_cols + ['up_label', 'future_20d_return', 'date_idx']].dropna()
    print(f"  训练样本: {len(df_train)}")
    print(f"  上涨比例: {df_train['up_label'].mean():.4f}")

    # === 3. Walk-Forward 评估 ===
    print("\n[3] Walk-Forward 评估 ...")
    # 划分：用 DAY_2294-2594 做 train, DAY_2595-2794 做 validation
    wf_train_mask = df_train['date_idx'] < 2594
    df_tr = df_train[wf_train_mask]
    df_va = df_train[~wf_train_mask]
    print(f"  walk-forward train: {len(df_tr)}, val: {len(df_va)}")

    X_tr = df_tr[feat_cols].values
    y_tr = df_tr['up_label'].values
    X_va = df_va[feat_cols].values
    y_va = df_va['up_label'].values

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
    m = lgb.train(p, train_set, num_boost_round=1500,
                  valid_sets=[train_set, val_set], valid_names=["train", "valid"],
                  callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100, verbose=False)])
    pred_va = m.predict(X_va, num_iteration=m.best_iteration)
    auc_wf = roc_auc_score(y_va, pred_va)
    brier_wf = brier_score_loss(y_va, pred_va)
    print(f"  Walk-Forward: AUC={auc_wf:.4f}, Brier={brier_wf:.4f}")

    # === 4. 用全量最近数据训练 final ===
    print("\n[4] 训练 final (全量最近 500 天) ...")
    Xtr_all = df_train[feat_cols].values
    ytr_all = df_train['up_label'].values
    final_model = lgb.train(p, lgb.Dataset(Xtr_all, label=ytr_all),
                            num_boost_round=int(m.best_iteration * 1.0))
    print(f"  final num_boost = {m.best_iteration}")

    # === 5. 测试集特征 ===
    print("\n[5] 测试集特征 ...")
    test = test.sort_values(['code', 'date']).reset_index(drop=True)
    test_rows = []
    for code, sub in test.groupby('code'):
        f = {'code': code}
        n = len(sub)
        for w in [3, 5, 10, 20, 40, 60]:
            if n > w:
                f[f'logret_{w}'] = np.log(sub['close'].iloc[-1] / sub['close'].iloc[-w-1])
            else:
                f[f'logret_{w}'] = 0
        rets = sub['close'].pct_change().dropna()
        f['vol_20'] = rets.iloc[-20:].std() if len(rets) > 1 else 0
        f['vol_60'] = rets.std() if len(rets) > 1 else 0
        ma20 = sub['close'].iloc[-20:].mean() if n >= 20 else sub['close'].mean()
        f['close_to_ma20'] = sub['close'].iloc[-1] / (ma20 + 1e-9) - 1
        vol_ma20 = sub['volume'].iloc[-20:].mean() if n >= 20 else sub['volume'].mean()
        f['vol_ratio_20'] = sub['volume'].iloc[-1] / (vol_ma20 + 1e-9)
        f['hl_range'] = (sub['high'] - sub['low']).mean() / (sub['close'].mean() + 1e-9)
        f['gap'] = (sub['open'] / sub['close'].shift(1) - 1).mean()
        # streak
        sign = np.sign(sub['close'].pct_change().fillna(0))
        streak = 0
        for s in sign:
            if s > 0:
                streak = max(1, streak + 1) if streak > 0 else 1
            elif s < 0:
                streak = min(-1, streak - 1) if streak < 0 else -1
            else:
                streak = 0
        f['streak'] = streak
        test_rows.append(f)
    test_df = pd.DataFrame(test_rows)
    Xte = test_df[feat_cols].values
    test_pred_lgb = final_model.predict(Xte)

    # === 6. 多重反转信号 ===
    print("\n[6] 多重反转信号 ...")
    # 反转动量（取负后归一化）
    rev_5 = norm_rank(-test_df['logret_5'].values)
    rev_10 = norm_rank(-test_df['logret_10'].values)
    rev_20 = norm_rank(-test_df['logret_20'].values)
    rev_40 = norm_rank(-test_df['logret_40'].values)
    rev_streak = norm_rank(-test_df['streak'].values)  # 连跌优先
    # 波动率倒数（低波动的资产更稳定）
    inv_vol = norm_rank(-test_df['vol_20'].values)
    # 与 20 日均线的关系
    rev_ma = norm_rank(-test_df['close_to_ma20'].values)  # 远低于均线的优先

    # 综合反转信号
    rev_signal = (rev_5 + rev_10 + rev_20 + rev_40 + rev_streak + inv_vol + rev_ma) / 7
    # 归一化
    rev_signal = norm_rank(rev_signal)

    # === 7. 集成 LGB + 反转信号 ===
    print("\n[7] 集成 ...")
    lgb_norm = norm_rank(test_pred_lgb)
    # 不同权重试验
    best_w_lgb = 0.5
    best_w_rev = 0.5
    # 用 walk-forward 评估找最优权重
    best_score = -1
    best_w = 0.5
    pred_va_lgb = pred_va  # LGB on validation
    # 反转信号在 validation 上的预测
    # 先计算 validation 的反转信号（构造类似的特征）
    df_va_recent = df_va.copy()
    rev_va = (
        norm_rank(-df_va_recent['logret_5'].values) +
        norm_rank(-df_va_recent['logret_10'].values) +
        norm_rank(-df_va_recent['logret_20'].values) +
        norm_rank(-df_va_recent['logret_40'].values) +
        norm_rank(-df_va_recent['streak'].values) +
        norm_rank(-df_va_recent['vol_20'].values) +
        norm_rank(-df_va_recent['close_to_ma20'].values)
    ) / 7
    rev_va = norm_rank(rev_va)
    lgb_va_norm = norm_rank(pred_va_lgb)

    print("\n  找最优权重组合 (walk-forward val):")
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        combo = w * lgb_va_norm + (1 - w) * rev_va
        # 评估 Brier
        b = brier_score_loss(y_va, combo)
        a = roc_auc_score(y_va, combo)
        # 还要看 top-5 命中
        order = np.argsort(-combo)
        top5_hit = np.mean(y_va[order[:5]])
        # 综合"分数"
        score = a - b
        if score > best_score:
            best_score = score
            best_w = w
            best_b = b
            best_a = a
            best_top5 = top5_hit
        print(f"    w_lgb={w:.1f}: AUC={a:.4f}, Brier={b:.4f}, top5_hit={top5_hit:.4f}")
    print(f"  best w_lgb = {best_w}, AUC={best_a:.4f}, Brier={best_b:.4f}, top5_hit={best_top5:.4f}")

    # 用最优权重生成最终 submission
    final_up = best_w * lgb_norm + (1 - best_w) * rev_signal
    final_up = norm_rank(final_up)

    sub = pd.DataFrame({
        'code': test_df['code'].values,
        'up_factor': final_up
    }).sort_values('code').reset_index(drop=True)
    sub.to_csv(SUB / out_csv_name, index=False, encoding='utf-8')
    print(f"\n[Submit] -> {SUB / out_csv_name}")
    print(sub['up_factor'].describe())
    top5 = sub.nlargest(5, 'up_factor')
    print(f"top-5: {top5['code'].tolist()}")

    # 报告
    report = {
        'walk_forward': {
            'auc': float(auc_wf),
            'brier': float(brier_wf),
        },
        'best_weight_lgb': float(best_w),
        'best_combo': {
            'auc': float(best_a),
            'brier': float(best_b),
            'top5_hit': float(best_top5),
        },
        'submission_stats': {
            'mean': float(sub['up_factor'].mean()),
            'std': float(sub['up_factor'].std()),
            'range': [float(sub['up_factor'].min()), float(sub['up_factor'].max())],
        }
    }
    with open(OUT / "v7_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
