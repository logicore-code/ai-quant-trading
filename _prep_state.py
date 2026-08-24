"""
run_v5.py
=========
针对真实评估的版本：
- 时间衰减样本权重：最近样本权重高（最接近测试集时点）
- 简化模型：单 LightGBM + 简单 stacking（去掉容易过拟合的多样性）
- 核心特征：只用 30 维最稳定的形态特征
- 强概率校准：Isotonic + 集成
- 半监督：用测试集形态做"市场状态匹配"加权
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
from scipy.stats import rankdata

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
OUT.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)


# 30 维核心特征（剔除过拟合的）
CORE_FEATURES = [
    # 末态 (5)
    'w_close_last', 'w_close_max', 'w_close_min', 'w_range_total', 'w_last_to_max',
    # 多尺度动量 (8)
    'm_ret_3', 'm_ret_5', 'm_ret_10', 'm_ret_20', 'm_ret_40',
    'm_logret_5', 'm_logret_10', 'm_logret_20',
    # 反转 (3)
    'mom_diff_2_5', 'mom_diff_5_20', 'mom_diff_3_10',
    # 波动率 (5)
    'v_std_all', 'v_std_5', 'v_std_10', 'v_std_20', 'v_gk_mean',
    # 量价 (3)
    'q_pv_corr_5', 'q_vol_ratio_5', 'q_vol_last_to_5',
    # 趋势 (3)
    't_rsi_14', 't_adx_approx', 't_di_diff',
    # 形态 (3)
    'p_amp_mean', 'p_amp_max', 'p_final_streak',
]


def select_long_stocks(train, n_stocks, min_history):
    counts = train.groupby('code').size()
    long_codes = counts[counts >= min_history].index[:n_stocks].tolist()
    return train[train['code'].isin(long_codes)].copy()


def extract_features(samples, train, desc=""):
    code_to_sub = {c: g for c, g in train.groupby("code", sort=False)}
    rows = []
    t0 = time.time()
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
        f["ctx_end_idx"] = int(row["ctx_end_date"].replace("DAY_", ""))
        f["sample_idx"] = i
        rows.append(f)
        if (i + 1) % 5000 == 0:
            print(f"  [{desc}] feat {i+1}/{len(samples)}  ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def extract_test_features(test, feat_cols):
    rows = []
    for code, sub in test.groupby("code"):
        f = window_features_v2(sub)
        f["code"] = code
        rows.append(f)
    df = pd.DataFrame(rows)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


def main(
    n_stocks=4375,
    n_per_stock=8,
    min_history=80,
    n_splits=5,
    num_boost=1200,
    seed=42,
    time_decay=True,
    state_match=True,
    out_csv_name="submission_v5.csv",
):
    t0 = time.time()
    print("=" * 70)
    print(f"[FCPFF v5] 时间加权 + 简化模型 + 状态匹配")
    print("=" * 70)

    print("\n[1] 加载数据 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"  train: {train.shape}, test: {test.shape}")

    print(f"\n[2] 选股 + 构造截断样本 ...")
    train_sub = select_long_stocks(train, n_stocks, min_history)
    print(f"  rows: {len(train_sub)}, codes: {train_sub['code'].nunique()}")
    samples = build_truncated_dataset(
        train_sub, n_per_stock=n_per_stock, min_history=min_history, seed=seed,
    )
    print(f"  samples: {len(samples)}")

    print(f"\n[3] 抽取窗口特征 ...")
    Xtr_df = extract_features(samples, train_sub, "pretrain")
    all_feats = [c for c in Xtr_df.columns
                 if c not in {"up_label", "future_20d_return", "sample_idx", "ctx_end_idx"}]
    # 只用核心特征
    feat_cols = [c for c in CORE_FEATURES if c in all_feats]
    print(f"  特征数: {len(feat_cols)} (核心)")
    Xtr = Xtr_df[feat_cols].values.astype(np.float32)
    ytr_cls = Xtr_df["up_label"].values.astype(np.int32)
    ytr_reg = Xtr_df["future_20d_return"].values.astype(np.float32)
    ctx_end_idx = Xtr_df["ctx_end_idx"].values.astype(np.int32)

    # 时间衰减权重：越接近测试集（DAY_2795-2814）权重越高
    if time_decay:
        # 样本结束日距离测试集起始日 (DAY_2795) 越近越好
        # 距 = 2795 - ctx_end_idx, 越大越老
        dist = 2795 - ctx_end_idx
        # 距离 <= 0 的是"未来"，丢弃
        valid = dist >= 0
        # 用指数衰减: w = exp(-dist / 500)，500 天 = e 倍衰减
        weights = np.exp(-dist.astype(np.float32) / 500.0)
        weights = weights * valid
        # 归一化
        weights = weights / (weights.sum() + 1e-9) * len(weights)
        # 重新构造训练集
        Xtr = Xtr[valid]
        ytr_cls = ytr_cls[valid]
        ytr_reg = ytr_reg[valid]
        weights_train = weights[valid]
        print(f"  时间加权后样本: {len(Xtr)}")
    else:
        weights_train = np.ones(len(Xtr), dtype=np.float32)

    print(f"\n[4] 抽取测试集特征 ...")
    Xte_df = extract_test_features(test, feat_cols)
    Xte = Xte_df[feat_cols].values.astype(np.float32)
    print(f"  X_te: {Xte.shape}")

    # 市场状态匹配：用测试集特征均值/标准差做"状态向量"
    if state_match:
        # 测试集状态：每只资产的 30 维特征
        # 训练集状态：每条样本的 30 维特征
        # 用 cosine 距离找最相似的 K 个
        from sklearn.metrics.pairwise import cosine_similarity
        # 标准化
        sc_state = StandardScaler()
        Xtr_n = sc_state.fit_transform(np.nan_to_num(Xtr, nan=0.0))
        Xte_n = sc_state.transform(np.nan_to_num(Xte, nan=0.0))
        # 对每个测试样本找训练集 top-1000
        sims = cosine_similarity(Xte_n, Xtr_n)  # (1500, n_train)
        # 对训练集每个样本：它在多少个测试样本的 top-1000 内？
        n_test = Xte_n.shape[0]
        top_k = min(1000, Xtr_n.shape[0])
        # 高效：sort sims[:, :] 取 top_k
        top_idx = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
        # 计算每个训练样本被选中的次数
        in_count = np.zeros(len(Xtr), dtype=np.int32)
        for i in range(n_test):
            in_count[top_idx[i]] += 1
        # 状态匹配权重：被选越多权重越大
        state_w = in_count.astype(np.float32) / n_test * 5 + 1  # 0-5+1
        # 合并时间权重和状态权重
        weights_train = weights_train * state_w
        print(f"  状态匹配加权: 平均 weight = {weights_train.mean():.2f}")

    print(f"\n[5] OOF 训练 5 折 (LightGBM) ...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(Xtr))
    test_pred = np.zeros(len(Xte))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
        print(f"  -- fold {fold+1}/{n_splits} --")
        X_tr, y_tr, w_tr = Xtr[tr_idx], ytr_cls[tr_idx], weights_train[tr_idx]
        X_va, y_va = Xtr[va_idx], ytr_cls[va_idx]
        X_tr_s = np.nan_to_num(X_tr, nan=0.0)
        X_va_s = np.nan_to_num(X_va, nan=0.0)
        X_te_s = np.nan_to_num(Xte, nan=0.0)

        p = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.03,
            "num_leaves": 31,  # 减少（避免过拟合）
            "min_data_in_leaf": 200,  # 增大
            "feature_fraction": 0.6,
            "bagging_fraction": 0.6,
            "bagging_freq": 5,
            "lambda_l1": 1.0,  # 增强正则
            "lambda_l2": 1.0,
            "verbose": -1,
            "n_jobs": -1,
        }
        train_set = lgb.Dataset(X_tr_s, label=y_tr, weight=w_tr)
        val_set = lgb.Dataset(X_va_s, label=y_va, reference=train_set)
        m = lgb.train(
            p, train_set, num_boost_round=num_boost,
            valid_sets=[train_set, val_set], valid_names=["train", "valid"],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100, verbose=False)],
        )
        oof[va_idx] = m.predict(X_va_s, num_iteration=m.best_iteration)
        test_pred += m.predict(X_te_s, num_iteration=m.best_iteration) / n_splits

    print(f"\n[6] OOF 评估")
    auc = roc_auc_score(ytr_cls, oof)
    brier = brier_score_loss(ytr_cls, oof)
    print(f"  OOF AUC = {auc:.4f}, Brier = {brier:.4f}")

    # 校准
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    iso.fit(oof, ytr_cls)
    oof_calib = iso.predict(oof)
    test_calib = iso.predict(test_pred)
    auc_c = roc_auc_score(ytr_cls, oof_calib)
    brier_c = brier_score_loss(ytr_cls, oof_calib)
    print(f"  Calib AUC = {auc_c:.4f}, Brier = {brier_c:.4f}")

    # Top-5 模拟
    rng = np.random.default_rng(0)
    n_sim = 1000
    n = len(oof_calib)
    hits_baseline, hits_model = [], []
    for _ in range(n_sim):
        idx = rng.integers(0, n, size=n)
        p_ = oof_calib[idx]
        y_ = ytr_cls[idx]
        order = np.argsort(-p_)[:5]
        hits_model.append(np.mean(y_[order]))
        rand = rng.choice(n, size=5, replace=False)
        hits_baseline.append(np.mean(y_[rand]))
    print(f"  Top-5 baseline = {np.mean(hits_baseline):.4f}, model = {np.mean(hits_model):.4f}")

    # 写提交
    sub = pd.DataFrame({
        "code": Xte_df["code"].values,
        "up_factor": test_calib,
    }).sort_values("code").reset_index(drop=True)
    out_path = SUB / out_csv_name
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[Submit] -> {out_path}")
    print(sub["up_factor"].describe())

    # 报告
    report = {
        "config": {
            "n_stocks": n_stocks, "n_per_stock": n_per_stock,
            "min_history": min_history, "n_splits": n_splits,
            "num_boost": num_boost, "time_decay": time_decay,
            "state_match": state_match, "n_features": len(feat_cols),
        },
        "metrics": {
            "oof_auc": float(auc), "oof_brier": float(brier),
            "calib_auc": float(auc_c), "calib_brier": float(brier_c),
        },
        "top5": {
            "baseline": float(np.mean(hits_baseline)),
            "model": float(np.mean(hits_model)),
        },
        "elapsed_sec": float(time.time() - t0),
    }
    with open(OUT / "v5_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] report -> {OUT / 'v5_report.json'}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_stocks", type=int, default=4375)
    p.add_argument("--n_per_stock", type=int, default=8)
    p.add_argument("--min_history", type=int, default=80)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--num_boost", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_time_decay", action="store_true")
    p.add_argument("--no_state_match", action="store_true")
    p.add_argument("--out", type=str, default="submission_v5.csv")
    args = p.parse_args()
    main(
        n_stocks=args.n_stocks, n_per_stock=args.n_per_stock,
        min_history=args.min_history, n_splits=args.n_splits,
        num_boost=args.num_boost, seed=args.seed,
        time_decay=not args.no_time_decay,
        state_match=not args.no_state_match,
        out_csv_name=args.out,
    )
