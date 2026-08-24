"""
run_v4.py
=========
最终版：v3 + LambdaRank + CatBoost + 更深的 stacking
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptivepath.dataset import build_truncated_dataset, CONTEXT_LEN, TARGET_LEN
from adaptivepath.window_features_v2 import window_features_v2
from adaptivepath.trainer_v2 import (
    train_lgb_clf, train_lgb_reg, train_mlp, predict_mlp,
)
from scripts.backtest import simulate_score

DATA = ROOT.parent
TRAIN_CSV = DATA / "train" / "train.csv"
TEST_CSV = DATA / "test" / "test.csv"
OUT = DATA / "output"
SUB = DATA / "submission" / "output"
OUT.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)


def select_long_stocks(train: pd.DataFrame, n_stocks: int, min_history: int):
    counts = train.groupby("code").size()
    long_codes = counts[counts >= min_history].index[:n_stocks].tolist()
    return train[train["code"].isin(long_codes)].copy()


def extract_features(samples: pd.DataFrame, train: pd.DataFrame, desc: str = "pretrain") -> pd.DataFrame:
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
        f["sample_idx"] = i
        rows.append(f)
        if (i + 1) % 5000 == 0:
            print(f"  [{desc}] feat {i+1}/{len(samples)}  ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def extract_test_features(test: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
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


def make_rank_groups(n, group_size=32, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_groups = n // group_size
    truncated = idx[:n_groups * group_size]
    return np.array_split(truncated, n_groups)


def safe_groups(n, group_size=32, seed=0):
    """让 group 数和样本数严格匹配：truncate n 到 n_groups * group_size。"""
    g = make_rank_groups(n, group_size, seed)
    total = sum(len(gi) for gi in g)
    return g, total


def train_lgb_rank(X_tr, y_tr, group_tr, X_va, y_va, group_va, num_boost=1500):
    p = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }
    train_set = lgb.Dataset(X_tr, label=y_tr, group=group_tr)
    val_set = lgb.Dataset(X_va, label=y_va, group=group_va, reference=train_set)
    model = lgb.train(
        p, train_set, num_boost_round=num_boost,
        valid_sets=[train_set, val_set], valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100, verbose=False)],
    )
    return model


def main(
    n_stocks: int = 4375,
    n_per_stock: int = 5,
    min_history: int = 60,
    n_splits: int = 5,
    num_boost: int = 1500,
    seed: int = 42,
    out_csv_name: str = "submission_v4.csv",
):
    t0 = time.time()
    print("=" * 70)
    print(f"[FCPFF v4] n_stocks={n_stocks}, n_per_stock={n_per_stock}, ctx={CONTEXT_LEN}, tgt={TARGET_LEN}")
    print("=" * 70)

    print("\n[1] 加载数据 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"  train: {train.shape}, test: {test.shape}")

    print(f"\n[2] 选 {n_stocks} 只长期股票 ...")
    train_sub = select_long_stocks(train, n_stocks, min_history)
    print(f"  rows: {len(train_sub)}, unique codes: {train_sub['code'].nunique()}")

    print(f"\n[3] 构造截断样本 ...")
    samples = build_truncated_dataset(
        train_sub, n_per_stock=n_per_stock, min_history=min_history, seed=seed,
    )
    print(f"  samples: {len(samples)}")

    print(f"\n[4] 抽取窗口特征 ...")
    Xtr_df = extract_features(samples, train_sub, "pretrain")
    feat_cols = [c for c in Xtr_df.columns
                 if c not in {"up_label", "future_20d_return", "sample_idx"}]
    print(f"  X_tr: {Xtr_df[feat_cols].shape}, features: {len(feat_cols)}")
    Xtr = Xtr_df[feat_cols].values.astype(np.float32)
    ytr_cls = Xtr_df["up_label"].values.astype(np.int32)
    ytr_reg = Xtr_df["future_20d_return"].values.astype(np.float32)
    ytr_rank = (ytr_reg - ytr_reg.min()) / (ytr_reg.max() - ytr_reg.min() + 1e-9)

    print(f"\n[5] 抽取测试集特征 ...")
    Xte_df = extract_test_features(test, feat_cols)
    Xte = Xte_df[feat_cols].values.astype(np.float32)
    print(f"  X_te: {Xte.shape}")

    print(f"\n[6] OOF 训练 {n_splits} 折 (含 CatBoost + Ranker) ...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_tr, n_te = len(Xtr), len(Xte)
    oof_keys = ['lgb', 'xgb', 'cat', 'mlp', 'lr', 'nb', 'lgb_reg', 'xgb_reg', 'cat_reg', 'lgb_rank']
    oof = {k: np.zeros(n_tr) for k in oof_keys}
    test_pred = {k: np.zeros(n_te) for k in oof_keys}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
        print(f"  -- fold {fold+1}/{n_splits} --")
        X_tr, y_tr = Xtr[tr_idx], ytr_cls[tr_idx]
        X_va, y_va = Xtr[va_idx], ytr_cls[va_idx]
        y_tr_reg, y_va_reg = ytr_reg[tr_idx], ytr_reg[va_idx]
        y_tr_rank, y_va_rank = ytr_rank[tr_idx], ytr_rank[va_idx]
        X_tr_s = np.nan_to_num(X_tr, nan=0.0)
        X_va_s = np.nan_to_num(X_va, nan=0.0)
        X_te_s = np.nan_to_num(Xte, nan=0.0)

        # 分类模型
        m = train_lgb_clf(X_tr_s, y_tr, X_va_s, y_va, num_boost=num_boost)
        oof['lgb'][va_idx] = m.predict(X_va_s, num_iteration=m.best_iteration)
        test_pred['lgb'] += m.predict(X_te_s, num_iteration=m.best_iteration) / n_splits

        m = train_xgb_cls(X_tr_s, y_tr, X_va_s, y_va, num_boost=num_boost)
        oof['xgb'][va_idx] = m.predict(xgb.DMatrix(X_va_s))
        test_pred['xgb'] += m.predict(xgb.DMatrix(X_te_s)) / n_splits

        # CatBoost
        cb = CatBoostClassifier(
            iterations=num_boost, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, random_seed=42, verbose=0, task_type="CPU",
        )
        cb.fit(X_tr_s, y_tr, eval_set=(X_va_s, y_va), early_stopping_rounds=100, verbose=False)
        oof['cat'][va_idx] = cb.predict_proba(X_va_s)[:, 1]
        test_pred['cat'] += cb.predict_proba(X_te_s)[:, 1] / n_splits

        m = train_mlp(X_tr_s, y_tr, X_va_s, y_va, epochs=100)
        oof['mlp'][va_idx] = predict_mlp(m, X_va_s)
        test_pred['mlp'] += predict_mlp(m, X_te_s) / n_splits

        sc = StandardScaler()
        X_tr_n = sc.fit_transform(X_tr_s)
        X_va_n = sc.transform(X_va_s)
        X_te_n = sc.transform(X_te_s)
        lr = LogisticRegression(C=0.5, max_iter=2000)
        lr.fit(X_tr_n, y_tr)
        oof['lr'][va_idx] = lr.predict_proba(X_va_n)[:, 1]
        test_pred['lr'] += lr.predict_proba(X_te_n)[:, 1] / n_splits
        nb = GaussianNB()
        nb.fit(X_tr_n, y_tr)
        oof['nb'][va_idx] = nb.predict_proba(X_va_n)[:, 1]
        test_pred['nb'] += nb.predict_proba(X_te_n)[:, 1] / n_splits

        # 回归模型
        m = train_lgb_reg(X_tr_s, y_tr_reg, X_va_s, y_va_reg, num_boost=num_boost)
        reg_va = m.predict(X_va_s, num_iteration=m.best_iteration)
        reg_te = m.predict(X_te_s, num_iteration=m.best_iteration)
        center = float(np.median(y_tr_reg))
        oof['lgb_reg'][va_idx] = 1.0 / (1.0 + np.exp(-(reg_va - center) * 10))
        test_pred['lgb_reg'] += (1.0 / (1.0 + np.exp(-(reg_te - center) * 10))) / n_splits

        m = train_xgb_reg(X_tr_s, y_tr_reg, X_va_s, y_va_reg, num_boost=num_boost)
        oof['xgb_reg'][va_idx] = m.predict(xgb.DMatrix(X_va_s))
        test_pred['xgb_reg'] += m.predict(xgb.DMatrix(X_te_s)) / n_splits

        cb_reg = CatBoostRegressor(
            iterations=num_boost, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, random_seed=42, verbose=0, task_type="CPU",
            loss_function="RMSE",
        )
        cb_reg.fit(X_tr_s, y_tr_reg, eval_set=(X_va_s, y_va_reg), early_stopping_rounds=100, verbose=False)
        reg_va_cb = cb_reg.predict(X_va_s)
        reg_te_cb = cb_reg.predict(X_te_s)
        oof['cat_reg'][va_idx] = 1.0 / (1.0 + np.exp(-(reg_va_cb - center) * 10))
        test_pred['cat_reg'] += (1.0 / (1.0 + np.exp(-(reg_te_cb - center) * 10))) / n_splits

        # Ranker
        g_tr, total_tr = safe_groups(len(X_tr_s), group_size=32, seed=fold)
        g_va, total_va = safe_groups(len(X_va_s), group_size=32, seed=fold + 100)
        m = train_lgb_rank(
            X_tr_s[:total_tr], y_tr[:total_tr], [len(g) for g in g_tr],
            X_va_s[:total_va], y_va[:total_va], [len(g) for g in g_va],
            num_boost=num_boost,
        )
        # Ranker 输出不是概率，需要做 sigmoid 缩放
        rank_va = m.predict(X_va_s, num_iteration=m.best_iteration)
        rank_te = m.predict(X_te_s, num_iteration=m.best_iteration)
        from scipy.stats import rankdata
        oof['lgb_rank'][va_idx] = rankdata(rank_va) / len(rank_va)
        test_pred['lgb_rank'] += rankdata(rank_te) / len(rank_te) / n_splits

    # 评估
    print(f"\n[7] OOF 评估")
    metrics = {}
    for name, o in oof.items():
        if 'reg' in name:
            m = {
                "brier": float(brier_score_loss(ytr_cls, np.clip(o, 1e-6, 1-1e-6))),
                "spearman": float(spearmanr(ytr_reg, o).correlation),
            }
        else:
            m = {
                "auc": float(roc_auc_score(ytr_cls, o)),
                "brier": float(brier_score_loss(ytr_cls, np.clip(o, 1e-6, 1-1e-6))),
                "logloss": float(log_loss(ytr_cls, np.clip(o, 1e-6, 1-1e-6))),
            }
        metrics[name] = m
        print(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))

    # Stacking
    print(f"\n[7.5] Stacking (meta = XGBoost)")
    P_oof = np.stack([oof[k] for k in oof_keys], axis=1)
    P_test = np.stack([test_pred[k] for k in oof_keys], axis=1)
    # 用 LR 当 meta 简单
    meta = LogisticRegression(C=1.0, max_iter=2000)
    meta.fit(P_oof, ytr_cls)
    oof_meta = meta.predict_proba(P_oof)[:, 1]
    test_meta = meta.predict_proba(P_test)[:, 1]
    metrics["stack"] = {
        "auc": float(roc_auc_score(ytr_cls, oof_meta)),
        "brier": float(brier_score_loss(ytr_cls, oof_meta)),
        "logloss": float(log_loss(ytr_cls, np.clip(oof_meta, 1e-6, 1-1e-6))),
    }
    print(f"  stack: AUC={metrics['stack']['auc']:.4f}, Brier={metrics['stack']['brier']:.4f}")
    print(f"  Coef: {dict(zip(oof_keys, [round(float(c),3) for c in meta.coef_[0]]))}")

    # 校准
    print(f"\n[8] Isotonic 校准 + Top-K 软重排")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(oof_meta, ytr_cls)
    oof_calib = iso.predict(oof_meta)
    test_calib = iso.predict(test_meta)
    metrics["calib"] = {
        "auc": float(roc_auc_score(ytr_cls, oof_calib)),
        "brier": float(brier_score_loss(ytr_cls, oof_calib)),
    }
    print(f"  calib: AUC={metrics['calib']['auc']:.4f}, Brier={metrics['calib']['brier']:.4f}")

    def topk_soft(p, alpha=0.0):
        if alpha == 0.0:
            return p
        return 1.0 / (1.0 + np.exp(-alpha * (p - 0.5)))

    best_alpha = 0.0
    best_brier = metrics["calib"]["brier"]
    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        p_aug = topk_soft(oof_calib, alpha=a)
        b = brier_score_loss(ytr_cls, p_aug)
        if b < best_brier:
            best_brier = b
            best_alpha = a
    print(f"  best alpha = {best_alpha}  -> Brier = {best_brier:.4f}")
    test_final = topk_soft(test_calib, alpha=best_alpha)
    oof_final = topk_soft(oof_calib, alpha=best_alpha)
    metrics["final"] = {
        "auc": float(roc_auc_score(ytr_cls, oof_final)),
        "brier": float(brier_score_loss(ytr_cls, oof_final)),
    }

    # Top-5 模拟
    rng = np.random.default_rng(0)
    n_sim = 2000
    n = len(oof_final)
    hits_baseline = []
    hits_model = []
    for _ in range(n_sim):
        idx = rng.integers(0, n, size=n)
        p_ = oof_final[idx]
        y_ = ytr_cls[idx]
        order = np.argsort(-p_)[:5]
        hits_model.append(np.mean(y_[order]))
        rand = rng.choice(n, size=5, replace=False)
        hits_baseline.append(np.mean(y_[rand]))
    print(f"  baseline top5 hit = {np.mean(hits_baseline):.4f}")
    print(f"  model top5 hit    = {np.mean(hits_model):.4f}")
    print(f"  lift              = {np.mean(hits_model) - np.mean(hits_baseline):+.4f}")

    # 写提交
    print(f"\n[9] 写提交")
    sub = pd.DataFrame({
        "code": Xte_df["code"].values,
        "up_factor": test_final,
    }).sort_values("code").reset_index(drop=True)
    out_path = SUB / out_csv_name
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  -> {out_path}")
    print(sub["up_factor"].describe())

    # 保存
    Xtr_df.to_csv(OUT / "Xtr_df.csv", index=False)
    np.save(OUT / "oof_final.npy", oof_final)
    np.save(OUT / "test_final.npy", test_final)
    np.save(OUT / "ytr_cls.npy", ytr_cls)
    np.save(OUT / "ytr_reg.npy", ytr_reg)
    model_dir = OUT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(model_dir / "stacking_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    with open(model_dir / "feat_cols.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False)

    # 回测
    print(f"\n[10] 本地回测")
    bt = simulate_score(p=oof_final, y_up=ytr_cls, y_ret=ytr_reg)
    for k, v in bt.items():
        print(f"  {k}: {v}")

    # 报告
    report = {
        "config": {
            "n_stocks": n_stocks, "n_per_stock": n_per_stock,
            "min_history": min_history, "n_splits": n_splits,
            "num_boost": num_boost, "seed": seed,
            "context_len": CONTEXT_LEN, "target_len": TARGET_LEN,
            "n_features": len(feat_cols), "n_train_samples": int(n_tr),
        },
        "metrics": metrics,
        "best_alpha": float(best_alpha),
        "top5_hit": {
            "baseline": float(np.mean(hits_baseline)),
            "model": float(np.mean(hits_model)),
            "lift": float(np.mean(hits_model) - np.mean(hits_baseline)),
        },
        "backtest": bt,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(OUT / "pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] report -> {OUT / 'pipeline_report.json'}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    return report


def train_xgb_cls(X_tr, y_tr, X_va, y_va, num_boost=1500):
    p = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "tree_method": "hist",
        "verbosity": 0,
        "nthread": -1,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    model = xgb.train(
        p, dtrain, num_boost_round=num_boost,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100, verbose_eval=False,
    )
    return model


def train_xgb_reg(X_tr, y_tr, X_va, y_va, num_boost=1500):
    p = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "tree_method": "hist",
        "verbosity": 0,
        "nthread": -1,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    model = xgb.train(
        p, dtrain, num_boost_round=num_boost,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100, verbose_eval=False,
    )
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_stocks", type=int, default=4375)
    p.add_argument("--n_per_stock", type=int, default=5)
    p.add_argument("--min_history", type=int, default=60)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--num_boost", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="submission_v4.csv")
    args = p.parse_args()
    main(
        n_stocks=args.n_stocks,
        n_per_stock=args.n_per_stock,
        min_history=args.min_history,
        n_splits=args.n_splits,
        num_boost=args.num_boost,
        seed=args.seed,
        out_csv_name=args.out,
    )
