"""
run_final.py
============
最终版：v4 + Seed Bagging (3 个种子平均)
"""
from __future__ import annotations

import argparse
import json
import os
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
from adaptivepath.trainer_v2 import train_lgb_clf, train_lgb_reg, train_mlp, predict_mlp
from scripts.backtest import simulate_score
from scripts.run_v4 import (
    select_long_stocks, extract_features, extract_test_features,
    safe_groups, train_lgb_rank, train_xgb_cls, train_xgb_reg,
)

DATA = ROOT.parent
TRAIN_CSV = DATA / "train" / "train.csv"
TEST_CSV = DATA / "test" / "test.csv"
OUT = DATA / "output"
SUB = DATA / "submission" / "output"
OUT.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)


def run_one_seed(Xtr, Xte, ytr_cls, ytr_reg, n_splits, num_boost, seed):
    """对单个种子跑完整 v4 流水线，返回 (oof_meta, test_meta)"""
    print(f"\n=== seed={seed} ===")
    n_tr, n_te = len(Xtr), len(Xte)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # 共享 KFold 切分!
    oof_keys = ['lgb', 'xgb', 'cat', 'mlp', 'lr', 'nb', 'lgb_reg', 'xgb_reg', 'cat_reg', 'lgb_rank']
    oof = {k: np.zeros(n_tr) for k in oof_keys}
    test_pred = {k: np.zeros(n_te) for k in oof_keys}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
        print(f"  [seed{seed}] fold {fold+1}/{n_splits}")
        X_tr, y_tr = Xtr[tr_idx], ytr_cls[tr_idx]
        X_va, y_va = Xtr[va_idx], ytr_cls[va_idx]
        y_tr_reg, y_va_reg = ytr_reg[tr_idx], ytr_reg[va_idx]
        X_tr_s = np.nan_to_num(X_tr, nan=0.0)
        X_va_s = np.nan_to_num(X_va, nan=0.0)
        X_te_s = np.nan_to_num(Xte, nan=0.0)

        m = train_lgb_clf(X_tr_s, y_tr, X_va_s, y_va, num_boost=num_boost)
        oof['lgb'][va_idx] = m.predict(X_va_s, num_iteration=m.best_iteration)
        test_pred['lgb'] += m.predict(X_te_s, num_iteration=m.best_iteration) / n_splits

        m = train_xgb_cls(X_tr_s, y_tr, X_va_s, y_va, num_boost=num_boost)
        oof['xgb'][va_idx] = m.predict(xgb.DMatrix(X_va_s))
        test_pred['xgb'] += m.predict(xgb.DMatrix(X_te_s)) / n_splits

        cb = CatBoostClassifier(
            iterations=num_boost, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, random_seed=seed, verbose=0,
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
            l2_leaf_reg=3, random_seed=seed, verbose=0, loss_function="RMSE",
        )
        cb_reg.fit(X_tr_s, y_tr_reg, eval_set=(X_va_s, y_va_reg), early_stopping_rounds=100, verbose=False)
        reg_va_cb = cb_reg.predict(X_va_s)
        reg_te_cb = cb_reg.predict(X_te_s)
        oof['cat_reg'][va_idx] = 1.0 / (1.0 + np.exp(-(reg_va_cb - center) * 10))
        test_pred['cat_reg'] += (1.0 / (1.0 + np.exp(-(reg_te_cb - center) * 10))) / n_splits

        g_tr, total_tr = safe_groups(len(X_tr_s), group_size=32, seed=fold + seed * 10)
        g_va, total_va = safe_groups(len(X_va_s), group_size=32, seed=fold + 100 + seed * 10)
        m = train_lgb_rank(
            X_tr_s[:total_tr], y_tr[:total_tr], [len(g) for g in g_tr],
            X_va_s[:total_va], y_va[:total_va], [len(g) for g in g_va],
            num_boost=num_boost,
        )
        rank_va = m.predict(X_va_s, num_iteration=m.best_iteration)
        rank_te = m.predict(X_te_s, num_iteration=m.best_iteration)
        from scipy.stats import rankdata
        oof['lgb_rank'][va_idx] = rankdata(rank_va) / len(rank_va)
        test_pred['lgb_rank'] += rankdata(rank_te) / len(rank_te) / n_splits

    # Stacking
    P_oof = np.stack([oof[k] for k in oof_keys], axis=1)
    P_test = np.stack([test_pred[k] for k in oof_keys], axis=1)
    meta = LogisticRegression(C=1.0, max_iter=2000)
    meta.fit(P_oof, ytr_cls)
    oof_meta = meta.predict_proba(P_oof)[:, 1]
    test_meta = meta.predict_proba(P_test)[:, 1]
    print(f"  [seed{seed}] OOF AUC = {roc_auc_score(ytr_cls, oof_meta):.4f}, "
          f"Brier = {brier_score_loss(ytr_cls, oof_meta):.4f}")
    return {
        "oof_meta": oof_meta,
        "test_meta": test_meta,
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_tr, n_te = len(Xtr), len(Xte)
    oof_keys = ['lgb', 'xgb', 'cat', 'mlp', 'lr', 'nb', 'lgb_reg', 'xgb_reg', 'cat_reg', 'lgb_rank']
    oof = {k: np.zeros(n_tr) for k in oof_keys}
    test_pred = {k: np.zeros(n_te) for k in oof_keys}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
        print(f"  [seed{seed}] fold {fold+1}/{n_splits}")
        X_tr, y_tr = Xtr[tr_idx], ytr_cls[tr_idx]
        X_va, y_va = Xtr[va_idx], ytr_cls[va_idx]
        y_tr_reg, y_va_reg = ytr_reg[tr_idx], ytr_reg[va_idx]
        X_tr_s = np.nan_to_num(X_tr, nan=0.0)
        X_va_s = np.nan_to_num(X_va, nan=0.0)
        X_te_s = np.nan_to_num(Xte, nan=0.0)

        m = train_lgb_clf(X_tr_s, y_tr, X_va_s, y_va, num_boost=num_boost)
        oof['lgb'][va_idx] = m.predict(X_va_s, num_iteration=m.best_iteration)
        test_pred['lgb'] += m.predict(X_te_s, num_iteration=m.best_iteration) / n_splits

        m = train_xgb_cls(X_tr_s, y_tr, X_va_s, y_va, num_boost=num_boost)
        oof['xgb'][va_idx] = m.predict(xgb.DMatrix(X_va_s))
        test_pred['xgb'] += m.predict(xgb.DMatrix(X_te_s)) / n_splits

        cb = CatBoostClassifier(
            iterations=num_boost, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, random_seed=seed, verbose=0,
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
            l2_leaf_reg=3, random_seed=seed, verbose=0, loss_function="RMSE",
        )
        cb_reg.fit(X_tr_s, y_tr_reg, eval_set=(X_va_s, y_va_reg), early_stopping_rounds=100, verbose=False)
        reg_va_cb = cb_reg.predict(X_va_s)
        reg_te_cb = cb_reg.predict(X_te_s)
        oof['cat_reg'][va_idx] = 1.0 / (1.0 + np.exp(-(reg_va_cb - center) * 10))
        test_pred['cat_reg'] += (1.0 / (1.0 + np.exp(-(reg_te_cb - center) * 10))) / n_splits

        g_tr, total_tr = safe_groups(len(X_tr_s), group_size=32, seed=fold + seed * 10)
        g_va, total_va = safe_groups(len(X_va_s), group_size=32, seed=fold + 100 + seed * 10)
        m = train_lgb_rank(
            X_tr_s[:total_tr], y_tr[:total_tr], [len(g) for g in g_tr],
            X_va_s[:total_va], y_va[:total_va], [len(g) for g in g_va],
            num_boost=num_boost,
        )
        rank_va = m.predict(X_va_s, num_iteration=m.best_iteration)
        rank_te = m.predict(X_te_s, num_iteration=m.best_iteration)
        from scipy.stats import rankdata
        oof['lgb_rank'][va_idx] = rankdata(rank_va) / len(rank_va)
        test_pred['lgb_rank'] += rankdata(rank_te) / len(rank_te) / n_splits

    # Stacking
    P_oof = np.stack([oof[k] for k in oof_keys], axis=1)
    P_test = np.stack([test_pred[k] for k in oof_keys], axis=1)
    meta = LogisticRegression(C=1.0, max_iter=2000)
    meta.fit(P_oof, ytr_cls)
    oof_meta = meta.predict_proba(P_oof)[:, 1]
    test_meta = meta.predict_proba(P_test)[:, 1]
    print(f"  [seed{seed}] OOF AUC = {roc_auc_score(ytr_cls, oof_meta):.4f}, "
          f"Brier = {brier_score_loss(ytr_cls, oof_meta):.4f}")
    return {
        "oof_meta": oof_meta,
        "test_meta": test_meta,
        "ytr_cls": ytr_cls,
        "ytr_reg": ytr_reg,
        "Xtr_df": Xtr_df,
        "Xte_df": Xte_df,
    }


def main(
    n_stocks: int = 4375,
    n_per_stock: int = 5,
    min_history: int = 60,
    n_splits: int = 5,
    num_boost: int = 1500,
    seeds: list = (42, 123, 2024),
    out_csv_name: str = "submission_final.csv",
):
    t0 = time.time()
    print("=" * 70)
    print(f"[FCPFF FINAL] seed_bagging over {len(seeds)} seeds: {seeds}")
    print("=" * 70)

    # 先准备共享数据
    print("\n[Pre] 加载数据 + 构造样本 + 抽特征 (只用一次) ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train_sub = select_long_stocks(train, n_stocks, min_history)
    samples = build_truncated_dataset(
        train_sub, n_per_stock=n_per_stock, min_history=min_history, seed=42,  # 固定一个 seed
    )
    Xtr_df = extract_features(samples, train_sub, "shared")
    feat_cols = [c for c in Xtr_df.columns
                 if c not in {"up_label", "future_20d_return", "sample_idx"}]
    Xtr = Xtr_df[feat_cols].values.astype(np.float32)
    ytr_cls = Xtr_df["up_label"].values.astype(np.int32)
    ytr_reg = Xtr_df["future_20d_return"].values.astype(np.float32)
    Xte_df = extract_test_features(test, feat_cols)
    Xte = Xte_df[feat_cols].values.astype(np.float32)
    print(f"  X_tr: {Xtr.shape}, X_te: {Xte.shape}")

    all_oof = []
    all_test = []
    for s in seeds:
        r = run_one_seed(Xtr, Xte, ytr_cls, ytr_reg, n_splits, num_boost, s)
        all_oof.append(r["oof_meta"])
        all_test.append(r["test_meta"])

    # 平均
    oof_meta_avg = np.mean(all_oof, axis=0)
    test_meta_avg = np.mean(all_test, axis=0)
    print(f"\n[Bagging] OOF AUC = {roc_auc_score(ytr_cls, oof_meta_avg):.4f}, "
          f"Brier = {brier_score_loss(ytr_cls, oof_meta_avg):.4f}")

    # Isotonic
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    iso.fit(oof_meta_avg, ytr_cls)
    oof_calib = iso.predict(oof_meta_avg)
    test_calib = iso.predict(test_meta_avg)
    print(f"[Calib] OOF AUC = {roc_auc_score(ytr_cls, oof_calib):.4f}, "
          f"Brier = {brier_score_loss(ytr_cls, oof_calib):.4f}")

    # Top-K soft
    def topk_soft(p, alpha=0.0):
        if alpha == 0.0:
            return p
        return 1.0 / (1.0 + np.exp(-alpha * (p - 0.5)))

    best_alpha = 0.0
    best_brier = brier_score_loss(ytr_cls, oof_calib)
    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        p_aug = topk_soft(oof_calib, alpha=a)
        b = brier_score_loss(ytr_cls, p_aug)
        if b < best_brier:
            best_brier = b
            best_alpha = a
    print(f"[TopK] best alpha = {best_alpha}, Brier = {best_brier:.4f}")
    oof_final = topk_soft(oof_calib, alpha=best_alpha)
    test_final = topk_soft(test_calib, alpha=best_alpha)

    # Top-5 模拟
    rng = np.random.default_rng(0)
    n_sim = 2000
    n = len(oof_final)
    hits_baseline, hits_model = [], []
    for _ in range(n_sim):
        idx = rng.integers(0, n, size=n)
        p_ = oof_final[idx]
        y_ = ytr_cls[idx]
        order = np.argsort(-p_)[:5]
        hits_model.append(np.mean(y_[order]))
        rand = rng.choice(n, size=5, replace=False)
        hits_baseline.append(np.mean(y_[rand]))
    print(f"[Top5] baseline = {np.mean(hits_baseline):.4f}, "
          f"model = {np.mean(hits_model):.4f}, "
          f"lift = {np.mean(hits_model) - np.mean(hits_baseline):+.4f}")

    # 写提交
    sub = pd.DataFrame({
        "code": Xte_df["code"].values,
        "up_factor": test_final,
    }).sort_values("code").reset_index(drop=True)
    out_path = SUB / out_csv_name
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[Submit] -> {out_path}")
    print(sub["up_factor"].describe())

    # 也写一份 submission.csv（最终提交）
    final_path = SUB / "submission.csv"
    sub.to_csv(final_path, index=False, encoding="utf-8")
    print(f"[Submit] final -> {final_path}")

    # 回测
    bt = simulate_score(p=oof_final, y_up=ytr_cls, y_ret=ytr_reg)
    print(f"\n[Backtest]")
    for k, v in bt.items():
        print(f"  {k}: {v}")

    # 报告
    report = {
        "config": {
            "n_stocks": n_stocks, "n_per_stock": n_per_stock,
            "min_history": min_history, "n_splits": n_splits,
            "num_boost": num_boost, "seeds": list(seeds),
            "context_len": CONTEXT_LEN, "target_len": TARGET_LEN,
        },
        "metrics": {
            "bagging_auc": float(roc_auc_score(ytr_cls, oof_meta_avg)),
            "bagging_brier": float(brier_score_loss(ytr_cls, oof_meta_avg)),
            "calib_auc": float(roc_auc_score(ytr_cls, oof_calib)),
            "calib_brier": float(brier_score_loss(ytr_cls, oof_calib)),
            "final_auc": float(roc_auc_score(ytr_cls, oof_final)),
            "final_brier": float(brier_score_loss(ytr_cls, oof_final)),
        },
        "best_alpha": float(best_alpha),
        "top5_hit": {
            "baseline": float(np.mean(hits_baseline)),
            "model": float(np.mean(hits_model)),
            "lift": float(np.mean(hits_model) - np.mean(hits_baseline)),
        },
        "backtest": bt,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(OUT / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] report -> {OUT / 'final_report.json'}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_stocks", type=int, default=4375)
    p.add_argument("--n_per_stock", type=int, default=5)
    p.add_argument("--min_history", type=int, default=60)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--num_boost", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--out", type=str, default="submission_final.csv")
    args = p.parse_args()
    main(
        n_stocks=args.n_stocks,
        n_per_stock=args.n_per_stock,
        min_history=args.min_history,
        n_splits=args.n_splits,
        num_boost=args.num_boost,
        seeds=tuple(args.seeds),
        out_csv_name=args.out,
    )
