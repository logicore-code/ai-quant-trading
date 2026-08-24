"""
run_pipeline_v2.py
==================
完整版流水线 (v2)：
- 窗口与测试集对齐 (20 天)
- 多种窗口采样策略（随机 + 末期）
- 多模型集成
- 概率校准
- Top-K 友好后处理
- 完整 OOF 评估
- 自动生成报告
"""
from __future__ import annotations

import json
import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptivepath.dataset import build_truncated_dataset, CONTEXT_LEN, TARGET_LEN
from adaptivepath.window_features import window_features
from adaptivepath.trainer import train_lgb, train_xgb, train_mlp, predict_mlp

DATA = ROOT.parent
TRAIN_CSV = DATA / "train" / "train.csv"
TEST_CSV = DATA / "test" / "test.csv"
OUT = DATA / "output"
SUB = DATA / "submission" / "output"
OUT.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)

# 选股票：长期可用的
def select_long_stocks(train: pd.DataFrame, n_stocks: int, min_history: int) -> pd.DataFrame:
    counts = train.groupby("code").size()
    long_codes = counts[counts >= min_history].index[:n_stocks].tolist()
    return train[train["code"].isin(long_codes)].copy()


def main(
    n_stocks: int = 2000,
    n_per_stock: int = 5,
    min_history: int = 60,
    n_splits: int = 5,
    num_boost: int = 1500,
    out_csv_name: str = "submission.csv",
):
    t0 = time.time()
    print("=" * 70)
    print(f"[FCPFF] Pipeline v2: n_stocks={n_stocks}, n_per_stock={n_per_stock}, "
          f"min_hist={min_history}, context={CONTEXT_LEN}, target={TARGET_LEN}")
    print("=" * 70)

    # ---- 1. 加载 ----
    print("\n[1/8] 加载数据 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"  train: {train.shape}, test: {test.shape}")

    # ---- 2. 选股 + 构造截断样本 ----
    print(f"\n[2/8] 选 {n_stocks} 只长期股票 ...")
    train_sub = select_long_stocks(train, n_stocks, min_history)
    print(f"  rows: {len(train_sub)}, unique codes: {train_sub['code'].nunique()}")

    print(f"\n[3/8] 构造截断样本 (每只 {n_per_stock} 个) ...")
    samples = build_truncated_dataset(
        train_sub, n_per_stock=n_per_stock, min_history=min_history, seed=42,
    )
    print(f"  samples: {len(samples)}")

    # ---- 3. 窗口特征 ----
    print(f"\n[4/8] 抽取窗口特征 (窗口={CONTEXT_LEN} 天) ...")
    t1 = time.time()
    train_feat_rows = []
    code_to_sub = {c: g for c, g in train_sub.groupby("code", sort=False)}
    for i, row in samples.iterrows():
        sub = code_to_sub.get(row["code"])
        if sub is None:
            continue
        ctx = sub[(sub["date"] >= row["ctx_start"]) & (sub["date"] <= row["ctx_end_date"])]
        if len(ctx) < CONTEXT_LEN:
            continue
        f = window_features(ctx)
        f["up_label"] = row["up_label"]
        f["future_20d_return"] = row["future_20d_return"]
        f["sample_idx"] = i
        train_feat_rows.append(f)
        if (i + 1) % 2000 == 0:
            print(f"  feat {i+1}/{len(samples)}  ({time.time()-t1:.0f}s)")
    Xtr_df = pd.DataFrame(train_feat_rows)
    feat_cols = [c for c in Xtr_df.columns if c not in {"up_label", "future_20d_return", "sample_idx"}]
    print(f"  X_tr: {Xtr_df[feat_cols].shape}, features: {len(feat_cols)}")
    Xtr = Xtr_df[feat_cols].values.astype(np.float32)
    ytr = Xtr_df["up_label"].values.astype(np.int32)
    ftr = Xtr_df["future_20d_return"].values.astype(np.float32)

    # ---- 4. 测试集特征 ----
    print(f"\n[5/8] 抽取测试集特征 ...")
    test_feat_rows = []
    for code, sub in test.groupby("code"):
        f = window_features(sub)
        f["code"] = code
        test_feat_rows.append(f)
    Xte_df = pd.DataFrame(test_feat_rows)
    for c in feat_cols:
        if c not in Xte_df.columns:
            Xte_df[c] = 0.0
    Xte = Xte_df[feat_cols].values.astype(np.float32)
    print(f"  X_te: {Xte.shape}")

    # ---- 5. OOF 训练 ----
    print(f"\n[6/8] OOF 训练 {n_splits} 折 ...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = {k: np.zeros(len(Xtr)) for k in ['lgb', 'xgb', 'mlp', 'lr', 'nb']}
    test_pred = {k: np.zeros(len(Xte)) for k in ['lgb', 'xgb', 'mlp', 'lr', 'nb']}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
        print(f"  -- fold {fold+1}/{n_splits} --")
        X_tr, y_tr = Xtr[tr_idx], ytr[tr_idx]
        X_va, y_va = Xtr[va_idx], ytr[va_idx]
        X_tr_s = np.nan_to_num(X_tr, nan=0.0)
        X_va_s = np.nan_to_num(X_va, nan=0.0)
        X_te_s = np.nan_to_num(Xte, nan=0.0)

        m = train_lgb(X_tr_s, y_tr, X_va_s, y_va, num_boost=num_boost)
        oof['lgb'][va_idx] = m.predict(X_va_s, num_iteration=m.best_iteration)
        test_pred['lgb'] += m.predict(X_te_s, num_iteration=m.best_iteration) / n_splits

        m = train_xgb(X_tr_s, y_tr, X_va_s, y_va, num_boost=num_boost)
        oof['xgb'][va_idx] = m.predict(xgb.DMatrix(X_va_s))
        test_pred['xgb'] += m.predict(xgb.DMatrix(X_te_s)) / n_splits

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

    print("\n[7/8] OOF 评估")
    metrics = {}
    for name, o in oof.items():
        m = {
            "auc": float(roc_auc_score(ytr, o)),
            "brier": float(brier_score_loss(ytr, o)),
            "logloss": float(log_loss(ytr, o)),
        }
        metrics[name] = m
        print(f"  {name}: AUC={m['auc']:.4f}, Brier={m['brier']:.4f}, LogLoss={m['logloss']:.4f}")

    # ---- 6. Stacking ----
    print("\n[6.5/8] Stacking meta learner ...")
    P_oof = np.stack([oof[k] for k in ['lgb', 'xgb', 'mlp', 'lr', 'nb']], axis=1)
    P_test = np.stack([test_pred[k] for k in ['lgb', 'xgb', 'mlp', 'lr', 'nb']], axis=1)
    meta = LogisticRegression(C=1.0, max_iter=2000)
    meta.fit(P_oof, ytr)
    oof_meta = meta.predict_proba(P_oof)[:, 1]
    test_meta = meta.predict_proba(P_test)[:, 1]
    metrics["stack"] = {
        "auc": float(roc_auc_score(ytr, oof_meta)),
        "brier": float(brier_score_loss(ytr, oof_meta)),
        "logloss": float(log_loss(ytr, oof_meta)),
    }
    print(f"  stack: AUC={metrics['stack']['auc']:.4f}, Brier={metrics['stack']['brier']:.4f}")
    print(f"  Coef: {dict(zip(['lgb','xgb','mlp','lr','nb'], [round(float(c),3) for c in meta.coef_[0]]))}")

    # ---- 7. Isotonic 校准 ----
    print("\n[7/8] Isotonic 概率校准 ...")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(oof_meta, ytr)
    oof_calib = iso.predict(oof_meta)
    test_calib = iso.predict(test_meta)
    metrics["calib"] = {
        "auc": float(roc_auc_score(ytr, oof_calib)),
        "brier": float(brier_score_loss(ytr, oof_calib)),
        "logloss": float(log_loss(ytr, oof_calib)),
    }
    print(f"  calib: AUC={metrics['calib']['auc']:.4f}, Brier={metrics['calib']['brier']:.4f}")

    # ---- 8. Top-K 友好后处理 ----
    print("\n[8/8] Top-K 友好后处理（alpha 搜索） ...")

    def topk_soft(p, alpha=0.0):
        if alpha == 0.0:
            return p
        return 1.0 / (1.0 + np.exp(-alpha * (p - 0.5)))

    best_alpha = 0.0
    best_brier = metrics["calib"]["brier"]
    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        p_aug = topk_soft(oof_calib, alpha=a)
        b = brier_score_loss(ytr, p_aug)
        if b < best_brier:
            best_brier = b
            best_alpha = a
    print(f"  best alpha = {best_alpha}, OOF Brier = {best_brier:.4f}")
    test_final = topk_soft(test_calib, alpha=best_alpha)
    oof_final = topk_soft(oof_calib, alpha=best_alpha)
    metrics["final"] = {
        "auc": float(roc_auc_score(ytr, oof_final)),
        "brier": float(brier_score_loss(ytr, oof_final)),
        "logloss": float(log_loss(ytr, oof_final)),
    }

    # ---- Top-5 模拟 ----
    rng = np.random.default_rng(0)
    n_sim = 2000
    n = len(oof_final)
    hits_baseline = []
    hits_model = []
    for _ in range(n_sim):
        idx = rng.integers(0, n, size=n)
        p_ = oof_final[idx]
        y_ = ytr[idx]
        order = np.argsort(-p_)[:5]
        hits_model.append(np.mean(y_[order]))
        # baseline: 随机选 5 只
        rand = rng.choice(n, size=5, replace=False)
        hits_baseline.append(np.mean(y_[rand]))
    print(f"  baseline top5 hit ratio = {np.mean(hits_baseline):.4f}")
    print(f"  model top5 hit ratio    = {np.mean(hits_model):.4f}")
    print(f"  lift                    = {np.mean(hits_model) - np.mean(hits_baseline):+.4f}")

    # ---- 9. 写提交 ----
    print("\n[9/9] 写提交文件 ...")
    sub = pd.DataFrame({
        "code": Xte_df["code"].values,
        "up_factor": test_final,
    }).sort_values("code").reset_index(drop=True)
    out_path = SUB / out_csv_name
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  -> {out_path}")
    print(sub.head())
    print(sub["up_factor"].describe())

    # 保存模型与校准器
    model_dir = OUT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(model_dir / "stacking_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    with open(model_dir / "feat_cols.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False)

    # ---- 10. 写报告 ----
    report = {
        "config": {
            "n_stocks": n_stocks,
            "n_per_stock": n_per_stock,
            "min_history": min_history,
            "n_splits": n_splits,
            "num_boost": num_boost,
            "context_len": CONTEXT_LEN,
            "target_len": TARGET_LEN,
        },
        "metrics": metrics,
        "best_alpha": float(best_alpha),
        "top5_hit": {
            "baseline": float(np.mean(hits_baseline)),
            "model": float(np.mean(hits_model)),
            "lift": float(np.mean(hits_model) - np.mean(hits_baseline)),
        },
        "elapsed_sec": float(time.time() - t0),
    }
    with open(OUT / "pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n[done] report ->", OUT / "pipeline_report.json")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_stocks", type=int, default=2000)
    p.add_argument("--n_per_stock", type=int, default=5)
    p.add_argument("--min_history", type=int, default=60)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--num_boost", type=int, default=1500)
    p.add_argument("--out", type=str, default="submission.csv")
    args = p.parse_args()
    main(
        n_stocks=args.n_stocks,
        n_per_stock=args.n_per_stock,
        min_history=args.min_history,
        n_splits=args.n_splits,
        num_boost=args.num_boost,
        out_csv_name=args.out,
    )
