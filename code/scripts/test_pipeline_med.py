"""
medium pipeline test
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ROOT = Path(r'E:\智能量化投资策略建模挑战赛\code')
sys.path.insert(0, str(ROOT))

from adaptivepath.dataset import build_truncated_dataset
from adaptivepath.window_features import window_features
from adaptivepath.trainer import train_lgb, train_xgb, train_mlp, predict_mlp

DATA = ROOT.parent
OUT = DATA / "output"
SUB = DATA / "submission" / "output"

t0 = time.time()
print("[1] 加载")
train = pd.read_csv(DATA / "train" / "train.csv")
test = pd.read_csv(DATA / "test" / "test.csv")

print("\n[2] 选 1000 只训练股票")
counts = train.groupby("code").size()
long_codes = counts[counts >= 200].index[:1000].tolist()
train_sub = train[train["code"].isin(long_codes)].copy()
print(f"  selected {len(long_codes)} stocks, rows: {len(train_sub)}")

print("\n[3] 构造截断样本 (每只 2 个)")
samples = build_truncated_dataset(train_sub, n_per_stock=2, min_history=100, seed=42)
print(f"  samples: {len(samples)}")

print("\n[4] 窗口特征")
train_feat_rows = []
for i, row in samples.iterrows():
    sub = train_sub[(train_sub["code"] == row["code"]) &
                    (train_sub["date"] >= row["ctx_start"]) &
                    (train_sub["date"] <= row["ctx_end_date"])]
    if len(sub) < 50:
        continue
    f = window_features(sub)
    f["up_label"] = row["up_label"]
    f["future_20d_return"] = row["future_20d_return"]
    train_feat_rows.append(f)
    if (i + 1) % 1000 == 0:
        print(f"  feat {i+1}/{len(samples)}")
Xtr_df = pd.DataFrame(train_feat_rows)
feat_cols = [c for c in Xtr_df.columns if c not in {"up_label", "future_20d_return"}]
print(f"  X_tr: {Xtr_df[feat_cols].shape}")
Xtr = Xtr_df[feat_cols].values.astype(np.float32)
ytr = Xtr_df["up_label"].values.astype(np.int32)

print("\n[5] 测试集特征")
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

print("\n[6] OOF 训练 5 折")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
oof = {k: np.zeros(len(Xtr)) for k in ['lgb','xgb','mlp','lr','nb']}
test_pred = {k: np.zeros(len(Xte)) for k in ['lgb','xgb','mlp','lr','nb']}

for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
    print(f"  -- fold {fold+1}/{n_splits} --")
    X_tr, y_tr = Xtr[tr_idx], ytr[tr_idx]
    X_va, y_va = Xtr[va_idx], ytr[va_idx]
    X_tr_s = np.nan_to_num(X_tr, nan=0.0)
    X_va_s = np.nan_to_num(X_va, nan=0.0)
    X_te_s = np.nan_to_num(Xte, nan=0.0)

    m = train_lgb(X_tr_s, y_tr, X_va_s, y_va, num_boost=800)
    oof['lgb'][va_idx] = m.predict(X_va_s, num_iteration=m.best_iteration)
    test_pred['lgb'] += m.predict(X_te_s, num_iteration=m.best_iteration) / n_splits

    m = train_xgb(X_tr_s, y_tr, X_va_s, y_va, num_boost=800)
    oof['xgb'][va_idx] = m.predict(xgb.DMatrix(X_va_s))
    test_pred['xgb'] += m.predict(xgb.DMatrix(X_te_s)) / n_splits

    m = train_mlp(X_tr_s, y_tr, X_va_s, y_va, epochs=80)
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

print("\n[7] OOF 评估")
for name, o in oof.items():
    print(f"  {name}: AUC={roc_auc_score(ytr, o):.4f}, Brier={brier_score_loss(ytr, o):.4f}")

print("\n[8] Stacking")
P_oof = np.stack([oof[k] for k in ['lgb','xgb','mlp','lr','nb']], axis=1)
P_test = np.stack([test_pred[k] for k in ['lgb','xgb','mlp','lr','nb']], axis=1)
meta = LogisticRegression(C=1.0, max_iter=2000)
meta.fit(P_oof, ytr)
oof_meta = meta.predict_proba(P_oof)[:, 1]
test_meta = meta.predict_proba(P_test)[:, 1]
print(f"  stack: AUC={roc_auc_score(ytr, oof_meta):.4f}, Brier={brier_score_loss(ytr, oof_meta):.4f}")
print(f"  Coef: {dict(zip(['lgb','xgb','mlp','lr','nb'], meta.coef_[0]))}")

print("\n[9] Isotonic 校准")
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
iso.fit(oof_meta, ytr)
oof_calib = iso.predict(oof_meta)
test_calib = iso.predict(test_meta)
print(f"  calib: AUC={roc_auc_score(ytr, oof_calib):.4f}, Brier={brier_score_loss(ytr, oof_calib):.4f}")

print("\n[10] 写提交")
sub = pd.DataFrame({
    "code": Xte_df["code"].values,
    "up_factor": test_calib,
}).sort_values("code").reset_index(drop=True)
sub.to_csv(SUB / "submission_med.csv", index=False, encoding="utf-8")
print(sub.head())
print(f"  -> {SUB / 'submission_med.csv'}")
print(f"  time: {time.time()-t0:.1f}s")
