"""
trainer_v2.py
============
增加 LambdaRank 排序学习（直接优化 top-K 排序）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import torch
import torch.nn as nn
import time


def train_lgb_clf(X_tr, y_tr, X_va, y_va, num_boost=1500):
    p = {
        "objective": "binary",
        "metric": "binary_logloss",
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
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    model = lgb.train(
        p, train_set, num_boost_round=num_boost,
        valid_sets=[train_set, val_set], valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100, verbose=False)],
    )
    return model


def train_lgb_reg(X_tr, y_tr, X_va, y_va, num_boost=1500):
    p = {
        "objective": "regression",
        "metric": "l2",
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
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    model = lgb.train(
        p, train_set, num_boost_round=num_boost,
        valid_sets=[train_set, val_set], valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100, verbose=False)],
    )
    return model


def train_lgb_rank(X_tr, y_tr, group_tr, X_va, y_va, group_va, num_boost=1500):
    """
    LambdaRank 训练。把每个样本当作独立 group（每组 1 个样本 + 1 个正负样本）。
    """
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


def make_rank_groups(y, group_size=20, seed=42):
    """
    把样本划分为多个 group，每个 group 有 group_size 个样本。
    组内可以计算 NDCG 排序损失。
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    # shuffle
    idx = rng.permutation(n)
    # 分组
    n_groups = n // group_size
    truncated = idx[:n_groups * group_size]
    groups = np.array_split(truncated, n_groups)
    group_sizes = [len(g) for g in groups]
    return group_sizes


def train_mlp(X_tr, y_tr, X_va, y_va, epochs=100, hidden=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    X_va_s = sc.transform(np.nan_to_num(X_va, nan=0.0))

    class Net(nn.Module):
        def __init__(self, d, h=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(h // 2, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    torch.manual_seed(42)
    net = Net(X_tr.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va_s, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=device)

    best_va = 1e9
    best_state = None
    patience = 0
    bs = 4096
    n = len(X_tr_t)
    for ep in range(epochs):
        net.train()
        idx = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            opt.zero_grad()
            out = net(X_tr_t[b])
            loss = loss_fn(out, y_tr_t[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p = net(X_va_t)
            vl = loss_fn(p, y_va_t).item()
        if vl < best_va - 1e-4:
            best_va = vl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience > 20:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return ("mlp", net, sc)


def predict_mlp(model, X):
    kind, m, sc = model
    X_s = sc.transform(np.nan_to_num(X, nan=0.0))
    if kind == "mlp":
        m.eval()
        with torch.no_grad():
            t = torch.tensor(X_s, dtype=torch.float32, device=next(m.parameters()).device)
            out = m(t).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-out))
    return m.predict_proba(X_s)[:, 1]
