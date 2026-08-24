"""
trainer.py
==========
多模型训练与集成：
- LightGBM (gb) -- 主要
- XGBoost (xgb) -- 主要
- MLP (mlp) -- 主要
- 朴素贝叶斯 (nb) -- 辅助
- 逻辑回归 (lr) -- 辅助

每个模型训练两套：
- "rank" 版：优化排序（AUC / NDCG），用于 top5 选择
- "calib" 版：优化概率准确性（logloss / Brier），用于 Brier Score
集成时用 Stacking（out-of-fold 预测作为 meta feature）。
"""
from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import lightgbm as lgb
import xgboost as xgb


# ----------------- 单模型定义 -----------------

def train_lgb(X_tr, y_tr, X_va=None, y_va=None, params=None, num_boost=1500):
    """LightGBM 训练（概率 + 排序）"""
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
    if params:
        p.update(params)
    train_set = lgb.Dataset(X_tr, label=y_tr)
    valid_sets = [train_set]
    valid_names = ["train"]
    if X_va is not None and y_va is not None:
        valid_sets.append(lgb.Dataset(X_va, label=y_va, reference=train_set))
        valid_names.append("valid")
    callbacks = [lgb.log_evaluation(0)]
    model = lgb.train(
        p, train_set,
        num_boost_round=num_boost,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


def train_xgb(X_tr, y_tr, X_va=None, y_va=None, num_boost=1500):
    """XGBoost 训练"""
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
    evals = [(dtrain, "train")]
    dval = None
    if X_va is not None and y_va is not None:
        dval = xgb.DMatrix(X_va, label=y_va)
        evals.append((dval, "valid"))
    model = xgb.train(
        p, dtrain, num_boost_round=num_boost,
        evals=evals, early_stopping_rounds=100 if dval is not None else None,
        verbose_eval=False,
    )
    return model


def train_mlp(X_tr, y_tr, X_va=None, y_va=None, epochs=200, hidden=128):
    """MLP（基于 PyTorch，如不可用则 fallback 到 sklearn MLP）"""
    try:
        import torch
        import torch.nn as nn
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(np.nan_to_num(X_tr, nan=0.0))
        X_va_s = sc.transform(np.nan_to_num(X_va, nan=0.0)) if X_va is not None else None

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
        if X_va_s is not None:
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
            if X_va_s is not None:
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
    except Exception as e:
        print(f"[train_mlp] torch not available: {e}, fallback to sklearn")
        from sklearn.neural_network import MLPClassifier
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(np.nan_to_num(X_tr, nan=0.0))
        X_va_s = sc.transform(np.nan_to_num(X_va, nan=0.0)) if X_va is not None else None
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42, early_stopping=True)
        clf.fit(X_tr_s, y_tr)
        return ("mlp_sk", clf, sc)


def predict_mlp(model, X):
    kind, m, sc = model
    X_s = sc.transform(np.nan_to_num(X, nan=0.0))
    if kind == "mlp":
        import torch
        m.eval()
        with torch.no_grad():
            t = torch.tensor(X_s, dtype=torch.float32, device=next(m.parameters()).device)
            out = m(t).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-out))
    else:
        return m.predict_proba(X_s)[:, 1]


# ----------------- 集成 -----------------

class Ensemble:
    """
    简单加权集成（OOF 拟合 meta 权重）。
    基础模型：lgb, xgb, mlp, lr, nb
    优化目标：logloss / Brier
    """
    def __init__(self, base_models: Dict[str, object], feature_names: List[str]):
        self.base_models = base_models
        self.feature_names = feature_names
        # meta 权重（per-model, 可后续学习）
        self.weights = {k: 1.0 / len(base_models) for k in base_models}

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for name, m in self.base_models.items():
            if name == "lgb":
                p = m.predict(X, num_iteration=getattr(m, "best_iteration", None))
            elif name == "xgb":
                # m 是 Booster
                p = m.predict(xgb.DMatrix(X))
            elif name == "mlp":
                p = predict_mlp(m, X)
            elif name == "lr":
                X_s = np.nan_to_num(X, nan=0.0)
                p = m.predict_proba(X_s)[:, 1]
            elif name == "nb":
                X_s = np.nan_to_num(X, nan=0.0)
                p = m.predict_proba(X_s)[:, 1]
            else:
                raise ValueError(f"unknown model {name}")
            preds.append(p)
        P = np.stack(preds, axis=1)  # (N, K)
        w = np.array([self.weights[k] for k in self.base_models])
        return P @ w

    def fit_meta(self, oof_preds: np.ndarray, y: np.ndarray):
        """
        在 OOF 预测上拟合一个简单逻辑回归作为 meta。
        """
        meta = LogisticRegression(max_iter=2000, C=1.0)
        meta.fit(oof_preds, y)
        self.meta = meta
        # 也把 LR 系数作为权重记录
        self.weights = {k: float(meta.coef_[0][i]) for i, k in enumerate(self.base_models)}
        return meta

    def predict_with_meta(self, X: np.ndarray) -> np.ndarray:
        base_preds = []
        for name, m in self.base_models.items():
            if name == "lgb":
                p = m.predict(X, num_iteration=getattr(m, "best_iteration", None))
            elif name == "xgb":
                p = m.predict(xgb.DMatrix(X))
            elif name == "mlp":
                p = predict_mlp(m, X)
            elif name == "lr":
                X_s = np.nan_to_num(X, nan=0.0)
                p = m.predict_proba(X_s)[:, 1]
            elif name == "nb":
                X_s = np.nan_to_num(X, nan=0.0)
                p = m.predict_proba(X_s)[:, 1]
            else:
                continue
            base_preds.append(p)
        P = np.stack(base_preds, axis=1)
        return self.meta.predict_proba(P)[:, 1]
