"""
wf_eval.py
==========
关键诊断：在最近 200 天上做 walk-forward 评估，模拟"测试集"评估

策略：
- 训练：DAY_0001-2694 的样本
- 验证：DAY_2695-2794 的样本（最接近"测试集"时点）
- 用 v6 / v8b / v9 三种 up_factor 生成方式比较
"""
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(r'E:\智能量化投资策略建模挑战赛\code')
sys.path.insert(0, str(ROOT))

from adaptivepath.window_features_v2 import window_features_v2

DATA = Path(r'E:\智能量化投资策略建模挑战赛')
TRAIN_CSV = DATA / "train" / "train.csv"
TEST_CSV = DATA / "test" / "test.csv"
OUT = DATA / "output"
SUB = DATA / "submission" / "output"


def main():
    t0 = time.time()
    print("=" * 70)
    print("[WF EVAL] 关键诊断：walk-forward 评估")
    print("=" * 70)

    # === 1. 加载 ===
    print("\n[1] 加载 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train['date_idx'] = train['date'].str.replace('DAY_', '', regex=False).astype(int)

    # === 2. 构造训练样本 + 验证样本 ===
    # 训练: DAY_0001-2694 的 (20+20) 窗口（对应"未来 20 日"在 2620-2714 之间）
    # 验证: DAY_2695-2794 的 (20+20) 窗口（对应"未来 20 日"在 2715-2794 之间）
    # 这模拟"测试集"在 2795 之后的情况
    print("\n[2] 构造 walk-forward 样本 ...")
    train_sorted = train.sort_values(['code', 'date']).reset_index(drop=True)
    code_to_data = {c: g.reset_index(drop=True) for c, g in train_sorted.groupby('code')}

    rng = np.random.default_rng(42)

    def make_windows(start_day_min, start_day_max, n_per_stock=3):
        """在 [start_day_min, start_day_max] 范围内取每只股票的 n 个 (20+20) 窗口"""
        rows = []
        for code, data in code_to_data.items():
            n = len(data)
            if n < 80:
                continue
            for s_start in range(start_day_min, min(start_day_max, n - 40)):
                ctx = data.iloc[s_start:s_start+20]
                tgt = data.iloc[s_start+20:s_start+40]
                if len(ctx) < 20 or len(tgt) < 20:
                    continue
                ctx_start = ctx['date'].iloc[0]
                ctx_end = ctx['date'].iloc[-1]
                tgt_start = tgt['date'].iloc[0]
                tgt_end = tgt['date'].iloc[-1]
                tgt_start_close = tgt['close'].iloc[0]
                tgt_end_close = tgt['close'].iloc[-1]
                future_20d_return = tgt_end_close / (tgt_start_close + 1e-9) - 1.0
                rows.append({
                    'code': code,
                    'ctx_start': ctx_start,
                    'ctx_end': ctx_end,
                    'tgt_start': tgt_start,
                    'tgt_end': tgt_end,
                    'future_20d_return': float(future_20d_return),
                    'up_label': int(future_20d_return > 0),
                })
        return pd.DataFrame(rows)

    # 训练样本: ctx 在 DAY_0001-2494
    train_windows = make_windows(40, 2495, n_per_stock=3)
    print(f"  训练样本 (DAY_0001-2494): {len(train_windows)}")

    # 验证样本: ctx 在 DAY_2495-2774（对应的 tgt 未来 20 日在 2515-2794）
    val_windows = make_windows(2495, 2775, n_per_stock=2)
    print(f"  验证样本 (DAY_2495-2774): {len(val_windows)}")

    # === 3. 提取训练/验证特征 ===
    print("\n[3] 提取特征 ...")
    def extract(windows, name="train"):
        rows = []
        for i, row in windows.iterrows():
            data = code_to_data.get(row['code'])
            if data is None:
                continue
            ctx = data[(data['date'] >= row['ctx_start']) & (data['date'] <= row['ctx_end'])]
            if len(ctx) < 20:
                continue
            f = window_features_v2(ctx)
            f['code'] = row['code']
            f['future_20d_return'] = row['future_20d_return']
            f['up_label'] = row['up_label']
            rows.append(f)
            if (i + 1) % 10000 == 0:
                print(f"  {name} feat {i+1}/{len(windows)}")
        return pd.DataFrame(rows)

    train_feat = extract(train_windows, "train")
    val_feat = extract(val_windows, "val")
    print(f"  train: {train_feat.shape}, val: {val_feat.shape}")

    # === 4. 准备特征 ===
    common = [c for c in val_feat.columns if c in train_feat.columns and c not in ('code', 'up_label', 'future_20d_return') and train_feat[c].dtype != object]
    print(f"  共同特征: {len(common)}")
    Xtr = train_feat[common].fillna(0).values
    Xva = val_feat[common].fillna(0).values
    ytr_cls = train_feat['up_label'].values
    yva_cls = val_feat['up_label'].values
    ytr_reg = train_feat['future_20d_return'].values
    yva_reg = val_feat['future_20d_return'].values

    sc = StandardScaler()
    Xtr_n = sc.fit_transform(Xtr)
    Xva_n = sc.transform(Xva)

    # === 5. 评估各种 up_factor 生成方式 ===
    print("\n[5] 评估各种 up_factor ...")

    # 5.1 全 0.5
    p_05 = np.full(len(yva_cls), 0.5)

    # 5.2 反转动量
    val_feat_recent = val_feat.copy()
    rev_5 = rankdata(-val_feat_recent['m_logret_5'].fillna(0).values) / len(val_feat_recent)
    rev_10 = rankdata(-val_feat_recent['m_logret_10'].fillna(0).values) / len(val_feat_recent)
    rev_20 = rankdata(-val_feat_recent['m_logret_20'].fillna(0).values) / len(val_feat_recent)
    p_rev = (rev_5 + rev_10 + rev_20) / 3

    # 5.3 LightGBM
    p = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200,
        "feature_fraction": 0.6, "bagging_fraction": 0.6, "bagging_freq": 5,
        "lambda_l1": 1.0, "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1,
    }
    train_set = lgb.Dataset(Xtr_n, label=ytr_cls)
    val_set = lgb.Dataset(Xva_n, label=yva_cls, reference=train_set)
    m = lgb.train(p, train_set, num_boost_round=1500,
                  valid_sets=[train_set, val_set], valid_names=["train", "valid"],
                  callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100, verbose=False)])
    p_lgb = m.predict(Xva_n, num_iteration=m.best_iteration)

    # 5.4 KNN (k=100)
    nn = NearestNeighbors(n_neighbors=100, metric='cosine', n_jobs=-1)
    nn.fit(Xtr_n)
    dists, idxs = nn.kneighbors(Xva_n)
    w = 1.0 / (dists + 1e-6)
    w = w / w.sum(axis=1, keepdims=True)
    p_knn_ret = np.zeros(len(Xva_n))
    for j in range(len(Xva_n)):
        p_knn_ret[j] = np.sum(w[j] * ytr_reg[idxs[j]])
    center = np.median(ytr_reg)
    p_knn = 1.0 / (1.0 + np.exp(-(p_knn_ret - center) * 10))
    p_knn = (p_knn - p_knn.min()) / (p_knn.max() - p_knn.min() + 1e-9)

    # 5.5 集成 v6 风格 (0.5 LGB + 0.5 反转)
    p_v6 = 0.5 * rankdata(p_lgb) / len(p_lgb) + 0.5 * p_rev

    # 5.6 集成 v8b 风格 (0.4 KNN + 0.6 反转)
    p_v8b = 0.4 * p_knn + 0.6 * p_rev

    # 5.7 v9 风格 (0.5 v6 + 0.5 v8b)
    p_v9 = 0.5 * rankdata(p_v6) / len(p_v6) + 0.5 * rankdata(p_v8b) / len(p_v8b)
    p_v9 = (p_v9 - p_v9.min()) / (p_v9.max() - p_v9.min() + 1e-9)

    # === 6. 评估每个方案 ===
    print("\n[6] 评估（按 AUC 和 Brier）...")
    results = {}
    for name, p in [
        ('all_0.5', p_05),
        ('reversal', p_rev),
        ('LGB', p_lgb),
        ('KNN', p_knn),
        ('v6 (LGB+rev)', p_v6),
        ('v8b (KNN+rev)', p_v8b),
        ('v9 (v6+v8b)', p_v9),
    ]:
        try:
            auc = roc_auc_score(yva_cls, p)
            brier = brier_score_loss(yva_cls, np.clip(p, 1e-6, 1-1e-6))
            sp = spearmanr(p, yva_reg).correlation
            # top-5 命中率
            order = np.argsort(-p)[:5]
            top5_hit = float(np.mean(yva_cls[order]))
            results[name] = {
                'auc': float(auc), 'brier': float(brier),
                'spearman': float(sp), 'top5_hit': top5_hit,
            }
            print(f"  {name:20s}: AUC={auc:.4f}, Brier={brier:.4f}, "
                  f"Spearman={sp:+.4f}, top5_hit={top5_hit:.4f}")
        except Exception as e:
            print(f"  {name}: {e}")

    # 保存
    with open(OUT / "wf_eval.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
