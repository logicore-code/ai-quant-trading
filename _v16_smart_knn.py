"""
v16_smart_knn.py
================
增强版 KNN：用全期训练 + 多 K 集成 + 形态投票

之前 v8b 用最近 200 天 OOF AUC 0.572
v16 用全期 11 年训练 + 多 K + 多指标投票
"""
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT = Path(r'E:\智能量化投资策略建模挑战赛\code')
sys.path.insert(0, str(ROOT))

from adaptivepath.window_features_v2 import window_features_v2

DATA = Path(r'E:\智能量化投资策略建模挑战赛')
TRAIN_CSV = DATA / "train" / "train.csv"
TEST_CSV = DATA / "test" / "test.csv"
OUT = DATA / "output"
SUB = DATA / "submission" / "output"


def main(out_csv_name="submission_v16.csv"):
    t0 = time.time()
    print("=" * 70)
    print("[v16] 增强 KNN: 全期训练 + 多 K 集成 + 形态投票")
    print("=" * 70)

    # === 1. 加载 ===
    print("\n[1] 加载 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train['date_idx'] = train['date'].str.replace('DAY_', '', regex=False).astype(int)
    print(f"  train: {train.shape}, test: {test.shape}")

    # === 2. 构造训练集 (20+20) 窗口 ===
    print("\n[2] 构造训练窗口 (每隔 5 天一个) ...")
    train = train.sort_values(['code', 'date']).reset_index(drop=True)
    code_to_data = {c: g.reset_index(drop=True) for c, g in train.groupby('code')}
    rng = np.random.default_rng(42)

    train_windows = []
    for code, data in code_to_data.items():
        n = len(data)
        if n < 60:
            continue
        # 每只股票 8 个窗口 (每隔 (n-40)/8 天)
        if n - 40 < 8:
            continue
        step = max(1, (n - 40) // 8)
        for s in range(0, n - 40, step):
            ctx = data.iloc[s:s+20]
            tgt = data.iloc[s+20:s+40]
            if len(ctx) < 20 or len(tgt) < 20:
                continue
            tgt_start_close = tgt['close'].iloc[0]
            tgt_end_close = tgt['close'].iloc[-1]
            future_20d_return = tgt_end_close / (tgt_start_close + 1e-9) - 1.0
            train_windows.append({
                'code': code,
                'ctx_start': ctx['date'].iloc[0],
                'ctx_end': ctx['date'].iloc[-1],
                'future_20d_return': float(future_20d_return),
                'up_label': int(future_20d_return > 0),
            })
    windows_df = pd.DataFrame(train_windows)
    print(f"  训练窗口数: {len(windows_df)}")

    # === 3. 提取训练特征 ===
    print("\n[3] 训练特征抽取 ...")
    train_rows = []
    for i, row in windows_df.iterrows():
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
        train_rows.append(f)
        if (i + 1) % 5000 == 0:
            print(f"  train feat {i+1}/{len(windows_df)}")
    train_feat = pd.DataFrame(train_rows)
    print(f"  train features: {train_feat.shape}")

    # === 4. 测试特征 ===
    print("\n[4] 测试特征 ...")
    test_rows = []
    for code, sub in test.groupby('code'):
        f = window_features_v2(sub)
        f['code'] = code
        test_rows.append(f)
    test_feat = pd.DataFrame(test_rows)
    print(f"  test features: {test_feat.shape}")

    # === 5. 特征对齐 ===
    common = [c for c in test_feat.columns if c in train_feat.columns and c not in ('code', 'up_label', 'future_20d_return') and train_feat[c].dtype != object]
    print(f"  共同特征: {len(common)}")
    Xtr = train_feat[common].fillna(0).values
    Xte = test_feat[common].fillna(0).values
    # 替换 inf
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=1e10, neginf=-1e10)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=1e10, neginf=-1e10)
    # 限制极端值
    Xtr = np.clip(Xtr, -1e6, 1e6)
    Xte = np.clip(Xte, -1e6, 1e6)
    ytr_ret = train_feat['future_20d_return'].values
    ytr_cls = train_feat['up_label'].values

    sc = StandardScaler()
    Xtr_n = sc.fit_transform(Xtr)
    Xte_n = sc.transform(Xte)

    # === 6. 多 K KNN 集成 ===
    print("\n[6] 多 K KNN 集成 ...")
    K_list = [50, 100, 200, 500]
    pred_ret_multi = np.zeros(len(Xte))
    for k in K_list:
        nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
        nn.fit(Xtr_n)
        dists, idxs = nn.kneighbors(Xte_n)
        w = 1.0 / (dists + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        for j in range(len(Xte_n)):
            pred_ret_multi[j] += np.sum(w[j] * ytr_ret[idxs[j]])
    pred_ret_multi /= len(K_list)

    # === 7. 转化为 up_factor ===
    center = np.median(ytr_ret)
    pred_up_knn = 1.0 / (1.0 + np.exp(-(pred_ret_multi - center) * 10))
    pred_up_knn = (pred_up_knn - pred_up_knn.min()) / (pred_up_knn.max() - pred_up_knn.min() + 1e-9)

    # === 8. 反转信号 ===
    print("\n[8] 反转信号 ...")
    rev_5 = rankdata(-test_feat['m_logret_5'].fillna(0).values) / len(test_feat)
    rev_10 = rankdata(-test_feat['m_logret_10'].fillna(0).values) / len(test_feat)
    rev_20 = rankdata(-test_feat['m_logret_20'].fillna(0).values) / len(test_feat)
    rev_streak = rankdata(-test_feat['p_final_streak'].fillna(0).values) / len(test_feat)
    rev_signal = (rev_5 + rev_10 + rev_20 + rev_streak) / 4

    # === 9. 集成 KNN + 反转 ===
    print("\n[9] 集成 KNN + 反转 (0.5 + 0.5) ...")
    final_up = 0.5 * pred_up_knn + 0.5 * rev_signal

    # === 10. 强中心化 ±0.3 ===
    final_up = 0.5 + 0.3 * (final_up - 0.5) * 2

    sub = pd.DataFrame({'code': test_feat['code'].values, 'up_factor': final_up})
    sub = sub.sort_values('code').reset_index(drop=True)
    sub.to_csv(SUB / out_csv_name, index=False, encoding='utf-8')
    print(f"\n[Submit] -> {SUB / out_csv_name}")
    print(sub['up_factor'].describe())
    top5 = sub.nlargest(5, 'up_factor')
    print(f"top-5: {top5['code'].tolist()}")

    report = {
        'method': 'multi-K KNN (K=50,100,200,500) + reversal + centralize',
        'K_list': K_list,
        'n_train_windows': len(train_feat),
    }
    with open(OUT / "v16_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
