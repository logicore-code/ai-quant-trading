"""
run_v8.py
=========
终极版：KNN-based 转导学习

核心思想：
- 每只测试资产有 20 日 OHLCV 形态
- 在训练集中找"最相似"的 (20 日窗口 + 未来 20 日) 样本
- 用这些相似样本的"未来 20 日收益"均值作为预测

这比 LightGBM 简单但更接近"模式迁移"的本质。

实现：
1. 提取测试集 30 维形态特征
2. 提取训练集大量"20 日窗口"的形态特征 + 对应"未来 20 日收益"
3. 对每个测试样本找 top-100 最相似的训练样本
4. 预测 = top-100 训练样本的"未来 20 日收益"均值（转化为 up_factor）
"""
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(r'E:\智能量化投资策略建模挑战赛\code')
sys.path.insert(0, str(ROOT))

from adaptivepath.window_features_v2 import window_features_v2

DATA = Path(r'E:\智能量化投资策略建模挑战赛')
TRAIN_CSV = DATA / "train" / "train.csv"
TEST_CSV = DATA / "test" / "test.csv"
OUT = DATA / "output"
SUB = DATA / "submission" / "output"


def main(out_csv_name="submission_v8.csv"):
    t0 = time.time()
    print("=" * 70)
    print("[FCPFF v8] KNN 转导学习 + 反转信号")
    print("=" * 70)

    # === 1. 加载 ===
    print("\n[1] 加载 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train['date_idx'] = train['date'].str.replace('DAY_', '', regex=False).astype(int)
    print(f"  train: {train.shape}, test: {test.shape}")

    # === 2. 构造训练集：滑动 (20+20) 窗口 ===
    print("\n[2] 构造训练集 (20+20) 窗口 ...")
    train = train.sort_values(['code', 'date']).reset_index(drop=True)

    # 对每只股票，每隔 20 天取一个窗口（避免重叠 + 时间多样）
    train_windows = []
    code_to_data = {c: g.sort_values('date').reset_index(drop=True) for c, g in train.groupby('code')}

    for code, data in code_to_data.items():
        n = len(data)
        if n < 60:  # 至少 20+20+20
            continue
        # 每隔 5 天取一个起点（提高样本多样性）
        for s in range(40, n - 40, 5):
            ctx = data.iloc[s:s+20]
            tgt = data.iloc[s+20:s+40]
            if len(ctx) < 20 or len(tgt) < 20:
                continue
            ctx_start_date = ctx['date'].iloc[0]
            ctx_end_date = ctx['date'].iloc[-1]
            tgt_end_date = tgt['date'].iloc[-1]
            tgt_start_close = tgt['close'].iloc[0]
            tgt_end_close = tgt['close'].iloc[-1]
            future_20d_return = tgt_end_close / (tgt_start_close + 1e-9) - 1.0
            up_label = int(future_20d_return > 0)
            # 关键：记录窗口结束日（用于时间衰减）
            ctx_end_idx = int(data['date_idx'].iloc[s+19])
            train_windows.append({
                'code': code,
                'ctx_start': ctx_start_date,
                'ctx_end': ctx_end_date,
                'ctx_end_idx': ctx_end_idx,
                'future_20d_return': future_20d_return,
                'up_label': up_label,
            })
    windows_df = pd.DataFrame(train_windows)
    print(f"  训练窗口数: {len(windows_df)}")

    # === 3. 提取训练集窗口特征 ===
    print("\n[3] 提取训练窗口特征 ...")
    train_feat_rows = []
    t1 = time.time()
    for i, row in windows_df.iterrows():
        data = code_to_data.get(row['code'])
        if data is None:
            continue
        ctx = data[(data['date'] >= row['ctx_start']) & (data['date'] <= row['ctx_end'])]
        if len(ctx) < 20:
            continue
        f = window_features_v2(ctx)
        f['code'] = row['code']
        f['ctx_end_idx'] = row['ctx_end_idx']
        f['future_20d_return'] = row['future_20d_return']
        f['up_label'] = row['up_label']
        train_feat_rows.append(f)
        if (i + 1) % 5000 == 0:
            print(f"  train feat {i+1}/{len(windows_df)}  ({time.time()-t1:.0f}s)")
    train_feat_df = pd.DataFrame(train_feat_rows)
    print(f"  train features: {train_feat_df.shape}")

    # === 4. 提取测试集特征 ===
    print("\n[4] 提取测试集特征 ...")
    test_feat_rows = []
    for code, sub in test.groupby('code'):
        f = window_features_v2(sub)
        f['code'] = code
        test_feat_rows.append(f)
    test_feat_df = pd.DataFrame(test_feat_rows)
    print(f"  test features: {test_feat_df.shape}")

    # === 5. 特征对齐与标准化 ===
    feat_cols = [c for c in test_feat_df.columns if c != 'code']
    # 训练集也只保留这些
    train_X = train_feat_df[feat_cols].fillna(0).values
    test_X = test_feat_df[feat_cols].fillna(0).values
    print(f"  特征维度: {len(feat_cols)}")

    sc = StandardScaler()
    train_X_n = sc.fit_transform(train_X)
    test_X_n = sc.transform(test_X)

    # === 6. KNN 找最近邻 ===
    print("\n[6] KNN 找最近邻 ...")
    from sklearn.neighbors import NearestNeighbors
    # 用 cosine 距离
    nn = NearestNeighbors(n_neighbors=200, metric='cosine', n_jobs=-1)
    nn.fit(train_X_n)
    dists, idxs = nn.kneighbors(test_X_n)
    print(f"  dists shape: {dists.shape}, idxs shape: {idxs.shape}")

    # === 7. 加权 KNN 预测 ===
    print("\n[7] 加权 KNN 预测 ...")
    # 用距离的倒数作为权重
    weights = 1.0 / (dists + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # 训练集的 future_20d_return 与 up_label
    ytr_reg = train_feat_df['future_20d_return'].values
    ytr_cls = train_feat_df['up_label'].values
    ctx_end_idx_tr = train_feat_df['ctx_end_idx'].values

    # 测试集预测：top-k 训练样本的 future_20d_return 加权均值
    pred_ret = np.zeros(len(test_X_n))
    for i in range(len(test_X_n)):
        knn_idx = idxs[i]
        knn_w = weights[i]
        pred_ret[i] = np.sum(knn_w * ytr_reg[knn_idx])

    # 转成 up_factor：用 sigmoid 中心化
    # 中位收益作为中心
    center = np.median(ytr_reg)
    pred_up = 1.0 / (1.0 + np.exp(-(pred_ret - center) * 10))
    # 拉到 [0, 1]
    pred_up = (pred_up - pred_up.min()) / (pred_up.max() - pred_up.min() + 1e-9)

    # === 8. 反转信号加权（基于训练集发现）===
    print("\n[8] 反转信号加权 ...")
    # 重新计算反转信号（用 rank 归一化）
    rev_5 = rankdata(-test_feat_df['logret_5'].fillna(0).values) / len(test_feat_df)
    rev_20 = rankdata(-test_feat_df['logret_20'].fillna(0).values) / len(test_feat_df)
    rev_streak = rankdata(-test_feat_df['p_final_streak'].fillna(0).values) / len(test_feat_df)
    rev_signal = (rev_5 + rev_20 + rev_streak) / 3

    # === 9. 集成 KNN + 反转 ===
    print("\n[9] 集成 KNN + 反转 ...")
    # 用不同 K 和权重做敏感性分析
    best_pred = None
    best_w = 0.5
    best_score = -np.inf

    # 直接用 0.5 + 0.5 集成
    final_up = 0.5 * pred_up + 0.5 * rev_signal

    # === 10. 写入 ===
    sub = pd.DataFrame({
        'code': test_feat_df['code'].values,
        'up_factor': final_up
    }).sort_values('code').reset_index(drop=True)
    sub.to_csv(SUB / out_csv_name, index=False, encoding='utf-8')
    print(f"\n[Submit] -> {SUB / out_csv_name}")
    print(sub['up_factor'].describe())
    top5 = sub.nlargest(5, 'up_factor')
    print(f"top-5: {top5['code'].tolist()}")

    # 报告
    report = {
        'method': 'KNN-transductive + reversal signal',
        'n_train_windows': len(train_feat_df),
        'n_features': len(feat_cols),
        'submission_stats': {
            'mean': float(sub['up_factor'].mean()),
            'std': float(sub['up_factor'].std()),
            'range': [float(sub['up_factor'].min()), float(sub['up_factor'].max())],
        }
    }
    with open(OUT / "v8_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
