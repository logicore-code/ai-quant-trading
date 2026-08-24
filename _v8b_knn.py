"""
v8b_knn.py
==========
轻量 KNN 转导：用最少的训练样本验证反转信号的有效性。

策略：
- 只用最近 200 天的样本
- 每只股票每 20 天取一个 (20+20) 窗口
- 总样本约 4000-5000（之前 v8 数据量太大）
- KNN + 反转信号集成
"""
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

ROOT = Path(r'E:\智能量化投资策略建模挑战赛\code')
sys.path.insert(0, str(ROOT))

from adaptivepath.window_features_v2 import window_features_v2

DATA = Path(r'E:\智能量化投资策略建模挑战赛')
TRAIN_CSV = DATA / "train" / "train.csv"
TEST_CSV = DATA / "test" / "test.csv"
OUT = DATA / "output"
SUB = DATA / "submission" / "output"


def main(out_csv_name="submission_v8b.csv"):
    t0 = time.time()
    print("=" * 70)
    print("[v8b] 轻量 KNN + 反转信号")
    print("=" * 70)

    # === 1. 加载 ===
    print("\n[1] 加载 ...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train['date_idx'] = train['date'].str.replace('DAY_', '', regex=False).astype(int)
    print(f"  train: {train.shape}, test: {test.shape}")

    # === 2. 用最近 200 天 (DAY_2594-2794)，每只股票取 5 个 (20+20) 窗口 ===
    print("\n[2] 构造最近 200 天的训练窗口 ...")
    train_recent = train[train['date_idx'] >= 2594].copy()
    train_recent = train_recent.sort_values(['code', 'date']).reset_index(drop=True)
    code_to_data = {c: g.reset_index(drop=True) for c, g in train_recent.groupby('code')}

    rng = np.random.default_rng(42)
    train_windows = []
    for code, data in code_to_data.items():
        n = len(data)
        if n < 60:
            continue
        # 5 个窗口，起点随机分布在前 60% 区间
        starts = rng.choice(range(0, n - 40), size=min(5, n - 40), replace=False)
        starts = sorted(starts)
        for s in starts:
            ctx = data.iloc[s:s+20]
            tgt = data.iloc[s+20:s+40]
            if len(ctx) < 20 or len(tgt) < 20:
                continue
            ctx_start_close = ctx['close'].iloc[0]
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
    print("\n[3] 提取训练特征 ...")
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
        f['future_20d_return'] = row['future_20d_return']
        f['up_label'] = row['up_label']
        train_feat_rows.append(f)
        if (i + 1) % 2000 == 0:
            print(f"  train feat {i+1}/{len(windows_df)}  ({time.time()-t1:.0f}s)")
    train_feat_df = pd.DataFrame(train_feat_rows)
    print(f"  train features: {train_feat_df.shape}")

    # === 4. 提取测试特征 ===
    print("\n[4] 提取测试特征 ...")
    test_feat_rows = []
    for code, sub in test.groupby('code'):
        f = window_features_v2(sub)
        f['code'] = code
        test_feat_rows.append(f)
    test_feat_df = pd.DataFrame(test_feat_rows)
    print(f"  test features: {test_feat_df.shape}")

    # === 5. 特征对齐 ===
    common = [c for c in test_feat_df.columns if c in train_feat_df.columns and c not in ('code', 'up_label', 'future_20d_return')]
    common = [c for c in common if train_feat_df[c].dtype != object]
    print(f"  共同数值特征: {len(common)}")

    Xtr = train_feat_df[common].fillna(0).values
    Xte = test_feat_df[common].fillna(0).values
    ytr_reg = train_feat_df['future_20d_return'].values
    ytr_cls = train_feat_df['up_label'].values

    sc = StandardScaler()
    Xtr_n = sc.fit_transform(Xtr)
    Xte_n = sc.transform(Xte)

    # === 6. OOF 评估 KNN ===
    print("\n[6] OOF 评估 KNN (5 折) ...")
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_reg = np.zeros(len(Xtr_n))
    for tr_idx, va_idx in kf.split(Xtr_n):
        nn = NearestNeighbors(n_neighbors=100, metric='cosine', n_jobs=-1)
        nn.fit(Xtr_n[tr_idx])
        dists, idxs = nn.kneighbors(Xtr_n[va_idx])
        w = 1.0 / (dists + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        for j, vi in enumerate(va_idx):
            oof_reg[vi] = np.sum(w[j] * ytr_reg[tr_idx[idxs[j]]])

    # 评估 OOF
    from scipy.stats import spearmanr
    sp, _ = spearmanr(oof_reg, ytr_reg)
    # 转化为 up_label
    center = np.median(ytr_reg)
    oof_up = 1.0 / (1.0 + np.exp(-(oof_reg - center) * 10))
    auc = roc_auc_score(ytr_cls, oof_up)
    brier = brier_score_loss(ytr_cls, np.clip(oof_up, 1e-6, 1-1e-6))
    print(f"  KNN OOF: Spearman vs ret = {sp:.4f}, AUC = {auc:.4f}, Brier = {brier:.4f}")

    # === 7. 在全部训练数据上 KNN 预测测试集 ===
    print("\n[7] KNN 预测测试集 ...")
    nn = NearestNeighbors(n_neighbors=100, metric='cosine', n_jobs=-1)
    nn.fit(Xtr_n)
    dists, idxs = nn.kneighbors(Xte_n)
    w = 1.0 / (dists + 1e-6)
    w = w / w.sum(axis=1, keepdims=True)
    pred_ret = np.zeros(len(Xte_n))
    for j in range(len(Xte_n)):
        pred_ret[j] = np.sum(w[j] * ytr_reg[idxs[j]])
    pred_up_knn = 1.0 / (1.0 + np.exp(-(pred_ret - center) * 10))
    pred_up_knn = (pred_up_knn - pred_up_knn.min()) / (pred_up_knn.max() - pred_up_knn.min() + 1e-9)

    # === 8. 反转信号 ===
    print("\n[8] 反转信号 ...")
    # 找反转信号用的特征名（window_features_v2 用 m_logret_X）
    rev_cols = {
        '5': [c for c in test_feat_df.columns if 'm_logret_5' in c or 'm_ret_5' in c or 'logret_5' in c],
        '10': [c for c in test_feat_df.columns if 'm_logret_10' in c or 'm_ret_10' in c or 'logret_10' in c],
        '20': [c for c in test_feat_df.columns if 'm_logret_20' in c or 'm_ret_20' in c or 'logret_20' in c],
    }
    print(f"  rev cols: {rev_cols}")
    rev_signals = []
    for w in ['5', '10', '20']:
        cs = rev_cols[w]
        if cs:
            # 取第一个匹配的列
            x = -test_feat_df[cs[0]].fillna(0).values
            rev_signals.append(rankdata(x) / len(test_feat_df))
        else:
            rev_signals.append(np.full(len(test_feat_df), 0.5))
    # streak
    streak_col = 'p_final_streak' if 'p_final_streak' in test_feat_df.columns else None
    if streak_col:
        rev_signals.append(rankdata(-test_feat_df[streak_col].fillna(0).values) / len(test_feat_df))
    # close_to_ma
    ma_col = 'close_z_20' if 'close_z_20' in test_feat_df.columns else 'pos_in_20'
    if ma_col in test_feat_df.columns:
        rev_signals.append(rankdata(-test_feat_df[ma_col].fillna(0).values) / len(test_feat_df))
    rev_signal = np.mean(rev_signals, axis=0)

    # === 9. 集成 KNN + 反转 ===
    print("\n[9] 集成 (KNN 0.4 + 反转 0.6) ...")
    final_up = 0.4 * pred_up_knn + 0.6 * rev_signal

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
        'method': 'KNN(transductive) + reversal signal',
        'n_train_windows': len(train_feat_df),
        'n_features': len(common),
        'oof_metrics': {
            'knn_auc': float(auc),
            'knn_brier': float(brier),
            'knn_spearman': float(sp),
        },
        'submission_stats': {
            'mean': float(sub['up_factor'].mean()),
            'std': float(sub['up_factor'].std()),
            'range': [float(sub['up_factor'].min()), float(sub['up_factor'].max())],
        }
    }
    with open(OUT / "v8b_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
