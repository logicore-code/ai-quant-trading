"""
test_signals.py
==============
在训练集上测试各种朴素信号对"未来 20 日收益"的预测力。
这告诉我们哪些信号是真实有效的。
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr

DATA = r'E:\智能量化投资策略建模挑战赛'
print('Loading train...')
train = pd.read_csv(os.path.join(DATA, 'train/train.csv'), usecols=['code','date','open','high','low','close','volume'])
print(f'  shape: {train.shape}')

# 对每只股票，按时间排序，计算 20 日窗口的"形态特征"和"未来 20 日收益"
print('\nComputing per-stock features...')
train = train.sort_values(['code','date']).reset_index(drop=True)

# 未来 20 日收益
train['next_20d_close'] = train.groupby('code')['close'].shift(-20)
train['future_20d_return'] = train['next_20d_close'] / train['close'] - 1.0

# 20 日动量（已实现）
train['logret_20'] = train.groupby('code')['close'].transform(
    lambda s: np.log(s / s.shift(20))
)
# 5 日动量
train['logret_5'] = train.groupby('code')['close'].transform(
    lambda s: np.log(s / s.shift(5))
)
# 10 日动量
train['logret_10'] = train.groupby('code')['close'].transform(
    lambda s: np.log(s / s.shift(10))
)
# 20 日波动率
train['vol_20'] = train.groupby('code')['close'].transform(
    lambda s: s.pct_change().rolling(20, min_periods=10).std()
)
# 20 日量比
train['vol_ma20'] = train.groupby('code')['volume'].transform(
    lambda s: s.rolling(20, min_periods=1).mean()
)
train['vol_ratio_20'] = train['volume'] / (train['vol_ma20'] + 1e-9)
# 20 日均价
train['close_ma20'] = train.groupby('code')['close'].transform(
    lambda s: s.rolling(20, min_periods=1).mean()
)
train['close_to_ma20'] = train['close'] / (train['close_ma20'] + 1e-9) - 1

# 计算 Spearman 相关系数（衡量排序能力）
print('\n=== 信号对未来 20 日收益的 Spearman 相关性 ===')
# 删除 NaN
df = train[['future_20d_return', 'logret_20', 'logret_5', 'logret_10', 'vol_20', 'vol_ratio_20', 'close_to_ma20']].dropna()
print(f'有效样本: {len(df)}')

for col in ['logret_20', 'logret_5', 'logret_10', 'vol_20', 'vol_ratio_20', 'close_to_ma20']:
    sp, _ = spearmanr(df[col], df['future_20d_return'])
    pe, _ = pearsonr(df[col], df['future_20d_return'])
    print(f'  {col:20s}: Spearman={sp:+.4f}, Pearson={pe:+.4f}')

# 分时段看
print('\n=== 按时间分段: logret_20 -> future_20d_return 相关性 ===')
train['date_idx'] = train['date'].str.replace('DAY_', '').astype(int)
for start in [0, 500, 1000, 1500, 2000, 2500, 2700]:
    end = start + 300
    sub = train[(train['date_idx'] >= start) & (train['date_idx'] < end)]
    sub = sub[['future_20d_return', 'logret_20', 'logret_5', 'logret_10']].dropna()
    if len(sub) > 100:
        sp_20, _ = spearmanr(sub['logret_20'], sub['future_20d_return'])
        sp_5, _ = spearmanr(sub['logret_5'], sub['future_20d_return'])
        sp_10, _ = spearmanr(sub['logret_10'], sub['future_20d_return'])
        print(f'  DAY_{start:04d}-{end:04d}: mom20={sp_20:+.4f}, mom5={sp_5:+.4f}, mom10={sp_10:+.4f}')

# 最后 200 天的相关性
print('\n=== 最后 200 天 (DAY_2594-2794) ===')
sub = train[train['date_idx'] >= 2594]
sub = sub[['future_20d_return', 'logret_20', 'logret_5', 'logret_10', 'vol_20']].dropna()
if len(sub) > 100:
    print(f'  n={len(sub)}')
    for col in ['logret_20', 'logret_5', 'logret_10', 'vol_20']:
        sp, _ = spearmanr(sub[col], sub['future_20d_return'])
        print(f'  {col}: Spearman={sp:+.4f}')

# 反向？
print('\n=== 反向信号 (反转策略) ===')
for col in ['logret_20', 'logret_5', 'logret_10']:
    neg = -sub[col]
    sp_neg, _ = spearmanr(neg, sub['future_20d_return'])
    sp_orig, _ = spearmanr(sub[col], sub['future_20d_return'])
    print(f'  {col}: 正向={sp_orig:+.4f}, 反向={sp_neg:+.4f}')
