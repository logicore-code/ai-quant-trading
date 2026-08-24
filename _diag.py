import pandas as pd
import numpy as np
import os

test = pd.read_csv(r'test\test.csv')
train = pd.read_csv(r'train\train.csv', usecols=['code','date','close','volume'])

# 1. 测试集特征
test_starts = test.groupby('code')['close'].first()
print('Test 特征:')
print(f'  test 首日 close 全为 100? {(test_starts == 100).all()}')
print(f'  test 资产数: {test["code"].nunique()}')
print(f'  test 日期范围: {test["date"].min()} - {test["date"].max()}')

# 2. 训练集最近 20 天的市场状态
train_recent = train[train['date'].isin([f'DAY_{i:04d}' for i in range(2775, 2795)])]
print(f'\n训练集最近 20 天 (DAY_2775-2794):')
print(f'  rows: {len(train_recent)}, codes: {train_recent["code"].nunique()}')
print(f'  close 均值: {train_recent["close"].mean():.2f}')
print(f'  close 标准差: {train_recent["close"].std():.2f}')

# 3. 测试集 vs 训练集最近 20 天的日收益率
test_rets = test.groupby('code')['close'].pct_change().dropna()
train_recent_rets = train_recent.sort_values(['code','date']).groupby('code')['close'].pct_change().dropna()
print(f'\n日收益率:')
print(f'  train (DAY_2775-2794): mean={train_recent_rets.mean():.4f}, std={train_recent_rets.std():.4f}')
print(f'  test  (DAY_0001-0020): mean={test_rets.mean():.4f}, std={test_rets.std():.4f}')

# 4. 训练集整体的"最近 20 天" vs "早期 20 天"
for window in [(1, 20), (200, 220), (500, 520), (1000, 1020), (2000, 2020), (2700, 2720)]:
    s, e = window
    sub = train[train['date'].isin([f'DAY_{i:04d}' for i in range(s, e+1)])]
    rets = sub.sort_values(['code','date']).groupby('code')['close'].pct_change().dropna()
    print(f'  train DAY_{s:04d}-{e:04d}: rets mean={rets.mean():.4f}, std={rets.std():.4f}')
