"""
test_simple_strategies.py
=========================
对比多个朴素但稳健的策略:
1. 全 0.5（字典序前 5）
2. 20 日动量（close[-1]/close[0] 归一化）
3. 5 日动量
4. 反向 20 日动量（卖近期涨的，买近期跌的）
5. 波动率倒数（低波动的资产）
6. v5 状态匹配

看哪个的 up_factor 排名最合理。
"""
import pandas as pd
import numpy as np
import os

DATA = r'E:\智能量化投资策略建模挑战赛'
test = pd.read_csv(os.path.join(DATA, 'test/test.csv'))

# 计算每只资产的 20 日特征
def get_asset_features(sub):
    sub = sub.sort_values('date').reset_index(drop=True)
    n = len(sub)
    base = sub['close'].iloc[0] + 1e-9
    close_n = sub['close'] / base

    features = {}
    features['close_last'] = close_n.iloc[-1]
    features['logret_20'] = np.log(close_n.iloc[-1] / close_n.iloc[0] + 1e-9)
    features['logret_5'] = np.log(close_n.iloc[-1] / close_n.iloc[-5] + 1e-9) if n >= 5 else 0
    features['logret_10'] = np.log(close_n.iloc[-1] / close_n.iloc[-10] + 1e-9) if n >= 10 else 0
    # 波动率
    rets = sub['close'].pct_change().dropna()
    features['vol_20'] = rets.std() if len(rets) > 1 else 0
    # 最大回撤
    cum = close_n.cummax()
    dd = (close_n - cum) / cum
    features['mdd'] = dd.min()
    # 振幅均值
    amp = (sub['high'] - sub['low']) / (sub['close'] + 1e-9)
    features['amp_mean'] = amp.mean()
    # 量比
    vol = sub['volume']
    vol_ma5 = vol.rolling(5, min_periods=1).mean().iloc[-1]
    vol_ma_full = vol.mean()
    features['vol_ratio_5'] = vol_ma5 / (vol_ma_full + 1e-9)
    return features

rows = []
for code, sub in test.groupby('code'):
    f = get_asset_features(sub)
    f['code'] = code
    rows.append(f)

df = pd.DataFrame(rows)
print(f'Assets: {len(df)}')
print(df.describe())

# === 朴素策略 ===

# S1: 全 0.5
s1 = pd.DataFrame({'code': df['code'], 'up_factor': 0.5})

# S2: 20 日动量做 up_factor
df['mom_20'] = df['logret_20']
print(f'\n20 日动量: mean={df["mom_20"].mean():.4f}, std={df["mom_20"].std():.4f}')
print(f'  正数: {(df["mom_20"] > 0).sum()} 只')
print(f'  负数: {(df["mom_20"] < 0).sum()} 只')

# 把动量归一化到 [0, 1]
def normalize_to_prob(x):
    """把 x 归一化到 [0, 1] 区间，0.5 是中位数"""
    sorted_x = np.sort(x)
    n = len(sorted_x)
    # 用 rank 归一化
    ranks = np.argsort(np.argsort(x)) + 1
    return ranks / n

s2 = pd.DataFrame({
    'code': df['code'],
    'up_factor': normalize_to_prob(df['logret_20'].values)
})

# S3: 5 日动量
s3 = pd.DataFrame({
    'code': df['code'],
    'up_factor': normalize_to_prob(df['logret_5'].values)
})

# S4: 反向 20 日动量（买跌卖涨）
s4 = pd.DataFrame({
    'code': df['code'],
    'up_factor': 1 - normalize_to_prob(df['logret_20'].values)
})

# S5: 低波动率优先（认为低波动更可能继续）
inv_vol = -df['vol_20']
s5 = pd.DataFrame({
    'code': df['code'],
    'up_factor': normalize_to_prob(inv_vol.values)
})

# S6: 大振幅优先（认为大振幅更可能反弹）
s6 = pd.DataFrame({
    'code': df['code'],
    'up_factor': normalize_to_prob(df['amp_mean'].values)
})

# S7: v5 当前提交
s7 = pd.read_csv(r'E:\智能量化投资策略建模挑战赛\submission\output\submission_v5.csv')
s7 = s7[['code', 'up_factor']]

# 打印 top-5
for name, s in [('S1-0.5', s1), ('S2-mom20', s2), ('S3-mom5', s3),
                ('S4-rev20', s4), ('S5-lowvol', s5), ('S6-amp', s6),
                ('S7-v5', s7)]:
    top5 = s.sort_values('up_factor', ascending=False).head(5)
    print(f'\n=== {name} ===')
    print(f'  up_factor: mean={s["up_factor"].mean():.3f}, '
          f'std={s["up_factor"].std():.3f}, '
          f'range=[{s["up_factor"].min():.3f}, {s["up_factor"].max():.3f}]')
    print(f'  top-5: {top5["code"].tolist()}')
