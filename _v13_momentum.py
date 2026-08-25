"""
v13_momentum.py (修正：用 5/10/15 日动量)
"""
import pandas as pd
import numpy as np
from scipy.stats import rankdata

test = pd.read_csv(r'E:\智能量化投资策略建模挑战赛\test\test.csv')
test = test.sort_values(['code', 'date']).reset_index(drop=True)

codes = test['code'].drop_duplicates().sort_values().values
n = len(codes)

# 提取每个资产的特征
logret_3 = np.zeros(n)
logret_5 = np.zeros(n)
logret_10 = np.zeros(n)
logret_15 = np.zeros(n)
vol_5 = np.zeros(n)
vol_10 = np.zeros(n)
ma_diff = np.zeros(n)
streak = np.zeros(n)  # 连涨/连跌

for i, code in enumerate(codes):
    sub = test[test['code'] == code].sort_values('date').reset_index(drop=True)
    close = sub['close'].values
    L = len(close)
    if L < 5:
        continue
    base = close[0]
    # 动量
    logret_3[i] = np.log(close[-1] / close[-4]) if L >= 4 else 0
    logret_5[i] = np.log(close[-1] / close[-6]) if L >= 6 else 0
    logret_10[i] = np.log(close[-1] / close[-11]) if L >= 11 else 0
    logret_15[i] = np.log(close[-1] / close[-16]) if L >= 16 else 0
    # 波动率
    rets = np.diff(close) / close[:-1]
    vol_5[i] = np.std(rets[-5:]) if len(rets) >= 5 else 0
    vol_10[i] = np.std(rets[-10:]) if len(rets) >= 10 else 0
    # 与均线关系
    ma5 = np.mean(close[-5:]) if L >= 5 else close[-1]
    ma10 = np.mean(close[-10:]) if L >= 10 else close[-1]
    ma_diff[i] = close[-1] / ma10 - 1
    # 连涨/连跌
    sign = np.sign(np.diff(close))
    s = 0
    for x in sign:
        if x > 0:
            s = max(1, s + 1) if s > 0 else 1
        elif x < 0:
            s = min(-1, s - 1) if s < 0 else -1
        else:
            s = 0
    streak[i] = s

print(f'Signal stats:')
print(f'  logret_5: mean={logret_5.mean():.4f}, std={logret_5.std():.4f}')
print(f'  logret_10: mean={logret_10.mean():.4f}, std={logret_10.std():.4f}')
print(f'  logret_15: mean={logret_15.mean():.4f}, std={logret_15.std():.4f}')

# Rank normalization
def rank_norm(x):
    return (rankdata(x) - 1) / (len(x) - 1)

# 多信号
m3 = rank_norm(logret_3)
m5 = rank_norm(logret_5)
m10 = rank_norm(logret_10)
m15 = rank_norm(logret_15)
ma_rk = rank_norm(ma_diff)
lowvol = 1 - rank_norm(vol_10)
# 反向 streak (跌了多天 -> up_factor 高)
streak_rev = 1 - rank_norm(streak)

# 加权
signal = 0.25 * m5 + 0.25 * m10 + 0.20 * ma_rk + 0.15 * lowvol + 0.15 * streak_rev

# 多个版本
for alpha, name in [(0.05, 'v13_weak'), (0.15, 'v13'), (0.30, 'v13_strong')]:
    up = 0.5 + alpha * (signal - 0.5) * 2
    sub = pd.DataFrame({'code': codes, 'up_factor': up})
    sub = sub.sort_values('code').reset_index(drop=True)
    sub.to_csv(rf'E:\智能量化投资策略建模挑战赛\submission\output\submission_{name}.csv',
               index=False, encoding='utf-8')
    print(f'\n{name} (±{alpha}):')
    print(f'  up_factor: mean={up.mean():.4f}, std={up.std():.4f}, range=[{up.min():.4f}, {up.max():.4f}]')
    print(f'  top-5: {sub.nlargest(5, "up_factor")["code"].tolist()}')
