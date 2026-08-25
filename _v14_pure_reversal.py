"""
v14_pure_reversal.py
====================
纯反转策略：避开机器学习，只用"反转信号"

诊断：
- v4 (LGB集成) 真实 0.018
- v9 (反转+LGB) 真实 -0.01178
- v13 (动量+无LGB) 真实 -0.12621
- → 机器学习部分贡献 0.018
- → 反转信号比动量好 10 倍
- → 纯反转应该比 v9 好（避开 LGB 的过拟合）

策略：
- 多窗口反转（5/10/15 日）
- 多信号集成（动量取反 + 连跌 + 波动率 + 偏离度）
- 强中心化 ±0.3
- 不引入机器学习
"""
import pandas as pd
import numpy as np
from scipy.stats import rankdata

test = pd.read_csv(r'E:\智能量化投资策略建模挑战赛\test\test.csv')
test = test.sort_values(['code', 'date']).reset_index(drop=True)

codes = test['code'].drop_duplicates().sort_values().values
n = len(codes)

# === 提取每只资产的特征 ===
logret_3 = np.zeros(n)
logret_5 = np.zeros(n)
logret_10 = np.zeros(n)
logret_15 = np.zeros(n)
vol_5 = np.zeros(n)
vol_10 = np.zeros(n)
ma_diff = np.zeros(n)  # close/ma10 - 1
streak = np.zeros(n)  # 连涨/连跌（正数=连涨，负数=连跌）

for i, code in enumerate(codes):
    sub = test[test['code'] == code].sort_values('date').reset_index(drop=True)
    close = sub['close'].values
    L = len(close)
    if L < 5:
        continue
    # 动量
    logret_3[i] = np.log(close[-1] / close[-4]) if L >= 4 else 0
    logret_5[i] = np.log(close[-1] / close[-6]) if L >= 6 else 0
    logret_10[i] = np.log(close[-1] / close[-11]) if L >= 11 else 0
    logret_15[i] = np.log(close[-1] / close[-16]) if L >= 16 else 0
    # 波动率
    rets = np.diff(close) / close[:-1]
    vol_5[i] = np.std(rets[-5:]) if len(rets) >= 5 else 0
    vol_10[i] = np.std(rets[-10:]) if len(rets) >= 10 else 0
    # 均线
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

print(f'logret_5: mean={logret_5.mean():.4f}, std={logret_5.std():.4f}')
print(f'logret_10: mean={logret_10.mean():.4f}, std={logret_10.std():.4f}')
print(f'streak: mean={streak.mean():.4f}, std={streak.std():.4f}')

# === Rank normalization ===
def rnk(x):
    return (rankdata(x) - 1) / (len(x) - 1)

# === 多信号反转 ===
# 关键：所有信号都"取反"——过去涨的反而 up_factor 低
r3 = 1 - rnk(logret_3)   # 5 日涨的 up_factor 低
r5 = 1 - rnk(logret_5)
r10 = 1 - rnk(logret_10)
r15 = 1 - rnk(logret_15)
# 远低于均线 = 反弹机会
rev_ma = 1 - rnk(ma_diff)  # 与均线偏离度取反
# 连跌的优先
rev_streak = 1 - rnk(streak)
# 高波动往往预示反转
high_vol = rnk(vol_10)

# === 加权 ===
signal = 0.25 * r5 + 0.20 * r10 + 0.15 * r15 + 0.20 * rev_ma + 0.15 * rev_streak + 0.05 * high_vol

# === 强中心化 ±0.3 ===
up_factor = 0.5 + 0.3 * (signal - 0.5) * 2

print(f'\nv14 (±0.3):')
print(f'  up_factor: mean={up_factor.mean():.4f}, std={up_factor.std():.4f}')
print(f'  range: [{up_factor.min():.4f}, {up_factor.max():.4f}]')
top5_idx = np.argsort(-up_factor)[:5]
print(f'  top-5: {[codes[i] for i in top5_idx]}')

sub = pd.DataFrame({'code': codes, 'up_factor': up_factor})
sub = sub.sort_values('code').reset_index(drop=True)
sub.to_csv(r'E:\智能量化投资策略建模挑战赛\submission\output\submission_v14.csv',
           index=False, encoding='utf-8')

# 也存一个更激进的 ±0.4 版本
up_factor_aggressive = 0.5 + 0.4 * (signal - 0.5) * 2
sub_agg = pd.DataFrame({'code': codes, 'up_factor': up_factor_aggressive})
sub_agg = sub_agg.sort_values('code').reset_index(drop=True)
sub_agg.to_csv(r'E:\智能量化投资策略建模挑战赛\submission\output\submission_v14_agg.csv',
               index=False, encoding='utf-8')

print(f'\nv14_agg (±0.4):')
print(f'  up_factor: mean={up_factor_aggressive.mean():.4f}, std={up_factor_aggressive.std():.4f}')
top5_idx2 = np.argsort(-up_factor_aggressive)[:5]
print(f'  top-5: {[codes[i] for i in top5_idx2]}')

# 弱版本 ±0.2
up_factor_weak = 0.5 + 0.2 * (signal - 0.5) * 2
sub_weak = pd.DataFrame({'code': codes, 'up_factor': up_factor_weak})
sub_weak = sub_weak.sort_values('code').reset_index(drop=True)
sub_weak.to_csv(r'E:\智能量化投资策略建模挑战赛\submission\output\submission_v14_weak.csv',
               index=False, encoding='utf-8')
print(f'\nv14_weak (±0.2) 已生成')
print(f'  top-5: {sub_weak.nlargest(5, "up_factor")["code"].tolist()}')
