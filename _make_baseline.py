"""
make_all_baselines.py
====================
生成多个 baseline 提交备用
"""
import pandas as pd
import numpy as np

# 读 test.csv 拿 code 顺序
test = pd.read_csv(r'E:\智能量化投资策略建模挑战赛\test\test.csv')
codes = test['code'].drop_duplicates().sort_values().reset_index(drop=True)
n = len(codes)
print(f'Total assets: {n}')

# === Baseline 1: 全 0.5 ===
b1 = pd.DataFrame({'code': codes, 'up_factor': 0.5})
b1.to_csv(r'E:\智能量化投资策略建模挑战赛\submission\output\submission_all05.csv', index=False, encoding='utf-8')
print('B1: all 0.5, top-5 by code order:', b1.sort_values('up_factor', ascending=False).head(5)['code'].tolist())

# === Baseline 2: 0.5 + 弱噪声 (避免字典序) ===
np.random.seed(42)
b2 = pd.DataFrame({
    'code': codes,
    'up_factor': 0.5 + np.random.uniform(-0.01, 0.01, n)  # 微小扰动
})
b2.to_csv(r'E:\智能量化投资策略建模挑战赛\submission\output\submission_all05_noise.csv', index=False, encoding='utf-8')
print('B2: 0.5 + tiny noise, top-5:', b2.nlargest(5, 'up_factor')['code'].tolist())

# === Baseline 3: 中性动量 (20 日动量弱权重) ===
test_2 = test.copy()
test_2['logret_20'] = test_2.groupby('code')['close'].transform(lambda s: np.log(s / s.shift(20)))
test_2['logret_5'] = test_2.groupby('code')['close'].transform(lambda s: np.log(s / s.shift(5)))
last_20 = test_2.groupby('code')['logret_20'].last()
last_5 = test_2.groupby('code')['logret_5'].last()
mom = (last_5 + last_20) / 2
mom_rank = (mom.rank() - 1) / (n - 1)  # 0 到 1
# 加一点动量，但中心化在 0.5
b3 = pd.DataFrame({
    'code': codes,
    'up_factor': 0.5 + 0.1 * (mom_rank - 0.5)  # ±0.05
})
b3.to_csv(r'E:\智能量化投资策略建模挑战赛\submission\output\submission_momentum.csv', index=False, encoding='utf-8')
print('B3: momentum weak, top-5:', b3.nlargest(5, 'up_factor')['code'].tolist())

# === Baseline 4: 反转动量 (20 日动量取反) ===
rev_rank = 1 - mom_rank
b4 = pd.DataFrame({
    'code': codes,
    'up_factor': 0.5 + 0.1 * (rev_rank - 0.5)
})
b4.to_csv(r'E:\智能量化投资策略建模挑战赛\submission\output\submission_reversal.csv', index=False, encoding='utf-8')
print('B4: reversal weak, top-5:', b4.nlargest(5, 'up_factor')['code'].tolist())

# === Baseline 5: 低波动率 ===
vols = test_2.groupby('code')['close'].transform(lambda s: s.pct_change().rolling(20, min_periods=5).std()).groupby(test_2['code']).last()
vol_rank = (vols.rank() - 1) / (n - 1)
# 低波动优先 (排名低的)
b5 = pd.DataFrame({
    'code': codes,
    'up_factor': 0.5 + 0.1 * (1 - vol_rank - 0.5)  # 低波动=高分
})
b5.to_csv(r'E:\智能量化投资策略建模挑战赛\submission\output\submission_lowvol.csv', index=False, encoding='utf-8')
print('B5: low vol, top-5:', b5.nlargest(5, 'up_factor')['code'].tolist())

print('\n所有 baseline 已生成')
