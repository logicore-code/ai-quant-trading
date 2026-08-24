"""
analyze_v4.py
============
分析 v4 的 up_factor 有什么问题。
对比几个 baseline:
- 全 0.5
- 随机 up_factor
- 极端 up_factor (0/1)
- v4 当前
- v5 当前

提交多个版本看哪个真实分数最高。
"""
import pandas as pd
import numpy as np
import os

# 读取 v4 和 v5 的提交
v4 = pd.read_csv(r'submission\output\submission.csv')
v5 = pd.read_csv(r'submission\output\submission_v5.csv')
print('v4 up_factor:')
print(v4['up_factor'].describe())
print()
print('v5 up_factor:')
print(v5['up_factor'].describe())
print()

# 看 v4 vs v5 的相关性
v4_v5 = v4.merge(v5, on='code', suffixes=('_v4', '_v5'))
print('v4 vs v5 correlation:')
print(v4_v5[['up_factor_v4', 'up_factor_v5']].corr())
print()

# 看 v4 的 top-5
v4_top5 = v4.nlargest(5, 'up_factor')
print('v4 top-5:')
print(v4_top5)
print()
v5_top5 = v5.nlargest(5, 'up_factor')
print('v5 top-5:')
print(v5_top5)
print()

# 创建几个 baseline
all_codes = v4['code'].values
n = len(all_codes)
print(f'Total assets: {n}')

# 1. 全 0.5
b1 = pd.DataFrame({'code': all_codes, 'up_factor': 0.5})
# 全 0.5 时，按 code 字典序选 top-5
b1_top5 = b1.sort_values('up_factor', ascending=False).head(5)
print('Baseline 全 0.5 top-5 (按字典序):')
print(b1_top5)
print()

# 2. 强信号：让极端 up_factor 集中在某些资产上
# 用测试集本身的"动量"作为 up_factor：m_ret_20 > 0
# 先简单模拟：随机给 up_factor
np.random.seed(42)
b2 = pd.DataFrame({
    'code': all_codes,
    'up_factor': np.random.uniform(0.3, 0.7, n)
})
print('Baseline 随机 up_factor:')
print(b2['up_factor'].describe())
