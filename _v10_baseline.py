"""
v10_baseline.py
===============
全 0.5 baseline — 最稳的提交
"""
import pandas as pd

# 读取 code 列表
test = pd.read_csv(r'E:\智能量化投资策略建模挑战赛\test\test.csv')
codes = test['code'].drop_duplicates().sort_values().reset_index(drop=True)

# 全 0.5
sub = pd.DataFrame({
    'code': codes,
    'up_factor': 0.5
})
sub.to_csv(r'E:\智能量化投资策略建模挑战赛\submission\output\submission.csv', index=False, encoding='utf-8')
print(f'生成 {len(sub)} 行全 0.5 提交')
print(sub.head())
print(sub['up_factor'].describe())
