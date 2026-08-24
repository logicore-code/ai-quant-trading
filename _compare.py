import pandas as pd
import numpy as np

# 看每个版本的 up_factor 分布
for name, path in [
    ('v4', r'submission\output\submission_v4_old.csv'),
    ('v6', r'submission\output\submission_v6.csv'),
    ('v8b', r'submission\output\submission_v8b.csv'),
    ('v9', r'submission\output\submission_v9.csv'),
    ('v11', r'submission\output\submission_v11.csv'),
    ('current', r'submission\output\submission.csv'),
]:
    df = pd.read_csv(path)
    up = df['up_factor']
    top5 = df.nlargest(5, 'up_factor')['code'].tolist()
    print(f'{name:8s}: mean={up.mean():.4f}, std={up.std():.4f}, '
          f'range=[{up.min():.4f}, {up.max():.4f}], top5={top5[:5]}')
