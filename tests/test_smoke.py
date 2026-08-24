"""
test_smoke.py
=============
冒烟测试：确保核心模块能跑 + 关键功能正确
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from adaptivepath.dataset import build_truncated_dataset
from adaptivepath.window_features_v2 import window_features_v2


def test_dataset_constructs():
    """测试截断预训练样本构造"""
    print("test_dataset_constructs ...")
    # 构造 mock 数据
    n = 199
    dates = [f"DAY_{i:04d}" for i in range(1, n + 1)]
    data = {
        "code": ["STOCK_A"] * n,
        "date": dates,
        "open": np.random.uniform(10, 20, n).tolist(),
        "high": np.random.uniform(15, 25, n).tolist(),
        "low": np.random.uniform(5, 15, n).tolist(),
        "close": np.random.uniform(10, 20, n).tolist(),
        "volume": np.random.uniform(1000, 100000, n).tolist(),
    }
    df = pd.DataFrame(data)
    samples = build_truncated_dataset(df, n_per_stock=2, min_history=60, seed=42)
    assert len(samples) > 0, "应该至少构造出一个样本"
    assert "future_20d_return" in samples.columns
    assert "up_label" in samples.columns
    print(f"  ok, samples={len(samples)}")


def test_window_features():
    """测试窗口特征抽取"""
    print("test_window_features ...")
    dates = [f"DAY_{i:04d}" for i in range(1, 25)]
    data = {
        "date": dates,
        "open": np.linspace(10, 12, 24),
        "high": np.linspace(10.5, 12.5, 24),
        "low": np.linspace(9.5, 11.5, 24),
        "close": np.linspace(10, 12, 24) + np.random.normal(0, 0.1, 24),
        "volume": np.random.uniform(1000, 10000, 24),
    }
    win = pd.DataFrame(data)
    f = window_features_v2(win)
    assert isinstance(f, dict)
    assert len(f) > 50, f"特征数应该 > 50, 实际 {len(f)}"
    # 一些关键特征
    for k in ["w_close_last", "m_ret_5", "m_ret_10", "v_std_5", "t_rsi_14"]:
        assert k in f, f"缺少特征 {k}"
    print(f"  ok, features={len(f)}")


def test_window_features_short():
    """测试短窗口（边界情况）"""
    print("test_window_features_short ...")
    dates = [f"DAY_{i:04d}" for i in range(1, 8)]
    data = {
        "date": dates,
        "open": [10, 11, 12, 13, 14, 15, 16],
        "high": [11, 12, 13, 14, 15, 16, 17],
        "low": [9, 10, 11, 12, 13, 14, 15],
        "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5],
        "volume": [1000] * 7,
    }
    win = pd.DataFrame(data)
    f = window_features_v2(win)
    # 应该不出错，返回的特征值可能都是 0
    assert isinstance(f, dict)
    print(f"  ok, features={len(f)}")


if __name__ == "__main__":
    test_dataset_constructs()
    test_window_features()
    test_window_features_short()
    print("\nAll tests passed!")
