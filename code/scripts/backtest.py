"""
backtest.py
===========
本地回测：模拟官方评分脚本的"top-5 加权 + 20 日收益 + 最大回撤 + Brier Score"。
但因为 OOF 标签已知（基于训练集截断窗口的"未来 20 日真实收益"），我们可以做"假回测"。

主要逻辑：
- 取 OOF 预测概率和 OOF 真实标签（up_label + future_20d_return）
- 把 OOF 预测视为"对 OOF 集 1500 个样本的 top-5 选择"
- 计算 top-5 加权收益、最大回撤、Brier Score、综合分
- 这给我们一个**OOF 上的"模拟综合分"**，与真实测试集的得分应当成正相关
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT.parent
OUT = DATA / "output" / "backtest"
OUT.mkdir(parents=True, exist_ok=True)


def simulate_score(
    p: np.ndarray,
    y_up: np.ndarray,
    y_ret: np.ndarray,
    cost: float = 0.002,
    n_sim: int = 1000,
    seed: int = 42,
) -> dict:
    """
    p: OOF 预测概率
    y_up: 实际方向（0/1）
    y_ret: 实际 20 日累计收益（连续）
    cost: 双边成本 0.2%
    """
    rng = np.random.default_rng(seed)
    n = len(p)

    # --- 全局指标 ---
    brier = float(np.mean((p - y_up) ** 2))
    p_above = float((y_ret > 0).mean())

    # --- top-5 模拟: 多次 bootstrap ---
    # 每次从全集有放回抽样 n 个样本，选 top-5
    top5_hits = []
    top5_returns = []  # 加权收益
    top5_returns_simple = []  # 等权收益
    for _ in range(n_sim):
        idx = rng.integers(0, n, size=n)
        p_i = p[idx]
        y_up_i = y_up[idx]
        y_ret_i = y_ret[idx]
        order = np.argsort(-p_i)[:5]
        # 命中: 实际为 1 的比例
        top5_hits.append(y_up_i[order].mean())
        # 加权收益: 按 p 归一化
        weights = p_i[order] / (p_i[order].sum() + 1e-9)
        # 应用成本
        ret_gross = (weights * y_ret_i[order]).sum()
        ret_net = ret_gross - cost  # 双边 0.2%
        top5_returns.append(ret_net)
        # 等权收益
        top5_returns_simple.append(y_ret_i[order].mean() - cost)

    # --- Brier on subset (N=4339 模拟) ---
    # 实际评分用 4339 只，OOF 集可能更大。简单地按 OOF 集统计。
    brier_subset = float(np.mean((p - y_up) ** 2))

    # --- 最大回撤: 用 top5 加权日收益序列模拟 ---
    # 由于我们只有"20 日累计收益"，无法计算日序列的最大回撤
    # 这里我们做一个"伪最大回撤"：用 y_ret 的分布估计
    # 实际综合评分中 MDD 是 top-5 组合的 20 日日净值最大回撤
    # 我们用 5 只资产的 (start, end) 估计一个"假 MDD"
    mdds = []
    for _ in range(n_sim):
        idx = rng.integers(0, n, size=n)
        p_i = p[idx]
        y_ret_i = y_ret[idx]
        order = np.argsort(-p_i)[:5]
        # 假设 5 只资产的日收益按均匀随机游走达到 y_ret
        # 每天均值 = y_ret / 20
        # 每天方差 = (y_ret / 20) 的数量级，估计为 (y_ret^2 / 20)
        # 简化: 用一个"最坏情况"近似 MDD = max(|y_ret|) * 0.5
        # 实际上还是用 y_ret 估计
        weights = p_i[order] / (p_i[order].sum() + 1e-9)
        worst_dd = -np.max(np.abs(weights * y_ret_i[order]))
        mdds.append(worst_dd)
    avg_mdd = float(np.mean(mdds))

    # --- 综合分 = 0.6 * ret + 0.2 * (1 - |mdd|) - 0.2 * brier ---
    # 累计收益率（top-5 加权）均值
    avg_ret = float(np.mean(top5_returns))
    # 平均回撤
    score = 0.6 * avg_ret + 0.2 * (1 - abs(avg_mdd)) - 0.2 * brier
    # 注: 这里 (1 - |mdd|) 是猜测的。原始赛题没给具体公式，假设是 linear。

    return {
        "brier": brier,
        "p_above_0": p_above,
        "top5_hit_mean": float(np.mean(top5_hits)),
        "top5_hit_std": float(np.std(top5_hits)),
        "top5_weighted_ret_mean": float(np.mean(top5_returns)),
        "top5_weighted_ret_std": float(np.std(top5_returns)),
        "top5_equal_ret_mean": float(np.mean(top5_returns_simple)),
        "top5_equal_ret_std": float(np.std(top5_returns_simple)),
        "avg_mdd": avg_mdd,
        "simulated_total_score": float(score),
        "n_samples": int(n),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", type=str, default=str(OUT.parent / "pipeline_report.json"))
    args = p.parse_args()

    # 加载 OOF 预测 + 真实标签
    # 我们在 run_full.py 中没有保存 OOF 预测，重新跑一次保存
    # 实际做法: 读 truncated_samples.csv 和 oof predictions
    # 这里为简化，从 Xtr_df.csv 读
    print("注意: 需要先运行 run_full.py --save_oof 才会保存 OOF 预测")
    # 占位: 如果存在 Xtr_df.csv 和 oof_preds.npy
    xtr_path = OUT.parent / "Xtr_df.csv"
    oof_path = OUT.parent / "oof_final.npy"
    if not (xtr_path.exists() and oof_path.exists()):
        print("没有保存的 OOF 数据，请先用 --save_oof 跑 run_full.py")
        return
    Xtr = pd.read_csv(xtr_path)
    oof_final = np.load(oof_path)
    print(f"Loaded {len(Xtr)} samples, OOF preds shape={oof_final.shape}")

    res = simulate_score(
        p=oof_final,
        y_up=Xtr["up_label"].values,
        y_ret=Xtr["future_20d_return"].values,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(OUT / "backtest_report.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
