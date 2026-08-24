"""
truncated_pretrain.py
=====================
核心创新：在训练集上做"截断预训练"（Truncated Pretraining）。

动机：
    比赛关键洞察：测试集和训练集是**完全独立的两组股票**。
    主办方从某 1500 只独立资产抽 20 日窗口作"测试集"，要求预测未来 20 日。
    我们从训练集 4375 只股票里，每只切多段 20 日窗口做"模拟测试样本"，
    用前面的历史做"上下文"，让模型学习"20 日价格形态 -> 未来 20 日方向"的映射。
    这是经典的元学习 / Few-shot / 模式迁移思想。

实现：
    - 对训练集 4375 只股票分别构造 (context, target) 滑动窗口样本
    - context: (60 天 OHLCV + 60 天统计特征)
    - target: 未来 20 日的累计收益率 / 上涨概率
    - 与最终测试集对齐：测试集是 20 天窗口 + 未来 20 天
    - 训练时我们用 "40 + 20" 模拟，推理时 20 天做 context

特点：
    - 不需要为每只股票都构造很多窗口（数据量太大）
    - 采用 "代表性抽样"：每只股票随机选 1-3 个窗口
    - 优先覆盖 长期可用的股票（>= 100 天）
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

# 窗口配置
# 关键: 与测试集对齐 — 测试集每只资产只有 20 天数据，要求预测未来 20 天
# 所以预训练时我们也用 20 天窗口学"20 天 -> 未来 20 天"的模式
CONTEXT_LEN = 20  # 用 20 天作为"上下文"（与测试集窗口长度一致）
TARGET_LEN = 20   # 预测未来 20 天


def build_truncated_dataset(
    train: pd.DataFrame,
    n_per_stock: int = 5,
    min_history: int = 60,  # 至少需要 20 + 20 + 20 历史 (前 20 用作特征)
    seed: int = 42,
) -> pd.DataFrame:
    """
    从 train 构造"截断预训练"样本。

    每条样本 = 一只股票的一个时间窗口，特征取自 [t-W, t) 的 OHLCV，标签取自 [t, t+T) 的累计收益。

    参数
    ----
    train : 训练集原始 DataFrame, columns=[code, date, open, high, low, close, volume]
    n_per_stock : 每只股票抽取的窗口数
    min_history : 至少需要多少历史天数
    """
    rng = np.random.default_rng(seed)
    train = train.sort_values(["code", "date"]).reset_index(drop=True)
    # 用 date 转 int 方便比较
    train["date_idx"] = train["date"].str.replace("DAY_", "").astype(int)

    # 分组
    samples = []
    by_code = train.groupby("code", sort=False)

    n_total = 0
    n_skip = 0
    for code, sub in by_code:
        n = len(sub)
        if n < min_history + TARGET_LEN:
            n_skip += 1
            continue
        # 在 [min_history, n - TARGET_LEN] 范围内均匀抽 n_per_stock 个起点
        end_min = min_history
        end_max = n - TARGET_LEN
        if end_max <= end_min:
            continue
        # 我们要的是"在 end 时刻，前 60 天是 context，end..end+20 是 target"
        # end 时刻 = 起点 + 60
        # 起点范围 [0, end_max - 60 - 1]，但起点需 >= min_history - 60
        start_min = max(0, end_min - CONTEXT_LEN)
        start_max = end_max - CONTEXT_LEN
        if start_max <= start_min:
            continue
        starts = np.sort(rng.integers(start_min, start_max, size=n_per_stock))
        for s in starts:
            ctx_start = s
            ctx_end = s + CONTEXT_LEN  # exclusive
            tgt_start = ctx_end
            tgt_end = ctx_end + TARGET_LEN
            ctx = sub.iloc[ctx_start:ctx_end]
            tgt = sub.iloc[tgt_start:tgt_end]
            if len(ctx) < CONTEXT_LEN or len(tgt) < TARGET_LEN:
                continue
            ctx_close_0 = ctx["close"].iloc[0]
            tgt_close_0 = tgt["close"].iloc[0]
            tgt_close_T = tgt["close"].iloc[-1]
            # 标签：未来 20 日累计收益率（基于 tgt 第 0 日的收盘价）
            future_20d_return = tgt_close_T / (tgt_close_0 + 1e-9) - 1.0
            up_label = int(future_20d_return > 0)
            samples.append({
                "code": code,
                "ctx_start": ctx["date"].iloc[0],
                "ctx_end_date": ctx["date"].iloc[-1],
                "tgt_start": tgt["date"].iloc[0],
                "tgt_end": tgt["date"].iloc[-1],
                "ctx_close_start": float(ctx_close_0),
                "ctx_close_end": float(ctx["close"].iloc[-1]),
                "tgt_close_start": float(tgt_close_0),
                "tgt_close_end": float(tgt_close_T),
                "future_20d_return": float(future_20d_return),
                "up_label": up_label,
            })
            n_total += 1

    print(f"[pretrain] skipped {n_skip} stocks with too short history")
    print(f"[pretrain] built {n_total} truncated samples from {len(by_code)} stocks")
    return pd.DataFrame(samples)


def aggregate_window_features(
    train: pd.DataFrame,
    samples: pd.DataFrame,
    feature_fn,
) -> pd.DataFrame:
    """
    对每条样本聚合"窗口级"特征。

    参数
    ----
    train : 训练集, 已排序
    samples : 由 build_truncated_dataset 构造
    feature_fn : 一个函数，接收一个 window DataFrame 返回一行特征 dict

    返回
    ----
    DataFrame, 每行对应一个 sample + 窗口聚合特征
    """
    train = train.sort_values(["code", "date"]).reset_index(drop=True)
    train["date_idx"] = train["date"].str.replace("DAY_", "").astype(int)
    by_code = train.groupby("code", sort=False)

    feature_rows = []
    for i, row in samples.iterrows():
        code = row["code"]
        ctx_start_date = row["ctx_start"]
        ctx_end_date = row["ctx_end_date"]
        sub = by_code.get_group(code)
        ctx = sub[(sub["date"] >= ctx_start_date) & (sub["date"] <= ctx_end_date)]
        if len(ctx) < CONTEXT_LEN:
            continue
        feat = feature_fn(ctx)
        feat["sample_idx"] = i
        feature_rows.append(feat)
        if (i + 1) % 5000 == 0:
            print(f"[aggregate] processed {i + 1} / {len(samples)}")

    return pd.DataFrame(feature_rows)


def make_test_features(
    test: pd.DataFrame,
    feature_fn,
) -> pd.DataFrame:
    """
    对测试集每只资产（20 天窗口）调用 feature_fn，得到一行特征。
    """
    test = test.sort_values(["code", "date"]).reset_index(drop=True)
    out = []
    for code, sub in test.groupby("code", sort=False):
        feat = feature_fn(sub)
        feat["code"] = code
        out.append(feat)
    return pd.DataFrame(out)
