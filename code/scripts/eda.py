"""
exploratory_data_analysis.py
============================
对 train.csv 与 test.csv 做全面探索性数据分析 (EDA)。

输出：
- output/eda/eda_report.txt
- output/eda/eda_summary.json
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT
TRAIN_CSV = DATA_DIR / "train" / "train.csv"
TEST_CSV = DATA_DIR / "test" / "test.csv"
OUT_DIR = DATA_DIR / "output" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("EDA - 智能量化投资策略建模挑战赛")
    print("=" * 70)

    # ============================================================
    # 1. 加载训练集
    # ============================================================
    print("\n[1] 加载 train.csv ...")
    train = pd.read_csv(TRAIN_CSV)
    train = train.sort_values(["code", "date"]).reset_index(drop=True)

    print(f"  shape = {train.shape}")
    print(f"  columns = {list(train.columns)}")
    print(f"  unique codes = {train['code'].nunique()}")
    print(f"  unique dates = {train['date'].nunique()}")
    print(f"  date range = {train['date'].min()} -> {train['date'].max()}")

    # ============================================================
    # 2. 加载测试集
    # ============================================================
    print("\n[2] 加载 test.csv ...")
    test = pd.read_csv(TEST_CSV)
    test = test.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"  shape = {test.shape}")
    print(f"  unique codes = {test['code'].nunique()}")
    print(f"  unique dates = {test['date'].nunique()}")
    print(f"  date range = {test['date'].min()} -> {test['date'].max()}")

    # ============================================================
    # 3. 缺失值与异常值
    # ============================================================
    print("\n[3] 缺失值与异常值统计")
    miss_train = train.isna().sum()
    miss_test = test.isna().sum()
    print("train NA:\n", miss_train)
    print("test NA:\n", miss_test)

    # 价格 <= 0 的异常
    neg_train = (train[["open", "high", "low", "close", "volume"]] <= 0).sum()
    print("train <=0 counts:\n", neg_train)
    neg_test = (test[["open", "high", "low", "close", "volume"]] <= 0).sum()
    print("test <=0 counts:\n", neg_test)

    # high < low 异常
    hl_train = (train["high"] < train["low"]).sum()
    hl_test = (test["high"] < test["low"]).sum()
    print(f"high<low: train={hl_train}, test={hl_test}")

    # ============================================================
    # 4. 时间覆盖与重采样
    # ============================================================
    print("\n[4] 时间覆盖")
    train_days = train.groupby("code").size().describe()
    print("train days per code:\n", train_days)
    test_days = test.groupby("code").size().describe()
    print("test days per code:\n", test_days)

    # ============================================================
    # 5. 价格统计
    # ============================================================
    print("\n[5] 价格统计")
    desc = train[["open", "high", "low", "close", "volume"]].describe()
    print(desc)

    # ============================================================
    # 6. 收益分布（按股票日内 / 跨日）
    # ============================================================
    print("\n[6] 收益分布")
    train["ret1"] = train.groupby("code")["close"].pct_change()
    train["ret5"] = train.groupby("code")["close"].pct_change(5)
    train["ret20"] = train.groupby("code")["close"].pct_change(20)
    print("ret1 stats:", train["ret1"].describe().to_dict())
    print("ret5 stats:", train["ret5"].describe().to_dict())
    print("ret20 stats:", train["ret20"].describe().to_dict())

    # 上涨概率（针对未来 20 日累计收益 > 0）
    train["future_20d_return"] = train.groupby("code")["close"].shift(-20) / train["close"] - 1.0
    up_ratio = (train["future_20d_return"] > 0).mean()
    print(f"训练集 P(未来 20 日累计收益 > 0) = {up_ratio:.4f}")
    # 按 close 数量级分层
    print(f"训练集 future_20d_return NaN 比例: {train['future_20d_return'].isna().mean():.4f}")

    # 区间收益分布
    buckets = [-np.inf, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, np.inf]
    labels = ["<-20%", "-20~-10%", "-10~-5%", "-5~0%", "0~5%", "5~10%", "10~20%", ">20%"]
    train["bucket"] = pd.cut(train["future_20d_return"], buckets, labels=labels)
    print("\n20-day return bucket distribution:")
    print(train["bucket"].value_counts(normalize=True).sort_index())

    # ============================================================
    # 7. 形态学观察：典型样本
    # ============================================================
    print("\n[7] 抽取样例股票的价格序列")
    sample_codes = train["code"].drop_duplicates().sample(3, random_state=42).tolist()
    for c in sample_codes:
        sub = train[train["code"] == c].copy()
        print(f"  {c}: {len(sub)} days, close[0]={sub['close'].iloc[0]:.2f}, "
              f"close[-1]={sub['close'].iloc[-1]:.2f}, "
              f"max={sub['close'].max():.2f}, min={sub['close'].min():.2f}")

    # ============================================================
    # 8. 测试集形态
    # ============================================================
    print("\n[8] 测试集形态")
    test_starts = test.groupby("code")["close"].first().describe()
    print("test first close:\n", test_starts)
    # 首日已归一化为 100
    print(f"test 首日 close 是否都接近 100: "
          f"{(test.groupby('code')['close'].first().between(99, 101)).mean():.4f}")

    test["ret1"] = test.groupby("code")["close"].pct_change()
    print("test ret1 stats:", test["ret1"].describe().to_dict())

    # ============================================================
    # 9. 测试集与训练集关系（同名 code 不代表同资产）
    # ============================================================
    print("\n[9] 测试集与训练集 code 关系")
    overlap = set(test["code"]).intersection(set(train["code"]))
    print(f"  train-test code overlap = {len(overlap)}")
    print("  (官方说明: 同名 code 不代表同一资产，不应拼接)")

    # ============================================================
    # 10. 关键发现汇总
    # ============================================================
    summary = {
        "train": {
            "rows": int(len(train)),
            "unique_codes": int(train["code"].nunique()),
            "unique_dates": int(train["date"].nunique()),
            "date_min": str(train["date"].min()),
            "date_max": str(train["date"].max()),
            "any_na": int(miss_train.sum()),
            "any_neg_price": int(neg_train.sum()),
            "any_high_lt_low": int(hl_train),
            "ret1_mean": float(train["ret1"].mean()),
            "ret1_std": float(train["ret1"].std()),
            "ret5_mean": float(train["ret5"].mean()),
            "ret20_mean": float(train["ret20"].mean()),
            "p_up_20d": float(up_ratio),
        },
        "test": {
            "rows": int(len(test)),
            "unique_codes": int(test["code"].nunique()),
            "unique_dates": int(test["date"].nunique()),
            "date_min": str(test["date"].min()),
            "date_max": str(test["date"].max()),
            "any_na": int(miss_test.sum()),
            "any_neg_price": int(neg_test.sum()),
            "any_high_lt_low": int(hl_test),
            "ret1_mean": float(test["ret1"].mean()),
            "ret1_std": float(test["ret1"].std()),
        },
        "overlap_codes": int(len(overlap)),
    }

    with open(OUT_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[done] EDA summary saved ->", OUT_DIR / "eda_summary.json")
    return summary


if __name__ == "__main__":
    main()
