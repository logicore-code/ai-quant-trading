"""
feature_engineering.py
======================
多时间尺度 Alpha 因子 + 价格形态特征。

设计原则：
- 测试集首日已归一化为 100，所以**绝对价格无意义**，**形态/比例** 才是关键
- 因此本模块不引入 "close 本身" 类特征，只引入：
    * 收益率类（动量/反转/均值回归）
    * 波动率类（已实现波动率、Garman-Klass、Parkinson）
    * 量价关系（量比、量价相关、OBV）
    * 趋势强度（ADX 类、RSI、布林带位置）
    * 形态特征（突破、回撤、跳空、振幅）
- 所有特征都对"窗口"友好，便于在滑动窗口上批量计算

输入：原始 OHLCV DataFrame
输出：与输入同长度的"特征" DataFrame（不含 label）
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# 一些可复用的工具
def _safe_div(a, b):
    """安全除法，避免除零。返回 0 当 b 接近 0。"""
    out = np.where(np.abs(b) > 1e-12, a / np.where(np.abs(b) > 1e-12, b, 1.0), 0.0)
    return out


def add_ret_features(df: pd.DataFrame, windows=(1, 2, 3, 5, 10, 20)) -> pd.DataFrame:
    """对 close 计算多个窗口的对数收益率 / 简单收益率。"""
    g = df.groupby("code", sort=False)["close"]
    for w in windows:
        df[f"ret_{w}"] = g.pct_change(w)
        # log return 近似正态，更稳
        df[f"logret_{w}"] = np.log(df["close"] / g.shift(w))
    return df


def add_meanrev_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """均值回归 / 偏离度特征：close 与 N 日均线的偏离。"""
    g = df.groupby("code", sort=False)["close"]
    for w in windows:
        ma = g.rolling(w, min_periods=2).mean().reset_index(level=0, drop=True)
        std = g.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
        df[f"ma_{w}"] = ma.values
        df[f"close_over_ma_{w}"] = df["close"] / ma.values - 1.0
        # z-score
        df[f"close_z_{w}"] = (df["close"] - ma.values) / (std.values + 1e-9)
    return df


def add_volatility_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """已实现波动率 + Garman-Klass + Parkinson。"""
    g_ret = df.groupby("code", sort=False)["logret_1"]
    for w in windows:
        # realized vol
        vol = g_ret.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
        df[f"vol_{w}"] = vol.values * np.sqrt(252)  # 年化

    # Garman-Klass: 0.5 * (log(H/L))^2 - (2*log2 - 1) * (log(C/O))^2
    log_hl = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], 0)
    log_co = np.log(df["close"] / df["open"]).replace([np.inf, -np.inf], 0)
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    df["garman_klass"] = gk
    # Parkinson: (1/(4*log2)) * (log(H/L))^2
    df["parkinson"] = (1.0 / (4 * np.log(2))) * log_hl ** 2

    # 5 日平均
    df["gk_5"] = df.groupby("code", sort=False)["garman_klass"].transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    df["parkinson_5"] = df.groupby("code", sort=False)["parkinson"].transform(
        lambda s: s.rolling(5, min_periods=2).mean()
    )
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """量价关系：量比、量价相关、OBV、换手。"""
    # 5/20 日量比
    g = df.groupby("code", sort=False)["volume"]
    df["vol_ma5"] = g.transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["vol_ma20"] = g.transform(lambda s: s.rolling(20, min_periods=2).mean())
    df["vol_ratio_5_20"] = df["vol_ma5"] / (df["vol_ma20"] + 1e-9)

    # 量价相关（10 日滚动）
    def _pvcorr(s_close, s_vol, w=10):
        # s_close 和 s_vol 同长度
        r = s_close.pct_change()
        out = pd.Series(index=s_close.index, dtype=float)
        for i in range(w, len(s_close) + 1):
            sub_r = r.iloc[i - w:i]
            sub_v = s_vol.iloc[i - w:i]
            if sub_r.std() > 1e-9 and sub_v.std() > 1e-9:
                out.iloc[i - 1] = sub_r.corr(sub_v)
            else:
                out.iloc[i - 1] = 0.0
        return out

    # 用 transform 加速（向量化 + 滑窗）
    g_close = df.groupby("code", sort=False)["close"]
    g_vol = df.groupby("code", sort=False)["volume"]
    # 速度优化：只对关键股票做相关，其他置 0
    # 这里直接用 rolling.corr
    df["ret1_for_corr"] = g_close.pct_change()
    df["pv_corr_10"] = df.groupby("code", sort=False).apply(
        lambda x: x["ret1_for_corr"].rolling(10, min_periods=5).corr(x["volume"])
    ).reset_index(level=0, drop=True)
    df.drop(columns=["ret1_for_corr"], inplace=True)

    # OBV
    sign = np.sign(df.groupby("code", sort=False)["close"].pct_change()).fillna(0)
    df["obv"] = (sign * df["volume"]).groupby(df["code"]).cumsum()
    # OBV 5 日变化
    df["obv_chg_5"] = df.groupby("code", sort=False)["obv"].pct_change(5)

    return df


def add_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI / MACD / Bollinger / ATR / ADX 类。"""
    g = df.groupby("code", sort=False)

    # 14 日 RSI
    def _rsi(s, w=14):
        delta = s.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        # Wilder smoothing
        roll_up = up.ewm(alpha=1.0 / w, adjust=False).mean()
        roll_down = down.ewm(alpha=1.0 / w, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-9)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi

    df["rsi_14"] = g["close"].transform(_rsi)

    # MACD (12, 26, 9)
    ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df.groupby("code", sort=False)["macd"].transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger (20, 2)
    ma20 = g["close"].transform(lambda s: s.rolling(20, min_periods=2).mean())
    std20 = g["close"].transform(lambda s: s.rolling(20, min_periods=2).std())
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_pos"] = (df["close"] - df["bb_lower"]) / ((df["bb_upper"] - df["bb_lower"]) + 1e-9)

    # ATR 14
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - g["close"].shift(1)).abs()
    low_close = (df["low"] - g["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.groupby(df["code"]).transform(lambda s: s.ewm(alpha=1.0 / 14, adjust=False).mean())
    df["atr_pct"] = df["atr_14"] / (df["close"] + 1e-9)

    # ADX 近似（趋势强度）
    up_move = df["high"] - g["high"].shift(1)
    down_move = g["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    df["plus_dm"] = plus_dm
    df["minus_dm"] = minus_dm
    df["plus_di"] = 100 * df.groupby("code", sort=False)["plus_dm"].transform(
        lambda s: s.ewm(alpha=1.0 / 14, adjust=False).mean()
    ) / (df["atr_14"] + 1e-9)
    df["minus_di"] = 100 * df.groupby("code", sort=False)["minus_dm"].transform(
        lambda s: s.ewm(alpha=1.0 / 14, adjust=False).mean()
    ) / (df["atr_14"] + 1e-9)
    df["dx"] = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"] + 1e-9)
    df["adx_14"] = df.groupby("code", sort=False)["dx"].transform(
        lambda s: s.ewm(alpha=1.0 / 14, adjust=False).mean()
    )
    return df


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """形态学特征：跳空、振幅、突破、回撤、十字星。"""
    g = df.groupby("code", sort=False)
    # 跳空（开盘相对昨日收盘的偏离）
    df["gap"] = (df["open"] / g["close"].shift(1) - 1.0)
    # 当日振幅
    df["amplitude"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    # 上影线 / 下影线
    body = (df["close"] - df["open"]).abs()
    df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (body + 1e-9)
    df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (body + 1e-9)
    # 十字星
    df["doji"] = (body / (df["high"] - df["low"] + 1e-9) < 0.1).astype(float)

    # 20 日新高/新低
    high20 = g["high"].transform(lambda s: s.rolling(20, min_periods=1).max())
    low20 = g["low"].transform(lambda s: s.rolling(20, min_periods=1).min())
    df["new_high_20"] = (df["close"] >= high20.shift(1)).astype(float)
    df["new_low_20"] = (df["close"] <= low20.shift(1)).astype(float)

    # 连涨/连跌天数
    sign = np.sign(g["close"].pct_change()).fillna(0)
    df["sign"] = sign
    df["streak"] = df.groupby("code", sort=False)["sign"].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    ) * sign

    # 趋势加速度（ret5 - ret5.shift(5)）
    df["ret5_accel"] = df["ret_5"] - g["ret_5"].shift(5)
    return df


def add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """时间衰减的近因特征：越近的日期权重越大。"""
    g = df.groupby("code", sort=False)
    # 5/10/20 日均收益（EMA）
    for w in [3, 5, 10, 20]:
        df[f"ret_ema_{w}"] = g["logret_1"].transform(
            lambda s: s.ewm(span=w, adjust=False).mean()
        )
    # 近期最大涨幅/最大回撤
    for w in [5, 10, 20]:
        df[f"max_ret_{w}"] = g["logret_1"].transform(
            lambda s: s.rolling(w, min_periods=1).max()
        )
        df[f"min_ret_{w}"] = g["logret_1"].transform(
            lambda s: s.rolling(w, min_periods=1).min()
        )
    return df


def add_cross_sectional_rank(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """横截面排名（每个交易日上 4375 只股票的相对位置）。"""
    for c in cols:
        if c not in df.columns:
            continue
        df[f"{c}_xrank"] = df.groupby("date")[c].rank(pct=True)
    return df


# 特征列表
def get_feature_columns(df: pd.DataFrame, exclude=("code", "date", "label", "up_factor", "future_20d_return")):
    return [c for c in df.columns if c not in exclude]


def build_features(df: pd.DataFrame, do_rank: bool = True) -> pd.DataFrame:
    """主入口：构造所有特征。"""
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = add_ret_features(df)
    df = add_meanrev_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_tech_indicators(df)
    df = add_pattern_features(df)
    df = add_recency_features(df)
    # 填补无穷/NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df
