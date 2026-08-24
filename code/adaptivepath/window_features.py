"""
window_features.py
==================
把一段窗口（测试集 20 天 / 截断预训练 60 天）的 OHLCV 序列浓缩为一行"窗口级"特征。

设计原则：
- 全部使用无量纲/相对化指标（因为测试集已归一化为 100）
- 涵盖：多尺度动量/反转/波动率/量价/趋势/形态/分形 等 50+ 维
- 与"市场典型统计模式"对齐，便于在训练集上学到的"模式"迁移到测试集
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _r(s, w):
    """滚动相关系数: 简单版"""
    return s.rolling(w, min_periods=2).corr(s.shift(1))


def _atr(high, low, close, w=14):
    """ATR 近似"""
    h_l = high - low
    h_c = (high - close.shift(1)).abs()
    l_c = (low - close.shift(1)).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / w, adjust=False).mean()


def _rsi(close, w=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1.0 / w, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / w, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)


def window_features(win: pd.DataFrame) -> Dict[str, float]:
    """
    输入：一段窗口 OHLCV DataFrame（按 date 升序，至少 5 行）
    输出：一行特征字典
    """
    out: Dict[str, float] = {}

    # 防止过短
    if len(win) < 5:
        return out

    open_ = win["open"].values
    high = win["high"].values
    low = win["low"].values
    close = win["close"].values
    vol = win["volume"].values
    n = len(close)

    # 基础比例归一化：以首日收盘价为基准
    base = close[0] + 1e-9
    open_n = open_ / base
    high_n = high / base
    low_n = low / base
    close_n = close / base

    # ==== 1. 末态特征 ====
    out["w_close_last"] = close_n[-1]
    out["w_close_max"] = float(close_n.max())
    out["w_close_min"] = float(close_n.min())
    out["w_close_max_idx"] = int(np.argmax(close_n)) / n
    out["w_close_min_idx"] = int(np.argmin(close_n)) / n
    out["w_range_total"] = out["w_close_max"] - out["w_close_min"]
    out["w_last_to_max"] = out["w_close_last"] - out["w_close_max"]
    out["w_last_to_min"] = out["w_close_last"] - out["w_close_min"]

    # ==== 2. 多尺度动量/反转 ====
    for w in [3, 5, 10, 15, 20, 30, 40, 60]:
        if n >= w:
            out[f"m_ret_{w}"] = close_n[-1] / close_n[-w] - 1.0
        else:
            out[f"m_ret_{w}"] = 0.0
    # logret
    for w in [3, 5, 10, 20]:
        if n >= w:
            out[f"m_logret_{w}"] = float(np.log(close_n[-1] / (close_n[-w] + 1e-9)))
        else:
            out[f"m_logret_{w}"] = 0.0

    # ==== 3. 反转信号：近 N 日 vs 远 N 日 ====
    for w_short, w_long in [(3, 10), (5, 20), (5, 10), (10, 20)]:
        if n >= w_long:
            r_short = close_n[-1] / close_n[-w_short] - 1.0 if n >= w_short else 0.0
            r_long = close_n[-1] / close_n[-w_long] - 1.0
            out[f"mom_diff_{w_short}_{w_long}"] = r_short - r_long
        else:
            out[f"mom_diff_{w_short}_{w_long}"] = 0.0

    # ==== 4. 波动率 ====
    logret = np.diff(np.log(close_n + 1e-9))
    out["v_std_all"] = float(np.std(logret)) if len(logret) > 1 else 0.0
    if n >= 5:
        out["v_std_5"] = float(np.std(logret[-5:]))
    else:
        out["v_std_5"] = 0.0
    if n >= 10:
        out["v_std_10"] = float(np.std(logret[-10:]))
    else:
        out["v_std_10"] = 0.0
    if n >= 20:
        out["v_std_20"] = float(np.std(logret[-20:]))
    else:
        out["v_std_20"] = 0.0
    # 波动率聚集：后段 / 前段
    if n >= 10:
        half = n // 2
        out["v_std_ratio_lr"] = (np.std(logret[half:]) + 1e-9) / (np.std(logret[:half]) + 1e-9)
    else:
        out["v_std_ratio_lr"] = 1.0
    # 峰度/偏度
    if len(logret) >= 10:
        m = np.mean(logret)
        s = np.std(logret) + 1e-9
        out["v_skew"] = float(np.mean(((logret - m) / s) ** 3))
        out["v_kurt"] = float(np.mean(((logret - m) / s) ** 4) - 3)
    else:
        out["v_skew"] = 0.0
        out["v_kurt"] = 0.0

    # Garman-Klass 平均
    log_hl = np.log((high + 1e-9) / (low + 1e-9))
    log_co = np.log((close + 1e-9) / (open_ + 1e-9))
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    out["v_gk_mean"] = float(np.mean(gk))
    out["v_gk_std"] = float(np.std(gk))

    # Parkinson
    out["v_parkinson_mean"] = float(np.mean((1.0 / (4 * np.log(2))) * log_hl ** 2))

    # ==== 5. 量价关系 ====
    vol_s = pd.Series(vol)
    logret_s = pd.Series(logret)
    if len(logret_s) >= 5:
        out["q_pv_corr_5"] = float(logret_s.rolling(5, min_periods=2).corr(vol_s).iloc[-1])
    else:
        out["q_pv_corr_5"] = 0.0
    if len(logret_s) >= 10:
        out["q_pv_corr_10"] = float(logret_s.rolling(10, min_periods=2).corr(vol_s).iloc[-1])
    else:
        out["q_pv_corr_10"] = 0.0

    # 量比
    vol_ma5 = vol_s.rolling(5, min_periods=1).mean().iloc[-1]
    vol_ma_full = vol_s.mean()
    out["q_vol_ratio_5"] = float(vol_ma5 / (vol_ma_full + 1e-9))
    out["q_vol_last"] = float(vol[-1] / (vol_ma5 + 1e-9))

    # 量趋势
    if n >= 10:
        out["q_vol_slope"] = float(np.polyfit(np.arange(10), vol[-10:], 1)[0] / (vol_ma5 + 1e-9))
    else:
        out["q_vol_slope"] = 0.0

    # ==== 6. 趋势强度 ====
    # ADX 简化
    if n >= 14:
        up_move = np.diff(high)
        down_move = -np.diff(low)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_approx = np.maximum(np.diff(high), 0) + np.maximum(-np.diff(low), 0)
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1.0 / 14, adjust=False).mean().iloc[-1] / (
            pd.Series(tr_approx).ewm(alpha=1.0 / 14, adjust=False).mean().iloc[-1] + 1e-9
        )
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1.0 / 14, adjust=False).mean().iloc[-1] / (
            pd.Series(tr_approx).ewm(alpha=1.0 / 14, adjust=False).mean().iloc[-1] + 1e-9
        )
        out["t_plus_di"] = float(plus_di)
        out["t_minus_di"] = float(minus_di)
        out["t_di_diff"] = float(plus_di - minus_di)
        out["t_adx_approx"] = float(abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9) * 100)
    else:
        out["t_plus_di"] = 0.0
        out["t_minus_di"] = 0.0
        out["t_di_diff"] = 0.0
        out["t_adx_approx"] = 0.0

    # RSI
    if n >= 14:
        out["t_rsi_14"] = float(_rsi(pd.Series(close), 14).iloc[-1])
    else:
        out["t_rsi_14"] = 50.0

    # ==== 7. 形态学 ====
    # 跳空
    gaps = (open_[1:] / close[:-1] - 1.0) if n > 1 else np.array([])
    if len(gaps) > 0:
        out["p_gap_mean"] = float(np.mean(np.abs(gaps)))
        out["p_gap_max"] = float(np.max(np.abs(gaps)))
        out["p_gap_pos_ratio"] = float(np.mean(gaps > 0))
    else:
        out["p_gap_mean"] = 0.0
        out["p_gap_max"] = 0.0
        out["p_gap_pos_ratio"] = 0.0

    # 振幅
    amp = (high - low) / (close + 1e-9)
    out["p_amp_mean"] = float(np.mean(amp))
    out["p_amp_max"] = float(np.max(amp))
    out["p_amp_last"] = float(amp[-1])

    # 上下影线
    body = np.abs(close - open_)
    upper = (high - np.maximum(close, open_)) / (body + 1e-9)
    lower = (np.minimum(close, open_) - low) / (body + 1e-9)
    out["p_upper_shadow"] = float(np.mean(upper))
    out["p_lower_shadow"] = float(np.mean(lower))

    # 突破信号
    if n >= 20:
        out["p_break_high_20"] = float(close[-1] >= np.max(high[-21:-1]))
        out["p_break_low_20"] = float(close[-1] <= np.min(low[-21:-1]))
    else:
        out["p_break_high_20"] = 0.0
        out["p_break_low_20"] = 0.0

    # 连涨/连跌
    sign = np.sign(np.diff(close))
    sign = np.concatenate([[0], sign])
    streak = 0
    max_up = 0
    max_dn = 0
    for s in sign:
        if s > 0:
            streak = max(1, streak + 1) if streak > 0 else 1
            max_up = max(max_up, streak)
        elif s < 0:
            streak = min(-1, streak - 1) if streak < 0 else -1
            max_dn = max(max_dn, abs(streak))
        else:
            streak = 0
    out["p_max_up_streak"] = float(max_up)
    out["p_max_dn_streak"] = float(max_dn)
    out["p_final_streak"] = float(streak)

    # ==== 8. Hurst-like（粗略估计） ====
    if n >= 16:
        diff_log = np.diff(np.log(close_n + 1e-9))
        # R/S 估计
        mean = np.mean(diff_log)
        cum = np.cumsum(diff_log - mean)
        r = float(np.max(cum) - np.min(cum))
        s = float(np.std(diff_log)) + 1e-9
        rs = r / s
        if rs > 0:
            out["h_rs"] = float(np.log(rs) / np.log(len(diff_log)))
        else:
            out["h_rs"] = 0.5
    else:
        out["h_rs"] = 0.5

    # ==== 9. 价格位置特征 ====
    if n >= 20:
        out["pos_in_20"] = float((close[-1] - np.min(low[-20:])) / (np.ptp(high[-20:]) + 1e-9))
    else:
        out["pos_in_20"] = 0.5
    if n >= 60:
        out["pos_in_60"] = float((close[-1] - np.min(low[-60:])) / (np.ptp(high[-60:]) + 1e-9))
    else:
        out["pos_in_60"] = 0.5

    # ==== 10. 自相关 / 趋势一致性 ====
    if n >= 10:
        out["a_autocorr_1"] = float(pd.Series(logret).autocorr(lag=1))
        out["a_autocorr_3"] = float(pd.Series(logret).autocorr(lag=3))
    else:
        out["a_autocorr_1"] = 0.0
        out["a_autocorr_3"] = 0.0

    # ==== 11. 末端与首端的对比 ====
    if n >= 5:
        out["end_vs_start_vol"] = float(np.std(logret[-5:]) / (np.std(logret) + 1e-9))
        out["end_vs_start_mom"] = float((close[-1] / close[-5] - 1.0) - (close[5] / close[0] - 1.0))
    else:
        out["end_vs_start_vol"] = 1.0
        out["end_vs_start_mom"] = 0.0

    # ==== 12. 微观结构 ====
    # 涨日占比
    out["m_up_ratio"] = float(np.mean(logret > 0))
    out["m_down_ratio"] = float(np.mean(logret < 0))
    out["m_zero_ratio"] = float(np.mean(logret == 0))
    # 平均日内振幅
    out["m_intra_range"] = float(np.mean((high - low) / (close + 1e-9)))

    return out
