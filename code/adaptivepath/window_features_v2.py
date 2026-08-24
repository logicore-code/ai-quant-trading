"""
window_features_v2.py
=====================
扩展版窗口特征：增加分形/形态/结构性特征
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def _rsi(close, w=14):
    delta = pd.Series(close).diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1.0 / w, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / w, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(s, span):
    return pd.Series(s).ewm(span=span, adjust=False).mean()


def _atr(high, low, close, w=14):
    h_l = np.array(high) - np.array(low)
    h_c = np.abs(np.array(high) - np.array(pd.Series(close).shift(1)))
    l_c = np.abs(np.array(low) - np.array(pd.Series(close).shift(1)))
    tr = pd.concat([pd.Series(h_l), pd.Series(h_c), pd.Series(l_c)], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / w, adjust=False).mean()


def window_features_v2(win: pd.DataFrame) -> Dict[str, float]:
    """
    输入: 一段窗口 OHLCV DataFrame (按 date 升序, 至少 5 行)
    输出: 一行扩展特征字典
    """
    out: Dict[str, float] = {}
    if len(win) < 5:
        return out

    open_ = win["open"].values
    high = win["high"].values
    low = win["low"].values
    close = win["close"].values
    vol = win["volume"].values
    n = len(close)

    # 归一化
    base = close[0] + 1e-9
    close_n = close / base

    # ========== 1. 末态特征 ==========
    out["w_close_last"] = close_n[-1]
    out["w_close_max"] = float(close_n.max())
    out["w_close_min"] = float(close_n.min())
    out["w_close_max_idx"] = int(np.argmax(close_n)) / n
    out["w_close_min_idx"] = int(np.argmin(close_n)) / n
    out["w_range_total"] = out["w_close_max"] - out["w_close_min"]
    out["w_last_to_max"] = close_n[-1] - out["w_close_max"]
    out["w_last_to_min"] = close_n[-1] - out["w_close_min"]

    # ========== 2. 多尺度动量 / 反转 ==========
    for w in [2, 3, 5, 7, 10, 14, 20, 30, 40, 60]:
        if n >= w:
            out[f"m_ret_{w}"] = close_n[-1] / close_n[-w] - 1.0
        else:
            out[f"m_ret_{w}"] = 0.0
    for w in [2, 5, 10, 20]:
        if n >= w:
            out[f"m_logret_{w}"] = float(np.log(close_n[-1] / (close_n[-w] + 1e-9)))
        else:
            out[f"m_logret_{w}"] = 0.0

    # 反转 / 趋势加速
    for ws, wl in [(2, 5), (3, 10), (5, 20), (5, 10), (10, 20)]:
        if n >= wl:
            rs = close_n[-1] / close_n[-ws] - 1.0 if n >= ws else 0.0
            rl = close_n[-1] / close_n[-wl] - 1.0
            out[f"mom_diff_{ws}_{wl}"] = rs - rl
        else:
            out[f"mom_diff_{ws}_{wl}"] = 0.0

    # ========== 3. 波动率 ==========
    logret = np.diff(np.log(close_n + 1e-9))
    out["v_std_all"] = float(np.std(logret)) if len(logret) > 1 else 0.0
    for w in [3, 5, 7, 10, 14, 20]:
        if n > w:
            out[f"v_std_{w}"] = float(np.std(logret[-w:]))
        else:
            out[f"v_std_{w}"] = 0.0
    # 波动率聚集
    if n >= 10:
        half = n // 2
        out["v_std_ratio_lr"] = (np.std(logret[half:]) + 1e-9) / (np.std(logret[:half]) + 1e-9)
    else:
        out["v_std_ratio_lr"] = 1.0
    # 波动率均值 / 方差
    if n >= 10:
        logret_s = pd.Series(logret)
        out["v_abs_mean"] = float(logret_s.abs().mean())
        out["v_abs_max"] = float(logret_s.abs().max())
    else:
        out["v_abs_mean"] = 0.0
        out["v_abs_max"] = 0.0
    # 偏度 / 峰度
    if len(logret) >= 10:
        m = np.mean(logret)
        s = np.std(logret) + 1e-9
        out["v_skew"] = float(np.mean(((logret - m) / s) ** 3))
        out["v_kurt"] = float(np.mean(((logret - m) / s) ** 4) - 3)
    else:
        out["v_skew"] = 0.0
        out["v_kurt"] = 0.0
    # Garman-Klass / Parkinson
    log_hl = np.log((high + 1e-9) / (low + 1e-9))
    log_co = np.log((close + 1e-9) / (open_ + 1e-9))
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    out["v_gk_mean"] = float(np.mean(gk))
    out["v_gk_std"] = float(np.std(gk))
    out["v_parkinson_mean"] = float(np.mean((1.0 / (4 * np.log(2))) * log_hl ** 2))

    # ========== 4. 量价关系 ==========
    vol_s = pd.Series(vol)
    logret_s = pd.Series(logret)
    for w in [3, 5, 10, 14, 20]:
        if len(logret_s) >= w:
            out[f"q_pv_corr_{w}"] = float(logret_s.rolling(w, min_periods=2).corr(vol_s).iloc[-1])
        else:
            out[f"q_pv_corr_{w}"] = 0.0
    for w in [3, 5, 10, 20]:
        vma = vol_s.rolling(w, min_periods=1).mean().iloc[-1]
        out[f"q_vol_ratio_{w}"] = float(vma / (vol_s.mean() + 1e-9))
        out[f"q_vol_last_to_{w}"] = float(vol[-1] / (vma + 1e-9))
    # 量趋势
    for w in [5, 10, 20]:
        if n >= w:
            out[f"q_vol_slope_{w}"] = float(np.polyfit(np.arange(w), vol[-w:], 1)[0] / (vol_s.rolling(w).mean().iloc[-1] + 1e-9))
        else:
            out[f"q_vol_slope_{w}"] = 0.0
    # OBV 简化
    sign = np.sign(np.diff(close))
    sign = np.concatenate([[0], sign])
    obv = np.cumsum(sign * vol)
    out["q_obv_last"] = float(obv[-1] / (obv.std() + 1e-9))
    if n >= 5:
        out["q_obv_chg_5"] = float((obv[-1] - obv[-5]) / (np.abs(obv[-5]) + 1e-9))
    else:
        out["q_obv_chg_5"] = 0.0

    # ========== 5. 趋势强度 (ADX) ==========
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

    # RSI / MACD
    if n >= 14:
        out["t_rsi_14"] = float(_rsi(close, 14).iloc[-1])
    else:
        out["t_rsi_14"] = 50.0
    if n >= 14:
        out["t_rsi_7"] = float(_rsi(close, 7).iloc[-1])
        out["t_rsi_21"] = float(_rsi(close, 21).iloc[-1])
    else:
        out["t_rsi_7"] = 50.0
        out["t_rsi_21"] = 50.0
    if n >= 26:
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        out["t_macd"] = float(ema12.iloc[-1] - ema26.iloc[-1])
        macd_sig = ema12 - ema26
        sig = _ema(macd_sig.values, 9)
        out["t_macd_signal"] = float(sig.iloc[-1])
        out["t_macd_hist"] = float(out["t_macd"] - out["t_macd_signal"])
    else:
        out["t_macd"] = 0.0
        out["t_macd_signal"] = 0.0
        out["t_macd_hist"] = 0.0

    # Bollinger
    if n >= 20:
        ma = pd.Series(close).rolling(20, min_periods=2).mean().iloc[-1]
        std = pd.Series(close).rolling(20, min_periods=2).std().iloc[-1]
        out["t_bb_pos"] = float((close[-1] - (ma - 2 * std)) / (4 * std + 1e-9))
        out["t_bb_width"] = float(4 * std / (ma + 1e-9))
    else:
        out["t_bb_pos"] = 0.5
        out["t_bb_width"] = 0.0

    # ATR
    if n >= 14:
        atr = _atr(high, low, close, 14)
        out["t_atr_14"] = float(atr.iloc[-1])
        out["t_atr_pct"] = float(atr.iloc[-1] / (close[-1] + 1e-9))
    else:
        out["t_atr_14"] = 0.0
        out["t_atr_pct"] = 0.0

    # ========== 6. 形态学 ==========
    gaps = (open_[1:] / close[:-1] - 1.0) if n > 1 else np.array([])
    if len(gaps) > 0:
        out["p_gap_mean"] = float(np.mean(np.abs(gaps)))
        out["p_gap_max"] = float(np.max(np.abs(gaps)))
        out["p_gap_pos_ratio"] = float(np.mean(gaps > 0))
        out["p_gap_neg_ratio"] = float(np.mean(gaps < 0))
    else:
        out["p_gap_mean"] = 0.0
        out["p_gap_max"] = 0.0
        out["p_gap_pos_ratio"] = 0.0
        out["p_gap_neg_ratio"] = 0.0

    amp = (high - low) / (close + 1e-9)
    out["p_amp_mean"] = float(np.mean(amp))
    out["p_amp_max"] = float(np.max(amp))
    out["p_amp_last"] = float(amp[-1])
    out["p_amp_std"] = float(np.std(amp))

    body = np.abs(close - open_)
    upper = (high - np.maximum(close, open_)) / (body + 1e-9)
    lower = (np.minimum(close, open_) - low) / (body + 1e-9)
    out["p_upper_shadow"] = float(np.mean(upper))
    out["p_lower_shadow"] = float(np.mean(lower))
    out["p_upper_shadow_last"] = float(upper[-1])
    out["p_lower_shadow_last"] = float(lower[-1])
    out["p_doji_ratio"] = float(np.mean(body / (high - low + 1e-9) < 0.1))

    # 突破信号
    for w in [5, 10, 20]:
        if n >= w + 1:
            out[f"p_break_high_{w}"] = float(close[-1] >= np.max(high[-w - 1:-1]))
            out[f"p_break_low_{w}"] = float(close[-1] <= np.min(low[-w - 1:-1]))
        else:
            out[f"p_break_high_{w}"] = 0.0
            out[f"p_break_low_{w}"] = 0.0

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
    out["p_streak_ratio"] = max_up / (max_up + max_dn + 1e-9)

    # ========== 7. Hurst-like (R/S) ==========
    if n >= 16:
        diff_log = np.diff(np.log(close_n + 1e-9))
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

    # ========== 8. 价格位置 ==========
    for w in [5, 10, 20, 30, 60]:
        if n >= w:
            out[f"pos_in_{w}"] = float((close[-1] - np.min(low[-w:])) / (np.ptp(high[-w:]) + 1e-9))
        else:
            out[f"pos_in_{w}"] = 0.5

    # ========== 9. 自相关 ==========
    for lag in [1, 2, 3, 5]:
        if n >= lag + 1:
            out[f"a_autocorr_{lag}"] = float(pd.Series(logret).autocorr(lag=lag))
        else:
            out[f"a_autocorr_{lag}"] = 0.0

    # ========== 10. 末端 / 首端对比 ==========
    if n >= 5:
        out["end_vs_start_vol"] = float(np.std(logret[-5:]) / (np.std(logret) + 1e-9))
        out["end_vs_start_mom"] = float((close[-1] / close[-5] - 1.0) - (close[5] / close[0] - 1.0))
    else:
        out["end_vs_start_vol"] = 1.0
        out["end_vs_start_mom"] = 0.0

    # ========== 11. 微观结构 ==========
    out["m_up_ratio"] = float(np.mean(logret > 0))
    out["m_down_ratio"] = float(np.mean(logret < 0))
    out["m_zero_ratio"] = float(np.mean(logret == 0))
    out["m_intra_range"] = float(np.mean((high - low) / (close + 1e-9)))
    # 最大单日涨幅 / 跌幅
    out["m_max_up"] = float(np.max(logret)) if len(logret) > 0 else 0.0
    out["m_max_dn"] = float(np.min(logret)) if len(logret) > 0 else 0.0

    # ========== 12. 形态识别: 头肩 / 双底 / 楔形 ==========
    # 简化：高低点序列的特征
    if n >= 10:
        # 找局部极值
        from scipy.signal import argrelextrema
        try:
            h_idx = argrelextrema(pd.Series(close_n), np.greater, order=2)[0]
            l_idx = argrelextrema(pd.Series(close_n), np.less, order=2)[0]
            out["m_num_peaks"] = float(len(h_idx))
            out["m_num_troughs"] = float(len(l_idx))
            # 平均峰高
            if len(h_idx) > 0:
                out["m_avg_peak_height"] = float(np.mean(close_n[h_idx]))
                out["m_last_peak_height"] = float(close_n[h_idx[-1]])
            else:
                out["m_avg_peak_height"] = 0.0
                out["m_last_peak_height"] = 0.0
            if len(l_idx) > 0:
                out["m_avg_trough_depth"] = float(np.mean(close_n[l_idx]))
                out["m_last_trough_depth"] = float(close_n[l_idx[-1]])
            else:
                out["m_avg_trough_depth"] = 0.0
                out["m_last_trough_depth"] = 0.0
        except Exception:
            out["m_num_peaks"] = 0.0
            out["m_num_troughs"] = 0.0
            out["m_avg_peak_height"] = 0.0
            out["m_last_peak_height"] = 0.0
            out["m_avg_trough_depth"] = 0.0
            out["m_last_trough_depth"] = 0.0
    else:
        out["m_num_peaks"] = 0.0
        out["m_num_troughs"] = 0.0

    return out
