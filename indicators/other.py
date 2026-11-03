"""其他指标：BIAS、PSY、DMI/ADX、斐波那契、枢轴点。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators._common import as_float, ma, require_window, safe_div, true_range, use_frame
from indicators.defaults import (
    BIAS_PERIODS,
    DMI_M,
    DMI_N,
    FIB_LEVELS,
    FIBONACCI_LOOKBACK,
    PSY_PERIOD,
)


def bias(df: pd.DataFrame, periods: tuple[int, ...] = BIAS_PERIODS, *, prepared: bool = False) -> pd.DataFrame:
    """乖离率，单位是百分比：(收盘价 - MA) / MA × 100。"""
    data = use_frame(df, prepared)
    close = as_float(data["close"], "close")
    columns = {}
    for period in periods:
        average = ma(close, period)
        columns[f"bias{int(period)}"] = safe_div(close - average, average) * 100
    return pd.DataFrame(columns, index=data.index)


def psy(df: pd.DataFrame, n: int = PSY_PERIOD, *, prepared: bool = False) -> pd.DataFrame:
    """心理线：N 日内收盘上涨天数 / N × 100。首根没有昨收，计为未上涨。"""
    data = use_frame(df, prepared)
    n = require_window(n)
    close = as_float(data["close"], "close")
    rising = (close > close.shift(1)).astype(float)
    values = rising.rolling(n, min_periods=n).sum() / n * 100
    return pd.DataFrame({"psy": values}, index=data.index)


def dmi(df: pd.DataFrame, n: int = DMI_N, m: int = DMI_M, *, prepared: bool = False) -> pd.DataFrame:
    """趋向指标。+DI / -DI 为 N 日方向运动除以 N 日真实波幅，ADX 为 DX 的 M 日均线。

    输出列 pdi、mdi 分别对应 +DI、-DI。
    """
    data = use_frame(df, prepared)
    n = require_window(n, name="DMI 周期")
    high = as_float(data["high"], "high")
    low = as_float(data["low"], "low")
    close = as_float(data["close"], "close")
    tr = true_range(high, low, close)
    up_move = high.diff().fillna(0.0)
    down_move = (low.shift(1) - low).fillna(0.0)
    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)

    tr_sum = tr.rolling(n, min_periods=n).sum()
    pdi = safe_div(plus_dm.rolling(n, min_periods=n).sum(), tr_sum) * 100
    mdi = safe_div(minus_dm.rolling(n, min_periods=n).sum(), tr_sum) * 100
    dx = safe_div((pdi - mdi).abs(), pdi + mdi) * 100
    adx = ma(dx, m)
    return pd.DataFrame({"pdi": pdi, "mdi": mdi, "adx": adx}, index=data.index)


def _fib_price(high: float, low: float, ratio: float, upward: bool) -> float:
    span = high - low
    if upward:
        if ratio <= 1:
            return high - span * ratio
        return high + span * (ratio - 1)
    if ratio <= 1:
        return low + span * ratio
    return low - span * (ratio - 1)


def fibonacci(df: pd.DataFrame, n: int = FIBONACCI_LOOKBACK, *, prepared: bool = False) -> pd.DataFrame:
    """用最近 N 根（含当日）的最高价、最低价划分回调与扩展。

    最低价出现得不晚于最高价时，视为向上波段：0 在高点，1 在低点，大于 1 的比例延伸到高点之上。
    最高价更早出现时，视为向下波段：0 在低点，1 在高点，扩展位在低点之下。
    同一价格多次出现时，取最早的那根。fib_trend 为 1 或 -1。
    """
    data = use_frame(df, prepared)
    n = require_window(n)
    high = as_float(data["high"], "high").to_numpy(dtype=float)
    low = as_float(data["low"], "low").to_numpy(dtype=float)
    size = len(high)
    columns = {name: np.full(size, np.nan) for name, _ in FIB_LEVELS}
    trend = np.full(size, np.nan)

    for i in range(n - 1, size):
        highs = high[i - n + 1 : i + 1]
        lows = low[i - n + 1 : i + 1]
        if np.isnan(highs).any() or np.isnan(lows).any():
            continue
        high_at = int(np.argmax(highs))
        low_at = int(np.argmin(lows))
        upward = low_at <= high_at
        trend[i] = 1.0 if upward else -1.0
        swing_high = float(highs[high_at])
        swing_low = float(lows[low_at])
        for name, ratio in FIB_LEVELS:
            columns[name][i] = _fib_price(swing_high, swing_low, ratio, upward)

    frame = pd.DataFrame(columns, index=data.index)
    frame.insert(0, "fib_trend", trend)
    return frame


def pivot_points(df: pd.DataFrame, *, prepared: bool = False) -> pd.DataFrame:
    """用前一根 K 线的高、低、收计算经典枢轴点。当日数值不使用当日价格。"""
    data = use_frame(df, prepared)
    high = as_float(data["high"], "high").shift(1)
    low = as_float(data["low"], "low").shift(1)
    close = as_float(data["close"], "close").shift(1)
    pivot = (high + low + close) / 3
    return pd.DataFrame(
        {
            "pivot_pp": pivot,
            "pivot_r1": 2 * pivot - low,
            "pivot_r2": pivot + (high - low),
            "pivot_r3": high + 2 * (pivot - low),
            "pivot_s1": 2 * pivot - high,
            "pivot_s2": pivot - (high - low),
            "pivot_s3": low - 2 * (high - pivot),
        },
        index=data.index,
    )
