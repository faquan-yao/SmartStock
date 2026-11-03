"""动量与超买超卖指标：RSI、KDJ、CCI、WR、ROC、MOM。"""

from __future__ import annotations

import pandas as pd

from indicators._common import as_float, avedev, ma, require_window, safe_div, sma, use_frame
from indicators.defaults import (
    CCI_PERIOD,
    KDJ_M1,
    KDJ_M2,
    KDJ_N,
    MOM_PERIOD,
    ROC_PERIOD,
    RSI_PERIODS,
    WR_PERIODS,
)


def rsi(df: pd.DataFrame, periods: tuple[int, ...] = RSI_PERIODS, *, prepared: bool = False) -> pd.DataFrame:
    """RSI = SMA(涨幅, N, 1) / SMA(|涨跌|, N, 1) × 100。"""
    data = use_frame(df, prepared)
    close = as_float(data["close"], "close")
    delta = close.diff()
    gain = delta.clip(lower=0)
    amplitude = delta.abs()
    columns = {}
    for period in periods:
        average_gain = sma(gain, period, 1)
        average_amplitude = sma(amplitude, period, 1)
        columns[f"rsi{int(period)}"] = safe_div(average_gain, average_amplitude) * 100
    return pd.DataFrame(columns, index=data.index)


def kdj(
    df: pd.DataFrame,
    n: int = KDJ_N,
    m1: int = KDJ_M1,
    m2: int = KDJ_M2,
    *,
    prepared: bool = False,
) -> pd.DataFrame:
    """RSV 为收盘价在 N 日高低区间中的位置。K、D 用 SMA(·, M, 1)，J = 3K - 2D。"""
    data = use_frame(df, prepared)
    n = require_window(n, name="KDJ 周期")
    high = as_float(data["high"], "high")
    low = as_float(data["low"], "low")
    close = as_float(data["close"], "close")
    lowest = low.rolling(n, min_periods=n).min()
    highest = high.rolling(n, min_periods=n).max()
    rsv = safe_div(close - lowest, highest - lowest) * 100
    k = sma(rsv, m1, 1)
    d = sma(k, m2, 1)
    j = 3 * k - 2 * d
    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j}, index=data.index)


def cci(df: pd.DataFrame, n: int = CCI_PERIOD, *, prepared: bool = False) -> pd.DataFrame:
    """CCI = (TYP - MA(TYP)) / (0.015 × AVEDEV(TYP))，TYP = (高+低+收) / 3。"""
    data = use_frame(df, prepared)
    high = as_float(data["high"], "high")
    low = as_float(data["low"], "low")
    close = as_float(data["close"], "close")
    typical = (high + low + close) / 3
    values = safe_div(typical - ma(typical, n), 0.015 * avedev(typical, n))
    return pd.DataFrame({"cci": values}, index=data.index)


def williams_r(df: pd.DataFrame, periods: tuple[int, ...] = WR_PERIODS, *, prepared: bool = False) -> pd.DataFrame:
    """威廉指标，取值 -100 到 0。接近 0 偏超买，接近 -100 偏超卖。"""
    data = use_frame(df, prepared)
    high = as_float(data["high"], "high")
    low = as_float(data["low"], "low")
    close = as_float(data["close"], "close")
    columns = {}
    for period in periods:
        period = require_window(period)
        highest = high.rolling(period, min_periods=period).max()
        lowest = low.rolling(period, min_periods=period).min()
        columns[f"wr{int(period)}"] = safe_div(highest - close, highest - lowest) * -100
    return pd.DataFrame(columns, index=data.index)


def roc(df: pd.DataFrame, n: int = ROC_PERIOD, *, prepared: bool = False) -> pd.DataFrame:
    """变动率，单位是百分比：(收盘价 / N 日前收盘价 - 1) × 100。"""
    data = use_frame(df, prepared)
    n = require_window(n)
    close = as_float(data["close"], "close")
    previous = close.shift(n)
    values = safe_div(close - previous, previous) * 100
    return pd.DataFrame({"roc": values}, index=data.index)


def momentum(df: pd.DataFrame, n: int = MOM_PERIOD, *, prepared: bool = False) -> pd.DataFrame:
    """动量：收盘价减 N 日前收盘价。"""
    data = use_frame(df, prepared)
    n = require_window(n)
    close = as_float(data["close"], "close")
    values = close - close.shift(n)
    return pd.DataFrame({"mom": values}, index=data.index)
