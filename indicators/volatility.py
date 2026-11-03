"""波动率与通道：ATR、唐奇安通道、轨道线。"""

from __future__ import annotations

import pandas as pd

from indicators._common import as_float, ma, require_window, true_range, use_frame
from indicators.defaults import ATR_PERIOD, DONCHIAN_PERIOD, ENVELOPE_PERIOD, ENVELOPE_WIDTH


def atr(df: pd.DataFrame, n: int = ATR_PERIOD, *, prepared: bool = False) -> pd.DataFrame:
    """真实波幅的 N 日算术平均。"""
    data = use_frame(df, prepared)
    high = as_float(data["high"], "high")
    low = as_float(data["low"], "low")
    close = as_float(data["close"], "close")
    values = ma(true_range(high, low, close), n)
    return pd.DataFrame({"atr": values}, index=data.index)


def donchian(df: pd.DataFrame, n: int = DONCHIAN_PERIOD, *, prepared: bool = False) -> pd.DataFrame:
    """唐奇安通道。上轨为 N 日最高价，下轨为 N 日最低价，中轨为二者中点。含当日。"""
    data = use_frame(df, prepared)
    n = require_window(n)
    high = as_float(data["high"], "high")
    low = as_float(data["low"], "low")
    upper = high.rolling(n, min_periods=n).max()
    lower = low.rolling(n, min_periods=n).min()
    return pd.DataFrame(
        {
            "donchian_upper": upper,
            "donchian_mid": (upper + lower) / 2,
            "donchian_lower": lower,
        },
        index=data.index,
    )


def envelope(
    df: pd.DataFrame,
    n: int = ENVELOPE_PERIOD,
    width: float = ENVELOPE_WIDTH,
    *,
    prepared: bool = False,
) -> pd.DataFrame:
    """轨道线。中轨为收盘价均线，上下轨为中轨 × (1 ± width)。width=0.06 表示 6%。"""
    data = use_frame(df, prepared)
    close = as_float(data["close"], "close")
    mid = ma(close, n)
    return pd.DataFrame(
        {
            "env_mid": mid,
            "env_upper": mid * (1 + width),
            "env_lower": mid * (1 - width),
        },
        index=data.index,
    )
