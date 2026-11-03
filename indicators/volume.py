"""成交量与资金类指标：均量、OBV、VWAP、MFI。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators._common import as_float, ma, require_window, use_frame
from indicators.defaults import MFI_PERIOD, VOL_MA_PERIODS


def volume_ma(df: pd.DataFrame, periods: tuple[int, ...] = VOL_MA_PERIODS, *, prepared: bool = False) -> pd.DataFrame:
    """成交量的简单移动平均。原始成交量仍保留在 volume 列。"""
    data = use_frame(df, prepared)
    volume = as_float(data["volume"], "volume")
    columns = {f"vol_ma{int(period)}": ma(volume, period) for period in periods}
    return pd.DataFrame(columns, index=data.index)


def obv(df: pd.DataFrame, *, prepared: bool = False) -> pd.DataFrame:
    """能量潮。收盘价上涨加成交量，下跌减成交量，持平不变。首根记 0。"""
    data = use_frame(df, prepared)
    close = as_float(data["close"], "close")
    volume = as_float(data["volume"], "volume").to_numpy(dtype=float)
    direction = np.sign(close.diff().to_numpy(dtype=float))
    if len(direction):
        direction[0] = 0.0
    direction = np.where(np.isnan(direction), 0.0, direction)
    values = np.cumsum(direction * volume)
    return pd.DataFrame({"obv": values}, index=data.index)


def vwap(df: pd.DataFrame, *, prepared: bool = False) -> pd.DataFrame:
    """累计成交量加权典型价。典型价 = (高+低+收) / 3。

    不使用 amount。日线样本是前复权价格，成交额仍是未复权口径，两者不能直接相除。
    """
    data = use_frame(df, prepared)
    high = as_float(data["high"], "high")
    low = as_float(data["low"], "low")
    close = as_float(data["close"], "close")
    volume = as_float(data["volume"], "volume").to_numpy(dtype=float)
    typical = ((high + low + close) / 3).to_numpy(dtype=float)
    cumulative_value = np.cumsum(typical * volume)
    cumulative_volume = np.cumsum(volume)
    values = np.full(len(typical), np.nan)
    np.divide(cumulative_value, cumulative_volume, out=values, where=cumulative_volume != 0)
    return pd.DataFrame({"vwap": values}, index=data.index)


def mfi(df: pd.DataFrame, n: int = MFI_PERIOD, *, prepared: bool = False) -> pd.DataFrame:
    """资金流量指标。典型价上升记正资金流，下降记负资金流，持平两边都不计。"""
    data = use_frame(df, prepared)
    n = require_window(n)
    high = as_float(data["high"], "high")
    low = as_float(data["low"], "low")
    close = as_float(data["close"], "close")
    volume = as_float(data["volume"], "volume")
    typical = (high + low + close) / 3
    previous = typical.shift(1)
    raw_flow = typical * volume
    positive = raw_flow.where(typical > previous, 0.0)
    negative = raw_flow.where(typical < previous, 0.0)
    # 首根没有昨典型价，比较结果为假，正负资金流都是 0。
    positive_sum = positive.rolling(n, min_periods=n).sum().to_numpy(dtype=float)
    negative_sum = negative.rolling(n, min_periods=n).sum().to_numpy(dtype=float)

    values = np.full(len(data), np.nan)
    valid = ~np.isnan(positive_sum) & ~np.isnan(negative_sum)
    positive_only = valid & (negative_sum == 0) & (positive_sum > 0)
    normal = valid & (negative_sum != 0)
    values[positive_only] = 100.0
    values[normal] = 100.0 - 100.0 / (1.0 + positive_sum[normal] / negative_sum[normal])
    return pd.DataFrame({"mfi": values}, index=data.index)
