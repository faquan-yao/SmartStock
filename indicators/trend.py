"""趋势类指标：MA、EMA、MACD、BOLL、SAR。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators._common import as_float, ema, ma, rolling_std_population, use_frame
from indicators.defaults import (
    BOLL_K,
    BOLL_PERIOD,
    EMA_PERIODS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MA_PERIODS,
    SAR_AF_MAX,
    SAR_AF_START,
    SAR_AF_STEP,
)


def moving_average(df: pd.DataFrame, periods: tuple[int, ...] = MA_PERIODS, *, prepared: bool = False) -> pd.DataFrame:
    data = use_frame(df, prepared)
    close = as_float(data["close"], "close")
    columns = {f"ma{int(period)}": ma(close, period) for period in periods}
    return pd.DataFrame(columns, index=data.index)


def exponential_moving_average(
    df: pd.DataFrame,
    periods: tuple[int, ...] = EMA_PERIODS,
    *,
    prepared: bool = False,
) -> pd.DataFrame:
    data = use_frame(df, prepared)
    close = as_float(data["close"], "close")
    columns = {f"ema{int(period)}": ema(close, period) for period in periods}
    return pd.DataFrame(columns, index=data.index)


def macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
    *,
    prepared: bool = False,
) -> pd.DataFrame:
    """DIF = EMA(fast) - EMA(slow)，DEA = EMA(DIF, signal)，柱 = 2×(DIF-DEA)。"""
    data = use_frame(df, prepared)
    close = as_float(data["close"], "close")
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = 2.0 * (dif - dea)
    return pd.DataFrame(
        {"macd_dif": dif, "macd_dea": dea, "macd_hist": hist},
        index=data.index,
    )


def bollinger(
    df: pd.DataFrame,
    n: int = BOLL_PERIOD,
    k: float = BOLL_K,
    *,
    prepared: bool = False,
) -> pd.DataFrame:
    """中轨为收盘价简单均线，上下轨为中轨 ± K 倍总体标准差。"""
    data = use_frame(df, prepared)
    close = as_float(data["close"], "close")
    mid = ma(close, n)
    std = rolling_std_population(close, n)
    return pd.DataFrame(
        {
            "boll_mid": mid,
            "boll_upper": mid + k * std,
            "boll_lower": mid - k * std,
        },
        index=data.index,
    )


def parabolic_sar(
    df: pd.DataFrame,
    af_start: float = SAR_AF_START,
    af_step: float = SAR_AF_STEP,
    af_max: float = SAR_AF_MAX,
    *,
    prepared: bool = False,
) -> pd.DataFrame:
    """抛物线转向。

    前两根 K 线中价决定初始方向。多头时 SAR 放在价格下方，空头时放在上方。
    加速因子在出现新极值之后的下一根 K 线才加大。价格触及 SAR 时，SAR 跳到此前极值并反向。
    sar_trend：1 为多头，-1 为空头。
    """
    data = use_frame(df, prepared)
    high = as_float(data["high"], "high").to_numpy(dtype=float)
    low = as_float(data["low"], "low").to_numpy(dtype=float)
    size = len(high)
    sar = np.full(size, np.nan)
    trend = np.full(size, np.nan)

    if size >= 2:
        up = (high[1] + low[1]) >= (high[0] + low[0])
        af = float(af_start)
        if up:
            sar[0] = low[0]
            extreme = high[0]
        else:
            sar[0] = high[0]
            extreme = low[0]
        trend[0] = 1.0 if up else -1.0

        for i in range(1, size):
            sar[i] = sar[i - 1] + af * (extreme - sar[i - 1])
            if up:
                sar[i] = min(sar[i], low[i - 1])
                if i >= 2:
                    sar[i] = min(sar[i], low[i - 2])
                if low[i] < sar[i]:
                    up = False
                    sar[i] = extreme
                    extreme = low[i]
                    af = float(af_start)
                elif high[i] > extreme:
                    extreme = high[i]
                    af = min(af + af_step, af_max)
            else:
                sar[i] = max(sar[i], high[i - 1])
                if i >= 2:
                    sar[i] = max(sar[i], high[i - 2])
                if high[i] > sar[i]:
                    up = True
                    sar[i] = extreme
                    extreme = high[i]
                    af = float(af_start)
                elif low[i] < extreme:
                    extreme = low[i]
                    af = min(af + af_step, af_max)
            trend[i] = 1.0 if up else -1.0

    return pd.DataFrame({"sar": sar, "sar_trend": trend}, index=data.index)
