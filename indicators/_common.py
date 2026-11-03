"""通达信口径的基础序列运算。"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def require_window(n: int, *, name: str = "周期") -> int:
    """周期必须是大于等于 1 的整数。布尔值不算整数。"""
    if isinstance(n, bool) or isinstance(n, np.bool_):
        raise ValueError(f"{name}必须是正整数")
    if isinstance(n, (int, np.integer)):
        value = int(n)
    else:
        raise ValueError(f"{name}必须是正整数")
    if value < 1:
        raise ValueError(f"{name}必须是正整数")
    return value


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """校验 OHLCV，并按 eob 升序返回新表。不修改调用方的数据。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("行情数据需要 pandas.DataFrame")

    renamed = df.copy()
    renamed.columns = [str(column).strip() for column in renamed.columns]
    missing = [column for column in OHLCV_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError("缺少行情列: " + ", ".join(missing))

    if "eob" in renamed.columns and len(renamed) > 0:
        parsed = pd.to_datetime(renamed["eob"], errors="coerce", utc=True)
        if bool(parsed.notna().all()):
            renamed = renamed.assign(_eob_key=parsed).sort_values("_eob_key", kind="mergesort")
            renamed = renamed.drop(columns="_eob_key")
        else:
            renamed = renamed.sort_values("eob", kind="mergesort")

    return renamed.reset_index(drop=True)


def use_frame(df: pd.DataFrame, prepared: bool) -> pd.DataFrame:
    if prepared:
        return df
    return prepare(df)


def as_float(series: pd.Series, name: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & converted.isna()
    if bool(bad.any()):
        raise ValueError(f"列 {name} 含有无法转为数值的数据")
    return converted.astype(float)


def ma(series: pd.Series, n: int) -> pd.Series:
    """简单移动平均。有效值从第 n 根开始。"""
    n = require_window(n)
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """通达信 EMA：首个有效值等于自身，其后 Y = (2X + (N-1)Y') / (N+1)。"""
    n = require_window(n)
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    alpha = 2.0 / (n + 1.0)
    prev = None
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        prev = value if prev is None else alpha * value + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=series.index, dtype=float)


def sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通达信 SMA(X, N, M)：首个有效值等于自身，其后 Y = (M*X + (N-M)*Y') / N。"""
    n = require_window(n, name="SMA 周期 N")
    m = require_window(m, name="SMA 权重 M")
    if m > n:
        raise ValueError("SMA 权重 M 不能大于周期 N")

    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    prev = None
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        prev = value if prev is None else (m * value + (n - m) * prev) / n
        out[i] = prev
    return pd.Series(out, index=series.index, dtype=float)


def rolling_std_population(series: pd.Series, n: int) -> pd.Series:
    """总体标准差，除以 N。"""
    n = require_window(n)
    return series.rolling(n, min_periods=n).std(ddof=0)


def avedev(series: pd.Series, n: int) -> pd.Series:
    """平均绝对偏差：窗口内 |X - MA(X)| 的算术平均。"""
    n = require_window(n)
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    for i in range(n - 1, len(values)):
        window = values[i - n + 1 : i + 1]
        if np.isnan(window).any():
            continue
        out[i] = np.mean(np.abs(window - np.mean(window)))
    return pd.Series(out, index=series.index, dtype=float)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """真实波幅。首根没有昨收，取最高价减最低价。"""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    tr = ranges.max(axis=1, skipna=True).astype(float)
    if len(tr):
        tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """分母为 0 或任一端为 NaN 时结果为 NaN，不产生无穷大。"""
    num = numerator.to_numpy(dtype=float)
    den = denominator.to_numpy(dtype=float)
    out = np.full(np.shape(num), np.nan, dtype=float)
    valid = (den != 0) & ~np.isnan(den) & ~np.isnan(num)
    np.divide(num, den, out=out, where=valid)
    return pd.Series(out, index=numerator.index, dtype=float)
