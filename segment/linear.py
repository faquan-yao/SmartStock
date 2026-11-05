"""自底向上的收盘价线性分段。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from segment.defaults import DEFAULT_MAX_ERROR


def linear_segment(df: pd.DataFrame, max_error: float = DEFAULT_MAX_ERROR) -> pd.DataFrame:
    """按收盘价把行情分成若干直线段，并写出每根 K 线所在直线的斜率。

    每段直线连接该段第一根和最后一根 K 线的收盘价，横轴是按时间排序后的序号。
    误差是段内收盘价偏离这条直线的最大值，再除以该点收盘价的绝对值。
    先把相邻两根 K 线各自连成一段，再不断合并误差最小的相邻两段，
    直到任何一次合并都会超过 max_error。

    返回按 eob 从早到晚排列的两列：

    - eob：输入中的结束时间
    - segment：该点所在直线的斜率，单位是每根 K 线的价格变化。
      同一条直线上的值相同。两段共用的端点写入后一段的斜率。
      只有一根 K 线时无法成线，该值为空。
    """
    limit = _require_max_error(max_error)
    ordered = _ordered_close(df)
    closes = _as_float(ordered["close"], "close").to_numpy(dtype=float)
    if not np.isfinite(closes).all():
        raise ValueError("列 close 含有空值或非有限数值")
    if np.any(closes == 0):
        raise ValueError("列 close 含有 0，无法计算相对误差")

    slopes = _slopes(closes, _segment_bounds(closes, limit))
    return pd.DataFrame({"eob": ordered["eob"].to_numpy(), "segment": slopes})


def _require_max_error(max_error: float) -> float:
    if isinstance(max_error, (bool, np.bool_)):
        raise ValueError("误差阈值必须是非负实数")
    try:
        value = float(max_error)
    except (TypeError, ValueError) as exc:
        raise ValueError("误差阈值必须是非负实数") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("误差阈值必须是非负实数")
    return value


def _as_float(series: pd.Series, name: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & converted.isna()
    if bool(bad.any()):
        raise ValueError(f"列 {name} 含有无法转为数值的数据")
    return converted.astype(float)


def _ordered_close(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("行情数据需要 pandas.DataFrame")
    renamed = df.copy()
    renamed.columns = [str(column).strip() for column in renamed.columns]
    missing = [column for column in ("eob", "close") if column not in renamed.columns]
    if missing:
        raise ValueError("缺少行情列: " + ", ".join(missing))

    data = renamed.loc[:, ["eob", "close"]]
    if len(data) == 0:
        return data.reset_index(drop=True)

    parsed = pd.to_datetime(data["eob"], errors="coerce", utc=True)
    if bool(parsed.notna().all()):
        data = data.assign(_eob_key=parsed).sort_values("_eob_key", kind="mergesort")
        data = data.drop(columns="_eob_key")
    else:
        data = data.sort_values("eob", kind="mergesort")
    return data.reset_index(drop=True)


def _segment_bounds(closes: np.ndarray, max_error: float) -> list[int]:
    """返回各段端点下标，含第一根和最后一根。"""
    n = len(closes)
    if n <= 1:
        return list(range(n))

    bounds = list(range(n))
    costs = [_max_relative_error(closes, bounds[i], bounds[i + 2]) for i in range(n - 2)]
    while costs:
        index = min(range(len(costs)), key=lambda i: (costs[i], i))
        if costs[index] > max_error:
            break
        del bounds[index + 1]
        del costs[index]
        if index > 0:
            costs[index - 1] = _max_relative_error(closes, bounds[index - 1], bounds[index + 1])
        if index < len(costs):
            costs[index] = _max_relative_error(closes, bounds[index], bounds[index + 2])
    return bounds


def _slopes(closes: np.ndarray, bounds: list[int]) -> np.ndarray:
    """同一条直线写入同一个斜率；共用端点写入后一条直线的斜率。"""
    slopes = np.full(len(closes), np.nan, dtype=float)
    if len(bounds) < 2:
        return slopes
    last_segment = len(bounds) - 2
    for index in range(len(bounds) - 1):
        start = bounds[index]
        end = bounds[index + 1]
        slope = (closes[end] - closes[start]) / (end - start)
        filled_until = end if index == last_segment else end - 1
        slopes[start : filled_until + 1] = slope
    return slopes


def _max_relative_error(closes: np.ndarray, start: int, end: int) -> float:
    """两端点连线对开区间内收盘价的最大相对偏离。相邻两点误差为 0。"""
    if end - start <= 1:
        return 0.0
    y0 = closes[start]
    y1 = closes[end]
    steps = np.arange(1, end - start)
    predicted = y0 + (y1 - y0) * (steps / (end - start))
    actual = closes[start + steps]
    return float(np.max(np.abs(actual - predicted) / np.abs(actual)))
