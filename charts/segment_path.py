"""把分段斜率还原成收盘价折线的端点。"""

from __future__ import annotations

import numpy as np


def segment_vertices(closes, slopes) -> tuple[np.ndarray, np.ndarray]:
    """返回折线端点的横坐标（K 线序号）和纵坐标（收盘价）。

    `slopes` 与 `segment.linear_segment` 的输出一致：同一条直线上的斜率相同，
    两段共用的端点保存后一段的斜率。端点之间按序号连成直线，正好经过两端收盘价。
    不足两根 K 线，或斜率全为空时，返回空数组。
    """
    close = np.asarray(closes, dtype=float)
    slope = np.asarray(slopes, dtype=float)
    if close.shape != slope.shape:
        raise ValueError("收盘价与分段斜率长度不一致")
    size = len(close)
    if size < 2 or not np.isfinite(slope).any():
        return np.array([], dtype=float), np.array([], dtype=float)

    bounds = [0]
    for index in range(1, size):
        if _changed(slope[index - 1], slope[index]):
            bounds.append(index)
    if bounds[-1] != size - 1:
        bounds.append(size - 1)
    points = np.asarray(bounds, dtype=int)
    return points.astype(float), close[points]


def break_on_trend_change(values, trend, reference=None, jump_ratio: float = 0.08) -> np.ndarray:
    """方向切换，或相对参考价出现过大跳变时，把该点断开，避免拉出竖线。"""
    out = np.array(values, dtype=float, copy=True)
    direction = np.asarray(trend, dtype=float)
    if out.shape != direction.shape:
        raise ValueError("数值与方向长度不一致")
    if len(out) < 2:
        return out
    previous_direction = direction[:-1]
    current_direction = direction[1:]
    changed = np.isfinite(previous_direction) & np.isfinite(current_direction) & (previous_direction != current_direction)
    if reference is not None:
        scale = np.abs(np.asarray(reference, dtype=float)[1:])
        scale = np.maximum(scale, 1e-9)
        previous_value = out[:-1]
        current_value = out[1:]
        both = np.isfinite(previous_value) & np.isfinite(current_value) & np.isfinite(scale)
        changed = changed | (both & (np.abs(current_value - previous_value) > jump_ratio * scale))
    out[np.flatnonzero(changed) + 1] = np.nan
    return out


def _changed(previous: float, current: float) -> bool:
    previous_missing = np.isnan(previous)
    current_missing = np.isnan(current)
    if previous_missing and current_missing:
        return False
    if previous_missing or current_missing:
        return True
    return bool(current != previous)
