"""把除 segment 以外的列变成网络可以用的数值矩阵。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.defaults import TARGET_COLUMN


class FeatureEncoder:
    """记录训练时的时间原点和文本类别，预测时按同一套规则编码。"""

    def __init__(
        self,
        feature_names: tuple[str, ...],
        datetime_origins: dict[str, str],
        categories: dict[str, tuple[str, ...]],
    ) -> None:
        self.feature_names = feature_names
        self.datetime_origins = datetime_origins
        self.categories = categories

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> FeatureEncoder:
        names = tuple(str(column) for column in frame.columns)
        origins: dict[str, str] = {}
        categories: dict[str, tuple[str, ...]] = {}
        for name in names:
            series = frame[name]
            if _is_datetime_column(name, series):
                parsed = pd.to_datetime(series, errors="coerce", utc=True)
                if parsed.notna().sum() == 0:
                    raise ValueError(f"列 {name} 无法解析为时间")
                origins[name] = parsed.min().isoformat()
            elif not pd.api.types.is_numeric_dtype(series):
                text = series.astype("string")
                seen: list[str] = []
                for value in text.tolist():
                    if pd.isna(value):
                        continue
                    text_value = str(value)
                    if text_value not in seen:
                        seen.append(text_value)
                categories[name] = tuple(seen)
        return cls(names, origins, categories)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError("缺少列: " + ", ".join(missing))
        columns: list[np.ndarray] = []
        for name in self.feature_names:
            series = frame[name]
            if name in self.datetime_origins:
                parsed = pd.to_datetime(series, errors="coerce", utc=True)
                origin = pd.Timestamp(self.datetime_origins[name])
                days = (parsed - origin).dt.total_seconds().to_numpy(dtype=float) / 86400.0
                columns.append(days)
            elif name in self.categories:
                mapping = {value: float(index) for index, value in enumerate(self.categories[name])}
                text = series.astype("string")
                coded = text.map(mapping).to_numpy(dtype=float)
                columns.append(coded)
            else:
                columns.append(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float))
        if not columns:
            raise ValueError("没有可用特征")
        return np.column_stack(columns)


def ordered_frame(df: pd.DataFrame) -> pd.DataFrame:
    """有 eob 时按时间从早到晚排列。不修改调用方的表。"""
    out = df.copy()
    if "eob" not in out.columns:
        return out.reset_index(drop=True)
    parsed = pd.to_datetime(out["eob"], errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("列 eob 含有无法解析的时间")
    out = out.assign(_order=parsed).sort_values("_order", kind="mergesort").drop(columns="_order")
    return out.reset_index(drop=True)


def target_values(frame: pd.DataFrame) -> np.ndarray:
    if TARGET_COLUMN not in frame.columns:
        raise ValueError("缺少列: segment")
    values = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("列 segment 含有空值或非有限数值")
    return values


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in frame.columns:
        return frame.copy()
    return frame.drop(columns=[TARGET_COLUMN])


def column_medians(values: np.ndarray) -> np.ndarray:
    fill = np.zeros(values.shape[1], dtype=float)
    for index in range(values.shape[1]):
        column = values[:, index]
        valid = column[np.isfinite(column)]
        if len(valid):
            fill[index] = float(np.median(valid))
    return fill


def fill_missing(values: np.ndarray, fill: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    missing = ~np.isfinite(out)
    if missing.any():
        rows, cols = np.nonzero(missing)
        out[rows, cols] = fill[cols]
    return out


def sliding_windows(values: np.ndarray, window: int) -> np.ndarray:
    """把连续 window 行拉直。每一行里，更早的 K 线排在前面。"""
    count, width = values.shape
    if window < 1:
        raise ValueError("窗口至少为 1")
    if count < window:
        raise ValueError("样本行数少于窗口")
    samples = count - window + 1
    windows = np.empty((samples, window * width), dtype=float)
    for offset in range(window):
        windows[:, offset * width : (offset + 1) * width] = values[offset : offset + samples]
    return windows


def _is_datetime_column(name: str, series: pd.Series) -> bool:
    if name == "eob" or pd.api.types.is_datetime64_any_dtype(series):
        return True
    return False
