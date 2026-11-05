"""收盘价线性分段入口。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from segment.defaults import DEFAULT_MAX_ERROR
from segment.linear import linear_segment

__all__ = [
    "DEFAULT_MAX_ERROR",
    "linear_segment",
    "load_history",
    "write_segments",
]


def load_history(path: str | Path) -> pd.DataFrame:
    """读取行情 CSV。"""
    return pd.read_csv(path)


def write_segments(
    input_path: str | Path,
    output_path: str | Path | None = None,
    max_error: float = DEFAULT_MAX_ERROR,
) -> Path:
    """读取 CSV，按收盘价做线性分段，并写入结果 CSV。返回输出路径。"""
    source = Path(input_path)
    if output_path is None:
        suffix = source.suffix or ".csv"
        destination = source.with_name(f"{source.stem}_segments{suffix}")
    else:
        destination = Path(output_path)
    result = linear_segment(load_history(source), max_error=max_error)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8")
    return destination
