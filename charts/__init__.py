"""技术指标与分段折线的图形入口。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from charts.draw import draw_chart
from charts.segment_path import segment_vertices
from charts.selection import (
    DEFAULT_INDICATORS,
    INDICATOR_NAMES,
    chart_panels,
    resolve_indicators,
)
from indicators._common import prepare
from indicators.momentum import cci, kdj, momentum, roc, rsi, williams_r
from indicators.other import bias, dmi, fibonacci, pivot_points, psy
from indicators.trend import bollinger, exponential_moving_average, macd, moving_average, parabolic_sar
from indicators.volatility import atr, donchian, envelope
from indicators.volume import mfi, obv, volume_ma, vwap
from segment import DEFAULT_MAX_ERROR, linear_segment

__all__ = [
    "DEFAULT_INDICATORS",
    "DEFAULT_MAX_ERROR",
    "INDICATOR_NAMES",
    "build_chart_frame",
    "chart_panels",
    "draw_chart",
    "plot_chart",
    "resolve_indicators",
    "segment_vertices",
    "write_chart",
]

_CALCULATORS = {
    "ma": moving_average,
    "ema": exponential_moving_average,
    "boll": bollinger,
    "sar": parabolic_sar,
    "vwap": vwap,
    "donchian": donchian,
    "envelope": envelope,
    "volume": volume_ma,
    "macd": macd,
    "kdj": kdj,
    "rsi": rsi,
    "cci": cci,
    "wr": williams_r,
    "roc": roc,
    "mom": momentum,
    "bias": bias,
    "mfi": mfi,
    "psy": psy,
    "dmi": dmi,
    "obv": obv,
    "atr": atr,
    "fib": fibonacci,
    "pivot": pivot_points,
}


def build_chart_frame(df: pd.DataFrame, max_error: float = DEFAULT_MAX_ERROR, indicators=None) -> pd.DataFrame:
    """计算选定指标，并按同一行序附上分段斜率。不修改调用方的表。"""
    names = resolve_indicators(indicators)
    out = prepare(df)
    for name in names:
        part = _CALCULATORS[name](out, prepared=True)
        for column in part.columns:
            out[column] = part[column].to_numpy()
    slopes = linear_segment(out, max_error=max_error)
    if len(out) != len(slopes):
        raise ValueError("指标与分段行数不一致")
    out["segment"] = slopes["segment"].to_numpy()
    return out


def plot_chart(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    max_error: float = DEFAULT_MAX_ERROR,
    title: str | None = None,
    indicators=None,
) -> Path:
    """计算选定指标和分段，按同花顺配色画图并保存 PNG。返回图片路径。"""
    frame = build_chart_frame(df, max_error=max_error, indicators=indicators)
    figure = draw_chart(frame, title=title or _title_from_frame(frame), indicators=indicators)
    destination = Path(output_path) if output_path is not None else _default_chart_path(frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120, facecolor=figure.get_facecolor())
    import matplotlib.pyplot as plt

    plt.close(figure)
    return destination


def write_chart(
    input_path: str | Path,
    output_path: str | Path | None = None,
    max_error: float = DEFAULT_MAX_ERROR,
    indicators=None,
) -> Path:
    """读取 CSV，绘制指标图。省略输出路径时在文件名后加 _chart.png。"""
    source = Path(input_path)
    if output_path is None:
        destination = source.with_name(f"{source.stem}_chart.png")
    else:
        destination = Path(output_path)
    frame = pd.read_csv(source)
    title = _title_from_frame(frame) if "symbol" in frame.columns else source.stem
    return plot_chart(frame, destination, max_error=max_error, title=title, indicators=indicators)


def _title_from_frame(frame: pd.DataFrame) -> str | None:
    if "symbol" not in frame.columns or len(frame) == 0:
        return None
    symbols = frame["symbol"].dropna().astype(str).unique()
    if len(symbols) == 1:
        return f"{symbols[0]} 技术指标"
    return None


def _default_chart_path(frame: pd.DataFrame) -> Path:
    if "symbol" in frame.columns and len(frame):
        symbols = frame["symbol"].dropna().astype(str).unique()
        if len(symbols) == 1:
            safe = symbols[0].replace(".", "_")
            return Path(f"{safe}_chart.png")
    return Path("chart.png")
