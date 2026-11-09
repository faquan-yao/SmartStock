"""主图、副图的绘制。"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=False)

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from charts.segment_path import break_on_trend_change, segment_vertices
from charts.selection import chart_panels, is_overlay, required_columns, resolve_indicators
from charts.style import (
    BG,
    CLOSE,
    DOWN,
    GUIDE,
    LINE_COLORS,
    MA_COLORS,
    SEGMENT,
    SPINE,
    TEXT,
    UP,
    VOL_COLORS,
    apply_axes,
    configure_font,
)
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume", "eob")

_FIB_LABELS = {
    "fib_0": "0%",
    "fib_236": "23.6%",
    "fib_382": "38.2%",
    "fib_500": "50%",
    "fib_618": "61.8%",
    "fib_786": "78.6%",
    "fib_1000": "100%",
    "fib_1272": "127.2%",
    "fib_1618": "161.8%",
    "fib_2618": "261.8%",
}
_PIVOT_LABELS = {
    "pivot_r3": "R3",
    "pivot_r2": "R2",
    "pivot_r1": "R1",
    "pivot_pp": "PP",
    "pivot_s1": "S1",
    "pivot_s2": "S2",
    "pivot_s3": "S3",
}


def require_chart_columns(frame, indicators=None) -> None:
    required = [*OHLCV_COLUMNS, "segment", *required_columns(indicators)]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("缺少绘图列: " + ", ".join(missing))


def draw_chart(frame, title: str | None = None, indicators=None, predictions=None) -> Figure:
    """按选定指标画白底图。K 线和分段折线始终绘制。返回 Figure，调用方负责关闭。

    predictions 含 eob 和 segment 时，在 K 线正下方单独加一块预测斜率折线图。
    横轴与 K 线对齐。纵轴标注这组斜率的最大值、最小值和 0，并在这三处画虚线。
    最小值大于 0 时不画最小值，最大值小于 0 时不画最大值。
    """
    names = resolve_indicators(indicators)
    require_chart_columns(frame, names)
    if len(frame) == 0:
        raise ValueError("没有可绘制的 K 线")

    configure_font()
    import matplotlib.pyplot as plt

    panels = list(chart_panels(names))
    drawers = [_draw_price]
    drawers.extend(_PANEL_DRAWERS[name] for name in names if not is_overlay(name))
    if predictions is not None and len(predictions) > 0:
        panels.insert(1, ("预测斜率", 1.15))
        drawers.insert(1, lambda ax, data, _selected: _draw_predicted_slopes(ax, data, predictions))
    ratios = [height for _title, height in panels]
    figure, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=(14, max(4.8, sum(ratios) * 0.92)),
        gridspec_kw={"height_ratios": ratios},
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(axes.ravel())
    figure.patch.set_facecolor(BG)
    figure.chart_axes = axes
    figure.suptitle(title or _default_title(frame), color=TEXT, fontsize=14)

    selected = set(names)
    for ax, (panel_name, _height), drawer in zip(axes, panels, drawers):
        drawer(ax, frame, selected)
        _finish_panel(ax, panel_name)

    _format_dates(axes[-1], frame["eob"])
    axes[0].set_xlim(-0.5, len(frame) - 0.5)
    return figure


def _default_title(frame) -> str:
    if "symbol" in frame.columns and len(frame):
        symbols = frame["symbol"].dropna().astype(str).unique()
        if len(symbols) == 1:
            return f"{symbols[0]} 技术指标"
    return "技术指标"


def _finish_panel(ax: Axes, title: str) -> None:
    apply_axes(ax)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT, pad=2)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.08, 1.0),
            fontsize=7,
            frameon=False,
            ncol=8,
            borderaxespad=0.0,
            labelcolor=TEXT,
        )
        for text in legend.get_texts():
            text.set_color(TEXT)
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span == 0:
        span = abs(ymax) * 0.05 or 1.0
    ax.set_ylim(ymin - span * 0.03, ymax + span * 0.04)


def _format_dates(ax: Axes, eob) -> None:
    import pandas as pd

    size = len(eob)
    if size == 0:
        return
    step = max(1, size // 8)
    ticks = list(range(0, size, step))
    if ticks[-1] != size - 1:
        ticks.append(size - 1)
    parsed = pd.to_datetime(eob, errors="coerce")
    labels = []
    for index in ticks:
        value = parsed.iloc[index]
        if pd.isna(value):
            labels.append(str(eob.iloc[index])[:10])
        else:
            labels.append(pd.Timestamp(value).strftime("%Y-%m-%d"))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8, color=TEXT)


def _plot_series(ax: Axes, values, label: str, color: str, **kwargs):
    series = np.asarray(values, dtype=float)
    if not np.isfinite(series).any():
        return None
    linewidth = kwargs.pop("linewidth", 1.0)
    line, = ax.plot(np.arange(len(series)), series, color=color, label=label, linewidth=linewidth, **kwargs)
    return line


def _guides(ax: Axes, levels: tuple[float, ...]) -> None:
    for level in levels:
        ax.axhline(level, color=GUIDE, linestyle="--", linewidth=0.6, zorder=0)


def _draw_kdj(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(ax, frame, (("kdj_k", "K", LINE_COLORS[0]), ("kdj_d", "D", LINE_COLORS[1]), ("kdj_j", "J", LINE_COLORS[2])), (20, 50, 80))


def _draw_rsi(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(
        ax,
        frame,
        (
            ("rsi6", "RSI6", LINE_COLORS[0]),
            ("rsi12", "RSI12", LINE_COLORS[1]),
            ("rsi14", "RSI14", LINE_COLORS[2]),
            ("rsi24", "RSI24", LINE_COLORS[3]),
        ),
        (30, 70),
    )


def _draw_cci(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(ax, frame, (("cci", "CCI", LINE_COLORS[0]),), (-100, 100))


def _draw_wr(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(ax, frame, (("wr6", "WR6", LINE_COLORS[0]), ("wr10", "WR10", LINE_COLORS[1])), (-20, -80))


def _draw_roc(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(ax, frame, (("roc", "ROC", LINE_COLORS[0]),), (0,))


def _draw_mom(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(ax, frame, (("mom", "MOM", LINE_COLORS[1]),), (0,))


def _draw_bias(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(
        ax,
        frame,
        (("bias6", "BIAS6", LINE_COLORS[0]), ("bias12", "BIAS12", LINE_COLORS[1]), ("bias24", "BIAS24", LINE_COLORS[2])),
        (0,),
    )


def _draw_mfi(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(ax, frame, (("mfi", "MFI", LINE_COLORS[0]),), (20, 80))


def _draw_psy(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(ax, frame, (("psy", "PSY", LINE_COLORS[0]),), (25, 75))


def _draw_dmi(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(ax, frame, (("pdi", "+DI", UP), ("mdi", "-DI", DOWN), ("adx", "ADX", LINE_COLORS[1])), ())


def _draw_obv(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(ax, frame, (("obv", "OBV", LINE_COLORS[0]),), ())


def _draw_atr(ax: Axes, frame, _selected=None) -> None:
    _draw_lines(ax, frame, (("atr", "ATR", LINE_COLORS[0]),), ())


def _draw_lines(ax: Axes, frame, series: tuple[tuple[str, str, str], ...], guides: tuple[float, ...]) -> None:
    _guides(ax, guides)
    for column, label, color in series:
        _plot_series(ax, frame[column], label, color)


def _draw_price(ax: Axes, frame, selected: set[str]) -> None:
    _candles(ax, frame)
    if "ma" in selected:
        for column, color in MA_COLORS.items():
            _plot_series(ax, frame[column], column.upper(), color, linewidth=1.0)
    if "ema" in selected:
        _plot_series(ax, frame["ema12"], "EMA12", LINE_COLORS[1], linewidth=1.0)
        _plot_series(ax, frame["ema26"], "EMA26", LINE_COLORS[4], linewidth=1.0)
    if "boll" in selected:
        _plot_series(ax, frame["boll_mid"], "BOLL中", CLOSE, linewidth=0.8, linestyle="--")
        _plot_series(ax, frame["boll_upper"], "BOLL上", "#e6c35c", linewidth=0.8)
        _plot_series(ax, frame["boll_lower"], "BOLL下", "#e6c35c", linewidth=0.8)
    if "sar" in selected:
        _draw_sar(ax, frame)
    if "donchian" in selected:
        _plot_series(ax, frame["donchian_upper"], "唐奇安上", "#7ec8ff", linewidth=0.7)
        _plot_series(ax, frame["donchian_mid"], "唐奇安中", "#7ec8ff", linewidth=0.7, linestyle="--")
        _plot_series(ax, frame["donchian_lower"], "唐奇安下", "#7ec8ff", linewidth=0.7)
    if "envelope" in selected:
        _plot_series(ax, frame["env_upper"], "轨道上", "#c080ff", linewidth=0.7)
        _plot_series(ax, frame["env_mid"], "轨道中", "#c080ff", linewidth=0.7, linestyle="--")
        _plot_series(ax, frame["env_lower"], "轨道下", "#c080ff", linewidth=0.7)
    if "vwap" in selected:
        _plot_series(ax, frame["vwap"], "VWAP", "#ff66cc", linewidth=1.0)
    closes = frame["close"].to_numpy(dtype=float)
    xs, ys = segment_vertices(closes, frame["segment"].to_numpy(dtype=float))
    if len(xs):
        ax.plot(xs, ys, color=SEGMENT, linewidth=1.8, label="分段", zorder=5)


def _draw_predicted_slopes(ax: Axes, frame, predictions) -> None:
    """在独立副图上画预测斜率折线。横坐标是 K 线主图里对应 eob 的序号。"""
    import pandas as pd
    from matplotlib.ticker import FixedLocator

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("预测结果需要 pandas.DataFrame")
    missing = [column for column in ("eob", "segment") if column not in predictions.columns]
    if missing:
        raise ValueError("预测结果缺少列: " + ", ".join(missing))

    xs, ys = _prediction_positions(frame["eob"], predictions)
    if len(xs) == 0:
        raise ValueError("预测结果的 eob 不在当前图中")

    order = np.argsort(xs)
    low = float(np.min(ys))
    high = float(np.max(ys))
    ticks = {0.0}
    if high > 0:
        ticks.add(high)
        ax.axhline(high, color=TEXT, linestyle="--", linewidth=0.8, zorder=0)
    if low < 0:
        ticks.add(low)
        ax.axhline(low, color=TEXT, linestyle="--", linewidth=0.8, zorder=0)
    ax.axhline(0, color=UP, linestyle="--", linewidth=0.8, zorder=0)
    ax.plot(xs[order], ys[order], color=LINE_COLORS[4], linewidth=1.2, label="预测斜率", zorder=3)
    ax.set_ylim(min(low, 0.0), max(high, 0.0))
    ax.yaxis.set_major_locator(FixedLocator(sorted(ticks)))


def _prediction_positions(frame_eob, predictions):
    import pandas as pd

    slopes = pd.to_numeric(predictions["segment"], errors="coerce").to_numpy(dtype=float)
    frame_time = pd.to_datetime(frame_eob, errors="coerce", utc=True)
    pred_time = pd.to_datetime(predictions["eob"], errors="coerce", utc=True)
    if bool(frame_time.notna().all()) and bool(pred_time.notna().all()):
        index_of = {value: index for index, value in enumerate(frame_time)}
        keys = list(pred_time)
    else:
        index_of = {str(value): index for index, value in enumerate(frame_eob)}
        keys = [str(value) for value in predictions["eob"]]

    xs = []
    ys = []
    for key, slope in zip(keys, slopes):
        index = index_of.get(key)
        if index is None or not np.isfinite(slope):
            continue
        xs.append(index)
        ys.append(float(slope))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _draw_sar(ax: Axes, frame) -> None:
    sar = frame["sar"].to_numpy(dtype=float)
    trend = frame["sar_trend"].to_numpy(dtype=float)
    x = np.arange(len(sar))
    up = np.isfinite(sar) & (trend == 1)
    down = np.isfinite(sar) & (trend == -1)
    if up.any():
        ax.scatter(x[up], sar[up], s=8, c=UP, marker="o", label="SAR多", zorder=4)
    if down.any():
        ax.scatter(x[down], sar[down], s=8, c=DOWN, marker="o", label="SAR空", zorder=4)


def _candles(ax: Axes, frame) -> None:
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    x = np.arange(len(close))
    colors = np.where(close >= open_, UP, DOWN)
    ax.vlines(x, low, high, colors=colors, linewidth=0.7, zorder=2)
    half = 0.3
    for index in range(len(close)):
        if not np.isfinite([open_[index], close[index]]).all():
            continue
        body_low = min(open_[index], close[index])
        body_high = max(open_[index], close[index])
        color = colors[index]
        if body_high == body_low:
            ax.hlines(body_low, index - half, index + half, color=color, linewidth=0.8, zorder=3)
            continue
        face = BG if close[index] >= open_[index] else color
        ax.add_patch(
            Rectangle(
                (index - half, body_low),
                half * 2,
                body_high - body_low,
                facecolor=face,
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
        )


def _draw_fibonacci(ax: Axes, frame, _selected=None) -> None:
    _plot_series(ax, frame["close"], "收盘价", CLOSE, linewidth=1.0)
    trend = frame["fib_trend"].to_numpy(dtype=float)
    fib_columns = [name for name in frame.columns if str(name).startswith("fib_") and name != "fib_trend"]
    for index, column in enumerate(fib_columns):
        series = break_on_trend_change(frame[column], trend, reference=frame["close"])
        _plot_series(ax, series, _FIB_LABELS.get(column, column), LINE_COLORS[index % len(LINE_COLORS)], linewidth=0.7)
    _draw_fib_trend(ax, frame)


def _draw_pivot(ax: Axes, frame, _selected=None) -> None:
    _plot_series(ax, frame["close"], "收盘价", CLOSE, linewidth=1.0)
    for column, label in _PIVOT_LABELS.items():
        color = UP if label.startswith("R") else DOWN if label.startswith("S") else CLOSE
        _plot_series(ax, frame[column], label, color, linewidth=0.7, linestyle="--")


def _draw_fib_trend(ax: Axes, frame) -> None:
    trend = frame["fib_trend"].to_numpy(dtype=float)
    if not np.isfinite(trend).any():
        return
    twin = ax.twinx()
    twin.plot(np.arange(len(trend)), trend, color="#888888", linewidth=0.8, drawstyle="steps-post", label="方向")
    twin.set_ylim(-1.5, 1.5)
    twin.set_yticks([-1, 1])
    twin.set_facecolor("none")
    twin.grid(False)
    twin.yaxis.tick_left()
    twin.tick_params(axis="y", colors="#888888", labelsize=8, labelleft=True, labelright=False)
    for spine in twin.spines.values():
        spine.set_color(SPINE)
    twin.spines["right"].set_visible(False)
    twin.set_ylabel("方向", color="#888888", fontsize=8)
    handles, labels = twin.get_legend_handles_labels()
    if handles:
        legend = twin.legend(handles, labels, loc="upper right", fontsize=7, frameon=False, labelcolor=TEXT)
        for text in legend.get_texts():
            text.set_color(TEXT)


def _draw_volume(ax: Axes, frame, _selected=None) -> None:
    volume = frame["volume"].to_numpy(dtype=float)
    open_ = frame["open"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    x = np.arange(len(volume))
    colors = np.where(close >= open_, UP, DOWN)
    valid = np.isfinite(volume)
    ax.bar(x[valid], volume[valid], width=0.6, color=colors[valid], label="成交量", zorder=2)
    for column, color in VOL_COLORS.items():
        _plot_series(ax, frame[column], column.replace("vol_ma", "VOL"), color, linewidth=1.0)


def _draw_macd(ax: Axes, frame, _selected=None) -> None:
    _guides(ax, (0,))
    hist = frame["macd_hist"].to_numpy(dtype=float)
    x = np.arange(len(hist))
    valid = np.isfinite(hist)
    colors = np.where(hist >= 0, UP, DOWN)
    ax.bar(x[valid], hist[valid], width=0.6, color=colors[valid], label="MACD", zorder=2)
    _plot_series(ax, frame["macd_dif"], "DIF", LINE_COLORS[0], linewidth=1.0)
    _plot_series(ax, frame["macd_dea"], "DEA", LINE_COLORS[1], linewidth=1.0)


_PANEL_DRAWERS = {
    "volume": _draw_volume,
    "macd": _draw_macd,
    "kdj": _draw_kdj,
    "rsi": _draw_rsi,
    "cci": _draw_cci,
    "wr": _draw_wr,
    "roc": _draw_roc,
    "mom": _draw_mom,
    "bias": _draw_bias,
    "mfi": _draw_mfi,
    "psy": _draw_psy,
    "dmi": _draw_dmi,
    "obv": _draw_obv,
    "atr": _draw_atr,
    "fib": _draw_fibonacci,
    "pivot": _draw_pivot,
}
