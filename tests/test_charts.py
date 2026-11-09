"""图形：折线端点、副图内容和输出文件。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from charts import (
    DEFAULT_INDICATORS,
    INDICATOR_NAMES,
    build_chart_frame,
    chart_panels,
    draw_chart,
    plot_chart,
    resolve_indicators,
    segment_vertices,
    write_chart,
)
from charts.segment_path import break_on_trend_change
from charts.style import BG, DOWN, SEGMENT, TEXT, UP
from indicators import indicator_column_names

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CANDIDATES = (
    ROOT / "data" / "SHSE_600031.csv",
    ROOT / "data" / "history_SHSE600000.csv",
)


def frame(rows, symbol=None) -> pd.DataFrame:
    data = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    data["eob"] = [f"2024-01-{day:02d}" for day in range(1, len(data) + 1)]
    if symbol is not None:
        data["symbol"] = symbol
    return data


def rising_then_falling() -> pd.DataFrame:
    closes = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]
    rows = []
    for index, close in enumerate(closes):
        # 偶数根阳线，奇数根阴线，收盘价序列本身不变。
        open_ = close - 0.2 if index % 2 == 0 else close + 0.2
        rows.append((open_, max(open_, close) + 0.2, min(open_, close) - 0.2, close, 100))
    return frame(rows, symbol="TEST.000001")


def line_by_label(ax, label: str):
    for line in ax.get_lines():
        if line.get_label() == label:
            return line
    raise AssertionError(f"找不到曲线 {label}")


def _horizontal_guides(ax):
    found = {}
    for line in ax.get_lines():
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if ydata.size >= 2 and np.allclose(ydata, ydata[0]):
            found[float(ydata[0])] = line
    return found


def test_segment_vertices_follow_shared_endpoint_rule():
    closes = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]
    slopes = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    xs, ys = segment_vertices(closes, slopes)
    assert xs.tolist() == [0, 3, 6]
    assert ys.tolist() == [1, 4, 1]
    for start, end in zip(xs[:-1], xs[1:]):
        start_index = int(start)
        end_index = int(end)
        expected = (closes[end_index] - closes[start_index]) / (end_index - start_index)
        assert slopes[start_index] == pytest.approx(expected)


def test_price_level_breaks_when_fib_direction_flips():
    got = break_on_trend_change([1.0, 1.0, 8.0, 8.0], [1.0, 1.0, -1.0, -1.0])
    assert got[0] == pytest.approx(1)
    assert np.isnan(got[2])
    assert got[3] == pytest.approx(8)
    jumped = break_on_trend_change([20.0, 20.0, 32.0, 32.0], [1.0, 1.0, 1.0, 1.0], reference=[20.0, 20.0, 20.0, 20.0])
    assert np.isnan(jumped[2])
    assert jumped[3] == pytest.approx(32)


def test_straight_segment_is_one_line_and_single_bar_draws_nothing():
    xs, ys = segment_vertices([1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 1, 1, 1])
    assert xs.tolist() == [0, 4]
    assert ys.tolist() == [1, 5]
    empty_x, empty_y = segment_vertices([10.0], [np.nan])
    assert empty_x.size == 0
    assert empty_y.size == 0


def test_chart_frame_appends_indicators_and_segment_without_mutating():
    data = rising_then_falling()
    original = data.copy()
    built = build_chart_frame(data, max_error=0.03)
    pd.testing.assert_frame_equal(data, original)
    assert list(built.columns[-1:]) == ["segment"]
    assert "ma5" in built.columns
    assert "macd_dif" in built.columns
    assert "vol_ma5" in built.columns
    assert "cci" not in built.columns
    assert built["segment"].tolist() == pytest.approx([1, 1, 1, -1, -1, -1, -1])
    everything = build_chart_frame(data, indicators="all")
    for column in indicator_column_names():
        assert column in everything.columns


def test_existing_indicator_columns_are_kept():
    data = rising_then_falling()
    data["ma5"] = 7.0
    data["macd_dif"] = -3.0
    data["segment"] = 0.5
    fresh = build_chart_frame(rising_then_falling(), max_error=0.03)
    built = build_chart_frame(data, max_error=0.03)
    assert built["ma5"].tolist() == pytest.approx([7.0] * len(data))
    assert built["ma10"].tolist() == pytest.approx(fresh["ma10"].tolist(), nan_ok=True)
    assert built["macd_dif"].tolist() == pytest.approx([-3.0] * len(data))
    assert built["macd_dea"].tolist() == pytest.approx(fresh["macd_dea"].tolist(), nan_ok=True)
    assert built["segment"].tolist() == pytest.approx([0.5] * len(data))


def test_draw_chart_panels_candles_and_segment_line():
    built = build_chart_frame(rising_then_falling(), max_error=0.03)
    figure = draw_chart(built)
    try:
        axes = figure.chart_axes
        assert [ax.get_title(loc="left") for ax in axes] == [title for title, _height in chart_panels()]
        assert DEFAULT_INDICATORS == ("ma", "volume", "macd")
        assert len(axes) == 3

        price = axes[0]
        patches = price.patches
        assert to_rgba(patches[0].get_edgecolor()) == pytest.approx(to_rgba(UP))
        assert to_rgba(patches[0].get_facecolor()) == pytest.approx(to_rgba(BG))
        assert to_rgba(patches[1].get_facecolor()) == pytest.approx(to_rgba(DOWN))

        segment = line_by_label(price, "分段")
        xs, ys = segment_vertices(built["close"], built["segment"])
        assert segment.get_xdata().tolist() == pytest.approx(xs.tolist())
        assert segment.get_ydata().tolist() == pytest.approx(ys.tolist())
        assert to_rgba(segment.get_color()) == pytest.approx(to_rgba(SEGMENT))

        ma5 = line_by_label(price, "MA5")
        assert ma5.get_ydata() == pytest.approx(built["ma5"].to_numpy(dtype=float), nan_ok=True)
        assert line_by_label(axes[2], "DIF").get_ydata() == pytest.approx(
            built["macd_dif"].to_numpy(dtype=float), nan_ok=True
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_predicted_slopes_scatter_at_the_matching_eob():
    built = build_chart_frame(rising_then_falling(), max_error=0.03)
    predictions = pd.DataFrame(
        {
            "eob": [built["eob"].iloc[2], built["eob"].iloc[4], built["eob"].iloc[5], built["eob"].iloc[6]],
            "segment": [40.0, -30.0, 0.0, 12.0],
        }
    )
    figure = draw_chart(built, predictions=predictions)
    try:
        titles = [ax.get_title(loc="left") for ax in figure.chart_axes]
        assert titles[:4] == ["K线", "预测斜率", "成交量", "MACD"]
        price = figure.chart_axes[0]
        slope_axis = figure.chart_axes[1]
        assert price.get_ylim()[1] < 20
        assert price.yaxis.get_ticks_position() == "left"
        assert slope_axis.yaxis.get_ticks_position() == "left"
        assert figure.chart_axes[2].yaxis.get_ticks_position() == "left"
        line = line_by_label(slope_axis, "预测斜率")
        assert line.get_xdata().tolist() == [2, 4, 5, 6]
        assert line.get_ydata().tolist() == pytest.approx([40, -30, 0, 12])
        assert slope_axis.get_yticks().tolist() == pytest.approx([-30, 0, 40])
        guides = _horizontal_guides(slope_axis)
        assert guides[40].get_linestyle() == "--"
        assert guides[-30].get_linestyle() == "--"
        assert guides[0].get_linestyle() == "--"
        assert to_rgba(guides[40].get_color()) == pytest.approx(to_rgba(TEXT))
        assert to_rgba(guides[-30].get_color()) == pytest.approx(to_rgba(TEXT))
        assert to_rgba(guides[0].get_color()) == pytest.approx(to_rgba(UP))
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)

    positive = predictions.copy()
    positive["segment"] = [4.0, 12.0, 7.0, 9.0]
    positive_figure = draw_chart(built, predictions=positive)
    try:
        positive_axis = positive_figure.chart_axes[1]
        assert positive_axis.get_yticks().tolist() == pytest.approx([0, 12])
        positive_guides = _horizontal_guides(positive_axis)
        assert 4.0 not in positive_guides
        assert to_rgba(positive_guides[12].get_color()) == pytest.approx(to_rgba(TEXT))
        assert to_rgba(positive_guides[0].get_color()) == pytest.approx(to_rgba(UP))
    finally:
        import matplotlib.pyplot as plt

        plt.close(positive_figure)

    negative = predictions.copy()
    negative["segment"] = [-12.0, -4.0, -7.0, -9.0]
    negative_figure = draw_chart(built, predictions=negative)
    try:
        negative_axis = negative_figure.chart_axes[1]
        assert negative_axis.get_yticks().tolist() == pytest.approx([-12, 0])
        negative_guides = _horizontal_guides(negative_axis)
        assert -4.0 not in negative_guides
        assert to_rgba(negative_guides[-12].get_color()) == pytest.approx(to_rgba(TEXT))
        assert to_rgba(negative_guides[0].get_color()) == pytest.approx(to_rgba(UP))
    finally:
        import matplotlib.pyplot as plt

        plt.close(negative_figure)

    missing = predictions.drop(columns=["segment"])
    with pytest.raises(ValueError, match="segment"):
        draw_chart(built, predictions=missing)
    unknown = predictions.copy()
    unknown["eob"] = ["1999-01-01", "1999-01-02", "1999-01-03", "1999-01-04"]
    with pytest.raises(ValueError, match="eob"):
        draw_chart(built, predictions=unknown)


def test_indicators_argument_selects_panels_and_rejects_unknown_names():
    assert resolve_indicators("kdj，rsi") == ("kdj", "rsi")
    assert resolve_indicators("all") == INDICATOR_NAMES
    with pytest.raises(ValueError, match="未知指标"):
        resolve_indicators("foo")

    built = build_chart_frame(rising_then_falling(), indicators=("kdj",))
    figure = draw_chart(built, indicators=("kdj",))
    try:
        assert [ax.get_title(loc="left") for ax in figure.chart_axes] == ["K线", "KDJ"]
        labels = {line.get_label() for line in figure.chart_axes[0].get_lines()}
        assert "分段" in labels
        assert "MA5" not in labels
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_fibonacci_panel_uses_a_direction_axis_instead_of_price_scale():
    built = build_chart_frame(rising_then_falling(), max_error=0, indicators=("fib",))
    # 短样本的方向值可能全空。这里只检查没有把 ±1 画进价格轴。
    figure = draw_chart(built, indicators=("fib",))
    try:
        price_labels = {line.get_label() for line in figure.chart_axes[1].get_lines()}
        assert "方向" not in price_labels
        assert "fib_trend" not in price_labels
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_missing_column_and_empty_frame_raise():
    with pytest.raises(ValueError, match="缺少绘图列"):
        draw_chart(frame([(1, 2, 0.5, 1.5, 10)]))
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "eob"])
    with pytest.raises(ValueError, match="没有可绘制的 K 线"):
        plot_chart(empty, "ignored.png")


def test_plot_chart_writes_png(tmp_path):
    destination = tmp_path / "out.png"
    saved = plot_chart(rising_then_falling(), destination, max_error=0.03)
    assert saved == destination
    assert destination.stat().st_size > 10_000


def test_write_chart_default_name_and_cli(tmp_path):
    source = tmp_path / "history_demo.csv"
    rising_then_falling().to_csv(source, index=False)
    destination = write_chart(source, max_error=0.03)
    assert destination.name == "history_demo_chart.png"
    assert destination.stat().st_size > 10_000
    destination.unlink()

    output = tmp_path / "cli.png"
    completed = subprocess.run(
        [sys.executable, "-m", "charts", str(source), "-o", str(output), "--max-error", "0.03"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(output)
    assert output.exists()


def test_sample_csv_smoke(tmp_path):
    sample = next((path for path in SAMPLE_CANDIDATES if path.exists()), None)
    if sample is None:
        pytest.skip("样例行情不在 data 目录")
    saved = write_chart(sample, tmp_path / "sample.png")
    assert saved.stat().st_size > 10_000
