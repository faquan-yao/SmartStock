"""自底向上线性分段：短序列锁定斜率和切点，再用样例 CSV 检查不变量。"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from segment import DEFAULT_MAX_ERROR, linear_segment, write_segments

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "SHSE_600031.csv"


def _frame(closes, eob=None) -> pd.DataFrame:
    data = pd.DataFrame({"close": closes})
    data["eob"] = eob if eob is not None else [f"2024-01-{day:02d}" for day in range(1, len(closes) + 1)]
    return data


def _max_relative_error(closes: np.ndarray, start: int, end: int) -> float:
    if end - start <= 1:
        return 0.0
    y0 = float(closes[start])
    y1 = float(closes[end])
    span = end - start
    error = 0.0
    for step in range(1, span):
        predicted = y0 + (y1 - y0) * (step / span)
        actual = float(closes[start + step])
        error = max(error, abs(actual - predicted) / abs(actual))
    return error


def test_default_threshold_is_three_percent():
    assert DEFAULT_MAX_ERROR == 0.03


def test_straight_line_slope_is_constant():
    got = linear_segment(_frame([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert got["segment"].tolist() == pytest.approx([1, 1, 1, 1, 1])


def test_peak_breakpoint_uses_the_following_slope():
    got = linear_segment(_frame([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]), max_error=0.03)
    assert got["segment"].tolist() == pytest.approx([1, 1, 1, -1, -1, -1, -1])


def test_large_threshold_merges_into_one_flat_slope():
    got = linear_segment(_frame([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]), max_error=0.75)
    assert got["segment"].tolist() == pytest.approx([0, 0, 0, 0, 0, 0, 0])


def test_zero_threshold_keeps_only_exact_lines():
    got = linear_segment(_frame([1.0, 2.0, 3.0, 2.5]), max_error=0)
    assert got["segment"].tolist() == pytest.approx([1, 1, -0.5, -0.5])


def test_two_point_segment_before_a_turn():
    got = linear_segment(_frame([1.0, 3.0, 2.0]), max_error=0)
    assert got["segment"].tolist() == pytest.approx([2, -1, -1])


def test_single_bar_slope_is_empty():
    got = linear_segment(_frame([10.0], eob=["2024-01-02"]))
    assert got["eob"].tolist() == ["2024-01-02"]
    assert math.isnan(got["segment"].iloc[0])


def test_empty_frame_keeps_columns():
    got = linear_segment(pd.DataFrame({"eob": [], "close": []}))
    assert got.columns.tolist() == ["eob", "segment"]
    assert got.empty


def test_rows_follow_eob_without_mutating_input():
    raw = _frame([3.0, 1.0, 2.0], eob=["2024-01-03", "2024-01-01", "2024-01-02"])
    original = raw.copy()
    got = linear_segment(raw, max_error=1)
    pd.testing.assert_frame_equal(raw, original)
    assert got["eob"].tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert got["segment"].tolist() == pytest.approx([1, 1, 1])


def test_missing_or_non_numeric_close_raises():
    with pytest.raises(ValueError, match="close"):
        linear_segment(pd.DataFrame({"eob": ["2024-01-01"]}))
    bad = _frame([1.0, 2.0])
    bad["close"] = ["x", "y"]
    with pytest.raises(ValueError, match="close"):
        linear_segment(bad)


def test_negative_or_bool_threshold_raises():
    with pytest.raises(ValueError, match="误差阈值"):
        linear_segment(_frame([1.0, 2.0]), max_error=-0.01)
    with pytest.raises(ValueError, match="误差阈值"):
        linear_segment(_frame([1.0, 2.0]), max_error=True)


def test_zero_close_raises():
    with pytest.raises(ValueError, match="0"):
        linear_segment(_frame([1.0, 0.0, 1.0]))


def test_write_segments_round_trip(tmp_path):
    source = tmp_path / "SHSE_600031.csv"
    _frame([1.0, 2.0, 3.0, 4.0, 5.0]).to_csv(source, index=False)
    destination = write_segments(source)
    assert destination.name == "SHSE_600031_segments.csv"
    saved = pd.read_csv(destination)
    assert saved["segment"].tolist() == pytest.approx([1, 1, 1, 1, 1])


def test_cli_writes_csv(tmp_path):
    source = tmp_path / "in.csv"
    destination = tmp_path / "out.csv"
    _frame([1.0, 2.0, 3.0]).to_csv(source, index=False)
    completed = subprocess.run(
        [sys.executable, "-m", "segment", str(source), "-o", str(destination), "--max-error", "0.03"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(destination)
    assert pd.read_csv(destination)["segment"].tolist() == pytest.approx([1, 1, 1])


def test_sample_csv_smoke():
    if not SAMPLE.exists():
        pytest.skip("样例行情不在 data/SHSE_600031.csv")
    history = pd.read_csv(SAMPLE)
    original = history.copy()
    got = linear_segment(history)
    pd.testing.assert_frame_equal(history, original)
    assert got.columns.tolist() == ["eob", "segment"]
    assert len(got) == len(history)
    assert got["eob"].tolist() == history["eob"].tolist()
    assert pd.to_datetime(got["eob"], utc=True).is_monotonic_increasing

    closes = history["close"].to_numpy(dtype=float)
    slopes = got["segment"].to_numpy(dtype=float)
    assert np.isfinite(slopes).all()
    starts = [0] + [index for index in range(1, len(slopes)) if slopes[index] != slopes[index - 1]]
    ends = [starts[index + 1] if index + 1 < len(starts) else len(slopes) - 1 for index in range(len(starts))]
    for index, start in enumerate(starts):
        end = ends[index]
        expected = (closes[end] - closes[start]) / (end - start)
        assert slopes[start] == pytest.approx(expected)
        filled_until = end - 1 if index + 1 < len(starts) else end
        assert np.all(slopes[start : filled_until + 1] == slopes[start])
        assert _max_relative_error(closes, start, end) <= DEFAULT_MAX_ERROR
    for index in range(len(starts) - 1):
        assert _max_relative_error(closes, starts[index], ends[index + 1]) > DEFAULT_MAX_ERROR
