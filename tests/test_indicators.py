"""手算短序列锁定公式，再用样例 CSV 做整表冒烟。"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from indicators import compute_all, indicator_column_names, write_indicators
from indicators._common import ema, ma, sma
from indicators.momentum import cci, kdj, momentum, roc, rsi, williams_r
from indicators.other import bias, dmi, fibonacci, pivot_points, psy
from indicators.trend import bollinger, macd, parabolic_sar
from indicators.volatility import atr, donchian, envelope
from indicators.volume import mfi, obv, volume_ma, vwap

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "history_SHSE600000.csv"

EXPECTED_COLUMNS = [
    "ma5", "ma10", "ma20", "ma60", "ma120", "ma250",
    "ema12", "ema26",
    "macd_dif", "macd_dea", "macd_hist",
    "boll_mid", "boll_upper", "boll_lower",
    "sar", "sar_trend",
    "rsi6", "rsi12", "rsi14", "rsi24",
    "kdj_k", "kdj_d", "kdj_j",
    "cci",
    "wr6", "wr10",
    "roc", "mom",
    "vol_ma5", "vol_ma10", "vol_ma20",
    "obv", "vwap", "mfi",
    "atr",
    "donchian_upper", "donchian_mid", "donchian_lower",
    "env_mid", "env_upper", "env_lower",
    "bias6", "bias12", "bias24",
    "psy",
    "pdi", "mdi", "adx",
    "fib_trend",
    "fib_0", "fib_236", "fib_382", "fib_500", "fib_618", "fib_786", "fib_1000",
    "fib_1272", "fib_1618", "fib_2618",
    "pivot_pp", "pivot_r1", "pivot_r2", "pivot_r3", "pivot_s1", "pivot_s2", "pivot_s3",
]


def frame(rows, eob=None, amount=None) -> pd.DataFrame:
    data = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    if eob is not None:
        data["eob"] = eob
    if amount is not None:
        data["amount"] = amount
    return data


def flat(closes, volume=1):
    return frame([(close, close, close, close, volume) for close in closes])


def test_ma_is_arithmetic_mean():
    got = ma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert got.iloc[:2].isna().all()
    assert got.iloc[2:].tolist() == pytest.approx([2, 3, 4])


def test_ema_uses_tongdaxin_seed():
    got = ema(pd.Series([1.0, 2.0, 3.0, 4.0]), 3)
    assert got.tolist() == pytest.approx([1, 1.5, 2.25, 3.125])


def test_ema_matches_adjust_false_span():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 8.0])
    assert ema(series, 5).tolist() == pytest.approx(series.ewm(span=5, adjust=False).mean().tolist())


def test_sma_uses_tongdaxin_weights():
    got = sma(pd.Series([1.0, 2.0, 3.0, 4.0]), 3, 1)
    assert got.tolist() == pytest.approx([1, 4 / 3, 17 / 9, 70 / 27])


def test_window_must_be_positive_integer():
    with pytest.raises(ValueError):
        ma(pd.Series([1.0, 2.0]), 0)
    with pytest.raises(ValueError):
        sma(pd.Series([1.0, 2.0]), 2, 3)
    with pytest.raises(ValueError):
        ma(pd.Series([1.0]), True)


def test_macd_histogram_is_twice_the_difference():
    out = macd(flat([1, 2, 3, 4, 5]), fast=2, slow=3, signal=2)
    assert out["macd_dif"].tolist() == pytest.approx([0, 1 / 6, 11 / 36, 85 / 216, 575 / 1296])
    assert out["macd_dea"].tolist() == pytest.approx([0, 1 / 9, 13 / 54, 37 / 108, 797 / 1944])
    assert out["macd_hist"].tolist() == pytest.approx((2 * (out["macd_dif"] - out["macd_dea"])).tolist())


def test_bollinger_uses_population_standard_deviation():
    out = bollinger(flat([1, 2, 3]), n=3, k=2)
    std = math.sqrt(2 / 3)
    assert out["boll_mid"].iloc[2] == pytest.approx(2)
    assert out["boll_upper"].iloc[2] == pytest.approx(2 + 2 * std)
    assert out["boll_lower"].iloc[2] == pytest.approx(2 - 2 * std)
    assert out["boll_upper"].iloc[2] != pytest.approx(4)


def test_sar_rises_with_new_highs_and_clamps_on_the_way_down():
    rising = parabolic_sar(frame([
        (9, 10, 8, 9, 1),
        (10, 11, 9, 10, 1),
        (11, 12, 10, 11, 1),
        (12, 13, 11, 12, 1),
        (11, 12, 9, 11, 1),
    ]))
    # 多头时 SAR 不能高于前两根最低价，所以前几根被压在 8。
    assert rising["sar"].tolist() == pytest.approx([8, 8, 8, 8.24, 8.6208])
    assert rising["sar_trend"].tolist() == pytest.approx([1, 1, 1, 1, 1])

    falling = parabolic_sar(frame([
        (10, 11, 9, 10, 1),
        (9, 10, 8, 9, 1),
        (8, 9.5, 7, 8, 1),
        (7.5, 9, 6, 7, 1),
    ]))
    assert falling["sar"].tolist() == pytest.approx([11, 11, 11, 10.76])
    assert falling["sar_trend"].tolist() == pytest.approx([-1, -1, -1, -1])


def test_sar_jumps_to_prior_extreme_on_reversal():
    out = parabolic_sar(frame([
        (9.5, 10, 9, 9.5, 1),
        (11.5, 12, 11, 11.5, 1),
        (9.5, 11, 8, 9, 1),
    ]))
    assert out["sar"].tolist() == pytest.approx([9, 9, 12])
    assert out["sar_trend"].tolist() == pytest.approx([1, 1, -1])


def test_rsi_smooths_gains_with_sma():
    out = rsi(flat([10, 11, 12, 11, 13]), periods=(3,))
    assert out["rsi3"].iloc[0] != out["rsi3"].iloc[0]
    assert out["rsi3"].iloc[1:].tolist() == pytest.approx([100, 100, 200 / 3, 250 / 3])


def test_kdj_and_flat_range():
    out = kdj(frame([
        (9, 10, 8, 9, 1),
        (11, 12, 9, 11, 1),
        (10.5, 11, 10, 10.5, 1),
        (12, 13, 11, 12, 1),
        (10, 12, 9, 10, 1),
    ]), n=3, m1=3, m2=3)
    assert out["kdj_k"].iloc[2:].tolist() == pytest.approx([62.5, 200 / 3, 475 / 9])
    assert out["kdj_d"].iloc[2:].tolist() == pytest.approx([62.5, 575 / 9, 1625 / 27])
    assert out["kdj_j"].iloc[2:].tolist() == pytest.approx([62.5, 650 / 9, 1025 / 27])

    flat_band = kdj(flat([5, 5, 5]), n=3)
    assert flat_band["kdj_k"].isna().all()


def test_cci_williams_roc_and_momentum():
    bars = frame([
        (9, 10, 8, 9, 1),
        (11, 12, 9, 11, 1),
        (10.5, 11, 10, 10.5, 1),
        (12, 13, 11, 12, 1),
        (10, 12, 9, 10, 1),
    ])
    assert cci(bars, n=3)["cci"].iloc[2] == pytest.approx(800 / 19)
    assert williams_r(bars, periods=(3,))["wr3"].iloc[2] == pytest.approx(-37.5)

    closes = flat([10, 11, 12, 11, 13])
    assert roc(closes, n=2)["roc"].iloc[2:].tolist() == pytest.approx([20, 0, 100 / 12])
    assert momentum(closes, n=2)["mom"].iloc[2:].tolist() == pytest.approx([2, 0, 1])


def test_volume_indicators():
    prices = flat([10, 10, 10, 10, 10], volume=1)
    prices["volume"] = [1, 2, 3, 4, 5]
    assert volume_ma(prices, periods=(3,))["vol_ma3"].iloc[2:].tolist() == pytest.approx([2, 3, 4])

    assert obv(frame([
        (10, 10, 10, 10, 100),
        (11, 11, 11, 11, 200),
        (11, 11, 11, 11, 50),
        (9, 9, 9, 9, 80),
    ]))["obv"].tolist() == pytest.approx([0, 200, 200, 120])

    weighted = vwap(frame([
        (9, 10, 8, 9, 100),
        (11, 12, 9, 11, 200),
    ], amount=[1e9, 1e9]))
    assert weighted["vwap"].tolist() == pytest.approx([9, 91 / 9])

    flow = mfi(frame([
        (10, 10, 10, 10, 1),
        (12, 12, 12, 12, 1),
        (11, 11, 11, 11, 1),
        (14, 14, 14, 14, 2),
    ]), n=2)
    assert flow["mfi"].iloc[0] != flow["mfi"].iloc[0]
    assert flow["mfi"].iloc[1:].tolist() == pytest.approx([100, 1200 / 23, 2800 / 39])


def test_atr_donchian_and_envelope():
    bars = frame([
        (9, 10, 8, 9, 1),
        (11, 12, 9, 11, 1),
        (10, 11, 10, 10, 1),
    ])
    assert atr(bars, n=2)["atr"].tolist() == pytest.approx([np.nan, 2.5, 2.0], nan_ok=True)
    channel = donchian(bars, n=3)
    assert channel["donchian_upper"].iloc[2] == pytest.approx(12)
    assert channel["donchian_lower"].iloc[2] == pytest.approx(8)
    assert channel["donchian_mid"].iloc[2] == pytest.approx(10)

    band = envelope(flat([1, 2, 3]), n=3, width=0.06)
    assert band["env_mid"].iloc[2] == pytest.approx(2)
    assert band["env_upper"].iloc[2] == pytest.approx(2.12)
    assert band["env_lower"].iloc[2] == pytest.approx(1.88)


def test_bias_and_psy():
    assert bias(flat([1, 2, 3]), periods=(3,))["bias3"].iloc[2] == pytest.approx(50)
    got = psy(flat([1, 2, 2, 3, 2]), n=3)["psy"]
    assert got.iloc[2:].tolist() == pytest.approx([100 / 3, 200 / 3, 100 / 3])


def test_dmi_direction_and_adx():
    out = dmi(frame([
        (9, 10, 8, 9, 1),
        (11, 12, 9, 11, 1),
        (12, 13, 11, 12, 1),
        (11, 13, 10, 11, 1),
        (10, 11, 9, 10, 1),
    ]), n=3, m=2)
    assert out["pdi"].iloc[2] == pytest.approx(300 / 7)
    assert out["mdi"].iloc[2] == pytest.approx(0)
    assert out["pdi"].iloc[3] == pytest.approx(37.5)
    assert out["mdi"].iloc[3] == pytest.approx(12.5)
    assert out["adx"].iloc[3] == pytest.approx(75)
    assert out["pdi"].iloc[4] == pytest.approx(100 / 7)
    assert out["mdi"].iloc[4] == pytest.approx(200 / 7)
    assert out["adx"].iloc[4] == pytest.approx(250 / 6)


def test_fibonacci_up_and_down_swings():
    out = fibonacci(frame([
        (1, 1, 1, 1, 1),
        (3, 5, 2, 3, 1),
        (3, 3, 2, 3, 1),
        (2, 4, 1, 2, 1),
    ]), n=3)
    assert out["fib_trend"].iloc[2] == pytest.approx(1)
    assert out.loc[2, ["fib_0", "fib_500", "fib_1000", "fib_1618"]].tolist() == pytest.approx(
        [5, 3, 1, 7.472]
    )
    assert out["fib_trend"].iloc[3] == pytest.approx(-1)
    assert out.loc[3, ["fib_0", "fib_500", "fib_1000", "fib_1618"]].tolist() == pytest.approx(
        [1, 3, 5, -1.472]
    )


def test_fibonacci_tie_uses_the_earliest_bar_as_up_swing():
    out = fibonacci(frame([
        (3, 5, 1, 3, 1),
        (4, 5, 2, 4, 1),
        (2.5, 3, 2, 2.5, 1),
    ]), n=3)
    assert out["fib_trend"].iloc[2] == pytest.approx(1)
    assert out["fib_0"].iloc[2] == pytest.approx(5)
    assert out["fib_1000"].iloc[2] == pytest.approx(1)


def test_pivot_points_use_previous_bar():
    out = pivot_points(frame([
        (9, 10, 8, 9, 1),
        (1, 1, 1, 1, 1),
    ]))
    assert out["pivot_pp"].iloc[0] != out["pivot_pp"].iloc[0]
    assert out.loc[1, ["pivot_pp", "pivot_r1", "pivot_r2", "pivot_r3", "pivot_s1", "pivot_s2", "pivot_s3"]].tolist() == pytest.approx(
        [9, 10, 11, 12, 8, 7, 6]
    )


def test_compute_all_keeps_original_columns_and_appends_indicators():
    out = compute_all(frame([(1, 2, 0.5, 1.5, 10)]))
    assert list(out.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert indicator_column_names() == EXPECTED_COLUMNS
    assert list(out.columns[5:]) == EXPECTED_COLUMNS


def test_compute_all_sorts_by_eob_without_mutating_input():
    data = frame(
        [(3, 3, 3, 3, 3), (1, 1, 1, 1, 1), (2, 2, 2, 2, 2)],
        eob=["2025-01-03 00:00:00+08:00", "2025-01-01 00:00:00+08:00", "2025-01-02 00:00:00+08:00"],
    )
    original = data.copy()
    out = compute_all(data)
    pd.testing.assert_frame_equal(data, original)
    assert out["close"].tolist() == [1, 2, 3]


def test_missing_or_non_numeric_columns_raise():
    with pytest.raises(ValueError, match="缺少行情列"):
        compute_all(pd.DataFrame({"close": [1.0]}))
    bad = frame([(1, 1, 1, 1, 1)])
    bad["close"] = ["x"]
    with pytest.raises(ValueError, match="close"):
        compute_all(bad)


def test_empty_frame_keeps_indicator_columns():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = compute_all(empty)
    assert len(out) == 0
    assert list(out.columns[5:]) == EXPECTED_COLUMNS


def test_write_indicators_round_trip(tmp_path):
    source = tmp_path / "history_SHSE600000.csv"
    flat([1, 2, 3]).to_csv(source, index=False)
    destination = write_indicators(source)
    assert destination.name == "history_SHSE600000_indicators.csv"
    saved = pd.read_csv(destination)
    assert "ma5" in saved.columns
    assert len(saved) == 3


def test_cli_writes_csv(tmp_path):
    source = tmp_path / "in.csv"
    destination = tmp_path / "out.csv"
    flat([1, 2]).to_csv(source, index=False)
    completed = subprocess.run(
        [sys.executable, "-m", "indicators", str(source), "-o", str(destination)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(destination)
    assert "rsi14" in pd.read_csv(destination).columns


def test_sample_history_smoke():
    if not SAMPLE.exists():
        pytest.skip("样例行情不在 data/history_SHSE600000.csv")
    raw = pd.read_csv(SAMPLE)
    out = compute_all(raw)
    assert len(out) == len(raw)
    assert list(out.columns[len(raw.columns):]) == EXPECTED_COLUMNS
    assert pd.to_datetime(out["eob"], utc=True).is_monotonic_increasing
    assert np.allclose(out["ma5"], out["close"].rolling(5).mean(), equal_nan=True)
    assert np.allclose(out["boll_mid"], out["ma20"], equal_nan=True)
    assert np.allclose(out["env_mid"], out["ma20"], equal_nan=True)
    assert np.allclose(out["macd_hist"], 2 * (out["macd_dif"] - out["macd_dea"]), equal_nan=True)
    assert np.allclose(out["kdj_j"], 3 * out["kdj_k"] - 2 * out["kdj_d"], equal_nan=True)
    assert not np.isinf(out[EXPECTED_COLUMNS].to_numpy(dtype=float)).any()
    rsi14 = out["rsi14"].dropna()
    wr6 = out["wr6"].dropna()
    mfi14 = out["mfi"].dropna()
    assert ((rsi14 >= 0) & (rsi14 <= 100)).all()
    assert ((wr6 >= -100) & (wr6 <= 0)).all()
    assert ((mfi14 >= 0) & (mfi14 <= 100)).all()
    assert set(np.unique(out["sar_trend"].dropna())).issubset({-1.0, 1.0})
    assert out["pivot_pp"].iloc[0] != out["pivot_pp"].iloc[0]
    if len(out) >= 250:
        assert out["ma250"].iloc[:249].isna().all()
        assert pd.notna(out["ma250"].iloc[249])
