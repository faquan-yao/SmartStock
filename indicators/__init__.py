"""技术指标计算入口。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from indicators._common import prepare
from indicators.defaults import (
    BIAS_PERIODS,
    EMA_PERIODS,
    FIB_LEVELS,
    MA_PERIODS,
    RSI_PERIODS,
    VOL_MA_PERIODS,
    WR_PERIODS,
)
from indicators.momentum import cci, kdj, momentum, roc, rsi, williams_r
from indicators.other import bias, dmi, fibonacci, pivot_points, psy
from indicators.trend import bollinger, exponential_moving_average, macd, moving_average, parabolic_sar
from indicators.volatility import atr, donchian, envelope
from indicators.volume import mfi, obv, volume_ma, vwap

__all__ = [
    "atr",
    "bias",
    "bollinger",
    "cci",
    "compute_all",
    "dmi",
    "donchian",
    "envelope",
    "exponential_moving_average",
    "fibonacci",
    "indicator_column_names",
    "kdj",
    "load_history",
    "macd",
    "mfi",
    "momentum",
    "moving_average",
    "obv",
    "parabolic_sar",
    "pivot_points",
    "psy",
    "roc",
    "rsi",
    "volume_ma",
    "vwap",
    "williams_r",
    "write_indicators",
]


def indicator_column_names() -> list[str]:
    """compute_all 追加的列名，顺序固定。"""
    names: list[str] = []
    names.extend(f"ma{period}" for period in MA_PERIODS)
    names.extend(f"ema{period}" for period in EMA_PERIODS)
    names.extend(["macd_dif", "macd_dea", "macd_hist"])
    names.extend(["boll_mid", "boll_upper", "boll_lower"])
    names.extend(["sar", "sar_trend"])
    names.extend(f"rsi{period}" for period in RSI_PERIODS)
    names.extend(["kdj_k", "kdj_d", "kdj_j"])
    names.append("cci")
    names.extend(f"wr{period}" for period in WR_PERIODS)
    names.extend(["roc", "mom"])
    names.extend(f"vol_ma{period}" for period in VOL_MA_PERIODS)
    names.extend(["obv", "vwap", "mfi"])
    names.append("atr")
    names.extend(["donchian_upper", "donchian_mid", "donchian_lower"])
    names.extend(["env_mid", "env_upper", "env_lower"])
    names.extend(f"bias{period}" for period in BIAS_PERIODS)
    names.append("psy")
    names.extend(["pdi", "mdi", "adx"])
    names.append("fib_trend")
    names.extend(name for name, _ratio in FIB_LEVELS)
    names.extend(
        ["pivot_pp", "pivot_r1", "pivot_r2", "pivot_r3", "pivot_s1", "pivot_s2", "pivot_s3"]
    )
    return names


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """计算文档中的全部技术指标。

    返回按 eob 升序排列的新表：原始列在前，指标列在后。数据不足的位置为 NaN。
    """
    data = prepare(df)
    parts = [
        moving_average(data, prepared=True),
        exponential_moving_average(data, prepared=True),
        macd(data, prepared=True),
        bollinger(data, prepared=True),
        parabolic_sar(data, prepared=True),
        rsi(data, prepared=True),
        kdj(data, prepared=True),
        cci(data, prepared=True),
        williams_r(data, prepared=True),
        roc(data, prepared=True),
        momentum(data, prepared=True),
        volume_ma(data, prepared=True),
        obv(data, prepared=True),
        vwap(data, prepared=True),
        mfi(data, prepared=True),
        atr(data, prepared=True),
        donchian(data, prepared=True),
        envelope(data, prepared=True),
        bias(data, prepared=True),
        psy(data, prepared=True),
        dmi(data, prepared=True),
        fibonacci(data, prepared=True),
        pivot_points(data, prepared=True),
    ]
    out = data.copy()
    for part in parts:
        for column in part.columns:
            out[column] = part[column].to_numpy()
    return out


def load_history(path: str | Path) -> pd.DataFrame:
    """读取行情 CSV。"""
    return pd.read_csv(path)


def write_indicators(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """读取 CSV，计算全部指标，并写入结果 CSV。返回输出路径。"""
    source = Path(input_path)
    if output_path is None:
        suffix = source.suffix or ".csv"
        destination = source.with_name(f"{source.stem}_indicators{suffix}")
    else:
        destination = Path(output_path)
    result = compute_all(load_history(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8")
    return destination
