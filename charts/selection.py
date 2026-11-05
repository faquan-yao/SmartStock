"""选择要绘制的指标。K 线和分段折线始终绘制，不在这个名单里。"""

from __future__ import annotations

from collections.abc import Sequence

from indicators.defaults import (
    BIAS_PERIODS,
    EMA_PERIODS,
    FIB_LEVELS,
    MA_PERIODS,
    RSI_PERIODS,
    VOL_MA_PERIODS,
    WR_PERIODS,
)

# 同花顺主图常见组合：均线、成交量、MACD。
DEFAULT_INDICATORS = ("ma", "volume", "macd")

# 名字 -> (副图标题, 高度, 是否叠在 K 线上, 所需列)
# 叠在 K 线上的指标不单独占副图。
_INDICATOR_TABLE: tuple[tuple[str, str, float, bool, tuple[str, ...]], ...] = (
    ("ma", "K线", 0.0, True, tuple(f"ma{period}" for period in MA_PERIODS)),
    ("ema", "K线", 0.0, True, tuple(f"ema{period}" for period in EMA_PERIODS)),
    ("boll", "K线", 0.0, True, ("boll_mid", "boll_upper", "boll_lower")),
    ("sar", "K线", 0.0, True, ("sar", "sar_trend")),
    ("vwap", "K线", 0.0, True, ("vwap",)),
    ("donchian", "K线", 0.0, True, ("donchian_upper", "donchian_mid", "donchian_lower")),
    ("envelope", "K线", 0.0, True, ("env_mid", "env_upper", "env_lower")),
    ("volume", "成交量", 1.15, False, tuple(f"vol_ma{period}" for period in VOL_MA_PERIODS)),
    ("macd", "MACD", 1.25, False, ("macd_dif", "macd_dea", "macd_hist")),
    ("kdj", "KDJ", 1.15, False, ("kdj_k", "kdj_d", "kdj_j")),
    ("rsi", "RSI", 1.15, False, tuple(f"rsi{period}" for period in RSI_PERIODS)),
    ("cci", "CCI", 1.1, False, ("cci",)),
    ("wr", "WR", 1.1, False, tuple(f"wr{period}" for period in WR_PERIODS)),
    ("roc", "ROC", 1.1, False, ("roc",)),
    ("mom", "MOM", 1.1, False, ("mom",)),
    ("bias", "BIAS", 1.1, False, tuple(f"bias{period}" for period in BIAS_PERIODS)),
    ("mfi", "MFI", 1.1, False, ("mfi",)),
    ("psy", "PSY", 1.1, False, ("psy",)),
    ("dmi", "DMI", 1.15, False, ("pdi", "mdi", "adx")),
    ("obv", "OBV", 1.1, False, ("obv",)),
    ("atr", "ATR", 1.1, False, ("atr",)),
    ("fib", "斐波那契", 2.2, False, ("fib_trend", *(name for name, _ratio in FIB_LEVELS))),
    ("pivot", "枢轴点", 1.6, False, ("pivot_pp", "pivot_r1", "pivot_r2", "pivot_r3", "pivot_s1", "pivot_s2", "pivot_s3")),
)

INDICATOR_NAMES = tuple(item[0] for item in _INDICATOR_TABLE)
_BY_NAME = {item[0]: item for item in _INDICATOR_TABLE}
KLINE_HEIGHT = 3.6


def resolve_indicators(indicators: str | Sequence[str] | None = None) -> tuple[str, ...]:
    """把参数解析成去重后的指标名，顺序与 INDICATOR_NAMES 一致。

    None 使用默认组合。字符串 "all" 表示全部。多个名字用逗号分隔。
    """
    if indicators is None:
        requested = list(DEFAULT_INDICATORS)
    elif isinstance(indicators, str):
        text = indicators.strip()
        if text.lower() == "all":
            return INDICATOR_NAMES
        requested = [part.strip().lower() for part in text.replace("，", ",").split(",") if part.strip()]
    else:
        requested = [str(item).strip().lower() for item in indicators if str(item).strip()]
    if not requested:
        raise ValueError("至少指定一个指标")
    unknown = [name for name in requested if name not in _BY_NAME]
    if unknown:
        choices = ", ".join(INDICATOR_NAMES)
        raise ValueError(f"未知指标: {', '.join(unknown)}。可选 {choices}，或 all")
    chosen = set(requested)
    return tuple(name for name in INDICATOR_NAMES if name in chosen)


def is_overlay(name: str) -> bool:
    return _BY_NAME[name][3]


def panel_title(name: str) -> str:
    return _BY_NAME[name][1]


def panel_height(name: str) -> float:
    return _BY_NAME[name][2]


def chart_panels(indicators: str | Sequence[str] | None = None) -> tuple[tuple[str, float], ...]:
    """返回从上到下的副图标题和高度。K 线永远在最上面。"""
    panels = [("K线", KLINE_HEIGHT)]
    for name in resolve_indicators(indicators):
        if not is_overlay(name):
            panels.append((panel_title(name), panel_height(name)))
    return tuple(panels)


def required_columns(indicators: str | Sequence[str] | None = None) -> tuple[str, ...]:
    names = resolve_indicators(indicators)
    columns: list[str] = []
    for name in names:
        columns.extend(_BY_NAME[name][4])
    return tuple(columns)
