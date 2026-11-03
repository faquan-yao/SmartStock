"""默认参数。周期与通达信常用设置对齐，并与使用说明中的参数表一致。"""

from __future__ import annotations

MA_PERIODS = (5, 10, 20, 60, 120, 250)
EMA_PERIODS = (12, 26)

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BOLL_PERIOD = 20
BOLL_K = 2.0

SAR_AF_START = 0.02
SAR_AF_STEP = 0.02
SAR_AF_MAX = 0.2

RSI_PERIODS = (6, 12, 14, 24)

KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3

CCI_PERIOD = 14
WR_PERIODS = (6, 10)

ROC_PERIOD = 12
MOM_PERIOD = 12

VOL_MA_PERIODS = (5, 10, 20)

MFI_PERIOD = 14
ATR_PERIOD = 14

DONCHIAN_PERIOD = 20
ENVELOPE_PERIOD = 20
ENVELOPE_WIDTH = 0.06

BIAS_PERIODS = (6, 12, 24)
PSY_PERIOD = 12

DMI_N = 14
DMI_M = 6

FIBONACCI_LOOKBACK = 60

# (列名, 比例)。小于等于 1 为回调位，大于 1 为扩展位。
FIB_LEVELS = (
    ("fib_0", 0.0),
    ("fib_236", 0.236),
    ("fib_382", 0.382),
    ("fib_500", 0.5),
    ("fib_618", 0.618),
    ("fib_786", 0.786),
    ("fib_1000", 1.0),
    ("fib_1272", 1.272),
    ("fib_1618", 1.618),
    ("fib_2618", 2.618),
)
