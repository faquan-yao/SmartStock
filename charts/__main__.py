"""命令行：python -m charts 输入.csv -o 输出.png"""

from __future__ import annotations

import argparse

from charts import DEFAULT_INDICATORS, DEFAULT_MAX_ERROR, INDICATOR_NAMES, write_chart


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按同花顺风格绘制 K 线、技术指标和分段折线")
    parser.add_argument("csv", help="输入行情 CSV，需包含 eob、open、high、low、close、volume")
    parser.add_argument("-o", "--output", help="输出 PNG。省略时在输入文件名后加 _chart.png")
    parser.add_argument(
        "--max-error",
        type=float,
        default=DEFAULT_MAX_ERROR,
        help="分段最大相对误差，默认 0.03",
    )
    parser.add_argument(
        "--indicators",
        default=",".join(DEFAULT_INDICATORS),
        help="要绘制的指标，逗号分隔。默认 ma,volume,macd。all 表示全部。可选: " + ", ".join(INDICATOR_NAMES),
    )
    args = parser.parse_args(argv)
    print(write_chart(args.csv, args.output, max_error=args.max_error, indicators=args.indicators))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
