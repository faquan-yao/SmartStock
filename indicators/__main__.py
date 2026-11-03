"""命令行：python -m indicators 输入.csv -o 输出.csv"""

from __future__ import annotations

import argparse

from indicators import write_indicators


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按通达信口径计算股票技术指标")
    parser.add_argument("csv", help="输入行情 CSV，需包含 open、high、low、close、volume")
    parser.add_argument("-o", "--output", help="输出 CSV。省略时在输入文件名后加 _indicators")
    args = parser.parse_args(argv)
    print(write_indicators(args.csv, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
