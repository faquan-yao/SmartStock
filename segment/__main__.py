"""命令行：python -m segment 输入.csv -o 输出.csv"""

from __future__ import annotations

import argparse

from segment import DEFAULT_MAX_ERROR, write_segments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按收盘价对行情做自底向上线性分段")
    parser.add_argument("csv", help="输入行情 CSV，需包含 eob、close")
    parser.add_argument("-o", "--output", help="输出 CSV。省略时在输入文件名后加 _segments")
    parser.add_argument(
        "--max-error",
        type=float,
        default=DEFAULT_MAX_ERROR,
        help="最大相对误差，默认 0.03（3%%）",
    )
    args = parser.parse_args(argv)
    print(write_segments(args.csv, args.output, max_error=args.max_error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
