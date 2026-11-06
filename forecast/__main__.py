"""命令行：python -m forecast train 输入.csv -o 模型.npz"""

from __future__ import annotations

import argparse

from forecast import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    DEFAULT_WINDOW,
    write_model,
    write_predictions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用多层感知机，根据行情和技术指标预测分段斜率")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练网络并保存 npz")
    train_parser.add_argument("csv", help="输入行情 CSV，需包含 segment，其余列作为特征")
    train_parser.add_argument("-o", "--output", help="输出 npz。省略时在输入文件名后加 _segment_model.npz")
    train_parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="窗口长度，默认 5")
    train_parser.add_argument("--hidden", default=",".join(str(size) for size in DEFAULT_HIDDEN), help="隐藏层宽度，逗号分隔，默认 32,16")
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练轮数，默认 300")
    train_parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="学习率，默认 0.01")
    train_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批大小，默认 32")
    train_parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="末尾验证集比例，默认 0.2")
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子，默认 7")

    predict_parser = subparsers.add_parser("predict", help="用已保存的网络写出预测斜率")
    predict_parser.add_argument("csv", help="输入行情 CSV，列要覆盖训练时用过的特征")
    predict_parser.add_argument("--model", required=True, help="train 写出的 npz")
    predict_parser.add_argument("-o", "--output", help="输出 CSV。省略时在输入文件名后加 _segment_pred.csv")

    args = parser.parse_args(argv)
    if args.command == "train":
        hidden = tuple(int(part) for part in args.hidden.split(",") if part.strip())
        print(
            write_model(
                args.csv,
                args.output,
                window=args.window,
                hidden=hidden,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )
        )
        return 0
    print(write_predictions(args.csv, args.model, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
