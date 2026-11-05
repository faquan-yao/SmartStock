"""策略运行日志：同时写到控制台和 logs/smartstock.log。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "smartstock", log_dir: str | Path | None = None) -> logging.Logger:
    """返回已配置的日志器。同名日志器只配置一次。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, _DATE)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    directory = Path(log_dir) if log_dir is not None else Path(__file__).resolve().parent / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        directory / "smartstock.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
