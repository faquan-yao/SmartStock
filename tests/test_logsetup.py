"""日志同时落到控制台和文件，同名日志器不重复挂处理器。"""

from __future__ import annotations

import logging

from logsetup import get_logger


def test_logger_writes_file_once(tmp_path, capsys):
    logger = get_logger("smartstock-test", tmp_path)
    again = get_logger("smartstock-test", tmp_path / "ignored")
    assert again is logger
    assert len(logger.handlers) == 2

    logger.info("处理完成 %s", "SHSE.600000")
    for handler in logger.handlers:
        handler.flush()

    text = (tmp_path / "smartstock.log").read_text(encoding="utf-8")
    assert "INFO" in text
    assert "处理完成 SHSE.600000" in text
    assert "处理完成 SHSE.600000" in capsys.readouterr().out

    logging.getLogger("smartstock-test").handlers.clear()
