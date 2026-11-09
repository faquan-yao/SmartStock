# coding=utf-8
from __future__ import print_function, absolute_import

import random

from gm.api import *
from datetime import datetime, timedelta
from indicators import compute_all
from segment import linear_segment
from charts import plot_chart
from logsetup import get_logger
from forecast import predict_segment, train_segment_model

logger = get_logger()

def get_normal_a_stock_list(skip_suspended: bool = True) -> list:
    """
    获取所有正常A股代码（剔除ST、科创板、北交所、创业板）
    :param skip_suspended: 是否剔除停牌股票
    :return: list, [SHSE.600000, SZSE.000001 ...]
    """
    # sec_type1=1010：股票；sec_type2=101001：A股
    df = get_symbols(
        sec_type1=1010,
        sec_type2=101001,
        df=True
    )

    # 1. 过滤ST股票
    df = df[df["is_st"] == False]

    # 2. 过滤停牌
    if skip_suspended:
        df = df[df["is_suspended"] == False]

    # 3. 过滤科创板 688开头、北交所 8开头、创业板 300开头
    def filter_board(symbol: str):
        # symbol样例 SHSE.688001 / SZSE.000001
        code = symbol.split(".")[-1]
        if code.startswith("688"):
            return False
        if code.startswith("8"):
            return False
        if code.startswith("300"):
            return False
        return True

    df = df[df["symbol"].apply(filter_board)]

    # 转为列表
    stock_list = df["symbol"].tolist()
    return stock_list

def init(context):
    symbols = get_normal_a_stock_list()
    logger.info("股票池 %s 只", len(symbols))
    samples = random.sample(symbols, 20)
    logger.info("随机选取其中20只股票进行研究：%s", samples)
    for symbol in samples:
        try:
            history_data = history_n(symbol, '1d', 625, fields='symbol, eob, open, close, low, high, volume, amount', adjust=ADJUST_PREV, df=True)
            if history_data is None or len(history_data) == 0:
                logger.warning("%s 没有日线，已跳过", symbol)
                continue
            history_data = compute_all(history_data)
            history_data = history_data.tail(375)
            segment_data = linear_segment(history_data)
            history_data = history_data.merge(segment_data[["eob", "segment"]], on="eob", how="left")
            train_set = history_data[:250]
            test_set = history_data[250:]
            model = train_segment_model(train_set, epochs=100)
            predicted = predict_segment(test_set, model)

            stem = str(symbol).replace(".", "_")
            png_path = f"data\\{stem}.png"
            csv_path = f"data\\{stem}.csv"
            plot_chart(test_set, png_path, predictions=predicted)
            test_set.to_csv(csv_path, index=False)
            logger.info("%s 完成，%s 行，图 %s，表 %s", symbol, len(test_set), png_path, csv_path)
        except Exception:
            logger.exception("%s 处理失败", symbol)
            continue


def on_tick(context, tick):
    logger.info("tick %s: %s, %s, %s", tick["symbol"], tick["open"], tick["high"], tick["low"])
    for quote in tick["quotes"]:
        logger.info("    %s, %s, %s, %s", quote["bid_p"], quote["bid_v"], quote["ask_p"], quote["ask_v"])


def on_bar(context, bars):
    for bar in bars:
        logger.info(
            "bar %s - %s: %s, %s, %s, %s, %s",
            bar["symbol"],
            bar["frequency"],
            bar["open"],
            bar["high"],
            bar["low"],
            bar["close"],
            bar["volume"],
        )
    logger.info("持仓 %s", context.account().positions())


if __name__ == '__main__':
    '''
        strategy_id策略ID, 由系统生成
        filename文件名, 请与本文件名保持一致
        mode运行模式, 实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID, 可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式, 不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
        backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
    '''
    backtest_start_time = str(datetime.now() - timedelta(days=5))[:19]
    backtest_end_time = str(datetime.now())[:19]
    run(strategy_id='c60678e8-ce8e-11f1-9e56-00ff1f0866ab',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='336730efc6dcc2f86609a78de822d21b5f0b4064',
        backtest_start_time='2025-01-01 08:00:00',
        backtest_end_time='2025-12-31 14:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
