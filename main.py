# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
from datetime import datetime, timedelta


def init(context):
    # 订阅浦发银行, bar频率为一天和一分钟
    # 订阅订阅多个频率的数据，可多次调用subscribe
    subscribe(symbols='SHSE.603893', frequency='1d')
    # subscribe(symbols='SHSE.603893', frequency='60s')
    # subscribe(symbols='SHSE.603893', frequency='tick')


def on_tick(context, tick):
    # 打印tick数据
    print(f'tick {tick["symbol"]}: {tick["open"]}, {tick["high"]}, {tick["low"]}', sep='\n')
    for quote in tick["quotes"]:
        print(f'\t\t{quote["bid_p"]}, {quote["bid_v"]}, {quote["ask_p"]}, {quote["ask_v"]}', sep='\n')


def on_bar(context, bars):
    for bar in bars:
        # 打印bar数据
        print(
            f'bar {bar["symbol"]} - {bar["frequency"]}: {bar["open"]}, {bar["high"]}, {bar["low"]}, {bar["close"]}, {bar["volume"]}',
            sep='\n')

    print(f"context positions {context.account().positions()}")


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
