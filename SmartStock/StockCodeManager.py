import tushare as ts
import pandas as pd
import re

class StockCodeManager:
    __stockcode = []

    def __init__(self):
        try:
            ts.set_token('c0cd3565a37a4a79c0f2c3c1536499e7a7dfefeda2b82f17196d1540')
            pro = ts.pro_api()
            data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
            print(data.tail())
            self.__stockcode = data['ts_code']
        except Exception as e:
            print(e)
            file = open("/home/faquanyao/work_space/stocks/sh_stock_code.txt", mode='r')
            lines = file.readlines()
            for l in lines:
                regex = re.compile("\w+")
                r = regex.findall(l)
                self.__stockcode.append(r[1]+".SH")

            file = open("/home/faquanyao/work_space/stocks/sz_stock_code.txt", mode='r')
            lines = file.readlines()
            for l in lines:
                regex = re.compile("\w+")
                r = regex.findall(l)
                self.__stockcode.append(r[1]+".SZ")

    def getStockCode(self):
        return self.__stockcode


if __name__ == "__main__":
    stockcode = StockCodeManager()
    code = stockcode.getStockCode()
    for i in code:
        print(i)