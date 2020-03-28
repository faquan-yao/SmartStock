from threading import BoundedSemaphore
import tushare as ts
import pandas as pd

from SmartStock.MySqlProxy import MySqlProxy
from SmartStock.StockCodeManager import StockCodeManager
import time
import logging

class DownloadManager:
    __logger = logging.getLogger("DownloadManager")
    __SqlManager = None
    __StockCodeManager = None
    __MaxThreadNum = 10

    #def __init__(self):

    def setSqlManager(self, mySqlManager):
        self.__SqlManager = mySqlManager

    def setStockCodeManager(self, stockCodeManager):
        self.__StockCodeManager = stockCodeManager

    def setMaxThreadNum(self, num):
        self.__MaxThreadNum = num

    def run(self):
        start_time = time.time()
        stockcodes = self.__StockCodeManager.getStockCode()
        pool_sema = BoundedSemaphore(self.__MaxThreadNum)

        for code in stockcodes:
            per_start_time = time.time()
            pool_sema.acquire()
            pro = ts.pro_api("c0cd3565a37a4a79c0f2c3c1536499e7a7dfefeda2b82f17196d1540")
            df = pro.daily(ts_code=str(code), start_date='20190101', end_date='20191231')
            df = ts.pro_bar(ts_code=str(code), adj='hfq', start_date='20190101', end_date='20191231')
            if df != None:
                df.to_csv("/home/faquanyao/work_space/stocks/data/"+code+".csv")
            pool_sema.release()
            per_end_time = time.time()
            self.__logger.info("Download stock data for {} cost {} seconds.".format(code, per_end_time-per_start_time))
        end_time = time.time()
        self.__logger.info("Processing time for {} is: {} seconds".format('download stock data', end_time - start_time))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    downloadManager = DownloadManager()
    downloadManager.setMaxThreadNum(5)
    mySqlProxy = MySqlProxy()
    stockCodeManager = StockCodeManager()
    downloadManager.setSqlManager(mySqlProxy)
    downloadManager.setStockCodeManager(stockCodeManager)
    downloadManager.run()

