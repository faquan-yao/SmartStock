package com.yaofaquan.SmartStock.StockDownload;

import com.yaofaquan.SmartStock.Application;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class DownloadManager {
    private final static String TUSHARE_URL = "http://api.waditu.com";
    private final CountDownLatch mCountDownLatch = new CountDownLatch(1);

    public void run() {
        Application.sLogger.debug("Start download stock code list.");
        StockCodeList list = new StockCodeList(mCountDownLatch);
        list.downLoad(list.getTuShareHttpsApiParams());
        try {
            mCountDownLatch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Application.sLogger.debug("Start download stock data.");
        ArrayList<String> stockCodeList = list.getStockCodeList();
        for (String code : stockCodeList) {
            Application.sLogger.debug("Download stock " + code);
            Stock stock = new Stock(code);
            Calendar end = Calendar.getInstance();
            Calendar start = Calendar.getInstance();
            start.add(Calendar.DATE, -100);
            stock.setStartDate(start.getTime());
            stock.setEndDate(end.getTime());

            Application.sLogger.debug(stock.getJson());

            stock.downLoad(stock.getJson());
            stock.saveToDb();
        }
    }
}
