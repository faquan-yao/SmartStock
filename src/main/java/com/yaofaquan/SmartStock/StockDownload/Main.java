package com.yaofaquan.SmartStock.StockDownload;

import org.apache.log4j.Logger;

import java.util.Calendar;

public class Main {
    public static final Logger sLogger = Logger.getLogger(Main.class.getSimpleName());
    private final static String TUSHARE_URL = "http://api.waditu.com";

    public static void main(String[] argv) {
        System.out.println("Hello world!");
        sLogger.info("Hello, log!");

        Stock stock = new Stock("300750.SZ");
        Calendar end = Calendar.getInstance();
        Calendar start = Calendar.getInstance();
        start.add(Calendar.DATE, -100);
        stock.setStartDate(start.getTime());
        stock.setEndDate(end.getTime());

        System.out.println(stock.getJson());

        stock.downLoad(stock.getJson());
        stock.saveToDb();
    }
}
