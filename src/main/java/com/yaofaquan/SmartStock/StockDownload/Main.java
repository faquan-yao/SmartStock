package com.yaofaquan.SmartStock.StockDownload;

import okhttp3.*;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONString;

import java.io.IOException;
import java.util.Calendar;
import java.util.Date;

public class Main {
    private final static String TUSHARE_URL = "http://api.waditu.com";
    public static void main(String[] argv) {
        System.out.println("Hello world!");

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
