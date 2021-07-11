package com.yaofaquan.SmartStock.StockDownload;

import org.json.JSONArray;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class Stock {
    private static final String TAG = "Stock";
    private static final String TUSHARE_URL = "http://api.waditu.com";
    private static final String TOKEN = "c0cd3565a37a4a79c0f2c3c1536499e7a7dfefeda2b82f17196d1540";
    private static final SimpleDateFormat sDateFormat = new SimpleDateFormat("YYYYMMDD");

    private String mStockCode = "";
    private String mApi = "daily";
    private Date mStartDate = new Date();
    private Date mEndDate = new Date();
    private String[] mFields = {"trade_date", "open", "high", "low", "close", "vol", "amount"};
    private String mAdj = "qfq";

    public Stock(String mStockCode) {
        this.mStockCode = mStockCode;
    }

    public void setStartDate(Date date) {
        mStartDate = date;
    }

    public void setEndDate(Date date) {
        mEndDate = date;
    }

    public String getJson() {
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("api_name", mApi);
        jsonObject.put("token", TOKEN);
        JSONObject params = new JSONObject();
        params.put("ts_code", mStockCode);
        params.put("start_date", sDateFormat.format(mStartDate));
        params.put("end_date", sDateFormat.format(mEndDate));
        jsonObject.put("params", params);
        JSONArray fields = new JSONArray();
        for (String field : mFields) {
            fields.put(field);
        }
        jsonObject.put("fields", fields);
        return jsonObject.toString();
    }
}
