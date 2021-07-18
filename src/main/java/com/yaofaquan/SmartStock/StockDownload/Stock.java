package com.yaofaquan.SmartStock.StockDownload;

import okhttp3.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.sql.*;
import java.sql.Connection;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class Stock {
    private static final String TAG = "Stock";
    private static final String TUSHARE_URL = "http://api.waditu.com";
    private static final String TOKEN = "c0cd3565a37a4a79c0f2c3c1536499e7a7dfefeda2b82f17196d1540";
    private static final SimpleDateFormat sDateFormat = new SimpleDateFormat("YYYYMMDD");

    private static final String DB_DRIVER = "com.mysql.cj.jdbc.Driver";
    private static final String DB_URL = "jdbc:mysql://120.79.203.124:3306/smart_stock?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC";
    private static final String DB_USER = "faquan.yao";
    private static final String DB_PWD = "999999";
    private static final String DB_CREATE_STOCK_DB = "CREATE TABLE IF NOT EXISTS Stock_%s (" +
            "trade_date VARCHAR(32) NOT NULL, " +
            "open DOUBLE NOT NULL, " +
            "high DOUBLE NOT NULL, " +
            "low DOUBLE NOT NULL, " +
            "close DOUBLE NOT NULL, " +
            "vol DOUBLE NOT NULL, " +
            "amount DOUBLE NOT NULL)";

    private String mStockCode = "";
    private String mApi = "daily";
    private Date mStartDate = new Date();
    private Date mEndDate = new Date();
    private String[] mFields = {"trade_date", "open", "high", "low", "close", "vol", "amount"};
    private String mAdj = "qfq";

    private JSONArray mJSONArrayData = null;

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

    public void downLoad(String requstContext) {
        MediaType mediaType = MediaType.parse("application/json; charset=utf-8");
        Request request = new Request.Builder()
                .url(TUSHARE_URL)
                .post(RequestBody.create(mediaType, requstContext))
                .build();
        OkHttpClient okHttpClient = new OkHttpClient();
        okHttpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {

            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                System.out.println(response.protocol() + " " +response.code() + " " + response.message());
                Headers headers = response.headers();
                for (int i = 0; i < headers.size(); i++) {
                    System.out.println(headers.name(i) + ":" + headers.value(i));
                }
                String rawData = response.body().string();
                System.out.println("onResponse: " + rawData);

                JSONObject jsonObject = new JSONObject(rawData);
                JSONObject data = jsonObject.getJSONObject("data");
                mJSONArrayData = data.getJSONArray("items");
                System.out.println(mJSONArrayData.toString());
                for (int i = 0; i < mJSONArrayData.length(); i++) {
                    JSONArray item = mJSONArrayData.getJSONArray(i);
                    for (int j = 0; j < item.length(); j++) {
                        if (j == 0) {
                            System.out.print(item.getString(0));
                        } else {
                            System.out.print(item.getDouble(j));
                        }
                        System.out.print("****");
                    }
                    System.out.println();
                }
            }
        });
    }

    public void saveToDb() {
        Connection conn = null;
        Statement statement = null;
        PreparedStatement preparedStatement = null;
        try {
            Class.forName(DB_DRIVER);
            conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PWD);
            statement = conn.createStatement();
            String dropTableSql = String.format("drop table if exists Stock_%s", mStockCode.replace('.', '_'));
            statement.execute(dropTableSql);
            String createTableSql = String.format(DB_CREATE_STOCK_DB, mStockCode.replace('.','_'));
            Main.sLogger.debug(createTableSql);
            statement.execute(createTableSql);
            statement.close();

            String insetDataSQL= String.format("insert into Stock_%s(trade_date, open, high, low, close, vol, amount)" +
                    " values(?, ?, ?, ?, ?, ?, ?)", mStockCode.replace('.', '_'));
            Main.sLogger.debug(insetDataSQL);
            preparedStatement = conn.prepareStatement(insetDataSQL);
            for (int i = 0; i < mJSONArrayData.length(); i++) {
                JSONArray item = mJSONArrayData.getJSONArray(i);
                Main.sLogger.debug(item.toString());
                preparedStatement.setString(1, item.getString(0));
                preparedStatement.setDouble(2, item.getDouble(1));
                preparedStatement.setDouble(3, item.getDouble(2));
                preparedStatement.setDouble(4, item.getDouble(3));
                preparedStatement.setDouble(5, item.getDouble(4));
                preparedStatement.setDouble(6, item.getDouble(5));
                preparedStatement.setDouble(7, item.getDouble(6));
                preparedStatement.execute();
            }
        } catch (ClassNotFoundException | SQLException e) {
            e.printStackTrace();
        } finally {
            if (statement != null) {
                try {
                    statement.close();
                } catch (SQLException throwables) {
                    throwables.printStackTrace();
                }
            }
            if (conn != null) {
                try {
                    conn.close();
                } catch (SQLException throwables) {
                    throwables.printStackTrace();
                }
            }
        }
    }
}
