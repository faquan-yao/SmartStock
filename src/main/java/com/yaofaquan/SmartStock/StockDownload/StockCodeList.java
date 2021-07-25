package com.yaofaquan.SmartStock.StockDownload;

import com.yaofaquan.SmartStock.Application;
import okhttp3.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.locks.Condition;

public class StockCodeList {
    private static final String TAG = "Stock";
    private static final String TUSHARE_URL = "http://api.waditu.com";
    private static final String TOKEN = "c0cd3565a37a4a79c0f2c3c1536499e7a7dfefeda2b82f17196d1540";

    private CountDownLatch mCountDownLatch = null;
    private String mApi = "stock_basic";
    private String[] mFields = {"ts_code", "symbol", "name","area", "industry", "list_date"};
    private JSONArray mJSONArrayData = null;
    private ArrayList<String> mStockList = new ArrayList<>();

    public StockCodeList(CountDownLatch countDownLatch) {
        mCountDownLatch = countDownLatch;
    }
    public String getTuShareHttpsApiParams() {
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("api_name", mApi);
        jsonObject.put("token", TOKEN);
        JSONObject params = new JSONObject();
        params.put("exchange", "");
        params.put("list_status", "L");
        jsonObject.put("params", params);
        JSONArray fields = new JSONArray();
        for (String s : mFields) {
            fields.put(s);
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
                Application.sLogger.error("Download stock code list failed.", e);
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                Headers headers = response.headers();
                for (int i = 0; i < headers.size(); i++) {
                    Application.sLogger.debug(headers.name(i) + ":" + headers.value(i));
                }
                String rawData = response.body().string();
                Application.sLogger.debug("onResponse: " + rawData);

                JSONObject jsonObject = new JSONObject(rawData);
                JSONObject data = jsonObject.getJSONObject("data");
                mJSONArrayData = data.getJSONArray("items");
                Application.sLogger.debug("Stock code list:");
                for (int i = 0; i < mJSONArrayData.length(); i++) {
                    JSONArray item = mJSONArrayData.getJSONArray(i);
                    mStockList.add(item.getString(0));
                    Application.sLogger.debug(item.getString(0));
                }
                mCountDownLatch.countDown();
            }
        });
    }

    public ArrayList getStockCodeList() {
        return mStockList;
    }
}
