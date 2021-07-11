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

        MediaType mediaType = MediaType.parse("application/json; charset=utf-8");
        Request request = new Request.Builder()
                .url(TUSHARE_URL)
                .post(RequestBody.create(mediaType, stock.getJson()))
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
                JSONArray items = data.getJSONArray("items");
                System.out.println(items.toString());
                for (int i = 0; i < items.length(); i++) {
                    JSONArray item = items.getJSONArray(i);
                    for (int j = 0; j < item.length(); j++) {
                        System.out.print(item.getString(j));
                        System.out.print("*");
                    }
                    System.out.println();
                }
            }
        });
    }
}
