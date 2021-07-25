package com.yaofaquan.SmartStock;

import com.sun.tools.javac.Main;
import com.yaofaquan.SmartStock.StockDownload.DownloadManager;
import org.apache.log4j.Logger;

import java.io.IOException;

public class Application {
    public static final Logger sLogger = Logger.getLogger(DownloadManager.class.getSimpleName());

    public static void main(String[] argv) {
        sLogger.debug("Start work.");
        DownloadManager downloadManager = new DownloadManager();
        Runnable runnable = () -> downloadManager.run();
        Thread thread = new Thread(runnable);
        thread.start();
        try {
            thread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            sLogger.debug("All work is done.");
        }
        try {
            System.in.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
