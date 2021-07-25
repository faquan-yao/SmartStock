package com.yaofaquan.SmartStock;

import java.io.IOException;

public class Application {
    public static void main(String[] argv) {
        System.out.println("Hello Java Application!");
        try {
            System.in.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
