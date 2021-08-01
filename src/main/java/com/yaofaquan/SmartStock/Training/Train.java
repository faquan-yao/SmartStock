package com.yaofaquan.SmartStock.Training;


import org.tensorflow.*;

public class Train {
    public static void main(String[] args) {
        System.out.println("Hello TensorFlow " + TensorFlow.version());

        Graph graph = new Graph();
        Operation x = graph.opBuilder("Const", "x")
                .setAttr("dtype", DataType.FLOAT)
                .setAttr("value", Tensor.create(3.0f))
                .build();
        Operation y = graph.opBuilder("Placeholder", "y")
                .setAttr("dtype", DataType.FLOAT)
                .build();

        Operation xy = graph.opBuilder("Mul", "xy")
                .addInput(x.output(0))
                .addInput(y.output(0))
                .build();

        Session session = new Session(graph);
        Tensor tensor = session.runner().fetch("xy").feed("x", Tensor.create(5.0f)).feed("y", Tensor.create(4.0f)).run().get(0);

        System.out.println(tensor.floatValue());
    }
}
