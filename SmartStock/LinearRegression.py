import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def LinearRegression(x_data, y_data):
    print("x_data have shape {} and y_data have shape {}"
          .format(x_data.shape, y_data.shape))
    plt.plot(x_data, y_data, '*')

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
    model.summary()

    model.compile(optimizer='adam', loss='mse')

    log = model.fit(x_data, y_data, epochs=100000)

    print(model.layers[0].get_weights())

    y_pred = model.predict(x_data)
    plt.plot(x_data, y_pred)
    plt.show()

    return model

if __name__ == "__main__":

    df = pd.read_csv("/home/faquanyao/work_space/stocks/data/000001.SZ.csv")
    print(df.shape)
    y = df['close']
    x = np.arange(0, len(y))
    LinearRegression(x, y)


