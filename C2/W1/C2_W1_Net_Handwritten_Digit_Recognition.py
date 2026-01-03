import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
import logging
from mutils import *

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


if __name__ == '__main__':
    X, y = load_data()

    # print(f"X.shape: {X.shape}")      # (1000, 400)
    # print(f"y.shape: {y.shape}")      # (1000, 1)
    # x_0 = X[0]
    # print(f"X[0].shape: {x_0.shape}")     # (400, 0)
    # print(f"the first element of X: {x_0}")

    # Net_display_train_data(X, y)

    # Sequential model
    model = Sequential(
        [
            tf.keras.Input(shape=(400, )),
            tf.keras.layers.Dense(25, activation='sigmoid'),
            tf.keras.layers.Dense(15, activation='sigmoid'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ], name = 'my_model'
    )

    # 参数结构
    # model.summary()
    # [layer1, layer2, layer3] = model.layers
    # # examine weights shapes
    # W1,b1 = layer1.get_weights()
    # W2,b2 = layer2.get_weights()
    # W3,b3 = layer3.get_weights()
    # print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    # print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    # print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    model.fit(
        X, y,
        epochs=20
    )

    prediction = model.predict(X[500].reshape(1, 400))
    print(f" predicting a one:  {prediction}")
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print(f"prediction after threshold: {yhat}")

    # Net_display_trained_data(X, y, model)

    model.save(r"model/Net_handwritten.keras")