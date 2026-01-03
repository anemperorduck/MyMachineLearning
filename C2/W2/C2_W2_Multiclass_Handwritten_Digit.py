import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
from autils import *
from lab_utils_softmax import plt_softmax
np.set_printoptions(precision=2)
from mutils import *

def my_softmax(z):
    """
    Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    ez = np.exp(z)
    a = ez/np.sum(ez)
    return a

if __name__ =='__main__':
    # plt_softmax(my_softmax)
    X,y = load_data()

    # display_Multiclass_train_data(X,y)

    # 神经网络构建
    tf.random.set_seed(1234)
    model = Sequential(
        [
            tf.keras.layers.InputLayer((400,)),
            tf.keras.layers.Dense(25, activation='relu', name='L1'),
            tf.keras.layers.Dense(15, activation='relu', name='L2'),
            tf.keras.layers.Dense(10, activation='linear', name='L3'),
        ],
        name='my_multiclass_model',
    )
    # 精度运算
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    # model.summary()

    layer1, layer2, layer3 = model.layers

    # W1, b1 = layer1.get_weights()
    # W2, b2 = layer2.get_weights()
    # W3, b3 = layer3.get_weights()
    # print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    # print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    # print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    history = model.fit(
        X, y,
        epochs=40,
    )
    # plot_loss_tf(history)

    model.save(r"model/multiclass_handwritten.keras")

    display_Muticlass_model_label(X, y, model)

    # Predict
    image_of_two = X[1015]
    prediction = model.predict(image_of_two.reshape(1, 400))  # prediction

    print(f" predicting a Two: \n{prediction}")     # [[-5.71  3.22  8.91  4.63 -8.57 -4.75 -0.88  5.92 -3.4  -5.88]]
    print(f" Largest Prediction index: {np.argmax(prediction)}")        # Largest Prediction index: 2

    # softmax 转换为可理解的概率序列
    prediction_p = tf.nn.softmax(prediction)

    print(f"predicting a Two. Probability vector: \n{prediction_p}")        # 概率序列
    print(f"Total of predictions: {np.sum(prediction_p):0.3f}")     # 概率综合为1
