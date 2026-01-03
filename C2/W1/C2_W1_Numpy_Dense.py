import numpy as np
from mutils import *
from autils import *
from tensorflow.keras.models import Sequential, load_model

def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)

    for i in range(units):
        w = W[:,i]
        z=np.dot(w,a_in) + b[i]
        a_out[i]=g(z)

    return a_out

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return(a3)

def my_dense_m(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (ndarray (m,j)) : m examples, j units
    """
    a_out = g(np.matmul(A_in, W) + b)
    return a_out

def my_sequential_m(X, W1, b1, W2, b2, W3, b3):
    A1 = my_dense_m(X,  W1, b1, sigmoid)
    A2 = my_dense_m(A1, W2, b2, sigmoid)
    A3 = my_dense_m(A2, W3, b3, sigmoid)
    return(A3)

if __name__ == '__main__':
    X, Y = load_data()
    net_model = load_model(r"model/net_handwritten.keras")
    Net_Numpy_display_trained_data(X, Y, net_model, my_sequential_m)



