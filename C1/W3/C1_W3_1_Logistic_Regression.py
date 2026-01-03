import numpy as np
import matplotlib.pyplot as plt
from C1_W3_utils import *
import copy
import math

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

def compute_cost(x, y, w, b, lambda_=1):
    m = x.shape[0]
    cost = 0.
    for i in range(m):
        z = np.dot(w, x[i]) + b
        f_wb = sigmoid(z)
        cost += (-y[i]*np.log(f_wb)) - (1-y[i])*np.log(1-f_wb)
    cost = cost/m

    return cost

def compute_gradient(x, y, w, b, lambda_=None):
    m,n = x.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid( np.dot(x[i], w)+b )
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i,j]        # **只更新第j个维度**
        dj_db += err
    dj_dw = dj_dw/m
    dj_db=dj_db/m

    return dj_dw, dj_db

def gradient_descent(x, y, w, b, cost_function, gradient_function, alpha, num_iters, lambda_=1):
    cost_history = []
    w_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 10000:
            cost = cost_function(x, y, w, b)
            cost_history.append(cost)

        if i%math.ceil(num_iters/10)==0 or i==(num_iters-1):
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(cost_history[-1]):8.2f}   ")

    return w, b, cost_history, w_history

def predict(x, w, b):
    m, n = x.shape
    p = np.zeros(m)

    for i in range(m):
        z_wb = np.dot(x[i], w)
        z_wb += b
        g_z = sigmoid(z_wb)

        p[i]=1 if g_z>0.5 else 0

    return p


if __name__ == "__main__":
    x_train, y_train = load_data("./data/ex2data1.txt")
    # plot_examine_data(x_train, y_train, "C1_W3_1_examine_data.png")

    np.random.seed(1)
    initial_w = 0.01 * (np.random.rand(2) - 0.5)
    initial_b = -8

    iterations = 1000
    alpha = 0.001

    w, b, cost_history, w_history = gradient_descent(
        x_train, y_train, initial_w, initial_b,
        compute_cost, compute_gradient,
        alpha, iterations, 0
    )

    # plot_decision_boundary(w, b, x_train, y_train, "C1_W3_1_decision_boundary.png")

    p = predict(x_train, w, b)
    print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))
