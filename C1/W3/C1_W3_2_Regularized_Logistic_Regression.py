import numpy as np
import matplotlib.pyplot as plt
from C1_W3_utils import *
import copy
import math

def compute_cost(x, y, w, b):
    m,_ = x.shape
    cost = 0.
    for i in range(m):
        z = np.dot(w, x[i]) + b
        f_wb = sig(z)
        cost += -y[i]*np.log(f_wb) - (1-y[i])*np.log(1-f_wb)

    cost /= m

    return cost

def compute_cost_reg(x, y, w, b, lambda_=1):
    m,_ = x.shape
    cost_reg = compute_cost(x, y, w, b)
    reg_term = (lambda_ / (2*m)) * np.sum(np.square(w))

    cost_reg += reg_term
    return cost_reg

def compute_gradient(x, y, w, b):
    m,n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        z = np.dot(w, x[i]) +b
        f_wb = sig(z)
        err = f_wb - y[i]

        for j in range(n):
            dj_dw[j] += err * x[i,j]
        dj_db += err
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def compute_gradient_reg(x, y, w, b, lambda_=1):
    m,n = x.shape
    dj_dw, dj_db = compute_gradient(x, y, w, b)

    for i in range(n):
        dj_dw[i] += (lambda_/m)*w[i]

    return dj_dw, dj_db

def gradient_descent_reg(x, y, w, b, cost_function, gradient_function, alpha, num_iters, lambda_=1):
    cost_history = []
    w_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x,y,w,b,lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 10000:
            cost = cost_function(x,y,w,b,lambda_)
            cost_history.append(cost)

        if i%math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(cost_history[-1]):8.2f}   ")

    return w, b, cost_history, w_history

def predict_reg(x, w, b):
    m,n = x.shape
    p = np.zeros(m)

    for i in range(m):
        z = np.dot(w, x[i]) + b
        f_wb = sig(z)

        p[i] = 1 if f_wb>0.5 else 0

    return p

if __name__ == '__main__':
    x_train, y_train = load_data("data/ex2data2.txt")
    # X_mapped, y_train = load_data("data/ex2data2.txt")
    # plot_examine_data(x_train, y_train, "C1_W3_2_examine_data.png")

    X_mapped = map_feature(x_train[:, 0], x_train[:, 1])
    # Initialize fitting parameters
    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 1.

    # Set regularization parameter lambda_ to 1 (you can try varying this)
    lambda_ = 0.01
    # Some gradient descent settings
    iterations = 10000
    alpha = 0.1

    w, b, J_history, _ = gradient_descent_reg(X_mapped, y_train, initial_w, initial_b,
                                          compute_cost_reg, compute_gradient_reg,
                                          alpha, iterations, lambda_)

    # plot_decision_boundary(w, b, X_mapped, y_train, "C1_W3_2_decision_boundary_without_map.png")

    p = predict_reg(X_mapped, w, b)
    print('Train Accuracy: %f'%(np.mean(p == y_train) * 100 ))