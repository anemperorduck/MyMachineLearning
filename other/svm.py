import math
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

np.random.seed(12)
num_observation = 50

x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observation)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observation)

x = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observation), np.ones(num_observation)))

y = np.where(y <= 0, -1, 1)

# plt.figure(figsize=(12, 8))
# plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.4)
# plt.show()


def Largrangian(w, alpha):
    first_part = np.sum(alpha)
    second_part = np.sum(np.dot(alpha*alpha*y*y*x.T, x))
    res = first_part-0.5*second_part
    return res

def gradient(w,x,y,b,lr):
    for i in range(500):
        for idx, x_i in enumerate(x):
            y_i = y[idx]
            cond = y_i * (np.dot(x_i ,w)-b>=1)
        if cond:
            w -= lr*2*w
        else:
            w -= lr*(2*w - np.dot(x_i, y_i))
            b -= lr*y_i
        
    return w, b

w, b, lr = np.random.random(x.shape[1]), 0, 0.0001
w, b = gradient(w,x,y,b,lr)

def predict(x,w,b):
    pred = np.dot(x,w)-b
    return np.sign(pred)

svm_pred = predict(x, w, b)
print(accuracy_score(y, svm_pred))