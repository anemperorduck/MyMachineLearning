import copy
import math
import matplotlib.pyplot as plt
import numpy as np


def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]

    return X,y


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.
    for i in range(m):
        fwb_i = np.dot(X[i], w) + b
        cost = cost + (fwb_i -y[i])**2
    cost = cost /(2* m)

    return np.squeeze(cost)     # 保证cost是标量


def compute_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros(n,)
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) +b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err* X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent_houses(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = X.shape[0]
    w = copy.deepcopy(w_in)
    b = b_in
    history = {'cost': [], 'params': [], 'grads': [], 'iter': []}
    save_interval = np.ceil(num_iters/10000)        # prevent resource exhaustion for long runs

    print(f"Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  ")
    print(f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|")

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J,w,b at each save interval for graphing
        if i ==0 or i % save_interval == 0:
            history["cost"].append(cost_function(X, y, w, b))
            history["params"].append([w,b])
            history["grads"].append([dj_dw,dj_db])
            history["iter"].append(i)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            cst = cost_function(X, y, w, b)
            print(
                f"{i:9d} "
                f"{cst:0.5e} "
                f"{w[0]: 0.1e} "
                f"{w[1]: 0.1e} "
                f"{w[2]: 0.1e} "
                f"{w[3]: 0.1e} "
                f"{b: 0.1e} "
                f"{dj_dw[0]: 0.1e} "
                f"{dj_dw[1]: 0.1e} "
                f"{dj_dw[2]: 0.1e} "
                f"{dj_dw[3]: 0.1e} "
                f"{dj_db: 0.1e}"
            )

    return w, b, history  # return w,b and history for graphing


def run_gradient_descent(X, y, alpha=1e-6, iterations=1000):
    m,n = X.shape
    initial_w = np.zeros(n)
    initial_b = 0.

    w_out, b_out, history = gradient_descent_houses(
        X, y, initial_w, initial_b,
        compute_cost, compute_gradient,
        alpha, iterations)

    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.2f}")
    return w_out, b_out, history


def z_score_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    x_norm = (X -mu) / sigma

    return x_norm, mu, sigma


if __name__ == '__main__':
    X_train, y_train = load_house_data()
    # print(f"X_train = \n{X_train}\n\ny_train = \n{y_train}")
    X_features = ['size(sqft)', 'bedroom', 'floors', 'age']

    def plot_sample():
        # fig --> 包含所有子图的图形对象;
        # ax -- > 包含每个子图的数组;
        fig, ax = plt.subplots(1, 4, figsize=(18,7), sharey=True)       # sharey --> 共享y轴
        for i in range(len(ax)):
            ax[i].scatter(X_train[:,i], y_train)
            ax[i].set_xlabel(X_features[i])
        ax[0].set_ylabel("Price(1000's)")

        plt.savefig('C1_W2_Lab03_Simple.png')
        plt.show()

    def plot_cost(X, y, history):
        ws = np.array([p[0] for p in history['params']])
        rng = max( abs(ws[:, 0].min()), abs(ws[:,0].max()) )
        wr = np.linspace(-rng+0.27, rng+0.27,20)
        cst = [compute_cost(X, y, np.array([wr[i], -32, -67, -1.46]), 221) for i in range(len(wr))]
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        ax[0].plot(history['iter'], history['cost'])
        ax[0].set_title("Cost to Iteration")
        ax[0].set_xlabel('iters')
        ax[0].set_ylabel('cost')

        ax[1].plot(wr, cst)
        ax[1].set_title("Cost vs w[0]")
        ax[1].set_xlabel("w[0]")
        ax[1].set_ylabel("Cost")
        ax[1].plot(ws[:, 0], history["cost"])

        plt.savefig('C1_W2_Lab03_w02Cost_1e-7.png')
        plt.show()

    # plot_sample()

    # _, _, history = run_gradient_descent(X_train, y_train, alpha=1e-7, iterations=10)
    # plot_cost(X_train, y_train, history)

    x_norm, x_mu, x_sigma = z_score_normalize_features(X_train)
    x_mean = (X_train - x_mu)
    def plot_normalized():
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        ax[0].scatter(X_train[:, 0], X_train[:, 3])
        ax[0].set_xlabel(X_features[0])
        ax[0].set_ylabel(X_features[3])
        ax[0].set_title("non-normalized")
        ax[0].axis('equal')

        ax[1].scatter(x_mean[:, 0], x_mean[:, 3])
        ax[1].set_xlabel(X_features[0])
        ax[0].set_ylabel(X_features[3])
        ax[1].set_title(r"X - $\mu$")
        ax[1].axis('equal')

        ax[2].scatter(x_norm[:, 0], x_norm[:, 3])
        ax[2].set_xlabel(X_features[0])
        ax[0].set_ylabel(X_features[3])
        ax[2].set_title(r"Z-score normalized")
        ax[2].axis('equal')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("distribution of features before, during, after normalization")
        plt.savefig("C1_W2_Lab03_X_normalized.png")
        plt.show()
    # plot_normalized()

    w_norm, b_norm, history = run_gradient_descent(x_norm, y_train, 1.0e-1, 1000)

    def plot_features_normalized():
        dlc = dict(dlblue='#0096ff', dlorange='#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
        m = x_norm.shape[0]
        yp = np.zeros(m)
        for i in range(m):
            yp[i] = np.dot(x_norm[i], w_norm) + b_norm

            # plot predictions and targets versus original features
        fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
        for i in range(len(ax)):
            ax[i].scatter(X_train[:, i], y_train, label='target')
            ax[i].set_xlabel(X_features[i])
            ax[i].scatter(X_train[:, i], yp, color=dlc["dlorange"], label='predict')
        ax[0].set_ylabel("Price");
        ax[0].legend();
        fig.suptitle("target versus prediction using z-score normalized model")
        plt.savefig("C1_W2_Lab03_features_normalized.png")
        plt.show()
    # plot_features_normalized()

    # First, normalize out example.
    x_house = np.array([1200, 3, 1, 40])
    x_house_norm = (x_house - x_mu) / x_sigma
    print(x_house_norm)
    x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
    print(
        f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict * 1000:0.0f}"
    )