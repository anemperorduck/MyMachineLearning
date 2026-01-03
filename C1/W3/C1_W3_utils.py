import numpy as np
import matplotlib.pyplot as plt

def sig(z):
    return 1 / (1 + np.exp(-z))

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y

def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)


def plot_data(x, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0

    # Plot examples
    plt.plot(x[positive, 0], x[positive, 1], 'k+', label=pos_label)
    plt.plot(x[negative, 0], x[negative, 1], 'yo', label=neg_label)


def plot_examine_data(x_train, y_train, save_path = r"output.png"):
    # Plot examples
    plot_data(x_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")
    # Set the y-axis label
    plt.ylabel('Exam 2 score')
    # Set the x-axis label
    plt.xlabel('Exam 1 score')
    plt.legend(loc="upper right")
    plt.savefig(save_path)
    plt.show()


def plot_decision_boundary(w, b, X, y, save_path="output.png"):
    plot_data(X[:, 0:2], y)

    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)

        plt.plot(plot_x, plot_y, c="b")
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sig(np.dot(map_feature(u[i], v[j]), w) + b)

        # important to transpose z before calling contour
        z = z.T

        # Plot z = 0
        plt.contour(u, v, z, levels=[0.5], colors="g")

    plt.savefig(save_path)
    plt.show()