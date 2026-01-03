import numpy as np
import matplotlib.pyplot as plt
from autils import *


def display_Multiclass_train_data(X, Y):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # You do not need to modify anything in this cell

    m, n = X.shape

    fig, axes = plt.subplots(8, 8, figsize=(5, 5))
    fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]

    # fig.tight_layout(pad=0.5)
    widgvis(fig)
    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Display the label above the image
        ax.set_title(Y[random_index, 0])
        ax.set_axis_off()
        fig.suptitle("Label, image", fontsize=14)
    plt.savefig(r"img/C2_W2_Multiclass_display_train_data.png")
    plt.show()



def display_Multiclass_trained_data(X, Y, model):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # You do not need to modify anything in this cell

    m, n = X.shape

    fig, axes = plt.subplots(8, 8, figsize=(5, 5))
    fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
    widgvis(fig)
    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1, 400))
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)

        # Display the label above the image
        ax.set_title(f"{Y[random_index, 0]},{yhat}", fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.savefig(r'img/C2_W2_Multiclass_trained_data.png')
    plt.show()


def display_Muticlass_model_label(X, Y, model):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    m, n = X.shape

    fig, axes = plt.subplots(8, 8, figsize=(5, 5))
    fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
    widgvis(fig)
    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1, 400))
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)

        # Display the label above the image
        ax.set_title(f"{Y[random_index, 0]},{yhat}", fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.savefig(r'img/C2_W2_Multiclass_model_label.png')
    plt.show()