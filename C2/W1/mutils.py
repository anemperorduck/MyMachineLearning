import numpy as np
import matplotlib.pyplot as plt


def Net_display_train_data(X, Y):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    m, n = X.shape
        
    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1)

    # flat展平为一维迭代器
    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Display the label above the image
        ax.set_title(Y[random_index,0])
        ax.set_axis_off()

    plt.savefig(r'img\C2_W1_Net_display_train_data.png')
    plt.show()


def Net_display_trained_data(X, Y, model):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    m, n = X.shape

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])  # [left, bottom, right, top]

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
        if prediction >= 0.5:
            yhat = 1
        else:
            yhat = 0

        # Display the label above the image
        ax.set_title(f"{Y[random_index, 0]},{yhat}")
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=16)
    plt.savefig(r"img/C2_W1_Net_display_trained_data")
    plt.show()



def Net_Numpy_display_trained_data(X, Y, net_model, my_sequential):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    m, n = X.shape

    layer1, layer2, layer3 = net_model.layers
    W1_tmp, b1_tmp = layer1.get_weights()
    W2_tmp, b2_tmp = layer2.get_weights()
    W3_tmp, b3_tmp = layer3.get_weights()

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])  # [left, bottom, right, top]

    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Predict using the Neural Network implemented in Numpy
        my_prediction = my_sequential(X[random_index], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp)
        my_yhat = int(my_prediction >= 0.5)

        # Predict using the Neural Network implemented in Tensorflow
        tf_prediction =net_model.predict(X[random_index].reshape(1, 400))
        tf_yhat = int(tf_prediction >= 0.5)

        # Display the label above the image
        ax.set_title(f"{Y[random_index, 0]},{tf_yhat},{my_yhat}")
        ax.set_axis_off()
    fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=16)
    plt.savefig(r"C2_W1_Net_Numpy_display_trained_data.png")
    plt.show()