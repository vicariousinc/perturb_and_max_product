# Generate data for Section 5.3 - Ising model: MNIST zeros
# And for Section 5.5 - Restricted Boltzmann machines, learning and sampling

import os

import numpy as np
from scipy.ndimage.morphology import binary_dilation
from sklearn.datasets import fetch_openml


def contour_mnist(X):
    # Contour are obtained by dilating the digits
    X = (X > 0.5).astype(int)
    s = np.zeros((3, 3, 3))
    s[1, 1, :3] = 1
    s[1, :3, 1] = 1
    X += binary_dilation(X, s)
    X[X == 1] = -1
    X[X == 0] = 1
    X[X == -1] = 0
    contour_X = np.ones((X.shape[0], 30, 30), int)
    contour_X[:, 1:-1, 1:-1] = X
    return contour_X


if __name__ == "__main__":
    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)

    # Get MNIST dataset
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    y = y.astype(np.uint8)
    X_train, X_test = X[:60000].reshape(-1, 28, 28), X[60000:].reshape(-1, 28, 28)
    y_train, y_test = y[:60000], y[60000:]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    np.savez_compressed(
        f"{data_folder}/mnist_digits_and_labels.npz",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # Get contour dataset
    X_train_contour = contour_mnist(X_train)
    X_test_contour = contour_mnist(X_test)

    np.savez_compressed(
        f"{data_folder}/noisy_mnist.npz",
        X_train=X_train_contour,
        y_train=y_train,
        X_test=X_test_contour,
        y_test=y_test,
    )
    print("Data generated for MNIST and contour MNIST")
