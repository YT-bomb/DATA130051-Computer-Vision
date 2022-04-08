# example of loading the mnist dataset
from tensorflow.keras.datasets import mnist
import numpy as np


def get_MNIST_data(
    num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True
):
    """
        Load the MNIST dataset from keras
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask].astype('float64')
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask].astype('float64')
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask].astype('float64')
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Package data into a dictionary
    return {
        "X_train": np.reshape(X_train, (num_training, -1)),
        "y_train": y_train,
        "X_val": np.reshape(X_val, (num_validation, -1)),
        "y_val": y_val,
        "X_test": np.reshape(X_test, (num_test, -1)),
        "y_test": y_test,
    }


if __name__ == "__main__":
    data = get_MNIST_data()
    print("Type of the train data: ", type(data["X_train"]))
    print("Shape of the X_train: ", data["X_train"].shape)