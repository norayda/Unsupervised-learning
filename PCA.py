import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tensorflow import keras


@dataclass
class Pca:
    X: np.ndarray
    n_components: int
    X_PCA = None
    main_composants = None

    def fit(self):
        # Center Data
        self.mean = np.mean(self.X, axis=0)
        X_centered = self.X - self.mean

        # Covariance
        cov = np.cov(X_centered.T)

        # Get EigenValues And Eigen Vectors
        eigen_val, eigen_vec = np.linalg.eigh(cov)

        # Align vector and values
        sorted_index = np.argsort(eigen_val)[::-1]
        eigen_val = eigen_val[sorted_index]
        eigen_vec = eigen_vec[:, sorted_index]

        # Get the first X most important composants
        main_composants = eigen_vec[:, 0:self.n_components]

        # Reduce data using main componants
        X_PCA = np.dot(main_composants.transpose(), X_centered.transpose()).transpose()
        self.X_PCA = X_PCA
        self.main_composants = main_composants
        return X_PCA

    def custom_cov(X):
        X -= X.mean(axis=0)
        nb_item = np.shape(X)[0] - 1
        cov = np.dot(X.T, X) / nb_item

    def encode(self, X):
        return np.dot(X, self.main_composants)

    def decode(self, X):
        return np.dot(X, np.transpose(self.main_composants)) + self.mean

def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(
        x_train, (np.shape(x_train)[0], np.shape(x_train)[1] * np.shape(x_train)[2])
    )
    x_train = x_train / 255.0
    return x_train, y_train

if __name__ == "__main__":
    X, Y = load_mnist()
    model = Pca(X=X[:1000], n_components=2)
    res = model.fit()

