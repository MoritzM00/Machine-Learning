import numpy as np
from ..validation import check_array


class PCA:
    # TODO: documentation
    # TODO: testing

    def __init__(self, n_components):
        self.n_components = n_components
        self.n_samples = None
        self.n_features = None
        self.components = None
        self.mean = None

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def transform(self, X):
        X = check_array(X)
        if self.mean is not None:
            X -= self.mean
        return np.dot(X, self.components.T)

    def inverse_transform(self, X):
        return np.dot(X, self.components) + self.mean

    def fit_transform(self, X, y=None):
        U, S, Vt = self._fit(X)
        U = U[:, :self.n_components]

        # X_new = X * V = U * S * V * Vt = U * S
        # because V (and U) are orthogonal matrices, i.e. V times V transpose is
        # equal to the (n_features, n_features) identity matrix
        U *= S[: self.n_components]
        return U

    def _fit(self, X):
        X = check_array(X)
        self.n_samples, self.n_features = X.shape

        # Center X
        self.mean = X.mean(axis=0)
        X -= self.mean

        # Singular Value Decomposition
        # U contains the left singular vectors, which are the eigenvectors of X(X.T)
        # Vt contains the right singular vectors, which are the eigenvectors of X.T X
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.components = Vt[, :self.n_components]

        return U, S, Vt
