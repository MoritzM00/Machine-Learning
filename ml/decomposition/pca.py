import numpy as np
from ..validation import check_array


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def transform(self, X):
        pass

    def inverse_transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        pass

    def _fit(self, X):
        pass
