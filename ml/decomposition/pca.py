import numpy as np
import numpy.linalg as la
from sklearn.base import BaseEstimator

from ml.base import TransformerMixin
from ml.validation import check_array


class PCA(TransformerMixin, BaseEstimator):
    """
    Principal Component Analysis.

    Attributes
    ----------
    n_components : int
        Number of principal components.
    n_samples_ : int
        Number of Samples.
    n_features_ : int
        Number of features.
    components_ : ndarray of shape (n_features, n_components)
        Principal Components.
    eigenvalues_ : ndarray of shape (n_components,)
        The eigenvalues of the covariance matrix of the training data.
    mean_ : float64
        Mean of the samples.
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def transform(self, X):
        """
        Transforms X using the principal components.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            The transformed input matrix `X`
        """
        X = check_array(X)
        if self.mean_ is not None:
            X -= self.mean_
        return np.dot(X, self.components_)

    def inverse_transform(self, X):
        """
        Undoes the transformation to `X`.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape  (n_samples, n_features)
            The original training data.
        """
        return np.dot(X, self.components_.T) + self.mean_

    def fit(self, X, y=None):
        """
        Fits the model using X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : Ignored
            No usage. (only for API consistency of fit method)

        Returns
        -------
        self : PCA
            The fitted model.
        """
        X = check_array(X)

        self.n_samples_, self.n_features_ = X.shape
        # Center X
        self.mean_ = X.mean(axis=0)
        X -= self.mean_

        # get the covariance matrix of the data
        # by calculation
        # cov = X.T.dot(X) / self.n_samples_
        # or use np.cov function
        cov = np.cov(X, rowvar=False)

        # eigendecomposition of the cov matrix
        eigenvalues, eigenvectors = la.eigh(cov)

        # sort the eigenvectors by eigenvalues from largest to smallest
        idx = eigenvalues.argsort()[::-1][: self.n_components]
        self.eigenvalues_ = eigenvalues[idx]
        self.components_ = eigenvectors[:, idx]
        return self
