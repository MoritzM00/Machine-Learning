import numpy as np
import numpy.linalg as la
from ml.validation import check_array


class PCA:
    """
    Principal Component Analysis.

    Attributes
    ----------
    n_components : int
        Number of principal components.
    n_samples : int
        Number of Samples.
    n_features : int
        Number of features.
    components : ndarray of shape (n_features, n_components)
        Principal Components.
    eigenvalues : ndarray of shape (n_components,)
        The eigenvalues of the covariance matrix of the training data.
    mean : float64
        Mean of the samples.
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.n_samples = None
        self.n_features = None
        self.components = None
        self.eigenvalues = None
        self.mean = None

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
        if self.mean is not None:
            X -= self.mean
        return np.dot(X, self.components)

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
        return np.dot(X, self.components.T) + self.mean

    def fit_transform(self, X, y=None):
        """
        Fits the model to X and transforms X.
        Equivalent to fit(X).transform(X)

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : Ignored
            No usage. (only for API consistency of fit method)

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            The transformed training data.
        """
        return self.fit(X).transform(X)

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
        self.n_samples, self.n_features = X.shape
        # Center X
        self.mean = X.mean(axis=0)
        X -= self.mean

        # get the covariance matrix of the data
        cov = np.cov(X, rowvar=False)
        # eigendecomposition of the cov matrix
        eigenvalues, eigenvectors = la.eigh(cov)

        # sort the eigenvectors by eigenvalues from largest to smallest
        idx = eigenvalues.argsort()[::-1][:self.n_components]
        self.eigenvalues = eigenvalues[idx]
        self.components = eigenvectors[:, idx]
        return self
