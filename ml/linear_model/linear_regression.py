from __future__ import annotations
import numpy as np
from math import sqrt

from numpy import ndarray

from ..validation import check_X_y, check_array, check_consistent_length
from ..metrics.regression import r2_score, adjusted_r2_score


class LinearRegression:
    """Ordinary least squares (OLS) linear linear_model

    We denote the multiple linear model as the following:
       y = ßX + e

    where y is the target vector and ß the coefficients of the model.
    X is called the design matrix and e is the error term.

    We assume that e is normally distributed with zero mean.
    X must have full rank.

    Attributes
    ----------
    beta : ndarray
        The coefficients of the model.
    X : ndarray
        Training data
    y : ndarray
        Target training data.
    n_samples : int
        Number of samples.
    n_features : int
        Number of regressors.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        self.beta = None
        self.X = None
        self.y = None
        self.n_samples = None
        self.n_features = None

    def fit(self, X, y) -> LinearRegression:
        """Fits the linear model using OLS.

        Solves the normal equation to calculate the coefficients in a
        closed form solution:

        .. math::
          \\beta = (X^T X)^{-1} \cdot X^T y

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : array_like, shape (n_samples,)
            Training target data.

        Returns
        -------
        LinearRegression
            The fitted model.
        """
        self.X, self.y = check_X_y(X, y)

        self.n_samples, self.n_features = X.shape
        # because a column will be added:
        self.n_features += 1

        # add column with ones for the intercept
        dummy_column = np.ones(shape=(self.n_samples, 1))
        self.X = np.concatenate((dummy_column, self.X), axis=1)

        # closed form solution: (X.T * X)^(-1) * X.T * y
        self.beta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.y))
        return self

    def predict(self, X, intercept_col=False) -> ndarray:
        """Predicts the target.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples.
        intercept_col : bool, default=False
            If True, then the first column of X must be a vectors of ones.

        Returns
        -------
        y_pred : ndarray
            The predicted values for `X`.

        """
        X = check_array(X)
        if intercept_col:
            y_pred = X.dot(self.beta)
        else:
            y_pred = X.dot(self.beta[1:]) + self.beta[0]
        return y_pred

    def score(self):
        """
        Return the score of the model, which is the r2 measure.

        Returns
        -------
        (r2, adj_r2) : (float, float)
            The score of the model.
        """
        y_pred = self.predict(self.X, intercept_col=True)
        r2 = r2_score(self.y, y_pred)
        adj_r2 = adjusted_r2_score(self.y, y_pred, n_features=self.n_features)
        return r2, adj_r2

    def residual_variance(self):
        """
        Calculates the empirical variance of the residuals

        Returns
        -------
        float64
            The empirical residual variance.

        Notes
        -----
        The total sum of squares (TSS) is equal to the explained variance (ESS)
        plus the residual variance (RSS):
        :math:`TSS = ESS + RSS`

        """
        residuals = (self.y - self.predict(self.X)) ** 2
        return np.sum(residuals) / (self.n_samples - self.n_features)
