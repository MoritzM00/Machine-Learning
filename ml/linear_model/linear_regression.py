from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_X_y

from ml.base import LinearModel, RegressorMixin


class LinearRegression(LinearModel, RegressorMixin):
    """
    Ordinary least squares (OLS) linear regression

    We denote the multiple linear model as the following:
       y = Xw + e

    where y is the target vector and ß the coefficients of the model.
    X is called the design matrix and e is the error term.

    We assume that e is normally distributed with zero mean.
    X must have full rank.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        The estimated coefficients of the model.
    intercept_ : float
        The intercept.
    n_samples : int
        Number of samples.
    n_features : int
        Number of regressors.
    """

    def fit(self, X, y) -> LinearRegression:
        """
        Fits the linear model using OLS.

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
        self.X_, self.y_ = check_X_y(X, y)

        self.n_samples_, self.n_features_ = self.X_.shape
        # because a column will be added:
        self.n_features_ += 1

        # add column with ones for the intercept
        intercept_column = np.ones(shape=(self.n_samples_, 1))
        self.X_ = np.concatenate((intercept_column, self.X_), axis=1)

        # closed form solution: (X.T * X)^(-1) * X.T * y
        self.coef_ = np.linalg.inv(self.X_.T.dot(self.X_)).dot(self.X_.T.dot(self.y_))
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
        return self

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
        residuals = (self.y_ - self.predict(self.X_)) ** 2
        return np.sum(residuals) / (self.n_samples_ - self.n_features_)
