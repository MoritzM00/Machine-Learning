from __future__ import annotations
import numpy as np
from math import sqrt

from ..base import LinearModel, RegressorMixin
from ..validation import check_X_y, check_array, check_consistent_length


class LinearRegression(LinearModel, RegressorMixin):
    """Ordinary least squares (OLS) linear regression

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
    residuals : ndarray
        Error terms.
    y_train : ndarray
        Target training data.
    T : int
        Number of samples.
    K : int
        Number of regressors.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        self.beta = None
        self.y_train = None
        self.residuals = None
        self.T = None
        self.K = None

    def fit(self, X, y) -> LinearRegression:
        """Fits the linear regression model using OLS.

        Solves the normal equation to calculate the coefficients in a
        closed form solution:

        .. math::
          \\beta = (X^T X)^{-1} \cdot X^T y

        Parameters
        ----------
        X : array_like, shape (T, K)
            Training data.
        y: array_like, shape (T,)
            Training target data.

        Returns
        -------
        LinearRegression
            The fitted model.
        """
        x_train, y_train = check_X_y(X, y)
        self.y_train = y_train

        self.T, self.K = x_train.shape
        # because a column will be added, we need to increment K
        self.K += 1

        # add column with ones for the intercept
        dummy_column = np.ones(shape=(self.T, 1))
        x_train = np.concatenate((dummy_column, x_train), axis=1)

        # closed form solution: (X.T * X)^(-1) * X.T * y
        self.beta = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T.dot(y_train))

        # calculate residuals
        y_hat = np.matmul(x_train, self.beta)
        self.residuals = self.y_train - y_hat
        return self

    def predict(self, X) -> float:
        """Predicts the target.

        Parameters
        ----------
        X : array_like, shape (K,)
            Samples.

        Returns
        -------
        float
            The predicted value for `x`

        """
        X = check_array(X)
        return np.dot(X, self.beta[1:]) + self.beta[0]

    def score(self) -> float:
        raise NotImplementedError

    def summary(self, verbose: bool = False) -> (float, float, float, float):
        """
        Calculates the following measures of goodness:
         - r-squared
         - adjusted r-squared
         - residual variance (RSS)
         - root mean squared error (RMSE)

        Parameters
        ----------
        verbose : bool, default=False
            if True, then the measures get printed as well

        Return
        -------
        4-tuple of float

        Notes
        -----
        The total sum of squares (TSS) is equal to the explained variance (ESS)
        plus the residual variance (RSS):
        :math:`TSS = ESS + RSS`

        As the r-squared measure increases artificially with the number of regressors, it is important
        to compare it with the adjusted r-squared, which eliminates this problem by scaling it with a factor
        that is anti-proportionally to the number of regressors in the model.
        If the adjusted r-squared is significantly lower than the r-squared
        the model may have too many regressors which do not have a significant impact on the models predictions.
        -> In a "good" model, both measures are close to each other.
        """
        var_residuals = np.dot(self.residuals, self.residuals) / (self.T - self.K)

        # Root mean squared error
        rmse = sqrt(1 / self.T * sum(self.residuals ** 2))

        y_mean = self.y_train.mean()
        r_squared = 1 - (sum(self.residuals ** 2)) / sum((self.y_train - y_mean) ** 2)
        adj_r_squared = 1 - (self.T - 1) / (self.T - self.K) * (1 - r_squared)

        if verbose:
            print(f"R-Squared = {r_squared:.4f}")
            print(f"Adjusted R-Squared = {adj_r_squared:.4f}")
            print(f"Residual variance (RSS) = {var_residuals:.4f}")
            print(f"Root Mean Squared Error = {rmse:.4f}")
        # TODO: return named tuple
        return r_squared, adj_r_squared, var_residuals, rmse
