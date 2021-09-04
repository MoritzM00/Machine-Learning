from __future__ import annotations
import numpy as np
import numpy.typing as npt
from math import sqrt

from ..base import LinearModel, RegressorMixin


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

    def fit(self, x_train: npt.ArrayLike, y_train: npt.ArrayLike) -> LinearRegression:
        """Fits the linear regression model using OLS.

        Solves the normal equation to calculate the coefficients in a
        closed form solution:

        .. math::
          \\beta = (X^T X)^{-1} \cdot X^T y

        Parameters
        ----------
        x_train : array_like
            Training data.
        y_train: array_like
            Training target data.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `X_train` is not 2D
            or if  `y_train` is not 1D.

        """
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.y_train = y_train

        if x_train.ndim == 1:
            # X must be at least two-dimensional to be used in numpy matmul function
            x_train = np.atleast_2d(x_train).T

        if y_train.ndim != 1:
            raise ValueError("y_train must be 1D")

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

    def predict(self, x: npt.ArrayLike) -> float:
        """Predicts the target for x.

        Parameters
        ----------
        x : array_like, shape (K,)
            Samples

        Returns
        -------
        float
            The predicted value for x.

        """

        x = np.array(x)
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        return np.dot(x, self.beta[1:]) + self.beta[0]

    def score(self) -> float:
        raise NotImplementedError

    def summary(self, verbose=False) -> (float, float, float, float):
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

        return r_squared, adj_r_squared, var_residuals, rmse
