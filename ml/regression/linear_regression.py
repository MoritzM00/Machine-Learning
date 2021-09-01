from __future__ import annotations
import numpy as np
import numpy.typing as npt
from math import sqrt

from ..base import LinearModel, RegressorMixin


class LinearRegression(LinearModel, RegressorMixin):
    r"""
    Ordinary least squares (OLS) linear regression model.

    We denote the multiple linear model as the following:

    .. math::
       y_t = \beta_1 + \beta_2 x_{t2} + \beta_3 x_{t3} + \cdots + \beta_K x_{tK} + e_t
    with :math:`t = 1, \ldots, T`
    and :math:`e_t` being the error term. We assume that :math:`e_t \sim N(0, \sigma^2)`

    In matrix notation, we can simply write

    .. math::
       y = \beta X + e
    where :math:`X \in \mathbb{R}^{TxK}` the design matrix
    and :math:`e, y \in \mathbb{R}^T`
    """

    def __init__(self):
        """
        Initializes the model.

        beta: includes bias, beta[0], and weights, beta[:1]
        residuals: error of the predicted values for y
        y_train: training data for the target vector y
        T: number of samples
        K: number of regressors
        """
        self.beta = None
        self.y_train = None
        self.residuals = None
        self.T = None
        self.K = None

    def fit(self, x_train: npt.ArrayLike, y_train: npt.ArrayLike) -> LinearRegression:
        """
        Fits the Model to X and y using the ordinary least squares estimator.
        The OLS estimator has a closed form solution:

        .. math::
          \\beta = (X^T X)^{-1} \cdot X^T y

        :param x_train: the Tx(K-1) regressor matrix, the k-th variable (the intercept) will be added by the model
        :param y_train: the target vector (T dimensional)
        :return: self
        :raises ValueError: if y_train is not 1D
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
        """
        Predicts y using the x data vector.

        :param x: a K-dimensional array
        :return: the predicted values for y

        :raises ValueError: if x is not 2D
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

        The total sum of squares (TSS) is equal to the explained variance (ESS)
        plus the residual variance (RSS):
        :math:`TSS = ESS + RSS`

        As the r-squared measure increases artificially with the number of regressors, it is important
        to compare it with the adjusted r-squared, which eliminates this problem by scaling it with a factor
        that is anti-proportionally to the number of regressors in the model.
        If the adjusted r-squared is significantly lower than the r-squared
        the model may have too many regressors which do not have a significant impact on the models predictions.
        -> In a "good" model, both measures are close to each other.

        :param verbose: if True, then the measures get printed as well
        :return: a 4-tuple of floats
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