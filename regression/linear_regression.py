import numpy as np
import numpy.typing as npt
from scipy import linalg
from math import sqrt


class LinearRegression:
    """
    Ordinary least squares (OLS) linear regression model

    b: Regressor vector
    X: Design matrix
    y: Target vector
    residuals: Residual vector
    T: Number of data points
    K: Number of regressors
    """

    def __init__(self):
        """
        Initializes the model.
        """
        self.b = None
        self.X = None
        self.y = None
        self.residuals = None
        self.T = None
        self.K = None

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> None:
        """
        Fits the Model to X and y using the ordinary least squares estimator.

        :param X: the Tx(K-1) regressor matrix, the k-th variable (the intercept) will be added by the model)
        :param y: the target vector (T dimensional)
        :return: None
        """
        self.X = np.array(X).T
        self.y = np.array(y)

        if self.X.ndim == 1:
            # X must be at least two-dimensional to be used in numpy matmul function
            self.X = np.atleast_2d(self.X).T

        self.T, self.K = self.X.shape
        # because a column will be added, we need to increment K
        self.K += 1

        # add column with ones for the intercept variable
        X_ = np.ones((self.T, self.K))
        X_[:, 1:] = self.X
        self.X = X_

        # calculate (X.T * X)^(-1) * X.T * y
        first = linalg.inv(np.matmul(X_.T, X_))
        second = np.matmul(X_.T, y)
        self.b = np.matmul(first, second)

        # calculate residuals
        y_hat = np.matmul(X_, self.b)
        self.residuals = self.y - y_hat

    def predict(self, x: np.array) -> float:
        """
        Predicts y from x.

        :param x: a K-dimensional array
        :return: the predicted values for y
        """
        if self.K == 1:
            return self.b[0] + self.b[1] * x
        else:
            return np.dot(x, self.b[1:]) + self.b[0]

    def summary(self, verbose=False) -> (float, float, float, float):
        """
        Calculates the following measures of goodness:
         - r-squared and adjusted r-squared
         - residual variance (RSS)
         - root mean squared error (RMSE)

        The total sum of squares (TSS) is equal to the explained variance (ESS)
        plus the residual variance (RSS)
        -> TSS = ESS + RSS

        As the r-squared measure increases artificially with the number of regressors, it is important
        to compare it with the adjusted r-squared, which eliminates this problem by scaling it with a factor
        that is anti-proportionally to the number of regressors in the model.
        If the adjusted r-squared is significantly lower than the r-squared
        the model may have too many regressors which do not have a significant impact on the models predictions.
        -> In a "good" model, both measures are close to each other.

        :param verbose: if True, then the measures get printed as well
        :return: a 4-tuple of floats
        """
        # Standardabweichung der Residuen
        var_residuals = np.dot(self.residuals, self.residuals) / (self.T - self.K)

        # Root mean squared error
        rmse = sqrt(1 / self.T * sum(self.residuals ** 2))

        y_mean = self.y.mean()
        r_squared = 1 - (sum(self.residuals ** 2)) / sum((self.y - y_mean) ** 2)
        adj_r_squared = 1 - (self.T - 1) / (self.T - self.K) * (1 - r_squared)

        if verbose:
            print(f"R-Squared = {r_squared:.4f}")
            print(f"Adjusted R-Squared = {adj_r_squared:.4f}")
            print(f"Residual variance (RSS) = {var_residuals:.4f}")
            print(f"Root Mean Squared Error = {rmse:.4f}")

        return r_squared, adj_r_squared, var_residuals, rmse
