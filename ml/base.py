from __future__ import annotations
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array


class LinearModel(BaseEstimator, metaclass=ABCMeta):
    """Base class for all linear models"""

    @abstractmethod
    def fit(self, X, y) -> LinearModel:
        """Fit model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : array_like, shape (n_samples,)
            Training target data

        Returns
        -------
        LinearModel
            The fitted model.
        """

    def predict(self, X) -> float:
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array_like, shape (n_features,)
            Samples.

        Returns
        -------
        float
            The predicted value.
        """
        check_is_fitted(self)
        X = check_array(X)
        return X.dot(self.coef_) + self.intercept_


class RegressorMixin:
    """
    Mixin for all regressors.

    Implements a score method, which returns the :math:R^2 score of the model.
    """

    _estimtor_type = "regressor"

    def score(self, X, y):
        """
        Returns the score, i.e. the :math:R^2 measure of the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The test data.
        y : ndarray of shape (n_features,)
            The true values of y for the test data.

        Returns
        -------
        float
            The ..math:R^2 score of the model.
        """
        from ml.metrics.regression import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)
