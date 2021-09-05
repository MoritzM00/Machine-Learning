from __future__ import annotations
from abc import ABCMeta, abstractmethod


class LinearModel(metaclass=ABCMeta):
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

    @abstractmethod
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
            The predicted value
        """


class RegressorMixin(metaclass=ABCMeta):
    """Mixin class for all regression estimators."""

    _estimator_type = "regressor"

    @abstractmethod
    def score(self) -> float:
        """
        Calculates the :math:`R^2` score of the model.

        Returns
        -------
        float
            the score of the model
        """
