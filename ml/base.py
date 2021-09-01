from __future__ import annotations
from abc import ABCMeta, abstractmethod

from numpy.typing import ArrayLike
import numpy as np


class LinearModel(metaclass=ABCMeta):
    """Base class for all linear models"""

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearModel:
        """Fit model."""

    @abstractmethod
    def predict(self, X: ArrayLike):
        """
        Predict using the linear model.

        :param X: samples

        :type X: array-like, shape (n_samples, n_features)

        :return: the predicted values
        :rtype: array, shape (n_samples,)
        """


class RegressorMixin(metaclass=ABCMeta):
    """Mixin class for all regression estimators."""

    _estimator_type = "regressor"

    @abstractmethod
    def score(self) -> float:
        """
        Returns the :math:`R^2` score of the model.

        :return: the score of the model
        """
