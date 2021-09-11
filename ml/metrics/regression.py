import numpy as np
from ..validation import check_array, check_consistent_length


def mean_absolute_error(y_true, y_pred):
    """
    Returns the mean absolute error.

    Parameters
    ----------
    y_true : array_like
        The correct targets.
    y_pred : array_like
        The estimated targets.

    Returns
    -------
    float
        The mean absolute error.
    """
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return np.average(y_true - y_pred)


def median_absolute_error(y_true, y_pred):
    """
    Returns the median absolute error.

    Parameters
    ----------
    y_true : array_like
        The correct targets.
    y_pred : array_like
        The estimated targets.

    Returns
    -------
    float
        The median absolute error.
    """
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return np.median(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred, root=False):
    """
    Returns the mean squared error (MSE) if root = False, else
    it returns the root mean squared error (RMSE)

    Parameters
    ----------
    y_true : array_like
        The correct targets.
    y_pred : array_like
        The estimated targets.
    root : bool, default=False
        If True, then the root mean squared error will be returned.

    Returns
    -------
    mse
        The (root) mean squared error.
    """
    y_true, y_pred = _validate_targets(y_true, y_pred)
    mse = np.average((y_true - y_pred) ** 2)
    if root:
        return np.sqrt(mse)
    return mse


def max_error(y_true, y_pred):
    """
    Returns the max absolute error

    Parameters
    ----------
    y_true : array_like
        The correct targets.
    y_pred : array_like
        The estimated targets.

    Returns
    -------
    float
        The max absolute error.
    """
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return np.max(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Returns the :math:`R^2` score.
    Worst score is 0 and best possible score is 1.

    Parameters
    ----------
    y_true : array_like
        The correct targets.
    y_pred : array_like
        The estimated targets.

    Returns
    -------
    float
        The r2 score.

    Notes
    -----
    As the r-squared measure increases artificially with the number of regressors, it is important
    to compare it with the adjusted r-squared, which eliminates this problem by scaling it with a factor
    that is anti-proportionally to the number of regressors in the model.
    If the adjusted r-squared is significantly lower than the r-squared
    the model may have too many regressors which do not have a significant impact on the models predictions.
    Therefore, in a "good" model, both measures are close to each other.
    """
    y_true, y_pred = _validate_targets(y_true, y_pred)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    total_sum_of_squares = np.sum((y_true - np.average(y_true) ** 2))
    return 1 - residual_sum_of_squares / total_sum_of_squares


def adjusted_r2_score(y_true, y_pred, n_features):
    """
    Returns the :math:`R^2` score.

    The higher the value, the better. Because of the scaling factor, unlike
    the R2_score, this measure is *not* in the interval [0, 1].

    Parameters
    ----------
    y_true : array_like
        The correct targets.
    y_pred : array_like
        The estimated targets.
    n_features : int
        The number of features in the model.

    Returns
    -------
    float
        The adjusted r2 score.
    """
    y_true, y_pred = _validate_targets(y_true, y_pred)
    n_samples: int = len(y_true)
    return 1 - ((n_samples - 1) / (n_samples - n_features)) * (1 - r2_score(y_true, y_pred))


def _validate_targets(y_true, y_pred):
    # TODO: Does not raise an error, if targets are not 1D, but it should
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)
    return y_true, y_pred
