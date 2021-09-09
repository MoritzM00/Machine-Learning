import numpy as np
from ..validation import check_array, check_consistent_length

# TODO: docstrings of metric functions
def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return np.average(y_true - y_pred)


def median_absolute_error(y_true, y_pred):
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return np.median(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred, root=False):
    y_true, y_pred = _validate_targets(y_true, y_pred)
    mse = np.average((y_true - y_pred) ** 2)
    if root:
        return np.sqrt(mse)
    return mse


def max_error(y_true, y_pred):
    y_true, y_pred = _validate_targets(y_true, y_pred)
    return np.max(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    y_true, y_pred = _validate_targets(y_true, y_pred)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    total_sum_of_squares = np.sum((y_true - np.average(y_true)))
    return 1 - residual_sum_of_squares / total_sum_of_squares


def adjusted_r2_score(y_true, y_pred, n_features):
    y_true, y_pred = _validate_targets(y_true, y_pred)
    n_samples: int = len(y_true)
    return 1 - ((n_samples - 1) / (n_samples - n_features)) * (1 - r2_score(y_true, y_pred))


def _validate_targets(y_true, y_pred):
    # TODO: Does not raise an error, if targets are not 1D, but it should
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)
    return y_true, y_pred
