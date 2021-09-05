import numpy as np
from numpy import ndarray

def check_X_y(X, y) -> (ndarray, ndarray):
    """
    Checks for valid X and y inputs.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    y : array_like, shape (n_samples,)

    Returns
    -------
    (ndarray, ndarray)
        the checked arrays

    Raises
    ------
    ValueError
        if X is not 2D
        or y is not 1D
        or if the number of rows in X is not equal to the length of y
    """
    X = np.array(X)
    y = np.array(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, instead got an {X.ndim}D array")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, instead got an {y.ndim}D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("mismatched dimensions of X and y")
    return X, y
