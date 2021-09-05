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
    X = check_2D(X, "X")
    y = np.array(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, instead got an {y.ndim}D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"mismatched dimensions of X and y:"
                         f" Rows in X = {X.shape[0]} != {y.shape[0]} = length of y")
    return X, y


def check_2D(array, arr_name: str = "") -> ndarray:
    """
    Checks if the given array is 2D and throws a ValueError otherwise.

    Parameters
    ----------
    array : array_like
        The array to check.
    arr_name : str, optional
        The name of the array to display in the error message.

    Returns
    -------
    ndarray
        The checked array.

    Raises
    ------
    ValueError
        If `array` is not 2D.
    """
    arr = np.array(array)
    if arr.ndim != 2:
        if arr_name:
            arr_name = " `" + arr_name + "`"
        msg = f"The input array{arr_name} must be 2D, instead got an {array.ndim}D array."
        raise ValueError(msg)
    return arr
