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
    X = check_array(X)
    y = check_array(y, ensure_2d=False)
    check_consistent_length(X, y)
    return X, y


def check_array(array, ensure_2d=True) -> ndarray:
    """
    Checks if the given array is 2D by default and throws a ValueError otherwise.

    Parameters
    ----------
    array : array_like
        The array to check.
    ensure_2d : bool, default=True
        True if the input array has to be 2D.

    Returns
    -------
    ndarray
        A copy of the array.
    """
    arr = np.array(array)
    if arr.ndim == 0:
        raise ValueError(f"Expected 2D array but got scalar array instead: {arr}")
    elif arr.ndim == 1 and ensure_2d:
        raise ValueError(f"Expected 2D array, but got 1D array instead: {arr}")
    elif arr.ndim > 2:
        raise ValueError(f"Expected 2D array, but got an {arr.ndim}D array instead: {arr}")
    return arr.copy()


def check_consistent_length(*arrays):
    """
    Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """
    arrays = [np.array(X) for X in arrays if X is not None]
    lengths = [X.shape[0] for X in arrays]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )
