import numpy as np
from scipy import stats
from sklearn.utils import check_array


def partial_correlation(x, y, u):
    """
    Calculates the partial correlation of x and y without the influence
    of u.

    This is useful if the correlation of two variables x and y is conditional to
    the fact that both x and y are correlated with a third variable u.
    For example if x is "consumption of ice at the beach", y is "Number of Shark
    Attacks at the beach" and u is the "Weather condition". Then it should be
    obvious that the correlation between x and y doesn't make any sense, but because
    both x and y are correlated with u, it will have a high correlation. This is
    also known as "spurious correlation".

    Parameters
    ----------
    x : array_like, shape (n,)
        Data.
    y : array_like, shape (n,)
        Data.
    u : array_like, shape (n,)
        The influence of this variable will be removed.

    Returns
    -------
    par_corr : float
        The partial correlation of x and y without the influence of u.
    """
    # r_xy is the estimated correlation between x and y
    r_xy, _ = stats.pearsonr(x, y)
    r_xu, _ = stats.pearsonr(x, u)
    r_yu, _ = stats.pearsonr(y, u)
    par_corr = (r_xy - r_xu * r_yu) / np.sqrt((1 - r_xu) ** 2 * (1 - r_yu) ** 2)
    return par_corr


def multiple_correlation(x, Y, rowvar=False):
    """
    Calculates the multiple correlation of the random variable x to the
    k random Variables in Y.

    The Variables in Y are expected to be in the columns, but you can specify
    it to be row variables with rowvar=True.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D Data.
    Y : array_like, shape (n, k)
        2-D Data.
    rowvar : bool, default=False
        If True, treat rows of Y as variables and columns as observations.

    Returns
    -------
    mult_corr : float
        The multiple correlation bewteen x and Y.
    """
    Y = check_array(Y)
    if rowvar:
        Y = Y.T
    # pairwise correlation of x and each variable in Y
    r_xY = np.array([corr for corr, _ in [stats.pearsonr(x, y) for y in Y.T]])
    Y_corr = np.corrcoef(Y, rowvar=False)
    mult_corr = np.sqrt(np.linalg.multi_dot([r_xY.T, np.linalg.inv(Y_corr), r_xY]))
    return mult_corr
