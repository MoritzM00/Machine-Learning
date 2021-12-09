from collections import namedtuple

import numpy as np
import numpy.linalg as la
from scipy import stats
from sklearn.utils import check_array

TestResult = namedtuple("TestResult", "statistic p_value")


def covariance_matrix_hypotest(X, exp_cov):
    """
    Tests whether the covariance matrix of the given Data `X` differs
    significantly from the expected covariance matrix `exp_cov`.

    Parameters
    ----------
    X : array_like, shape (n, p)
        The data.
    exp_cov : array_like, shape (n, p)
        The expected covariance matrix.

    Returns
    -------
    statistic : float
        The value of the test statistic
    p_value: float
        The p-value.
    """
    X = check_array(X, copy=True)
    n, p = X.shape
    cov = np.cov(X, rowvar=False)
    lndet_exp = np.log(la.det(exp_cov))
    lndet_data = np.log(la.det(cov))
    tr_data_inv_exp = np.trace(np.dot(cov, la.inv(exp_cov)))
    statistic = (n - 1) * (lndet_exp - lndet_data + tr_data_inv_exp - p)

    df = p * (p + 1) / 2
    p_value = stats.chi2.sf(x=statistic, df=df)
    return TestResult(statistic, p_value)


def simultan_mu_cov_hypotest(X, exp_mu, exp_cov):
    """
    Tests whether the empirical mean vector and covariance matrix
    is equal to the expected mean and covariance matrix `exp_mu` and `exp_cov`.

    Parameters
    ----------
    X : array_like, shape (n, p)
        The data.
    exp_mu : array_like, shape (p,)
        The expected mean vector.
    exp_cov : array_like, shape (n, p)
        The expected covariance matrix.

    Returns
    -------
    statistic : float
        The value of the test statistic
    p_value: float
        The p-value.
    """
    X = check_array(X, copy=True)
    n, p = X.shape
    cov = np.cov(X, rowvar=False)
    inv_exp_cov = la.inv(exp_cov)
    a = n * np.log(la.det(cov @ inv_exp_cov))
    b = n * np.trace(cov @ inv_exp_cov)
    statistic = -1 * p * n - a + b + mean_vector_hypotest(X, exp_mu, cov=exp_cov)[0]

    df = p + p * (p + 1) / 2
    p_value = stats.chi2.sf(x=statistic, df=df)
    return TestResult(statistic, p_value)


def mean_vector_hypotest(X, exp_mu, cov=None, alpha=None):
    """
    Tests whether the empirical mean vector of the given data set is equal to
    the expected mean vector `exp_mu`.

    Parameters
    ----------
    X : array_like, shape (n, p)
        The data.
    exp_mu : array_like, shape (p,)
        The expected mean vector.
    cov : array_like of shape (p,p), default=None
        If given, then it is assumed to be the ground truth covariance matrix. If cov=None (default)
        then the empirical covariance matrix is calculated.
    alpha

    Returns
    -------
    statistic : float
        The value of the test statistic
    p_value: float
        The p-value.
    """
    X = check_array(X, copy=True)
    exp_mu = check_array(exp_mu, ensure_2d=False)
    n, p = X.shape
    mean = np.mean(X, axis=0)
    if cov is None:
        cov = np.cov(X, rowvar=False)
        const = (n - p) / (p * (n - 1))
        rv = stats.f(p, n - p)
    else:
        const = 1
        rv = stats.chi2(df=p)

    statistic = (
        const * n * la.multi_dot([(mean - exp_mu).T, la.inv(cov), mean - exp_mu])
    )
    p_value = rv.sf(x=statistic)
    if alpha:
        print(
            f"Critical Value for Significance level {1 - alpha:.2%}: {rv.ppf(1 - alpha)}"
        )
    return TestResult(statistic, p_value)


def symmetry_hypotest(X, cov=None, alpha=None):
    """
    Tests whether all feature-wise means in the given data set are equal.

    H0: All expected means are equal.
    H1: At least one feature has a different expected mean.

    Parameters
    ----------
    X : array_like, shape (n, p)
        Samples
    cov : array_like, shape (p, p)
        Optional, default=None. The true covariance matrix (ground truth). If a
        ground truth covariance matrix is given, the test statistic is calculated differently.
    alpha : float, default=None
        Float in [0, 1] Interval. If given, print the critical value for the
        significance level 1 - alpha.

    Returns
    -------
    statistic : float
        The value of the test statistic
    p_value: float
        The p-value.
    """
    X = check_array(X, ensure_min_features=2, copy=True)
    Z = X[:, :-1] - X[:, -1].reshape(-1, 1)
    return mean_vector_hypotest(
        Z, exp_mu=np.zeros(X.shape[1] - 1), cov=cov, alpha=alpha
    )


def two_uncorrelated_features_hypotest(x, y):
    """
    Tests whether two features are uncorrelated.

    Parameters
    ----------
    x : array_like, shape (n,)
        Data of first feature.
    y : array_like, shape (n,)
        Data of second feature.

    Returns
    -------
    statistic : float
        The value of the test statistic
    p_value: float
        The p-value.
    """
    x = check_array(x, copy=True, ensure_2d=False)
    y = check_array(y, copy=True, ensure_2d=False)
    n = len(x)
    r, _ = stats.pearsonr(x, y)
    statistic = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
    rv = stats.t(n - 2)
    p_value = min(1 - rv.cdf(statistic), rv.cdf(statistic))
    return TestResult(statistic, p_value)


def uncorrelated_features_hypotest(X, is_corr_mtx=False, n_samples=-1):
    """
    Tests whether all features are uncorrelated.
    If the null hypothesis can be rejected for a given significance level,
    then there exists at least one pair of features that are not uncorrelated.

    H0: for all (i,j) with i != j, rho(X_i, X_j) = 0
    H1: it exists (i,j) with i != j so that rho(X_i, X_j) != 0

    This test requires that X follows a multivariate gaussian.

    Parameters
    ----------
    X : array_like, shape (n, p) or shape(p, p) if is_corr_mtx is True
        Samples of two features.
    is_corr_mtx : bool, default=False
        If true then `X` is treated as the empirical correlation matrix.
        Then, the parameter `n_samples` has to be set to the number of samples from
        which the correlation matrix was computed.
    n_samples : int, default=-1
        Has to be set to the number of samples if is_corr_mtx=True, else it is ignored.

    Returns
    -------
    statistic : float
        The value of the test statistic
    p_value: float
        The p-value.
    """
    X = check_array(X, copy=True)
    n, p = X.shape
    if is_corr_mtx:
        R = X
        n = n_samples
    else:
        R = np.corrcoef(X, rowvar=False)
    df = p * (p - 1) / 2
    statistic = -1 * (n - 1 - (2 * p + 5) / 6) * np.log(la.det(R))
    p_value = stats.chi2.sf(df=df, x=statistic)
    return TestResult(statistic, p_value)


def specific_correlation_hypotest(x, y, exp_rho):
    """
    Tests whether the correlation between x and y is significantly
    different from the expected correlation exp_rho.

    H0: The correlation is equal to exp_rho.
    H1: The correlation is not equal to exp_rho.

    This test requires that x and y are normally distributed.

    Parameters
    ----------
    x : array_like, shape (n,)
        Data of first feature.
    y : array_like, shape (n,)
        Data of second feature.
    exp_rho : float
        The expected correlation.

    Returns
    -------
    statistic : float
        The value of the test statistic
    p_value: float
        The p-value.
    """
    x = check_array(x, copy=True, ensure_2d=False)
    y = check_array(y, copy=True, ensure_2d=False)
    n = len(x)
    r, _ = stats.pearsonr(x, y)
    # the transformed value (Fishers Z Transformation) is approximately
    # normally distributed with variance 1/(n-3). for the mean, see below
    transformed_r = np.arctanh(r)

    # because E(g(x)) != g(E(x)) (here g is arctanh) the mean of the transformed
    # estimator is not just equal to arctanh(r)
    transformed_mu = np.arctanh(exp_rho) + exp_rho / (2 * (n - 1))

    statistic = (transformed_r - transformed_mu) * np.sqrt(n - 3)
    p_value = 2 * stats.norm.sf(statistic)
    return statistic, p_value
