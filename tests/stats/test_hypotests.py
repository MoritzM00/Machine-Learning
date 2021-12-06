import numpy as np

from ml.stats.hypotests import (
    covariance_matrix_hypotest,
    simultan_mu_cov_hypotest,
    mean_vector_hypotest,
    symmetry_hypotest,
    uncorrelated_features_hypotest,
    two_uncorrelated_features_hypotest
)


def test_covariance_matrix_test():
    X = [
        [630, 390, 720],
        [520, 410, 690],
        [550, 290, 590],
        [450, 190, 650],
        [490, 210, 800],
        [510, 290, 630],
        [530, 350, 700],
        [700, 500, 750],
        [410, 380, 610],
        [590, 290, 580]
    ]
    exp_cov = [[7000, 0, 0],
               [0, 7000, 0],
               [0, 0, 7000]]
    test_statistic, p_value = covariance_matrix_hypotest(X, exp_cov)



def test_mean_vector_test():
    X = [
        [66, 53, 57, 29, 32, 35, 39, 43, 40, 29, 30, 45],
        [77, 63, 58, 38, 36, 26, 27, 25, 25, 36, 28, 63],
        [76, 66, 64, 36, 35, 34, 31, 31, 31, 27, 34, 74]
    ]
    X = np.array(X).T
    simple_X = np.array([[43, 43, 43, 43], [43, 43, 43, 43]]).T
    exp_mu = np.array([43, 43])
    true_sigma = [[240, 170], [170, 240]]

    expected = 0.0000
    actual = mean_vector_hypotest(simple_X, exp_mu, cov=true_sigma)[0]
    assert np.isclose(expected, actual, atol=1e-4)

    expected = 36.0780
    actual = mean_vector_hypotest(simple_X, [0, 0], cov=true_sigma)[0]
    assert np.isclose(expected, actual, atol=1e-4)

    expected = 0.1136
    actual = mean_vector_hypotest(X[:, :2], exp_mu, cov=true_sigma, alpha=.05)[0]
    assert np.isclose(expected, actual, atol=1e-4)

    expected = 0.1039
    actual = mean_vector_hypotest(X[:, :2], exp_mu, alpha=.05)[0]
    assert np.isclose(expected, actual, atol=1e-4)


def test_symmetry_test():
    X = [
        [66, 53, 57, 29, 32, 35, 39, 43, 40, 29, 30, 45],
        [77, 63, 58, 38, 36, 26, 27, 25, 25, 36, 28, 63],
        [76, 66, 64, 36, 35, 34, 31, 31, 31, 27, 34, 74]
    ]
    X = np.array(X).T
    # assert something for symmetry test

    Y = [[175, 172, 188, 167],
         [170, 160, 170, 171],
         [163, 155, 178, 179],
         [180, 175, 168, 189]]
    expected = 2.1221
    actual = symmetry_hypotest(Y)[0]
    assert np.isclose(expected, actual, atol=1e-4)


def test_two_uncorrelated_features_hypotest():
    x = [2.8, 2.6, 2.4, 1.7, 2.5, 1.4, 2.1, 2.2, 1.8, 2.5]
    y = [36, 30, 39, 31, 38, 21, 41, 47, 40, 37]
    expected = 1.2299
    actual = two_uncorrelated_features_hypotest(x, y)[0]
    assert np.isclose(expected, actual, atol=1e-4)


def test_uncorrelated_features_hypotest():
    R = [[1, 0.855, 0.932],
         [0.855, 1, 0.871],
         [0.932, 0.871, 1]]
    expected = 141.0614
    actual = uncorrelated_features_hypotest(R, is_corr_mtx=True, n_samples=43)[0]
    assert np.isclose(expected, actual, atol=1e-4)
