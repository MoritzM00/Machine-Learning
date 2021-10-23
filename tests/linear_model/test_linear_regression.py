import math

import numpy as np
from numpy.testing import assert_allclose

from ml.linear_model import LinearRegression


def test_linear_regression_complex():
    y = [12.6, 13.1, 15.1, 15.1, 14.9, 16.1, 17.9, 21, 22.3, 21.9]
    X = [
        [117, 126.3, 134.4, 137.5, 141.7, 149.4, 158.4, 166.5, 177.1, 179.8],
        [84.5, 89.7, 96.2, 99.1, 103.2, 107.5, 114.1, 120.4, 126.8, 127.2],
        [3.1, 3.6, 2.3, 2.3, 0.9, 2.1, 1.5, 3.8, 3.6, 4.1],
    ]
    intercept = -8.7471
    coef = [-0.1819, 0.475, 0.7541]
    model = LinearRegression()
    X = np.array(X).T
    model.fit(X, y)
    assert_allclose(coef, model.coef_, rtol=1e-4)
    assert math.isclose(intercept, model.intercept_, rel_tol=1e-4)


def test_linear_regression_simple():
    y = [15, 19, 21, 22, 23, 28, 27, 29]
    X = [19, 21, 23, 26, 27, 31, 33, 36]
    X = np.reshape(X, (-1, 1))

    intercept = 1.94
    coef = 0.78

    model = LinearRegression().fit(X, y)
    assert_allclose(coef, model.coef_, rtol=1e-4)
    assert math.isclose(intercept, model.intercept_, rel_tol=1e-4)


def test_linear_regression_prediction_score():
    X = [[0], [1]]
    y = [0, 1]
    model = LinearRegression().fit(X, y)
    assert model.predict([[10]]) == 10

    X_test = [[2], [3]]
    y_test = [2, 3]

    assert model.score(X_test, y_test) == 1.0
