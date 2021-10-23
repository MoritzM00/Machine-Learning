import numpy as np
import pytest

from ml.validation import check_array, check_consistent_length, check_X_y


def test_not_2D():
    X = np.array([1, 2])
    with pytest.raises(ValueError):
        check_array(X)


def test_check_X_y():
    X = np.array([[1, 2, 3], [1, 2, 3]])
    y1 = np.array([[1]])
    y2 = np.array([1])
    with pytest.raises(ValueError):
        check_X_y(X, y1)
    with pytest.raises(ValueError):
        check_X_y(X, y2)


def test_check_consistent_length():
    with pytest.raises(ValueError):
        check_consistent_length([[1, 2]], [1, 2])
