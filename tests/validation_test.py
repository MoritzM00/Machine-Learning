import unittest
import numpy as np

from ml.validation import check_X_y, check_array, check_consistent_length


class TestValidation(unittest.TestCase):
    def test_not_2D(self):
        X = np.array([1, 2])
        self.assertRaises(
            ValueError,
            lambda: check_array(X)
        )

    def test_check_X_y(self):
        X = np.array(
            [[1, 2, 3],
             [1, 2, 3]]
        )
        y1 = np.array([[1]])
        y2 = np.array([1])
        self.assertRaises(
            ValueError,
            lambda: check_X_y(X, y1)
        )
        self.assertRaises(
            ValueError,
            lambda: check_X_y(X, y2)
        )

    def test_check_consistent_length(self):
        self.assertRaises(
            ValueError,
            lambda: check_consistent_length(
                [[1, 2]],
                [1, 2],
            )
        )


if __name__ == '__main__':
    unittest.main()
