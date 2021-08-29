import unittest
from regression.linear_regression import LinearRegression
import pandas as pd
import numpy as np


class TestLinearRegression(unittest.TestCase):
    def test_linear_regression(self):
        y = [12.6, 13.1, 15.1, 15.1, 14.9, 16.1, 17.9, 21, 22.3, 21.9, 21]
        X = [[117, 126.3, 134.4, 137.5, 141.7, 149.4, 158.4, 166.5, 177.1, 179.8, 183.8],
             [84.5, 89.7, 96.2, 99.1, 103.2, 107.5, 114.1, 120.4, 126.8, 127.2, 128.7],
             [3.1, 3.6, 2.3, 2.3, 0.9, 2.1, 1.5, 3.8, 3.6, 4.1, 1.9]]
        beta = [-8.6716, -0.1024, 0.3657, 0.6722]
        model = LinearRegression()
        model.fit(X, y)
        for b1, b2 in zip(beta, model.b):
            self.assertAlmostEqual(b1, b2, delta=0.01)

        y = [15, 19, 21, 22, 23, 28, 27, 29]
        X = [19, 21, 23, 26, 27, 31, 33, 36]

        beta = [1.94, 0.78]
        df = pd.DataFrame(np.array(X).T)

        model = LinearRegression()
        model.fit(X, y)
        for b1, b2 in zip(beta, model.b):
            self.assertAlmostEqual(b1, b2, delta=0.01)

        y = [12.6, 13.1, 15.1, 15.1, 14.9, 16.1, 17.9, 21, 22.3, 21.9, 21]
        X = [[117, 126.3, 134.4, 137.5, 141.7, 149.4, 158.4, 166.5, 177.1, 179.8, 183.8],
             [84.5, 89.7, 96.2, 99.1, 103.2, 107.5, 114.1, 120.4, 126.8, 127.2, 128.7]]

        model = LinearRegression()
        model.fit(X, y)


if __name__ == '__main__':
    unittest.main()
