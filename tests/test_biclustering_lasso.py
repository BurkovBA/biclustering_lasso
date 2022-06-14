import unittest

import numpy as np

from ..biclustering_lasso import lasso_coordinate_descent, biclustering_lasso_coordinate_descent


class BiclusteringLassoPrototypeTest(unittest.TestCase):
    """
    python3 -m unittest biclustering_lasso.tests.test_biclustering_lasso.BiclusteringLassoPrototypeTest
    """
    def test_biclustering_lasso_prototype(self):
        X = np.array([[2.1, 0.4, 1.2, 0.3, 1.1], [2.1, -0.7, 2.3, 0.4, 2.2], [2.4, 0.5, 3.2, 0.7, 3.3]])
        w = np.ones(8)

        w_lasso, gap, tol, n_iter = biclustering_lasso_coordinate_descent(
            w=w,
            alpha=5,
            X=X,
            max_iter=1000,
            tol=1e-4
        )

        print(f"w_lasso = {w_lasso}, gap = {gap}, tol = {tol}, n_iter = {n_iter}")


class LassoCoordinateDescentPrototypeTest(unittest.TestCase):
    def test_lasso_coordinate_descent_prototype(self):
        X = np.array([[2.1, 0.4, 1.2, 0.3, 1.1], [2.1, -0.7, 2.3, 0.4, 2.2], [2.4, 0.5, 3.2, 0.7, 3.3]])  # np.random.randn(3, 5)
        y = np.array([1, 2, 3])  # np.random.randn(3)
        w = np.zeros(5)

        print(X)

        w_lasso, gap, tol, n_iter = lasso_coordinate_descent(
            w=w,
            alpha=5,
            X=X,
            y=y,
            max_iter=1000,
            tol=1e-4
        )

        print(f"w_lasso = {w_lasso}, gap = {gap}, tol = {tol}, n_iter = {n_iter}")


if __name__ == "__main__":
    unittest.main()
