import unittest
import numpy as np

from utils.data_gen import (
    gaussian_mixture, regression_mixture, regression_with_missing_cov,
    add_outliers
)


class TestUtils(unittest.TestCase):

    def test_gaussian_mixture(self):
        X1 = gaussian_mixture(n_samples=10,
                              n_features=2,
                              mean=np.array([1, 2]),
                              sigma=1,
                              bgn_sigma=0.01,
                              subspace=None)
        assert X1.shape == (10, 2)
        assert np.all(X1 != 0)

        # check if the `subspace` parameter takes effect
        X2 = gaussian_mixture(n_samples=10,
                              n_features=2,
                              mean=np.array([1, 0]),
                              sigma=1,
                              bgn_sigma=0,
                              subspace=np.array([0]))
        assert X2.shape == (10, 2)
        assert np.all(X2[:, 1] == 0)
        assert np.all(X2[:, 0] != 0)

    def test_regression_mixture(self):
        X1, Y1 = regression_mixture(n_samples=10,
                                    n_features=2,
                                    mean=np.array([1, 2]),
                                    sigma=1,
                                    bgn_sigma=0.01,
                                    subspace=None)
        assert X1.shape == (10, 2)
        assert Y1.shape == (10,)
        assert np.all(X1 != 0)

        # check if the `subspace` parameter takes effect
        X2, Y2 = regression_mixture(n_samples=10,
                                    n_features=2,
                                    mean=np.array([1, 0]),
                                    sigma=1,
                                    bgn_sigma=0,
                                    subspace=np.array([0]))
        assert X2.shape == (10, 2)
        assert Y2.shape == (10,)
        assert np.all(X2[:, 1] == 0)
        assert np.all(X2[:, 0] != 0)

    def test_regression_with_missing_cov(self):
        X1, Y1 = regression_with_missing_cov(n_samples=10,
                                             n_features=2,
                                             mean=np.array([1, 2]),
                                             sigma=1,
                                             missing_prob=0.5,
                                             bgn_sigma=0.01,
                                             subspace=None)
        assert X1.shape == (10, 2)
        assert Y1.shape == (10,)
        assert np.all(X1 != 0)

        # check if the `subspace` parameter takes effect
        X2, Y2 = regression_mixture(n_samples=10,
                                    n_features=2,
                                    mean=np.array([1, 0]),
                                    sigma=1,
                                    bgn_sigma=0,
                                    subspace=np.array([0]))
        assert X2.shape == (10, 2)
        assert Y2.shape == (10,)
        assert np.all(np.logical_or(X2[:, 1] == 0, np.isnan(X2[:, 1])))
        assert np.all(X2[:, 0] != 0)

    def test_add_outliers(self):
        # check if the X is changed when `inplace` is True
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        X_orig = X.copy()

        X3 = add_outliers(X, n_outliers=2, replace=True, inplace=True)
        assert X3.shape == (3, 3)
        assert X3 is X
        assert np.any(X3 != X_orig, axis=1).sum() == 2  # check if exactly two rows are mutated

        # check if the X is not changed when `inplace` is False
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        X_orig = X.copy()
        X4 = add_outliers(X, n_outliers=2, replace=True, inplace=False)
        assert X4.shape == (3, 3)
        assert X4 is not X
        assert np.all(X_orig == X)
        assert np.any(X4 != X_orig, axis=1).sum() == 2  # check if exactly two rows are mutated

        # check X's shape is preserved when the `replace` is True
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        X1 = add_outliers(X, n_outliers=2, replace=True, inplace=False)
        assert X1.shape == (3, 3)
        assert np.any(X1 != X, axis=1).sum() == 2  # check if exactly two rows are mutated

        # check X's shape is changed when the `replace` is False
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        X2 = add_outliers(X, n_outliers=2, replace=False, inplace=False)
        assert X2.shape == (5, 3)

