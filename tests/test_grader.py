import unittest
import numpy as np

from utils.grader import (
    gaussian_mixture_grad, regression_mixture_grad, regression_without_cov_grad
)


class TestUtils(unittest.TestCase):

    def test_gaussian_mixture_grad(self):
        # X = Z * 1 + V
        # beta = 1
        X = np.array([[-1], [-1.1], [-0.9], [-1.2], [-0.8], [-1.3], [-0.95],
                      [1], [1.1], [0.9], [1.2], [0.8], [1.3], [0.95], [1.12]])
        init_beta = np.array([2])
        g = gaussian_mixture_grad(X, beta=init_beta, beta0=init_beta, sigma=0.3)
        assert g.shape == X.shape
        assert np.all(g < 0)

    def test_regression_mixture_grad(self):
        # Y = Z(2*X)+V
        # beta = 2
        X = np.array([[0], [-0.4], [-0.2], [-0.8], [-0.3], [-0.05],
                      [0.1], [0.4], [0.09], [1.2], [0.3], [0.05]])
        Y = np.array([0.1, 0.75, 0.35, -1.54, -0.58, -0.099,
                      -0.2, -0.8, 0.19, 2.38, -0.62, 0.098])
        init_beta = np.array([1])
        g = regression_mixture_grad(X, Y, beta=init_beta, beta0=init_beta, sigma=0.3)
        assert g.shape == X.shape
        assert np.all(g > -0.2)

    def test_regression_without_cov_grad(self):
        # Y = 2 * X + V
        X = np.array([[0], [-0.4], [np.nan], [-0.8], [-0.3], [np.nan],
                      [0.1], [np.nan], [0.09], [1.2], [np.nan], [0.05]])
        Y = np.array([0.1, -0.75, -0.35, -1.54, -0.58, -0.099,
                      0.2, 0.8, 0.19, 2.38, 0.62, 0.098])
        Z = np.logical_not(np.isnan(X))
        init_beta = np.array([1])
        g = regression_without_cov_grad(X, Y, beta=init_beta,
                                        beta0=init_beta, sigma=0.1)
        assert g.shape == X.shape
        assert np.all(g[Z.ravel()] > -0.1)

