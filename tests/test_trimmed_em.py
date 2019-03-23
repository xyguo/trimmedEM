import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from trimmedEM.trimmed_em import (
    supp, trimmed_em, trunc, d_trim, trim
)


class LRGrader(object):
    """Mock class for testing the trimmed_em function.
    implements the gradient for square loss of linear regression
    """
    def gradient(self, X, Y, P):
        Y_hat = X.dot(P)
        g = X * (Y_hat-Y)[:, np.newaxis]
        return -g


class TestUtils(unittest.TestCase):

    def test_supp(self):
        x = np.array([5, 2, 3, 1, 4, 0, 7, 9, 8])
        s_x = supp(x, 5) # s_x is the indices of elements 4, 5, 7, 8, 9
        assert_array_equal(s_x, np.array([4, 0, 6, 8, 7]))

    def test_trunc(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        S = np.array([0, 2, 5, 1])
        t_x = trunc(x, S)
        assert_array_equal(t_x, np.array([1, 2, 3, 0, 0, 6, 0, 0]))

    def test_trim(self):
        x = np.array([5, 2, 3, 1, 4, 0, 7, 9, 8])
        t_x = trim(x, 5)
        assert_array_equal(t_x, np.array([5, 0, 0, 0, 4, 0, 7, 9, 8]))

    def test_d_trim(self):
        X = np.array([[10,  2,  4, 16,  1],
                      [14, 19,  4,  8, 12],
                      [1,  3,  9, 13,  3],
                      [0, 19,  2,  9,  8],
                      [8,  5, 17, 11,  8],
                      [11, 13,  9,  8, 10]])
        d_X = d_trim(X, alpha=0.2)
        assert_array_almost_equal(d_X,
                                  np.array([(10 + 1 + 8 + 11) / 4,
                                            (3 + 19 + 5 + 13) / 4,
                                            (4 + 4 + 9 + 9) / 4,
                                            (13 + 9 + 11 + 8) / 4,
                                            (3 + 8 + 8 + 10) / 4]),
                                  decimal=5)

    def test_trimmed_em(self):
        X = np.array([[1], [2], [3], [4], [5], [6], [2], [3], [1.5], [2.6]])
        Y = np.array([2, 4, 6, 8.1, 10, 11.9, 3.9, 6, 3.1, 5.4])
        init_g = np.array([1.0])
        eta = 0.05
        alpha = 0.1
        sparsity = 1
        grader = LRGrader()

        g = trimmed_em(X=X, Y=Y, init_val=init_g,
                       n_iters=30, step_size=eta, sparsity=sparsity,
                       alpha=alpha, grader=grader)

        assert np.abs(g[0] - 2) < 0.05
