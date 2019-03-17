# -*- coding: utf-8 -*-
"""
Implementation for the main algorithm of paper

`Estimating High Dimensional Robust Mixture Model via TrimmedExpectation-Maximization Algorithm`
"""
# Author:  Xiangyu Guo  xiangyug[at]buffalo.edu

import numpy as np


class TrimmedEM(object):

    def __init__(self, n_iters=100, eta=1e-3, sparsity=50,
                 alpha=0.05, grader=None, init_val=None):
        """
        Trimmed Expectation Maximization
        :param n_iters: int. number of maximum iterations
        :param eta: double. fixed step size
        :param sparsity: int
        :param alpha: double. fraction of data to be trimmed when computing the gradient
        :param grader: object which has a method gradient(X, Y, P) -> G, where X is the array of samples,
                Y is the array of labels, P is the array of parameters, and G is the array of gradients
                (for updating the parameters) computed at each sample, which is of the same shape as
                (n_samples, n_params).
        :param init_val: init value for the parameter vector
        """
        self.n_iters_ = n_iters
        self.eta_ = eta
        self.sparsity_ = sparsity

        assert alpha < 0.5
        self.alpha_ = alpha
        self.beta_ = init_val

        # TODO: specify the def for grader
        self.grader_ = grader

    @property
    def beta(self):
        return self.beta_

    def fit(self, X, Y, init_val=None):
        """
        :param X: array of shape=(n_samples, n_features)
        :param Y: array of shape=(n_samples, n_outputs)
        :param init_val: array, init value of the parameter to be estimated
        :return:
        """
        beta = init_val if init_val else self.beta_
        if beta is None:
            raise ValueError("Parameter vector is not initialized properly")
        self.beta_ = trimmed_em(X, Y,
                                init_val=beta,
                                n_iters=self.n_iters_,
                                step_size=self.eta_,
                                sparsity=self.sparsity_,
                                alpha=self.alpha_,
                                grader=self.grader_
                                )

        return self


def supp(x, sparsity):
    """return the index of the top-sparsity largest elements in x"""
    return np.argsort(x)[-sparsity:]


def trimmed_em(X, Y, init_val, n_iters, step_size,
               sparsity, alpha, grader):
    """
    Trimmed Expectation Maximization algorithm
    :param X: array of shape=(n_samples, n_features)
    :param Y: array of shape=(n_samples, n_outputs)
    :param init_val: array, init value of the parameter to be estimated
    :param n_iters: int. number of maximum iterations
    :param step_size: double. fixed step size
    :param sparsity: int, number of dimensions to be preserved when computing the gradient
    :param alpha: double. fraction of data to be trimmed when computing the gradient
    :param grader: function of form grad_func(X, Y, P) -> G, where X is the array of samples,
            Y is the array of labels, P is the array of parameters, and G is the array of gradients
            (for updating the parameters) computed at each sample, which is of the same shape as
            (n_samples, n_params).
    :return:
    """
    S = supp(init_val, sparsity)
    beta = trunc(init_val, S)

    for t in range(n_iters):
        Q = grader.gradient(X, Y, beta)
        Q = d_trim(Q, alpha)
        beta_i = beta - step_size * Q
        beta = trim(beta_i, sparsity)
        # S_i = supp(beta_i, sparsity)
        # beta = trunc(beta_i, S_i)

    return beta


def trunc(x, S):
    """Truncate all indices of x except for S
    :param x: array of shape=(n_features,)
    :param S: array of indices that needs to be retained
    :return:
    """
    trunc_x = np.zeros(x.shape)
    trunc_x[S] = x[S]
    return trunc_x


def trim(x, sparsity):
    """retain the top-sparsity components of x
    (equivalent to `trunc(x, supp(x, sparsity))`
    """
    to_be_trimmed = np.argsort(x)[:x.shape[0]-sparsity]
    x[to_be_trimmed] = 0
    return x


def d_trim(V, alpha):
    """Dimensional alpha-trimmed estimator [Liu et al., 2019]
    :param V: array of shape=(n_samples, n_features), set of gradients
    :param alpha: fraction to be trimmed
    :return g: estimated gradient
    """
    assert alpha < 0.5
    n_samples, n_features = V.shape
    n_remove = min(int(n_samples * alpha), n_samples // 2)
    sorted_idxs = np.argsort(V, axis=0)
    # remove the largest and smallest alpha-fraction of samples on each dimension
    retained_idxs =sorted_idxs[n_remove: n_samples - n_remove]

    # TODO: too slow; try to avoid for-loop
    g = np.array([np.average(V[retained_idxs[:, i], i]) for i in range(n_features)])

    return g