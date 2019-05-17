# -*- coding: utf-8 -*-
"""
Implementation for the main algorithm of paper

`Estimating High Dimensional Robust Mixture Model via TrimmedExpectation-Maximization Algorithm`
"""
# Author: e78c3441e9ae030d3d335b90aaf253f9 (intentionally hidden for the review process)

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.exceptions import NotFittedError


class TrimmedEM(object):

    def __init__(self, n_iters=100, eta=1e-3, sparsity=50,
                 alpha=0.05, grader=None, init_val=None,
                 groundtruth=None, record_all_loss=False):
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
        :param groundtruth: the true value of the paramter. used only for comparing cost per iteration.
        :param record_all_loss: whether to record the loss for each iteration. not effective unless
            the `true_val` is not None.
        """
        self.n_iters_ = n_iters
        self.eta_ = eta
        self.sparsity_ = sparsity

        assert alpha < 0.5
        self.alpha_ = alpha
        self.beta_ = init_val

        # TODO: specify the def for grader
        self.grader_ = grader

        if record_all_loss and groundtruth is None:
            raise ValueError("Can't compute costs when `groundtruth` is unknown.")
        self.record_loss_ = record_all_loss
        self.groundtruth_ = groundtruth
        self.losses_ = None

    @property
    def beta(self):
        return self.beta_

    @property
    def iteration_losses(self):
        return self.losses_

    def loss(self, groundtruth):
        if self.beta_ is None:
            raise NotFittedError("Model hasn't been fitted")
        return euclidean(self.beta_, groundtruth)

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
        estimated_beta = trimmed_em(X, Y,
                                    init_val=beta,
                                    n_iters=self.n_iters_,
                                    step_size=self.eta_,
                                    sparsity=self.sparsity_,
                                    alpha=self.alpha_,
                                    grader=self.grader_,
                                    groundtruth=self.groundtruth_,
                                    return_costs=self.record_loss_
                                    )
        if self.record_loss_:
            self.beta_, self.losses_ = estimated_beta
        else:
            self.beta_ = estimated_beta
        return self


def supp(x, sparsity):
    """return the index of the top-sparsity largest elements in x"""
    return np.argsort(x)[-sparsity:]


def trimmed_em(X, Y, init_val, n_iters, step_size,
               sparsity, alpha, grader, groundtruth=None, return_costs=False):
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
    :param groundtruth: true value of the parameter to be estimated
    :param return_costs: whether to return cost for each iteration. not effective
        unless `groundtruth` is not None.
    :return:
    """
    S = supp(init_val, sparsity)
    beta = trunc(init_val, S)
    if return_costs and groundtruth is None:
        raise ValueError("Can't compute costs when `groundtruth` is unknown.")
    costs = []

    for t in range(n_iters):
        Q = grader.gradient(X, Y, beta)
        Q = d_trim(Q, alpha)
        # Q = np.clip(Q, a_min=-100, a_max=100)
        beta_i = beta + step_size * Q
        beta = trim(beta_i, sparsity)
        if return_costs:
            costs.append(euclidean(groundtruth, beta))
        # S_i = supp(beta_i, sparsity)
        # beta = trunc(beta_i, S_i)

    if return_costs:
        return beta, np.array(costs)
    else:
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
    g = np.average(np.array([V[retained_idxs[:, i], i] for i in range(n_features)]), axis=1)

    return g