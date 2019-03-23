# -*- coding: utf-8 -*-
"""Helper classes for calculating gradients for some simple modes"""
# Author: Xiangyu Guo     xiangyug[at]buffalo.edu

import numpy as np
from scipy.special import expit


class Grader(object):
    """Abstract class for gradient computer"""
    def gradient(self, X, Y, P):
        pass


class GMMGrader(Grader):
    """Gradient computer for Gaussian Mixture Model"""
    def __init__(self, sigma=1):
        self.sigma_ = sigma

    def gradient(self, X, Y, P):
        return gaussian_mixture_grad(X, P, P, self.sigma_)


class MRMGrader(Grader):
    """Gradient computer for Mixture of Regression Model"""
    def __init__(self, sigma=1):
        self.sigma_ = sigma

    def gradient(self, X, Y, P):
        return regression_mixture_grad(X, Y, P, P, self.sigma_)


class RMCGrader(Grader):
    """Gradient computer for Regression with Missing Covariance model"""
    def __init__(self, sigma=1):
        self.sigma_ = sigma

    def gradient(self, X, Y, P):
        return regression_without_cov_grad(X, Y, P, P, self.sigma_)


def gaussian_mixture_grad(X, beta, beta0, sigma):
    """Compute the gradient on data set X w.r.t. the estimated parameter
    for the Mixture of Gaussian model.
    :param X: array of (n_samples, n_features)
    :param beta: array of (n_features,), estimated parameter
    :param beta0: array of (n_features,), previously estimated parameter
    :param sigma: the fixed variance of the Gaussians
    :return G: array of (n_samples, n_features). Gradient at each sample.
    """
    w = expit(X.dot(beta)[:, np.newaxis] / (sigma ** 2))
    G = (2 * w - 1) * X - beta0
    return G


def regression_mixture_grad(X, Y, beta, beta0, sigma):
    """Compute the gradient on data set X w.r.t. the estimated parameter P
    for the Mixture of Regression model.
    :param X: array of (n_samples, n_features), data set
    :param Y: array of (n_samples,), objective of the regression
    :param beta: array of (n_features,), estimated parameter
    :param beta0: array of (n_features,), previously estimated parameter
    :param sigma: the fixed variance of the Gaussian noises
    :return G:
    """
    w = expit((X.dot(beta) * Y) / (sigma ** 2))
    G = (2 * w * Y)[:, np.newaxis] * X - X.dot(beta0)[:, np.newaxis] * X
    return G


def regression_without_cov_grad(X, Y, beta, beta0, sigma):
    Z = np.ones(shape=X.shape)
    Z = np.nan_to_num(Z)

    raise NotImplementedError("function not implemented")

