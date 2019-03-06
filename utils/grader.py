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


def gaussian_mixture_grad(X, mu, mu0, sigma):
    """Compute the gradient on data set X w.r.t. the estimated mean mean
    for the Mixture of Gaussian model.
    :param X: array of (n_samples, n_features)
    :param mu: array of (n_features,), estimated mean
    :param mu0: array of (n_features,), previously estimated mean
    :param sigma: the fixed variance of the Gaussians
    :return G: array of (n_samples, n_features). Gradient at each sample.
    """
    w = expit(X.dot(mu)[:, np.newaxis] / (sigma ** 2))
    G = (2 * w - 1) * X - mu0
    return G


def regression_mixture_grad(X, Y, mu, mu0, sigma):
    """Compute the gradient on data set X w.r.t. the estimated parameter P
    for the Mixture of Regression model.
    :param X: array of (n_samples, n_features), data set
    :param Y: array of (n_samples,), objective of the regression
    :param mu: array of (n_features,), estimated mean
    :param mu0: array of (n_features,), previously estimated mean
    :param sigma: the fixed variance of the Gaussian noises
    :return G:
    """
    w = expit((X.dot(mu) * Y) / (sigma ** 2))
    G = (2 * w * Y)[:, np.newaxis] * X - X.dot(mu0)[:, np.newaxis] * X
    return G



