# -*- coding: utf-8 -*-
"""Helper functions for creating synthesized data"""
# Author: Xiangyu Guo     xiangyug[at]buffalo.edu

import numpy as np
import scipy as sp


def gaussian_mixture(n_samples, n_features,
                     mean,
                     sigma=None,
                     n_outliers=0,
                     outliers_dist_factor=50,
                     random_state=None):
    """
    generate data according to gaussian mixture model
    :param n_samples: number of samples
    :param n_features: dimension of data generated
    :param n_outliers: number of outliers
    :param mean: array of shape=(n_features,)
    :param sigma: covariance of the Gaussian
    :param outliers_dist_factor:
    :return Y: array of shape=(n_samples, n_features)
    """
    np.random.seed(random_state)
    Y = np.zeros((n_samples, n_features))
    Z = np.random.choice([-1, +1], n_samples, replace=True, p=[1, 1])
    V = np.random.multivariate_normal(np.zeros(n_features),
                                      np.identity(n_features) * sigma,
                                      n_samples)

    Y[:] = Z[:, np.newaxis].dot(mean[np.newaxis, :]) + V
    #TODO: add desired number of outliers
    return Y


def regression_mixture(n_samples, n_features,
                       mean,
                       sigma_noise=None,
                       n_outliers=0,
                       outliers_dist_factor=50,
                       random_state=None):
    """
    generate data according to mixture of regression model
    :param n_samples: number of samples
    :param n_clusters: number of Gaussians
    :param n_outliers: number of outliers
    :param n_features: dimension of data generated
    :param mean: array of shape=(n_features, )
    :param sigma_noise: array of shape=(n_features, n_features)
        covariance matrix for background noise
    :param outliers_dist_factor:
    :return X: array of shape=(n_samples, n_features)
    """
    np.random.seed(random_state)
    X = np.random.multivariate_normal(np.zeros(n_features),
                                      np.identity(n_features),
                                      n_samples)
    Y = np.zeros((n_samples, ))
    Z = np.random.choice([-1, +1], n_samples, replace=True, p=[1, 1])
    V = np.random.normal(0, sigma_noise, n_samples)

    Y[:] = Z * (X.dot(mean)) + V

    #TODO: add desired number of outliers
    return X, Y


def add_outliers(X, n_outliers, dist_factor=10, random_state=None, return_index=False):
    """
    add outliers to a given data set: add large random shifts to some of the data in X
    :param X: array of shape=(n_samples, n_features)
    :param n_outliers:
    :param dist_factor: int,
        affects how far away the outliers distributed
    :param random_state:
    :param return_index: bool, whether to return
    :return X or (X, outliers):
        X: array of shape=(n_samples, n_features);
        outliers: array of shape=(n_outliers,), indices of outlier points in the returned data set.
    """
    if n_outliers == 0:
        return X
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    assert n_outliers < n_samples
    shift = max(np.mean(np.linalg.norm(X, axis=1)), 1.0)
    outlier_idxs = np.random.choice(n_samples, n_outliers, replace=False)
    outliers_shift = np.random.multivariate_normal(np.zeros(n_features),
                                                   np.identity(n_features) * dist_factor * shift,
                                                   n_outliers)
    outliers = X[outlier_idxs] + outliers_shift
    X = np.vstack((X, outliers))
    return (X, outlier_idxs) if return_index else X