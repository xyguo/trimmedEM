# -*- coding: utf-8 -*-
"""Helper functions for creating synthesized data"""
# Author: e78c3441e9ae030d3d335b90aaf253f9 (intentionally hidden for the review process)

import numpy as np
import scipy as sp


def gaussian_mixture(n_samples, n_features,
                     mean,
                     sigma=1,
                     subspace=None,
                     bgn_sigma=0,
                     random_state=None):
    """
    generate data according to gaussian mixture model
    :param n_samples: number of samples
    :param n_features: dimension of data generated
    :param mean: array of shape=(n_features,)
    :param sigma: covariance of the Gaussian
    :param subspace: None or array of shape=(n_effective_idxs,).
        The 'effective dimensions' of data.
        When subspace is a subset of [n_features], this results in sparse data set.
        When subspace is None, then it's equivalent to subspace = np.arange(n_features).
    :param bgn_sigma: float,
        The standard variance of the background noise (N(0, bg_noise_sigma));
        This noise appears across ALL dimensions (regardless of the `subspace` parameter)
    :param random_state:
    :return Y: array of shape=(n_samples, n_features)
    """
    np.random.seed(random_state)
    if subspace is not None:
        assert np.all(subspace < n_features) and np.all(subspace >= 0)
    Y = np.zeros((n_samples, n_features))
    Z = np.random.choice([-1, +1], n_samples, replace=True, p=[0.5, 0.5])
    V = np.zeros((n_samples, n_features))

    if subspace is None:
        V[:] = np.random.multivariate_normal(np.zeros(n_features),
                                             np.identity(n_features) * sigma,
                                             n_samples)
        Y[:] = Z[:, np.newaxis].dot(mean[np.newaxis, :]) + V
    else:
        n_effective_idxs = subspace.shape[0]
        V[:, subspace] = np.random.multivariate_normal(np.zeros(n_effective_idxs),
                                                       np.identity(n_effective_idxs) * sigma,
                                                       n_samples)
        Y[:] = Z[:, np.newaxis].dot(mean[np.newaxis, :]) + V
        if bgn_sigma > 0:
            background_noise = np.random.randn(n_samples, n_features) * bgn_sigma
            Y[:] += background_noise
    return Y


def regression_mixture(n_samples, n_features,
                       mean,
                       sigma=1,
                       subspace=None,
                       bgn_sigma=0,
                       random_state=None):
    """
    generate data according to mixture of regression model
    Y = Z \cdot X^T \beta + V
    :param n_samples: number of samples
    :param n_features: dimension of data generated
    :param mean: array of shape=(n_features, ), the `beta` in the model
    :param sigma: array of shape=(n_features, n_features)
        covariance matrix for noise (V) on observation variables
    :param subspace: None or array of shape=(n_effective_idxs,).
        The 'effective dimensions' of data.
        When subspace is a subset of [n_features], this results in sparse data set.
        When subspace is None, then it's equivalent to subspace = np.arange(n_features).
    :param bgn_sigma: float,
        The standard variance of the background noise (N(0, bg_noise_sigma));
        This noise appears across ALL dimensions (regardless of the `subspace` parameter)
    :return (X, Y): X is array of shape=(n_samples, n_features), Y is array of shape=(n_samples,)
    """
    np.random.seed(random_state)
    if subspace is not None:
        assert np.all(subspace < n_features) and np.all(subspace >= 0)
    X = np.zeros((n_samples, n_features))
    Y = np.zeros((n_samples, ))
    Z = np.random.choice([-1, +1], n_samples, replace=True, p=[0.5, 0.5])
    V = np.random.normal(0, sigma, n_samples)

    if subspace is None:
        X[:] = np.random.multivariate_normal(np.zeros(n_features),
                                             np.identity(n_features),
                                             n_samples)
    else:
        n_effective_idxs = subspace.shape[0]
        X[:, subspace] = np.random.multivariate_normal(np.zeros(n_effective_idxs),
                                                       np.identity(n_effective_idxs),
                                                       n_samples)
        if bgn_sigma > 0:
            background_noise = np.random.randn(n_samples, n_features) * bgn_sigma
            X[:] += background_noise

    Y[:] = Z * (X.dot(mean)) + V

    return X, Y


def regression_with_missing_cov(n_samples, n_features,
                                mean,
                                missing_prob=0.2,
                                sigma=1,
                                subspace=None,
                                bgn_sigma=0,
                                random_state=None):
    """
    generate data according to the regression-with-missing-covariates model
    Y = X^T \beta + V
    :param n_samples: number of samples
    :param n_features: dimension of data generated
    :param mean: array of shape=(n_features, ), the `beta` in the model
    :param missing_prob: float between 0 and 1.
        each dim of X will be unobservable w.p. `missing_prob` i.i.d.
    :param sigma: array of shape=(n_features, n_features)
        covariance matrix for the noise (V) on observation variables
    :param subspace: None or array of shape=(n_effective_idxs,).
        The 'effective dimensions' of data.
        When subspace is a subset of [n_features], this results in sparse data set.
        When subspace is None, then it's equivalent to subspace = np.arange(n_features).
    :param bgn_sigma: float,
        The standard variance of the background noise (N(0, bg_noise_sigma));
        This noise appears across ALL dimensions (regardless of the `subspace` parameter)
    :return (X, Y): X is array of shape=(n_samples, n_features), Y is array of shape=(n_samples,)
    """
    np.random.seed(random_state)
    if subspace is not None:
        assert np.all(subspace < n_features) and np.all(subspace >= 0)
    X = np.zeros((n_samples, n_features))
    Y = np.zeros((n_samples, ))
    Z = np.random.choice([1, np.nan], size=(n_samples, n_features),
                         replace=True, p=[1-missing_prob, missing_prob])
    V = np.random.normal(0, sigma, n_samples)

    if subspace is None:
        X[:] = np.random.multivariate_normal(np.zeros(n_features),
                                             np.identity(n_features),
                                             n_samples)
    else:
        n_effective_idxs = subspace.shape[0]
        X[:, subspace] = np.random.multivariate_normal(np.zeros(n_effective_idxs),
                                                       np.identity(n_effective_idxs),
                                                       n_samples)
        if bgn_sigma > 0:
            background_noise = np.random.randn(n_samples, n_features) * bgn_sigma
            X[:] += background_noise

    Y[:] = X.dot(mean) + V

    return Z * X, Y


def add_outliers(X, n_outliers, dist_factor=10, replace=True,
                 inplace=False, random_state=None, return_index=False):
    """
    add outliers to a given data set: add large random shifts to some of the data in X
    :param X: array of shape=(n_samples, n_features)
    :param n_outliers:
    :param dist_factor: int,
        affects how far away the outliers distributed
    :param replace: bool, if true then perturb the original data set rather than adding
        new points as outliers.
    :param inplace: bool, default False.
        Whether to modify X in-place or adding outliers to a copy of X.
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

    # calculate the shift
    X_valid = np.abs(X.ravel())
    X_valid = X_valid[np.logical_not(np.isnan(X_valid))]
    shift = max(np.max(X_valid) * np.sqrt(n_features), 1.0)

    outlier_idxs = np.random.choice(n_samples, n_outliers, replace=False)
    outliers_shift = np.random.multivariate_normal(np.zeros(n_features),
                                                   np.identity(n_features) * dist_factor * shift,
                                                   n_outliers)
    if inplace:
        # avoid modifying the input directly: remember, X is passed-in by reference
        X_copy = X
    else:
        # X_copy refers to the same object as X
        X_copy = X.copy()

    if replace:
        X_copy[outlier_idxs] = X_copy[outlier_idxs] + outliers_shift
    else:
        outliers = X_copy[outlier_idxs] + outliers_shift
        X_copy = np.vstack((X, outliers))
    return (X_copy, outlier_idxs) if return_index else X_copy