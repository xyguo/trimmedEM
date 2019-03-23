import sys
import numpy as np
from trimmedEM import TrimmedEM
from utils.grader import GMMGrader
from utils.data_gen import gaussian_mixture, add_outliers

# Experiment for Gaussian Mixture Model
epsilons = [0.05, 0.1, 0.15, 0.2]
sparsities = [100, 75, 50, 25]
n_samples = 2000
gmm_sigma = 1
dim = 200
true_sparsity = np.int(dim * 0.3)  # number of effective dimensions
effective_idxs = np.random.choice(dim, size=true_sparsity, replace=False)
true_beta = np.zeros(dim)
true_beta[effective_idxs] = 5
X = gaussian_mixture(n_samples=n_samples, n_features=dim,
                     mean=true_beta, sigma=gmm_sigma,
                     subspace=effective_idxs)
gmm_g = GMMGrader(sigma=gmm_sigma)
results_GMM = {
    'eps-1': np.array(epsilons),
    'err-1': [],
    'sparsity-1': np.array(sparsities),
    'n_samples-1': n_samples,
    'dim-1': dim,
    'true-beta-1': true_beta,
    'true-sparsity-1': true_sparsity,
}
n_repeats = 2

## type 1: error v.s. n_samples / (sparsity * log(dim)), for different epsilon
print("==================\ntype 1: error v.s. n_samples / (sparsity * log(dim)), for different epsilon")
for r in range(n_repeats):
    print("===\nRepeat {}".format(r))
    err_rates = [[] for _ in range(len(epsilons))]
    for i, eps in enumerate(results_GMM['eps-1']):
        n_outliers = np.int(results_GMM['n_samples-1'] * eps)
        X_corrupted = add_outliers(X, n_outliers=n_outliers, dist_factor=50)
        for s in results_GMM['sparsity-1']:
            beta0 = np.random.randn(results_GMM['dim-1'])
            model = TrimmedEM(n_iters=50, eta=0.1, sparsity=s,
                              alpha=0.2, grader=gmm_g,
                              init_val=beta0)
            model.fit(X_corrupted, Y=None)
            err_rates[i].append(model.loss(groundtruth=true_beta))
            print("eps={}, sparsity={}, loss={}".format(eps, s, err_rates[i][-1]))
    results_GMM['err-1'].append(np.array(err_rates))
results_GMM['err-1'] = np.array(results_GMM['err-1'])
print("err-1:")
print(np.mean(results_GMM['err-1'], axis=0) / np.linalg.norm(true_beta))

## type 2: error v.s. n_iterations, for different epsilon
print("===============\nType 2: error v.s. n_iterations, for different eps\n")
n_iters = 50
# results_MRM['eps-2'] = np.array(epsilons)
results_GMM['eps-2'] = np.array([0.2])
results_GMM['err-2'] = []
results_GMM['sparsity-2'] = 75
results_GMM['n_samples-2'] = n_samples
results_GMM['dim-2'] = dim
results_GMM['true-beta-2'] = true_beta
results_GMM['true-sparsity-2'] = true_sparsity

for r in range(n_repeats):
    print("===\nRepeat {}".format(r))
    err_rates = []
    for i, eps in enumerate(results_GMM['eps-2']):
        beta0 = np.random.randn(results_GMM['dim-2'])
        n_outliers = np.int(results_GMM['n_samples-2'] * eps)
        X_corrupted = add_outliers(X, n_outliers=n_outliers, dist_factor=50)
        model = TrimmedEM(n_iters=50, eta=0.1, sparsity=results_GMM['sparsity-2'],
                          alpha=0.3, grader=gmm_g,
                          init_val=beta0,
                          groundtruth=results_GMM['true-beta-2'], record_all_loss=True)
        model.fit(X_corrupted, Y=None)
        err_rates.append(model.iteration_losses)
        print("eps={}, final_loss={}".format(eps, err_rates[-1][-1]))
    results_GMM['err-2'].append(np.array(err_rates))
results_GMM['err-2'] = np.array(results_GMM['err-2'])
print("err-2:")
print(np.min(np.mean(results_GMM['err-2'], axis=0), axis=1) / np.linalg.norm(results_GMM['true-beta-2']))

## type III: error v.s. n_iterations, for different dim
print("=================\ntype III: error v.s. n_iterations, for different dim")
n_iters = 50
results_GMM['eps-3'] = 0.2
results_GMM['err-3'] = []
results_GMM['n_samples-3'] = n_samples
results_GMM['dim-3'] = np.array([i * 100 for i in range(2, 11)])
# results_MRM['sparsity-3'] = (results_MRM['dim-3'] * 0.3).astype(np.int)
results_GMM['sparsity-3'] = np.ones(9).astype(np.int) * 100
# results_MRM['true-sparsity-3'] = (results_MRM['dim-3'] * 0.3).astype(np.int)
results_GMM['true-sparsity-3'] = np.ones(9).astype(np.int) * 100
results_GMM['true-beta-3'] = []

n_outliers = np.int(results_GMM['n_samples-3'] * results_GMM['eps-3'])
for r in range(n_repeats):
    print("===\nRepeat {}".format(r))
    err_rates = []
    for i, d in enumerate(results_GMM['dim-3']):
        effective_idxs = np.random.choice(d, size=results_GMM['true-sparsity-3'][i], replace=False)
        true_beta = np.zeros(d)
        true_beta[effective_idxs] = 5
        if r == 0:
            results_GMM['true-beta-3'].append(true_beta)
        X = gaussian_mixture(n_samples=results_GMM['n_samples-3'],
                             n_features=d,
                             mean=true_beta,
                             sigma=gmm_sigma,
                             subspace=effective_idxs)
        X_corrupted = add_outliers(X, n_outliers=n_outliers, dist_factor=50)
        model = TrimmedEM(n_iters=50, eta=0.1,
                          sparsity=results_GMM['sparsity-3'][i],
                          alpha=0.2, grader=gmm_g,
                          init_val=np.random.randn(d),
                          groundtruth=true_beta, record_all_loss=True)
        model.fit(X_corrupted, Y=None)
        err_rates.append(model.iteration_losses)
        print("dim={}, final_loss={}".format(d, err_rates[-1][-1]))
    results_GMM['err-3'].append(np.array(err_rates))
results_GMM['err-3'] = np.array(results_GMM['err-3'])
true_beta_norm = np.array([np.linalg.norm(x) for x in results_GMM['true-beta-3']])
print("err-3:")
print(np.min(np.mean(results_GMM['err-3'], axis=0), axis=1) / true_beta_norm)

filename_GMM = "results_for_GMM_20190318"
np.savez(filename_GMM, **results_GMM)

