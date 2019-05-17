import sys
import numpy as np
from trimmedEM import TrimmedEM
from utils.grader import RMCGrader
from utils.data_gen import regression_with_missing_cov, add_outliers

# Experiment for the Regression with Missing Covariates (RMC) Model
epsilons = [0, 0.05, 0.1, 0.15, 0.2]
n_samples = 2000
rmc_sigma = 0.1
dim = 100
# sparsities = np.array([0.4, 0.2, 0.1, 0.05]) * dim
# sparsities = np.array([32, 24, 16, 12, 8, 4, 2])
sparsities = np.array([15, 13, 11, 9, 7, 5, 3])
sparsities = sparsities.astype(np.int)

rmc_g = RMCGrader(sigma=rmc_sigma)
results_RMC = {
    'eps-1': np.array(epsilons),
    'err-1': [],
    'sparsity-1': np.array(sparsities),
    'n_samples-1': n_samples,
    'dim-1': dim,
    'true-beta-1': [],
}
n_iters = 101
n_repeats = 20
## type 1: error v.s. n_samples / (sparsity * log(dim)), for different epsilon
print("==================\ntype 1: error v.s. n_samples / (sparsity * log(dim)), for different epsilon")
for r in range(n_repeats):
    print("===\nRepeat {}".format(r))
    err_rates = [[] for _ in range(len(epsilons))]
    for i, eps in enumerate(results_RMC['eps-1']):
        n_outliers = np.int(results_RMC['n_samples-1'] * eps)
        results_RMC['true-beta-1'].append([])
        for s in results_RMC['sparsity-1']:
            # re-generate data with different sparsity
            effective_idxs = np.random.choice(dim, size=s, replace=False)
            true_beta = np.zeros(dim)
            true_beta[effective_idxs] = 10
            X, Y = regression_with_missing_cov(n_samples=n_samples, n_features=dim,
                                               missing_prob=0.1,
                                               mean=true_beta, sigma=rmc_sigma,
                                               subspace=effective_idxs)
            results_RMC['true-beta-1'][i].append(true_beta)

            # corrupt the data
            XY = add_outliers(np.hstack((X, Y[:, np.newaxis])),
                              n_outliers=n_outliers,
                              dist_factor=50)
            X_corrupted, Y_corrupted = XY[:, :-1], XY[:, -1].ravel()

            # set initial point for gradient descent
            init_distortion = np.linalg.norm(true_beta) * \
                              np.random.randn(results_RMC['dim-1']) / \
                              (4 * np.sqrt(dim))
            beta0 = true_beta + init_distortion

            model = TrimmedEM(n_iters=n_iters,
                              eta=0.05, sparsity=s,
                              alpha=0.3, grader=rmc_g,
                              init_val=beta0)
            model.fit(X, Y_corrupted)
            err_rates[i].append(model.loss(groundtruth=true_beta))
            # print("eps={}, sparsity={}, loss={}, loss / true_beta={}"
            #       .format(eps, s, err_rates[i][-1], err_rates[i][-1] / np.linalg.norm(true_beta)))
    results_RMC['err-1'].append(np.array(err_rates))
results_RMC['err-1'] = np.array(results_RMC['err-1'])
results_RMC['true-beta-1'] = results_RMC['true-beta-1']
true_beta_norms = np.zeros(shape=(len(epsilons), len(sparsities)))
for e in range(len(epsilons)):
    for s in range(len(sparsities)):
        true_beta_norms[e,s] = np.linalg.norm(results_RMC['true-beta-1'][e][s])
print("err-1:")
print(np.mean(results_RMC['err-1'], axis=0) / true_beta_norms)

## type 2: error v.s. n_iterations, for different epsilon
print("===============\nType 2: error v.s. n_iterations, for different eps\n")
dim = 100
true_sparsity = 4
epsilons = [0, 0.05, 0.1, 0.15, 0.2]

results_RMC['eps-2'] = np.array(epsilons)
results_RMC['err-2'] = []
results_RMC['sparsity-2'] = true_sparsity
results_RMC['n_samples-2'] = n_samples
results_RMC['dim-2'] = dim
results_RMC['true-beta-2'] = []

for r in range(n_repeats):
    print("===\nRepeat {}".format(r))
    effective_idxs = np.random.choice(dim, size=true_sparsity, replace=False)
    true_beta = np.zeros(dim)
    true_beta[effective_idxs] = 10
    results_RMC['true-beta-2'].append(true_beta)
    X, Y = regression_with_missing_cov(n_samples=n_samples,
                                       n_features=dim,
                                       mean=true_beta,
                                       missing_prob=0.1,
                                       sigma=rmc_sigma,
                                       subspace=effective_idxs)
    err_rates = []
    for i, eps in enumerate(results_RMC['eps-2']):
        init_distortion = np.linalg.norm(results_RMC['true-beta-2']) * \
                          np.random.randn(results_RMC['dim-2']) / \
                          (10 * np.sqrt(results_RMC['dim-2']))
        beta0 = results_RMC['true-beta-2'][-1] + init_distortion
        n_outliers = np.int(results_RMC['n_samples-2'] * eps)

        # corrupt data arbitrarily
        XY = add_outliers(np.hstack((X, Y[:, np.newaxis])),
                          n_outliers=n_outliers,
                          dist_factor=50)
        X_corrupted, Y_corrupted = XY[:, :-1], XY[:, -1].ravel()

        model = TrimmedEM(n_iters=n_iters, eta=0.1,
                          sparsity=results_RMC['sparsity-2'],
                          alpha=0.3, grader=rmc_g,
                          init_val=beta0,
                          groundtruth=results_RMC['true-beta-2'][-1],
                          record_all_loss=True)
        model.fit(X_corrupted, Y_corrupted)
        err_rates.append(model.iteration_losses)
        # print("eps={}, final_loss={}".format(eps, err_rates[-1][-1]))
    results_RMC['err-2'].append(np.array(err_rates))
results_RMC['err-2'] = np.array(results_RMC['err-2'])
results_RMC['true-beta-2'] = np.array(results_RMC['true-beta-2'])
print("err-2:")
# print(np.min(np.mean(results_RMC['err-2'] /
#                      np.linalg.norm(results_RMC['true-beta-2'], axis=1)[:, np.newaxis, np.newaxis],
#                      axis=0),
#              axis=1))
print(np.mean(results_RMC['err-2'] /
              np.linalg.norm(results_RMC['true-beta-2'], axis=1)[:, np.newaxis, np.newaxis],
              axis=0)[:, -1])

## type III: error v.s. n_iterations, for different dim
print("=================\ntype III: error v.s. n_iterations, for different dim")
results_RMC['eps-3'] = 0.2
results_RMC['err-3'] = []
results_RMC['n_samples-3'] = n_samples
results_RMC['dim-3'] = np.array([i * 40 for i in range(2, 7)])
# results_RMC['dim-3'] = np.array([60])
# results_RMC['sparsity-3'] = (results_RMC['dim-3'] * 0.05).astype(np.int)
results_RMC['sparsity-3'] = 4
results_RMC['true-beta-3'] = []

n_outliers = np.int(results_RMC['n_samples-3'] * results_RMC['eps-3'])
for r in range(n_repeats):
    print("===\nRepeat {}".format(r))
    err_rates = []
    for i, d in enumerate(results_RMC['dim-3']):
        effective_idxs = np.random.choice(d, size=results_RMC['sparsity-3'], replace=False)
        true_beta = np.zeros(d)
        true_beta[effective_idxs] = 10
        init_distortion = np.linalg.norm(true_beta) * \
                          np.random.randn(d) / \
                          (4 * np.sqrt(d))
        beta0 = true_beta + init_distortion

        if r == 0:
            results_RMC['true-beta-3'].append(true_beta)
        X, Y = regression_with_missing_cov(n_samples=results_RMC['n_samples-3'],
                                           missing_prob=0.1,
                                           n_features=d,
                                           mean=true_beta,
                                           sigma=rmc_sigma,
                                           subspace=effective_idxs)
        # corrupt data arbitrarily
        XY = add_outliers(np.hstack((X, Y[:, np.newaxis])),
                          n_outliers=n_outliers,
                          dist_factor=50)
        X_corrupted, Y_corrupted = XY[:, :-1], XY[:, -1].ravel()

        model = TrimmedEM(n_iters=n_iters, eta=0.08,
                          sparsity=results_RMC['sparsity-3'],
                          alpha=0.3, grader=rmc_g,
                          init_val=beta0,
                          groundtruth=true_beta, record_all_loss=True)
        model.fit(X, Y_corrupted)
        err_rates.append(model.iteration_losses)
        # print("dim={}, final_loss={}".format(d, err_rates[-1][-1]))
    results_RMC['err-3'].append(np.array(err_rates))
results_RMC['err-3'] = np.array(results_RMC['err-3'])
true_beta_norm = np.array([np.linalg.norm(x) for x in results_RMC['true-beta-3']])
print("err-3:")
# print(np.min(np.mean(results_RMC['err-3'], axis=0), axis=1) / true_beta_norm)
print(np.mean(results_RMC['err-3'], axis=0)[:, -1] / true_beta_norm)

filename_RMC = "results_for_RMC_20190513"
np.savez(filename_RMC, **results_RMC)

sys.exit(0)