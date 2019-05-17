import sys
import numpy as np
from trimmedEM import TrimmedEM
from utils.grader import MRMGrader
from utils.data_gen import regression_mixture, add_outliers

# Experiment for Mixture of Regression Model
epsilons = [0, 0.05, 0.1, 0.15, 0.2]
# epsilons = [0]
n_samples = 2000
mrm_sigma = 0.2
dim = 100
# sparsities = np.array([0.4, 0.2, 0.1, 0.05]) * dim
# sparsities = np.array([32, 24, 16, 12, 8, 4, 2])
sparsities = np.array([15, 13, 11, 9, 7, 5, 3])
sparsities = sparsities.astype(np.int)

mrm_g = MRMGrader(sigma=mrm_sigma)
results_MRM = {
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
    for i, eps in enumerate(results_MRM['eps-1']):
        n_outliers = np.int(results_MRM['n_samples-1'] * eps)
        results_MRM['true-beta-1'].append([])
        for s in results_MRM['sparsity-1']:
            # re-generate data with different sparsity
            effective_idxs = np.random.choice(dim, size=s, replace=False)
            true_beta = np.zeros(dim)
            true_beta[effective_idxs] = 5
            results_MRM['true-beta-1'][i].append(true_beta)

            # generate data
            X, Y = regression_mixture(n_samples=n_samples, n_features=dim,
                                      mean=true_beta, sigma=mrm_sigma,
                                      subspace=effective_idxs)
            # corrupt data arbitrarily
            XY = add_outliers(np.hstack((X, Y[:, np.newaxis])),
                              n_outliers=n_outliers,
                              dist_factor=50)
            X_corrupted, Y_corrupted = XY[:, :-1], XY[:, -1].ravel()

            # set initial point for gradient descent
            init_distortion = np.linalg.norm(true_beta) * \
                              np.random.randn(results_MRM['dim-1']) / \
                              (4 * np.sqrt(dim))
            beta0 = true_beta + init_distortion

            model = TrimmedEM(n_iters=n_iters,
                              eta=0.1, sparsity=s,
                              alpha=0.2, grader=mrm_g,
                              init_val=beta0)
            model.fit(X_corrupted, Y_corrupted)
            err_rates[i].append(model.loss(groundtruth=true_beta))
            # print("eps={}, sparsity={}, loss={}, loss / true_beta={}"
            #       .format(eps, s, err_rates[i][-1], err_rates[i][-1] / np.linalg.norm(true_beta)))
    results_MRM['err-1'].append(np.array(err_rates))
results_MRM['err-1'] = np.array(results_MRM['err-1'])
results_MRM['true-beta-1'] = results_MRM['true-beta-1']
true_beta_norms = np.zeros(shape=(len(epsilons), len(sparsities)))
for e in range(len(epsilons)):
    for s in range(len(sparsities)):
        true_beta_norms[e,s] = np.linalg.norm(results_MRM['true-beta-1'][e][s])
print("err-1:")
print(np.mean(results_MRM['err-1'], axis=0) / true_beta_norms)

## type 2: error v.s. n_iterations, for different epsilon
print("===============\nType 2: error v.s. n_iterations, for different eps\n")
true_sparsity = 7
dim = 100
epsilons = [0, 0.05, 0.1, 0.15, 0.2]

results_MRM['eps-2'] = np.array(epsilons)
results_MRM['err-2'] = []
results_MRM['sparsity-2'] = true_sparsity
results_MRM['n_samples-2'] = n_samples
results_MRM['dim-2'] = dim
results_MRM['true-beta-2'] = []

for r in range(n_repeats):
    print("===\nRepeat {}".format(r))
    effective_idxs = np.random.choice(dim, size=true_sparsity, replace=False)
    true_beta = np.zeros(dim)
    true_beta[effective_idxs] = 50
    results_MRM['true-beta-2'].append(true_beta)
    X, Y = regression_mixture(n_samples=results_MRM['n_samples-2'],
                              n_features=results_MRM['dim-2'],
                              mean=true_beta, sigma=mrm_sigma,
                              subspace=effective_idxs)
    err_rates = []
    for i, eps in enumerate(results_MRM['eps-2']):
        init_distortion = np.linalg.norm(results_MRM['true-beta-2'][-1]) * \
                          np.random.randn(results_MRM['dim-2']) / \
                          (4 * np.sqrt(results_MRM['dim-2']))
        beta0 = results_MRM['true-beta-2'][-1] + init_distortion
        n_outliers = np.int(results_MRM['n_samples-2'] * eps)

        # corrupt data arbitrarily
        XY = add_outliers(np.hstack((X, Y[:, np.newaxis])),
                          n_outliers=n_outliers,
                          dist_factor=50)
        X_corrupted, Y_corrupted = XY[:, :-1], XY[:, -1].ravel()

        model = TrimmedEM(n_iters=n_iters, eta=0.1,
                          sparsity=results_MRM['sparsity-2'],
                          alpha=0.2, grader=mrm_g,
                          init_val=beta0,
                          groundtruth=results_MRM['true-beta-2'][-1],
                          record_all_loss=True)
        model.fit(X_corrupted, Y_corrupted)
        err_rates.append(model.iteration_losses)
        # print("eps={}, final_loss={}".format(eps, err_rates[-1][-1]))
    results_MRM['err-2'].append(np.array(err_rates))
results_MRM['err-2'] = np.array(results_MRM['err-2'])
results_MRM['true-beta-2'] = np.array(results_MRM['true-beta-2'])
print("err-2:")
# print(np.min(np.mean(results_MRM['err-2'] /
#                      np.linalg.norm(results_MRM['true-beta-2'], axis=1)[:, np.newaxis, np.newaxis],
#                      axis=0),
#              axis=1))
print(np.mean(results_MRM['err-2'] /
              np.linalg.norm(results_MRM['true-beta-2'], axis=1)[:, np.newaxis, np.newaxis],
              axis=0)[:, -1])

## type III: error v.s. n_iterations, for different dim
print("=================\ntype III: error v.s. n_iterations, for different dim")
results_MRM['eps-3'] = 0.2
results_MRM['err-3'] = []
results_MRM['n_samples-3'] = n_samples
results_MRM['dim-3'] = np.array([i * 40 for i in range(2, 7)])
# results_MRM['sparsity-3'] = (results_MRM['dim-3'] * 0.05).astype(np.int)
results_MRM['sparsity-3'] = 7
results_MRM['true-beta-3'] = []

n_outliers = np.int(results_MRM['n_samples-3'] * results_MRM['eps-3'])
for r in range(n_repeats):
    print("===\nRepeat {}".format(r))
    err_rates = []
    for i, d in enumerate(results_MRM['dim-3']):
        effective_idxs = np.random.choice(d, size=results_MRM['sparsity-3'],
                                          replace=False)
        true_beta = np.zeros(d)
        true_beta[effective_idxs] = 10
        init_distortion = np.linalg.norm(true_beta) * \
                          np.random.randn(d) / \
                          (4 * np.sqrt(d))
        beta0 = true_beta + init_distortion

        if r == 0:
            results_MRM['true-beta-3'].append(true_beta)
        X, Y = regression_mixture(n_samples=results_MRM['n_samples-3'],
                                  n_features=d,
                                  mean=true_beta,
                                  sigma=mrm_sigma,
                                  subspace=effective_idxs)

        # corrupt data arbitrarily
        XY = add_outliers(np.hstack((X, Y[:, np.newaxis])),
                          n_outliers=n_outliers,
                          dist_factor=50)
        X_corrupted, Y_corrupted = XY[:, :-1], XY[:, -1].ravel()

        model = TrimmedEM(n_iters=n_iters, eta=0.1,
                          sparsity=results_MRM['sparsity-3'],
                          alpha=0.2, grader=mrm_g,
                          init_val=beta0,
                          groundtruth=true_beta, record_all_loss=True)
        model.fit(X_corrupted, Y_corrupted)
        err_rates.append(model.iteration_losses)
        # print("dim={}, final_loss={}".format(d, err_rates[-1][-1]))
    results_MRM['err-3'].append(np.array(err_rates))
results_MRM['err-3'] = np.array(results_MRM['err-3'])
true_beta_norm = np.array([np.linalg.norm(x) for x in results_MRM['true-beta-3']])
print("err-3:")
# print(np.min(np.mean(results_MRM['err-3'], axis=0), axis=1) / true_beta_norm)
print(np.mean(results_MRM['err-3'], axis=0)[:, -1] / true_beta_norm)

filename_MRM = "results_for_MRM_20190502"
np.savez(filename_MRM, **results_MRM)

sys.exit(0)
