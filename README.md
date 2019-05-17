# Code implementation for paper "Estimating High Dimensional Robust Mixture Model via Trimmed Expectation-Maximization Algorithm"

# Remark

The parameters that affects the algorithm's performance the most are:

1. _epsilon_: The fraction of data being corrupted. Usually unknown in practice.
2. _alpha_: The parameter of the alpha-DTrim subroutine. This parameter determines
how much fraction of data should be ignored.
3. _sparsity_: The algorithm assumes the data resides in a much lower-dimension 
subspace of the input space, and use _sparsity_ to denote the dimension of this 
subspace. The true sparsity of data is usually unknown in practice. But in our 
experiment, we can assume it's known.

Assume the model is not mis-specified, when applying our algorithm, we should ensure 
the following to get a good performance:

1. _alpha >= epsilon_: _alpha_ should be an upperbound of the true corrupted fraction, 
otherwise the performance is still going to be heavily affected by outliers.
2. _sparsity >= true sparsity_: Otherwise, the algorithm cannot fit the data in all
its effective dimensions, which may result in high error on these dimensions.

Besides, the initial value of parameter is also important: According to our theoretical
result, the algorithm converges to the optimal only if the starting point is close enough
to the optimal solution.