LightGBM.jl
========

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

**LightGBM.jl** provides a Julia interface for Microsoft's [LightGBM](https://github.com/Microsoft/LightGBM). The package uses LightGBM's new C API to realize the best performance and allow fitting of large databases. The package supports all major operating systems (Windows, Linux, and Mac OS X).

## Installation
Install the latest version of LightGBM by following the installation steps on: (https://github.com/Microsoft/LightGBM/wiki/Installation-Guide).

Then add the package to Julia with:
```julia
Pkg.clone("https://github.com/Allardvm/LightGBM.jl.git")
```

To use the package, set the environment variable LIGHTGBM_PATH to point to the LightGBM directory prior to loading LightGBM.jl. This can be done for the duration of a single Julia session with:
```julia
ENV["LIGHTGBM_PATH"] = "../LightGBM"
```

To test the package, first set the environment variable LIGHTGBM_PATH and then call:
```julia
Pkg.test("LightGBM")
```

## Exports

### Functions

#### `fit(estimator, X, y[, test...]; [verbosity = 1])`
Fit the `estimator` with features data `X` and label `y` using the X-y pairs in `test` as
validation sets.

Return a dictionary with an entry for each validation set. Each entry of the dictionary is another
dictionary with an entry for each validation metric in the `estimator`. Each of these entries is an
array that holds the validation metric's value at each evaluation of the metric.

##### Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `test::Tuple{Matrix{TX},Vector{Ty}}...`: optionally contains one or more tuples of X-y pairs of
    the same types as `X` and `y` that should be used as validation sets.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.

#### `predict(estimator, X; [predict_type = 0, n_trees = -1, verbosity = 1])`
Return an array with the labels that the `estimator` predicts for features data `X`.

##### Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `X::Matrix{T<:Real}`: the features data.
* `predict_type::Integer`: keyword argument that controls the prediction type. `0` for normal
    scores with transform (if needed), `1` for raw scores, `2` for leaf indices.
* `n_trees::Integer`: keyword argument that sets the controls the number of trees used in the
    prediction. `< 0` for all available trees.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.

### Estimators

#### `LGBMRegression`
```julia
LGBMRegression(; [num_iterations = 10,
                  learning_rate = .1,
                  num_leaves = 127,
                  max_depth = -1,
                  tree_learner = "serial",
                  num_threads = Sys.CPU_CORES,
                  histogram_pool_size = -1.,
                  min_data_in_leaf = 100,
                  min_sum_hessian_in_leaf = 10.,
                  feature_fraction = 1.,
                  feature_fraction_seed = 2,
                  bagging_fraction = 1.,
                  bagging_freq = 0,
                  bagging_seed = 3,
                  early_stopping_round = 0,
                  max_bin = 255,
                  data_random_seed = 1,
                  is_sigmoid = true,
                  init_score = "",
                  is_pre_partition = false,
                  is_sparse = true,
                  two_round = false,
                  save_binary = false,
                  sigmoid = 1.,
                  is_unbalance = false,
                  max_position = 20,
                  label_gain = Float64[],
                  metric = ["l2"],
                  metric_freq = 1,
                  is_training_metric = false,
                  ndcg_at = Int[],
                  num_machines = 1,
                  local_listen_port = 12400,
                  time_out = 120,
                  machine_list_file = ""])
```
Return a LGBMRegression estimator.

#### `LGBMBinary`
```julia
LGBMBinary(; [num_iterations = 10,
              learning_rate = .1,
              num_leaves = 127,
              max_depth = -1,
              tree_learner = "serial",
              num_threads = Sys.CPU_CORES,
              histogram_pool_size = -1.,
              min_data_in_leaf = 100,
              min_sum_hessian_in_leaf = 10.,
              feature_fraction = 1.,
              feature_fraction_seed = 2,
              bagging_fraction = 1.,
              bagging_freq = 0,
              bagging_seed = 3,
              early_stopping_round = 0,
              max_bin = 255,
              data_random_seed = 1,
              is_sigmoid = true,
              init_score = "",
              is_pre_partition = false,
              is_sparse = true,
              two_round = false,
              save_binary = false,
              sigmoid = 1.,
              is_unbalance = false,
              max_position = 20,
              label_gain = Float64[],
              metric = ["binary_logloss"],
              metric_freq = 1,
              is_training_metric = false,
              ndcg_at = Int[],
              num_machines = 1,
              local_listen_port = 12400,
              time_out = 120,
              machine_list_file = ""])
```
Return a LGBMBinary estimator.

#### `LGBMLambdaRank`
```julia
LGBMLambdaRank(; [num_iterations = 10,
                  learning_rate = .1,
                  num_leaves = 127,
                  max_depth = -1,
                  tree_learner = "serial",
                  num_threads = Sys.CPU_CORES,
                  histogram_pool_size = -1.,
                  min_data_in_leaf = 100,
                  min_sum_hessian_in_leaf = 10.,
                  feature_fraction = 1.,
                  feature_fraction_seed = 2,
                  bagging_fraction = 1.,
                  bagging_freq = 0,
                  bagging_seed = 3,
                  early_stopping_round = 0,
                  max_bin = 255,
                  data_random_seed = 1,
                  is_sigmoid = true,
                  init_score = "",
                  is_pre_partition = false,
                  is_sparse = true,
                  two_round = false,
                  save_binary = false,
                  sigmoid = 1.,
                  is_unbalance = false,
                  max_position = 20,
                  label_gain = Float64[],
                  metric = ["ndcg"],
                  metric_freq = 1,
                  is_training_metric = false,
                  ndcg_at = Int[],
                  num_machines = 1,
                  local_listen_port = 12400,
                  time_out = 120,
                  machine_list_file = ""])
```
Return a LGBMLambdaRank estimator.

#### `LGBMMulticlass`
```julia
LGBMMulticlass(; [num_iterations = 10,
                  learning_rate = .1,
                  num_leaves = 127,
                  max_depth = -1,
                  tree_learner = "serial",
                  num_threads = Sys.CPU_CORES,
                  histogram_pool_size = -1.,
                  min_data_in_leaf = 100,
                  min_sum_hessian_in_leaf = 10.,
                  feature_fraction = 1.,
                  feature_fraction_seed = 2,
                  bagging_fraction = 1.,
                  bagging_freq = 0,
                  bagging_seed = 3,
                  early_stopping_round = 0,
                  max_bin = 255,
                  data_random_seed = 1,
                  is_sigmoid = true,
                  init_score = "",
                  is_pre_partition = false,
                  is_sparse = true,
                  two_round = false,
                  save_binary = false,
                  sigmoid = 1.,
                  is_unbalance = false,
                  max_position = 20,
                  label_gain = Float64[],
                  metric = ["multi_logloss"],
                  metric_freq = 1,
                  is_training_metric = false,
                  ndcg_at = Int[],
                  num_machines = 1,
                  local_listen_port = 12400,
                  time_out = 120,
                  machine_list_file = "",
                  num_class = 1])
```
Return a LGBMMulticlass estimator.