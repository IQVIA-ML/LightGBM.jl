LightGBM.jl
========

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

**LightGBM.jl** provides a Julia interface for Microsoft's [LightGBM](https://github.com/Microsoft/LightGBM).

## Installation
Install the latest version of LightGBM by following the installation steps on: (https://github.com/Microsoft/LightGBM/wiki/Installation-Guide).

Then add the package to Julia with:
```julia
Pkg.clone("https://github.com/Allardvm/LightGBM.jl.git")
```

To use the package, set the environment variable LIGHTGBM to point to the LightGBM binary. This can be done for the duration of a single Julia session with (include the .exe on Windows):
```julia
ENV["LIGHTGBM"] = "../lightgbm"
```

## Exports

### Functions

#### `fit(estimator, X, y[, test...])`
Fit the `estimator` with features data `X` and label `y` using the X-y pairs in `test` as
validation sets.

Return a dictionary with an entry for each validation set. Each entry of the dictionary is another
dictionary with an entry for each validation metric in the `estimator`. Each of these entries is an
array that holds the validation metric's value at each iteration.

##### Arguments
* `estimator::LightGBMEstimator`: the estimator to be fit.
* `X::Array{TX<:Real,2}`: the features data.
* `y::Array{Ty<:Real,1}`: the labels.
* `test::Tuple{Array{TX,2},Array{Ty,1}}...`: optionally contains one or more tuples of X-y pairs of
    the same types as `X` and `y` that should be used as validation sets.

#### `predict(estimator, X)`
Return an array with the labels that the `estimator` predicts for features data `X`.

##### Arguments
* `estimator::LightGBMEstimator`: the estimator to use in the prediction.
* `X::Array{T<:Real,2}`: the features data.

### Estimators 

#### `LightGBMRegression`

##### Constructor
```julia
LightGBMRegression(; [num_iterations = 10,
                      learning_rate = .1,
                      num_leaves = 127,
                      tree_learner = "serial",
                      num_threads = Sys.CPU_CORES,
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
Return a LightGBMRegression estimator.

#### LightGBMBinary
##### Constructor
```julia
LightGBMBinary(; [num_iterations = 10,
                  learning_rate = .1,
                  num_leaves = 127,
                  tree_learner = "serial",
                  num_threads = Sys.CPU_CORES,
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
Return a LightGBMBinary estimator.

#### LightGBMLambdaRank
##### Constructor
```julia
LightGBMLambdaRank(; [num_iterations = 10,
                      learning_rate = .1,
                      num_leaves = 127,
                      tree_learner = "serial",
                      num_threads = Sys.CPU_CORES,
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
Return a LightGBMBinary estimator.
