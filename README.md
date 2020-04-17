This package was originally authored by [Allardvm](https://github.com/Allardvm) and [wakakusa](https://github.com/wakakusa/)

LightGBM.jl
![CI](https://github.com/IQVIA-ML/LightGBM.jl/workflows/CI/badge.svg)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
========

**LightGBM.jl** provides a high-performance Julia interface for Microsoft's
[LightGBM](https://lightgbm.readthedocs.io/en/latest/).

The package adds a couple of convenience features:
* Automated cross-validation 
* Exhaustive grid search search procedure
* Integration with [MLJ](https://github.com/alan-turing-institute/MLJ.jl) (which also provides the above via different interfaces)

Additionally, the package automatically converts all LightGBM parameters that refer to indices 
(e.g. `categorical_feature`) from Julia's one-based indices to C's zero-based indices.

A majority of the C-interfaces are implemented. A few are known to be missing and are
[tracked.](https://github.com/IQVIA-ML/LightGBM.jl/issues)

All major operating systems (Windows, Linux, and Mac OS X) are supported. Julia versions 1.0+ are supported.

# Table of Contents
1. [Installation](#installation)
1. [Example](#a-simple-example-using-lightgbm-example-files)
1. [Exports](#exports)
1. [MLJ](#mlj-support)

# Installation
Please ensure your system meets the pre-requisites for LightGBM. This generally means ensuring
that `libomp` is installed and linkable on your system. See here for [Microsoft's installation guide.](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

Please note that the package actually downloads a [precompiled binary](https://github.com/microsoft/LightGBM/releases)
so you do not need to install LightGBM first. This is done as a user convenience, and support
will be added for supplying ones own LightGBM binary (for GPU acceleration, etc).

To add the package to Julia:
```julia
Pkg.add("LightGBM")
```

Running tests for the package requires the use of the LightGBM example files,
download and extract the [LightGBM source](https://github.com/microsoft/LightGBM/archive/v2.3.1.zip)
and set the enviroment variable `LIGHTGBM_EXAMPLES_PATH` to the root of the source installation.
Then you can run the tests by simply doing
```julia
Pkg.test("LightGBM")
```

# A simple example using LightGBM example files

First, download [LightGBM source](https://github.com/microsoft/LightGBM/archive/v2.3.1.zip) 
and untar it somewhere.

```bash
cd ~
wget https://github.com/microsoft/LightGBM/archive/v2.3.1.tar.gz
tar -xf v2.3.1.tar.gz
```

```julia
using LightGBM
using DelimitedFiles

LIGHTGBM_SOURCE = abspath("~/LightGBM-2.3.1")

# Load LightGBM's binary classification example.
binary_test = readdlm(joinpath(LIGHTGBM_SOURCE, "examples", "binary_classification", "binary.test"), '\t')
binary_train = readdlm(joinpath(LIGHTGBM_SOURCE, "examples", "binary_classification", "binary.train"), '\t')
X_train = binary_train[:, 2:end]
y_train = binary_train[:, 1]
X_test = binary_test[:, 2:end]
y_test = binary_test[:, 1]

# Create an estimator with the desired parameters—leave other parameters at the default values.
estimator = LGBMClassification(
    objective = "binary",
    num_iterations = 100,
    learning_rate = .1,
    early_stopping_round = 5,
    feature_fraction = .8,
    bagging_fraction = .9,
    bagging_freq = 1,
    num_leaves = 1000,
    num_class = 1,
    metric = ["auc", "binary_logloss"]
)

# Fit the estimator on the training data and return its scores for the test data.
fit!(estimator, X_train, y_train, (X_test, y_test))

# Predict arbitrary data with the estimator.
predict(estimator, X_train)

# Cross-validate using a two-fold cross-validation iterable providing training indices.
splits = (collect(1:3500), collect(3501:7000))
cv(estimator, X_train, y_train, splits)

# Exhaustive search on an iterable containing all combinations of learning_rate ∈ {.1, .2} and
# bagging_fraction ∈ {.8, .9}
params = [Dict(:learning_rate => learning_rate,
               :bagging_fraction => bagging_fraction) for
          learning_rate in (.1, .2),
          bagging_fraction in (.8, .9)]
search_cv(estimator, X_train, y_train, splits, params)

# Save and load the fitted model.
filename = pwd() * "/finished.model"
savemodel(estimator, filename)
loadmodel(estimator, filename)
```

# Exports

Note that a lot of parameters used within this module and in the code and examples are
exact matches with those from [LightGBM.](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
Not all of these are necessarily supported but see the guide for detailed explanations of what these
parameters do and their valid values.

## Functions

### `fit!(estimator, X, y[, test...]; [verbosity = 1, is_row_major = false])`
Fit the `estimator` with features data `X` and label `y` using the X-y pairs in `test` as
validation sets.

Return a dictionary with an entry for each validation set. Each entry of the dictionary is another
dictionary with an entry for each validation metric in the `estimator`. Each of these entries is an
array that holds the validation metric's value at each evaluation of the metric.

#### Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `test::Tuple{Matrix{TX},Vector{Ty}}...`: optionally contains one or more tuples of X-y pairs of
    the same types as `X` and `y` that should be used as validation sets.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
* `is_row_major::Bool`: keyword argument that indicates whether or not `X` is row-major. `true`
    indicates that it is row-major, `false` indicates that it is column-major (Julia's default).
* `weights::Vector{Tw<:Real}`: the training weights.
* `init_score::Vector{Ti<:Real}`: the init scores.

### `predict(estimator, X; [predict_type = 0, num_iterations = -1, verbosity = 1, is_row_major = false])`
Return an array with outputs
* Probabilities for binary or multiclass (with output being 2-d if multiclass)
* Regression predictions

### `predict_classes(multiclass_estimator, X; [predict_type = 0, num_iterations = -1, verbosity = 1, is_row_major = false, binary_threshold = 0.5])`
A convenience method for obtaining predicted classes from the `LGBMClassification` estimator.

#### Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `X::Matrix{T<:Real}`: the features data.
* `predict_type::Integer`: keyword argument that controls the prediction type. `0` for normal
    scores with transform (if needed), `1` for raw scores, `2` for leaf indices.
* `num_iterations::Integer`: keyword argument that sets the number of iterations of the model to
    use in the prediction. `< 0` for all iterations.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
* `is_row_major::Bool`: keyword argument that indicates whether or not `X` is row-major. `true`
    indicates that it is row-major, `false` indicates that it is column-major (Julia's default).
* `binary_threshold::Real`: The decision threshold to use for a binary classification
    (when using `binary` objective only, otherwise argmax decision)

### `cv(estimator, X, y, splits; [verbosity = 1])` (Experimental—interface may change)
Cross-validate the `estimator` with features data `X` and label `y`. The iterable `splits` provides
vectors of indices for the training dataset. The remaining indices are used to create the
validation dataset.

Return a dictionary with an entry for the validation dataset and, if the parameter
`is_training_metric` is set in the `estimator`, an entry for the training dataset. Each entry of
the dictionary is another dictionary with an entry for each validation metric in the `estimator`.
Each of these entries is an array that holds the validation metric's value for each dataset, at the
last valid iteration.

#### Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `splits`: the iterable providing arrays of indices for the training dataset.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.

### `search_cv(estimator, X, y, splits, params; [verbosity = 1])` (Experimental—interface may change)
Exhaustive search over the specified sets of parameter values for the `estimator` with features
data `X` and label `y`. The iterable `splits` provides vectors of indices for the training dataset.
The remaining indices are used to create the validation dataset.

Return an array with a tuple for each set of parameters value, where the first entry is a set of
parameter values and the second entry the cross-validation outcome of those values. This outcome is
a dictionary with an entry for the validation dataset and, if the parameter `is_training_metric` is
set in the `estimator`, an entry for the training dataset. Each entry of the dictionary is
another dictionary with an entry for each validation metric in the `estimator`. Each of these
entries is an array that holds the validation metric's value for each dataset, at the last valid
iteration.

#### Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `splits`: the iterable providing arrays of indices for the training dataset.
* `params`: the iterable providing dictionaries of pairs of parameters (Symbols) and values to
    configure the `estimator` with.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.

### `savemodel(estimator, filename; [num_iteration = -1])`
Save the fitted model in `estimator` as `filename`.

#### Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `filename::String`: the name of the file to save the model in.
* `num_iteration::Integer`: keyword argument that sets the number of iterations of the model that
    should be saved. `< 0` for all iterations.

### `loadmodel(estimator, filename)`
Load the fitted model `filename` into `estimator`. Note that this only loads the fitted model—not
the parameters or data of the estimator whose model was saved as `filename`.

#### Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `filename::String`: the name of the file that contains the model.

## Estimators

### `LGBMRegression <: LGBMEstimator`
```julia
LGBMRegression(;
    objective = "regression",
    num_iterations = 10,
    learning_rate = .1,
    num_leaves = 127,
    max_depth = -1,
    tree_learner = "serial",
    num_threads = Sys.CPU_THREADS,
    histogram_pool_size = -1.,
    min_data_in_leaf = 100,
    min_sum_hessian_in_leaf = 10.,
    lambda_l1 = 0.,
    lambda_l2 = 0.,
    min_gain_to_split = 0.,
    feature_fraction = 1.,
    feature_fraction_seed = 2,
    bagging_fraction = 1.,
    bagging_freq = 0,
    bagging_seed = 3,
    early_stopping_round = 0,
    max_bin = 255,
    data_random_seed = 1,
    init_score = "",
    is_sparse = true,
    save_binary = false,
    categorical_feature = Int[],
    is_unbalance = false,
    metric = ["l2"],
    metric_freq = 1,
    is_training_metric = false,
    ndcg_at = Int[],
    num_machines = 1,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_file = "",
    device_type="cpu",
)
```
Return an LGBMRegression estimator.

### `LGBMClassification <: LGBMEstimator`
```julia
LGBMClassification(;
    objective = "multiclass",
    num_iterations = 10,
    learning_rate = .1,
    num_leaves = 127,
    max_depth = -1,
    tree_learner = "serial",
    num_threads = Sys.CPU_THREADS,
    histogram_pool_size = -1.,
    min_data_in_leaf = 100,
    min_sum_hessian_in_leaf = 10.,
    lambda_l1 = 0.,
    lambda_l2 = 0.,
    min_gain_to_split = 0.,
    feature_fraction = 1.,
    feature_fraction_seed = 2,
    bagging_fraction = 1.,
    bagging_freq = 0,
    bagging_seed = 3,
    early_stopping_round = 0,
    max_bin = 255,
    data_random_seed = 1,
    init_score = "",
    is_sparse = true,
    save_binary = false,
    categorical_feature = Int[],
    is_unbalance = false,
    metric = ["multi_logloss"],
    metric_freq = 1,
    is_training_metric = false,
    ndcg_at = Int[],
    num_machines = 1,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_file = "",
    num_class = 2,
    device_type="cpu",
)
```
Return an LGBMClassification estimator.

# MLJ Support

This package has an interface to [MLJ](https://github.com/alan-turing-institute/MLJ.jl).
Exhaustive MLJ documentation is out of scope for here, however the main things are:

The MLJ interface models are
```julia
LightGBM.MLJInterface.LGBMClassifier
LightGBM.MLJInterface.LGBMRegressor
```

And these have the same interface parameters as the [estimators](#estimators)

The interface models are generally passed to `MLJBase.fit` or `MLJBase.machine`
and integrated as part of a larger MLJ pipeline. [An example is provided](https://alan-turing-institute.github.io/MLJTutorials/end-to-end/boston-lgbm/)
