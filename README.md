This package was originally authored by [Allardvm](https://github.com/Allardvm) and [wakakusa](https://github.com/wakakusa/)

LightGBM.jl
![CI](https://github.com/IQVIA-ML/LightGBM.jl/workflows/CI/badge.svg)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://IQVIA-ML.github.io/LightGBM.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://IQVIA-ML.github.io/LightGBM.jl/dev)
========

**LightGBM.jl** provides a high-performance Julia interface for Microsoft's
[LightGBM](https://lightgbm.readthedocs.io/en/latest/).

The package adds a couple of convenience features:
* Automated cross-validation
* Exhaustive grid search search procedure
* Integration with [MLJ](https://github.com/alan-turing-institute/MLJ.jl), which also provides the above via different interfaces (verified only on Julia 1.6+)

Additionally, the package automatically converts all LightGBM parameters that refer to indices
(e.g. `categorical_feature`) from Julia's one-based indices to C's zero-based indices.

A majority of the C-interfaces are implemented. A few are known to be missing and are
[tracked.](https://github.com/IQVIA-ML/LightGBM.jl/issues)

All major operating systems (Windows, Linux, and Mac OS X) are supported. Julia versions 1.6+ are supported.

# Table of Contents
1. [Installation](#installation)
1. [Example](#a-simple-example-using-lightgbm-example-files)
1. [MLJ](#mlj-support)

# Installation

To add the package to Julia:
```julia
Pkg.add("LightGBM")
```
This package uses [LightGBM_jll](https://github.com/JuliaBinaryWrappers/LightGBM_jll.jl) to package `lightgbm` binaries
so it works out-of-the-box.
## Tests
Running tests for the package requires the use of the LightGBM example files,
download and extract the [LightGBM source](https://github.com/microsoft/LightGBM/archive/v3.3.5.zip)
and set the environment variable `LIGHTGBM_EXAMPLES_PATH` to the root of the source installation.
Then you can run the tests by simply doing
```julia
Pkg.test("LightGBM")
```

To skip MLJ testing when running tests, set the env var `DISABLE_MLJ_TESTS` to anything. (You might want to do this to get the tests to run faster)

# A simple example using LightGBM example files

First, download [LightGBM source](https://github.com/microsoft/LightGBM/archive/v3.3.5.zip)
and untar it somewhere.

```bash
cd ~
wget https://github.com/microsoft/LightGBM/archive/v3.3.5.tar.gz
tar -xf v3.3.5.tar.gz
```

```julia
using LightGBM
using DelimitedFiles

LIGHTGBM_SOURCE = abspath("~/LightGBM-3.3.5")

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
loadmodel!(estimator, filename)
```
# LGBM Ranking Support

LightGBM.jl core includes a separate estimator `LGBMRanking` with parameters suitable for ranking applications as described in [group query](https://lightgbm.readthedocs.io/en/v3.3.5/Parameters.html#query-data). Similar to other
wrapper libraries it is possible to pass a one-dimensional array with `group` information parameter.

Here's an example of how to use `LGBMRanking`:


```julia
using LightGBM

# Create X_train Matrix
X_train = [
    0.3 0.6 0.9;
    0.1 0.4 0.7;
    0.5 0.8 1.1;
    0.3 0.6 0.9;
    0.7 1.0 1.3;
    0.2 0.5 0.8;
    0.1 0.4 0.7;
    0.4 0.7 1.0;
]

# Create X_test Matrix
X_test = [
    0.6 0.9 1.2;
    0.2 0.5 0.8;
]

# Create y_train and y_test arrays
y_train = [0, 0, 0, 0, 1, 0, 1, 1]
y_test = [0, 1]

# Create group_train and group_test arrays
group_train = [2, 2, 4]
group_test = [1, 1]

# Create ranker model
ranker = LightGBM.LGBMRanking(
    num_class = 1,
    objective = "lambdarank",
    metric = ["ndcg"],
    eval_at = [1, 3, 5, 10],
    learning_rate = 0.1,
    num_leaves = 31,
    min_data_in_leaf = 1,
)

# Fit the model
LightGBM.fit!(ranker, X_train, Vector(y_train), group = group_train)

# Predict the relevance scores for the test set
y_pred = LightGBM.predict(ranker, X_test)
   ```

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
and integrated as part of a larger MLJ pipeline. [An example is provided](https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-lgbm/)

MLJ Is only officially supported on 1.6+ (because this is what MLJ supports). Using older versions of the MLJ package may work, but your mileage may vary.

# Custom LightGBM binaries

This package uses [LightGBM_jll](https://github.com/JuliaBinaryWrappers/LightGBM_jll.jl) to package `lightgbm` binaries.
JLL packages use the [Artifacts system](https://pkgdocs.julialang.org/v1/artifacts/) to provide the files.
If you would like to override the existing files with your own binaries, you can follow the [overriding the artifacts](https://docs.binarybuilder.org/stable/jll/#Overriding-the-artifacts-in-JLL-packages) guidance.

## Contributors ✨

Please don't hesitate to add yourself when you contribute to CONTRIBUTORS.md.
