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
* Integration with [MLJ](https://github.com/alan-turing-institute/MLJ.jl) (which also provides the above via different interfaces)

Additionally, the package automatically converts all LightGBM parameters that refer to indices
(e.g. `categorical_feature`) from Julia's one-based indices to C's zero-based indices.

A majority of the C-interfaces are implemented. A few are known to be missing and are
[tracked.](https://github.com/IQVIA-ML/LightGBM.jl/issues)

All major operating systems (Windows, Linux, and Mac OS X) are supported. Julia versions 1.0+ are supported.

# Table of Contents
1. [Installation](#installation)
1. [Example](#a-simple-example-using-lightgbm-example-files)
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
loadmodel!(estimator, filename)
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
and integrated as part of a larger MLJ pipeline. [An example is provided](https://alan-turing-institute.github.io/DataScienceTutorials.jl/end-to-end/boston-lgbm/)

## Contributors ✨

The list of our Contributors can be found [here](CONTRIBUTORS.md).
Please don't hesitate to add yourself when you contribute.
