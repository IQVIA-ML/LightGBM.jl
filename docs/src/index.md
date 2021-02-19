LightGBM.jl
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
- [LightGBM.jl](#lightgbmjl)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [A simple example using LightGBM example files](#a-simple-example-using-lightgbm-example-files)
- [Parameters](#parameters)
- [MLJ Support](#mlj-support)

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

# Parameters

Note that a lot of parameters used within this module and in the code and examples are
exact matches with those from [LightGBM.](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
Not all of these are necessarily supported but see the guide for detailed explanations of what these
parameters do and their valid values.

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

# Custom LightGBM binaries
Though this package comes with a precompiled binary (`lib_lightgbm.so` for linux, `lib_lightgbm.dylib` for macos, `lib_lightgbm.dll` for windows, refer to [Microsoft's LightGBM release page](https://github.com/microsoft/LightGBM/releases)), a custom binary can be used with this package (we use `Libdl.dlopen` to do this). In order to do so, either:

  - Add the directory of your custom binary to the `Libdl.DL_LOAD_PATH` before calling `import LightGBM`, e.g.
      ```
      import Libdl
      push!(Libdl.DL_LOAD_PATH, "/path/to/your/lib_lightgbm/directory")

      import LightGBM
      ...
      ```
  - Specify the directory of your custom binary in the environment variables `LD_LIBRARY_PATH` (for linux), `DYLD_LIBRARY_PATH` (macos), `PATH` (windows), or place the custom binary file in the system search path

Note: `Libdl.DL_LOAD_PATH` will be first searched and used, then the system library paths. If no binaries are found, the program will fallback to using the precompiled binary

## Contributors ✨

The list of our Contributors can be found [here](CONTRIBUTORS.md).
Please don't hesitate to add yourself when you contribute.
