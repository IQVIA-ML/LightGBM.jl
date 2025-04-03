# MLJ Compliant Documentation Guidelines:
# - Follow the MLJModelInterface document string standard:
#   https://juliaai.github.io/MLJModelInterface.jl/dev/document_strings/#The-document-string-standard
# - Use Markdown formatting for clarity and consistency.
# - Maintain the existing structure for docstrings, including sections like
#   "Training data", "Operations", "Hyper-parameters", "Fitted parameters", and "Report".
# - Ensure examples are runnable and include necessary imports.
# - Use proper indentation and alignment for lists and code blocks.
# - Keep descriptions concise but informative, avoiding redundancy.

const LGBM_PARAMS_DOCS_LINK = "https://lightgbm.readthedocs.io/en/v3.3.5/Parameters.html"

"""
$(MLJModelInterface.doc_header(LGBMRegressor))

LightGBM, short for light gradient-boosting machine, is a
framework for gradient boosting based on decision tree algorithms and used for
classification, regression and other machine learning tasks, with a focus on
performance and scalability. This model in particular is used for various types of
regression tasks.

# Training data 

In MLJ or MLJBase, bind an instance `model` to data with 

  mach = machine(model, X, y) 

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of
  scitype `Continuous`; check the column scitypes with `schema(X)`; alternatively,
  `X` is any `AbstractMatrix` with `Continuous` elements; check the scitype with
  `scitype(X)`.
- y is a vector of targets whose items are of scitype `Continuous`. Check the
  scitype with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features
  `Xnew`, which should have the same scitype as `X` above.

# Hyper-parameters
See $LGBM_PARAMS_DOCS_LINK.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `fitresult`: Fitted model information, contains a `LGBMRegression` object, an
  empty vector, and the regressor with all its parameters

# Report

The fields of `report(mach)` are:

- `training_metrics`: A dictionary containing all training metrics.
- `importance`: A `namedtuple` containing:
    - `gain`: The total gain of each split used by the model
    - `split`: The number of times each feature is used by the model.

# Examples

```julia

using DataFrames
using MLJ

# load the model
LGBMRegressor = @load LGBMRegressor pkg=LightGBM 

X, y = @load_boston # a table and a vector 
X = DataFrame(X)
train, test = partition(collect(eachindex(y)), 0.70, shuffle=true)

first(X, 3)
lgb = LGBMRegressor() # initialise a model with default params
mach = machine(lgb, X[train, :], y[train]) |> fit!

predict(mach, X[test, :])

# access feature importances
model_report = report(mach)
gain_importance = model_report.importance.gain
split_importance = model_report.importance.split
```

See also
[LightGBM.jl](https://github.com/IQVIA-ML/LightGBM.jl) and
the unwrapped model type
[`LightGBM.LGBMRegression`](@ref)
"""
LGBMRegressor


"""
$(MLJModelInterface.doc_header(LGBMClassifier))

`LightGBM, short for light gradient-boosting machine, is a
framework for gradient boosting based on decision tree algorithms and used for
classification and other machine learning tasks, with a focus on
performance and scalability. This model in particular is used for various types of
classification tasks.

# Training data In MLJ or MLJBase, bind an instance `model` to data with 

  mach = machine(model, X, y) 

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of
  scitype `Continuous`; check the column scitypes with `schema(X)`; alternatively,
  `X` is any `AbstractMatrix` with `Continuous` elements; check the scitype with
  `scitype(X)`.
- y is a vector of targets whose items are of scitype `Continuous`. Check the
  scitype with scitype(y).

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features
  `Xnew`, which should have the same scitype as `X` above.

# Hyper-parameters
See $LGBM_PARAMS_DOCS_LINK.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `fitresult`: Fitted model information, contains a `LGBMClassification` object, a
  `CategoricalArray` of the input class names, and the classifier with all its
  parameters

# Report

The fields of `report(mach)` are:

- `training_metrics`: A dictionary containing all training metrics.
- `importance`: A `namedtuple` containing:
    - `gain`: The total gain of each split used by the model
    - `split`: The number of times each feature is used by the model.


# Examples

```julia

using DataFrames
using MLJ

# load the model
LGBMClassifier = @load LGBMClassifier pkg=LightGBM 

X, y = @load_iris 
X = DataFrame(X)
train, test = partition(collect(eachindex(y)), 0.70, shuffle=true)

first(X, 3)
lgb = LGBMClassifier() # initialise a model with default params
mach = machine(lgb, X[train, :], y[train]) |> fit!

predict(mach, X[test, :])

# access feature importances
model_report = report(mach)
gain_importance = model_report.importance.gain
split_importance = model_report.importance.split
```

See also
[LightGBM.jl](https://github.com/IQVIA-ML/LightGBM.jl) and
the unwrapped model type
[`LightGBM.LGBMClassification`](@ref)

"""
LGBMClassifier