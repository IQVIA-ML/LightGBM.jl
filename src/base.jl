"""
    fit(estimator, X, y[, test...]; [verbosity = 1])

Fit the `estimator` with features data `X` and label `y` using the X-y pairs in `test` as
validation sets.

Return a dictionary with an entry for each validation set. Each entry of the dictionary is another
dictionary with an entry for each validation metric in the `estimator`. Each of these entries is an
array that holds the validation metric's value at each iteration.

# Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `test::Tuple{Matrix{TX},Vector{Ty}}...`: optionally contains one or more tuples of X-y pairs of
    the same types as `X` and `y` that should be used as validation sets.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
"""
function fit{TX<:Real,Ty<:Real}(estimator::LGBMEstimator, X::Matrix{TX}, y::Vector{Ty},
                                test::Tuple{Matrix{TX},Vector{Ty}}...; verbosity::Integer = 1)
    return api_fit(estimator, X, y, test..., verbosity = verbosity)
end

"""
    predict(estimator, X; [predict_type = 0, n_trees = -1, verbosity = 1])

Return an array with the labels that the `estimator` predicts for features data `X`.

# Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `X::Matrix{T<:Real}`: the features data.
* `predict_type::Integer`: keyword argument that controls the prediction type. `0` for normal
    scores with transform (if needed), `1` for raw scores, `2` for leaf indices.
* `n_trees::Integer`: keyword argument that sets the controls the number of trees used in the
    prediction. `< 0` for all available trees.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
"""
function predict{T<:Real}(estimator::LGBMEstimator, X::Matrix{T}; predict_type::Integer = 0,
                          n_trees::Integer = -1, verbosity::Integer = 1)
    return api_predict(estimator, X, predict_type = predict_type, n_trees = n_trees,
                       verbosity = verbosity)
end

"""
    cv(estimator, X, y, cv; [verbosity = 1])

Cross-validate the `estimator` with features data `X` and label `y`. The iterator `cv` provides
vectors of indices for the training dataset. The remaining indices are used to create the
validation dataset.

Return a dictionary with an entry for the validation dataset and an entry for the training dataset,
if the parameter `is_training_metric` is set in the `estimator`. Each entry of the dictionary is
another dictionary with an entry for each validation metric in the `estimator`. Each of these
entries is an array that holds the validation metric's value for each dataset, at the last valid
iteration.

# Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `cv`: the iterator providing arrays of indices for the training dataset.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
"""
function cv{TX<:Real,Ty<:Real}(estimator::LGBMEstimator, X::Matrix{TX}, y::Vector{Ty}, cv;
                               verbosity::Integer = 1)
    api_cv(estimator, X, y, cv; verbosity = verbosity)
end
