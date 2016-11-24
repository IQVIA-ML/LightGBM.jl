"""
    fit(estimator, X, y[, test...])

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
"""
function fit{TX<:Real,Ty<:Real}(estimator::LGBMEstimator, X::Matrix{TX}, y::Vector{Ty},
                                test::Tuple{Matrix{TX},Vector{Ty}}...; verbosity::Integer = 1)
    return api_fit(estimator, X, y, test..., verbosity = verbosity)
end

"""
    predict(estimator, X)

Return an array with the labels that the `estimator` predicts for features data `X`.

# Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `X::Matrix{T<:Real}`: the features data.
"""
function predict{T<:Real}(estimator::LGBMEstimator, X::Matrix{T}; predict_type::Integer = 0,
                          n_trees::Integer = -1, verbosity::Integer = 1)
    return api_predict(estimator, X, predict_type = predict_type, n_trees = n_trees,
                       verbosity = verbosity)
end
