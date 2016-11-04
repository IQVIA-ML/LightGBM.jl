"""
    fit(estimator, X, y[, test...])

Fit the `estimator` with features data `X` and label `y` using the X-y pairs in `test` as
validation sets.

Return a dictionary with an entry for each validation set. Each entry of the dictionary is another
dictionary with an entry for each validation metric in the `estimator`. Each of these entries is an
array that holds the validation metric's value at each iteration.

# Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Array{TX<:Real,2}`: the features data.
* `y::Array{Ty<:Real,1}`: the labels.
* `test::Tuple{Array{TX,2},Array{Ty,1}}...`: optionally contains one or more tuples of X-y pairs of
    the same types as `X` and `y` that should be used as validation sets.
"""
function fit{TX<:Real,Ty<:Real}(estimator::LGBMEstimator, X::Array{TX,2}, y::Array{Ty,1},
                                test::Tuple{Array{TX,2},Array{Ty,1}}...; verbosity::Integer = 1)
    return cli_fit(estimator, X, y, test..., verbosity = verbosity)
end

"""
    predict(estimator, X)

Return an array with the labels that the `estimator` predicts for features data `X`.

# Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `X::Array{T<:Real,2}`: the features data.
"""
function predict{T<:Real}(estimator::LGBMEstimator, X::Array{T,2}; verbosity::Integer = 1)
    return cli_predict(estimator, X, verbosity = verbosity)
end

function storeresults!(results, estimator::LGBMEstimator, iter::Integer, test::String,
                       metric::String, score::Float64)
    if !haskey(results, test)
        results[test] = Dict{String,Array{Float64,1}}()
        results[test][metric] = Array(Float64, estimator.num_iterations)
    elseif !haskey(results[test], metric)
        results[test][metric] = Array(Float64, estimator.num_iterations)
    end
    results[test][metric][iter] = score

    return nothing
end

function shrinkresults!(results, last_retained_iter::Integer)
    for test_key in keys(results)
        test = results[test_key]
        for metric_key in keys(test)
            test[metric_key] = test[metric_key][1:last_retained_iter]
        end
    end

    return nothing
end
