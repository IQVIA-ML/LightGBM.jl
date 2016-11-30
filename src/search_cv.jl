"""
    search_cv(estimator, X, y, splits, params; [verbosity = 1])

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

# Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `splits`: the iterable providing arrays of indices for the training dataset.
* `params`: the iterable providing dictionaries of pairs of parameters (Symbols) and values to
    configure the `estimator` with.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
"""
function search_cv{TX<:Real,Ty<:Real}(estimator::LGBMEstimator, X::Matrix{TX}, y::Vector{Ty},
                                      splits, params; verbosity::Integer = 1)
    n_params = length(params)
    search_results = Array(Tuple{Dict{Symbol,ANY},Dict{String,Dict{String,Vector{Float64}}}}, n_params)
    for (param_idx, param) in enumerate(params)
        log_info(verbosity, "\nSearch: ", param_idx, "\n", param, "\n")
        search_estimator = deepcopy(estimator)
        foreach(param -> setfield!(search_estimator, param[1], param[2]), param)
        search_results[param_idx] = (param, cv(search_estimator, X, y, deepcopy(splits),
                                                    verbosity = verbosity))
    end

    log_info(verbosity, "\nSearch finished\n")
    for (param, results) in search_results
        log_info(verbosity, param, "\n")
        for dataset in keys(results)
            for metric in keys(results[dataset])
                log_info(verbosity, "- ", dataset, "'s ", metric,
                         " mean: ", mean(results[dataset][metric]),
                         ", std: ", std(results[dataset][metric]), "\n")
            end
        end
    end

    return search_results
end
