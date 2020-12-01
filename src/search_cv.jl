"""
    search_cv(estimator, X, y, splits, params; [verbosity = 1])

Exhaustive search over the specified sets of parameter values for the `estimator` with features
data `X` and label `y`. The iterable `splits` provides vectors of indices for the training dataset.
The remaining indices are used to create the validation dataset.
Alternatively, search_cv can be called with an input Dataset class


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
* `dataset::Dataset`: prepared dataset (either (X, y), or dataset needs to be specified as input)
* `splits`: the iterable providing arrays of indices for the training dataset.
* `params`: the iterable providing dictionaries of pairs of parameters (Symbols) and values to
    configure the `estimator` with.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
"""
function search_cv(estimator::LGBMEstimator, X::Matrix{TX}, y::Vector{Ty},
                                      splits, params; verbosity::Integer = 1) where {TX<:Real,Ty<:Real}

    ds_parameters = stringifyparams(estimator, DATASETPARAMS)
    full_ds = LGBM_DatasetCreateFromMat(X, ds_parameters)
    LGBM_DatasetSetField(full_ds, "label", y)
                                                                    
    return search_cv(estimator, full_ds, splits, params, verbosity = verbosity)
end

# Pass Dataset class directly. This will speed up the process if it is part of an iterative process and a pre-constructed dataset is available
function search_cv(
    estimator::LGBMEstimator, dataset::Dataset, splits, params; 
    verbosity::Integer = 1
)
    n_params = length(params)
    results = Array{Tuple{Dict{Symbol,Any},Dict{String,Dict{String,Vector{Float64}}}}}(undef,n_params)
    for (search_idx, search_params) in enumerate(params)
        log_info(verbosity, "\nSearch: ", search_idx, "\n", search_params, "\n")

        search_estimator = deepcopy(estimator)
        foreach(param -> setfield!(search_estimator, param[1], param[2]), search_params)
        search_results = cv(search_estimator, dataset, deepcopy(splits),
        verbosity = ifelse(verbosity == 1, 0, verbosity))

        results[search_idx] = (search_params, search_results)
        cv_logsummary(search_results, verbosity)
    end
    log_info(verbosity, "\nSearch finished\n")

    return results
end
