"""
    cv(estimator, X, y, splits; [verbosity = 1])

Cross-validate the `estimator` with features data `X` and label `y`. The iterable `splits` provides
vectors of indices for the training dataset. The remaining indices are used to create the
validation dataset.

Return a dictionary with an entry for the validation dataset and, if the parameter
`is_training_metric` is set in the `estimator`, an entry for the training dataset. Each entry of
the dictionary is another dictionary with an entry for each validation metric in the `estimator`.
Each of these entries is an array that holds the validation metric's value for each dataset, at the
last valid iteration.

# Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `splits`: the iterable providing arrays of indices for the training dataset.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
"""
function cv{TX<:Real,Ty<:Real}(estimator::LGBMEstimator, X::Matrix{TX}, y::Vector{Ty}, splits;
                               verbosity::Integer = 1)
    start_time = now()
    num_data = size(X)[1]
    ds_parameters = stringifyparams(estimator, datasetparams)
    bst_parameters = stringifyparams(estimator, boosterparams) * " verbosity=$verbosity"

    full_ds = LGBM_DatasetCreateFromMat(X, ds_parameters)
    LGBM_DatasetSetField(full_ds, "label", y)

    split_scores = Dict{String,Dict{String,Vector{Float64}}}()
    for (split_idx, train_inds) in enumerate(splits)
        log_info(verbosity, "\nCross-validation: ", split_idx, "\n")

        log_debug(verbosity, "Started creating LGBM training dataset ", split_idx, "\n")
        train_ds = LGBM_DatasetGetSubset(full_ds, train_inds, ds_parameters)

        log_debug(verbosity, "Started creating LGBM test dataset ", split_idx, "\n")
        test_inds = setdiff(1:num_data, train_inds)
        test_ds = LGBM_DatasetGetSubset(full_ds, test_inds, ds_parameters)

        log_debug(verbosity, "Started creating LGBM booster ", split_idx, "\n")
        estimator.booster = LGBM_BoosterCreate(train_ds, bst_parameters)
        LGBM_BoosterAddValidData(estimator.booster, test_ds)

        results = train(estimator, ["validation"], verbosity, start_time)

        for dataset in keys(results)
            dataset_results = results[dataset]
            if !haskey(split_scores, dataset)
                split_scores[dataset] = Dict{String,Vector{Float64}}()
                for metric in keys(dataset_results)
                    split_scores[dataset][metric] = [dataset_results[metric][end]]
                end
            else
                for metric in keys(dataset_results)
                    push!(split_scores[dataset][metric], dataset_results[metric][end])
                end
            end
        end
    end

    log_info(verbosity, "\nCross-validation finished\n")
    cv_logsummary(split_scores, verbosity)

    return split_scores
end

function cv_logsummary(cv_results::Dict{String,Dict{String,Vector{Float64}}}, verbosity::Integer)
    for dataset in keys(cv_results)
        for metric in keys(cv_results[dataset])
            log_info(verbosity, "- ", dataset, "'s ", metric,
                     " mean: ", mean(cv_results[dataset][metric]),
                     ", std: ", std(cv_results[dataset][metric]), "\n")
        end
    end
end
