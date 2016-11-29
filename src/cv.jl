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
    start_time = now()
    n_data = size(X)[1]
    ds_parameters = getparamstring(estimator, datasetparams)
    bst_parameters = getparamstring(estimator, boosterparams) * " verbosity=$verbosity"

    split_scores = Dict{String,Dict{String,Vector{Float64}}}()
    for (split_idx, train_inds) in enumerate(cv)
        log_info(verbosity, "\nCross-validation: ", split_idx, ", starting\n")

        log_debug(verbosity, "Started creating LGBM training dataset ", split_idx, "\n")
        train_X = X[train_inds, :]
        train_y = y[train_inds]
        train_ds = LGBM_CreateDatasetFromMat(train_X, ds_parameters)
        LGBM_DatasetSetField(train_ds, "label", train_y)

        log_debug(verbosity, "Started creating LGBM test dataset ", split_idx, "\n")
        test_inds = setdiff(1:n_data, train_inds)
        test_X = X[test_inds, :]
        test_y = y[test_inds]
        test_ds = LGBM_CreateDatasetFromMat(test_X, ds_parameters, train_ds)
        LGBM_DatasetSetField(test_ds, "label", test_y)

        log_debug(verbosity, "Started creating LGBM booster ", split_idx, "\n")
        estimator.booster = LGBM_BoosterCreate(train_ds, [test_ds], ["validation"], bst_parameters)

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
    for dataset in keys(split_scores)
        for metric in keys(split_scores[dataset])
            log_info(verbosity, "- ", dataset, "'s ", metric,
                     " mean: ", mean(split_scores[dataset][metric]),
                     ", std: ", std(split_scores[dataset][metric]), "\n")
        end
    end

    return split_scores
end
